# /src/fine_tuner_all.py
# 多模态青铜器年代分类微调脚本
# 基于预训练BLIP模型，添加跨模态注意力和分类头
# 支持冻结或微调预训练模型参数

import json
import time  
import numpy as np
from PIL import Image
from transformers import BlipModel, BlipConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BlipProcessor
from torch.optim.lr_scheduler import MultiStepLR
import os
import torchvision.transforms as transforms
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

# --------------------- 模型参数量统计工具函数 ---------------------
def count_parameters(model):
    """统计模型总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_params(num):
    """将参数量格式化为易读形式（如1.2B、560M、89K）"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return f"{num}"

# --------------------- 权重调度器 ---------------------
class AdaptiveWeightScheduler:
    def __init__(self, cls_start=0.85, cls_mid=0.65, cls_end=0.75,
                 align_start=0.15, align_mid=0.35, align_end=0.25,
                 mid_epoch=30, end_epoch=70):
        self.cls_start = cls_start
        self.cls_mid = cls_mid
        self.cls_end = cls_end
        self.align_start = align_start
        self.align_mid = align_mid
        self.align_end = align_end
        self.mid_epoch = mid_epoch
        self.end_epoch = end_epoch

    def get_weights(self, epoch):
        if epoch < self.mid_epoch:
            return self.cls_start, self.align_start
        elif epoch < self.end_epoch:
            return self.cls_mid, self.align_mid
        else:
            return self.cls_end, self.align_end


# --------------------- 对比损失 ---------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text_embeddings, image_embeddings):
        text_embeddings = nn.functional.normalize(text_embeddings, p=2, dim=1)
        image_embeddings = nn.functional.normalize(image_embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(text_embeddings, image_embeddings.T) / self.temperature
        batch_size = text_embeddings.size(0)
        labels_pos = torch.arange(batch_size).to(text_embeddings.device)
        loss = (self.criterion(sim_matrix, labels_pos) + self.criterion(sim_matrix.T, labels_pos)) / 2
        return loss


# --------------------- 交叉注意力模块 ---------------------
class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.text_self = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.image_self = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.text_cross = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.image_cross = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.text_norm1 = nn.LayerNorm(embed_dim)
        self.text_norm2 = nn.LayerNorm(embed_dim)
        self.image_norm1 = nn.LayerNorm(embed_dim)
        self.image_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_cross_attn_weights = None  # 保存跨模态注意力权重（多head）

    def forward(self, text_feat, image_feat):
        # text_feat: [B, seq_len_txt, dim]; image_feat: [B, seq_len_img, dim]
        
        # 1. 文本自注意力（text→text）
        t2, _ = self.text_self(
            text_feat.transpose(0, 1),  # [seq_len_txt, B, dim]
            text_feat.transpose(0, 1),
            text_feat.transpose(0, 1)
        )
        text_feat = self.text_norm1(text_feat + self.dropout(t2.transpose(0, 1)))

        # 2. 图像自注意力（image→image）
        i2, _ = self.image_self(
            image_feat.transpose(0, 1),  # [seq_len_img, B, dim]
            image_feat.transpose(0, 1),
            image_feat.transpose(0, 1)
        )
        image_feat = self.image_norm1(image_feat + self.dropout(i2.transpose(0, 1)))

        # 3. 文本-图像交叉注意力（text→image）
        t2, _ = self.text_cross(
            text_feat.transpose(0, 1),   # query: 文本序列
            image_feat.transpose(0, 1),  # key/value: 图像序列
            image_feat.transpose(0, 1)
        )
        text_feat = self.text_norm2(text_feat + self.dropout(t2.transpose(0, 1)))

        # 4. 图像-文本交叉注意力（image→text，保存跨模态注意力权重）
        i2, attn_weights = self.image_cross(
            image_feat.transpose(0, 1),  # query: 图像序列 [seq_len_img, B, dim]
            text_feat.transpose(0, 1),   # key: 文本序列 [seq_len_txt, B, dim]
            text_feat.transpose(0, 1),   # value: 文本序列 [seq_len_txt, B, dim]
            need_weights=True,           # 必须开启，才能获取注意力权重
            average_attn_weights=False   # 不平均head，保留所有head的权重
        )
        # 保存交叉注意力权重（shape: [B, num_heads, seq_len_img, seq_len_txt]）
        self.last_cross_attn_weights = attn_weights.detach() if attn_weights is not None else None

        image_feat = self.image_norm2(image_feat + self.dropout(i2.transpose(0, 1)))

        return text_feat, image_feat


# --------------------- 数据集 ---------------------
class MultimodalEraDataset(Dataset):
    def __init__(self, jsonl_path, era_labels_npy_path, processor, max_len=512):
        raw_labels = np.load(era_labels_npy_path)
        if raw_labels.dtype.type is np.str_ or raw_labels.dtype.type is np.object_:
            uniq = sorted(set(raw_labels.tolist()))
            self.era_to_id = {lbl: i for i, lbl in enumerate(uniq)}
            self.labels = np.array([self.era_to_id[lbl] for lbl in raw_labels])
        else:
            self.labels = raw_labels
            self.era_to_id = {v: v for v in np.unique(raw_labels)}

        self.num_classes = len(self.era_to_id)
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                text_fields = [
                    f"文物名：{obj.get('文物名', '未知')}",
                    f"分类：{obj.get('分类', '未知')}",
                    f"图案与纹样：{obj.get('图案与纹样', '无')}",
                    f"物件类型：{obj.get('物件类型', '未知')}"
                ]
                combined_text = "；".join(text_fields)
                image_path = obj["图片url"]
                label_id = self.labels[i]
                self.samples.append((image_path, combined_text, label_id))

        self.processor = processor
        self.max_len = max_len
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.15, 0.1)
        ])
        self.val_transform = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, text, label = self.samples[idx]
        image_path = Path(str(image_path).replace('\\', '/')).name
        image_path = Path('/workspace/gugong/data/raw/images') / image_path
        image = Image.open(image_path).convert("RGB")
        image = self.train_transform(image)
        inputs = self.processor(images=image, text=text, return_tensors="pt",
                                padding="max_length", truncation=True, max_length=self.max_len)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(label, dtype=torch.long)
        return inputs


# --------------------- 模型 ---------------------
class FineTuner(nn.Module):
    def __init__(self, model_name, num_classes, freeze_base=False, num_attention_heads=8):
        super().__init__()
        if model_name is not None:
            self.backbone = BlipModel.from_pretrained(model_name)
        else:
            self.backbone = BlipModel(BlipConfig())

        if freeze_base:
            for p in self.backbone.parameters():
                p.requires_grad = False

        img_dim = self.backbone.vision_model.config.hidden_size
        txt_dim = self.backbone.text_model.config.hidden_size
        self.text_proj = nn.Linear(txt_dim, img_dim) if txt_dim != img_dim else nn.Identity()

        self.cross_attn = CrossModalAttention(img_dim, num_attention_heads)
        self.classifier = nn.Sequential(
            nn.Linear(img_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

        self.last_self_attn = None   # 视觉自注意力权重（来自backbone）
        self.last_cross_attn = None  # 跨模态注意力权重（来自CrossModalAttention）

    def forward(self, pixel_values, input_ids, attention_mask):
        # 1. BLIP backbone前向传播（输出含注意力）
        out = self.backbone(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,  # 必须开启，才能获取自注意力
            return_dict=True
        )

        # 2. 保存视觉自注意力（取最后一层，做热力图用）
        vision_outputs = out.vision_model_output
        if vision_outputs is not None and hasattr(vision_outputs, "attentions") and vision_outputs.attentions:
            # vision_outputs.attentions 是列表，每个元素shape: [B, num_heads, seq_len, seq_len]
            self.last_self_attn = vision_outputs.attentions[-1].detach()  # 取最后一层自注意力
        else:
            self.last_self_attn = None

        # 3. 提取图像、文本序列特征
        img_seq = vision_outputs.last_hidden_state     # [B, N_patch+1, dim]
        txt_seq = self.text_proj(out.text_model_output.last_hidden_state)  # [B, L_text, dim]

        # 4. 交叉注意力计算（同时保存跨模态注意力）
        txt_feat, img_feat = self.cross_attn(txt_seq, img_seq)
        self.last_cross_attn = self.cross_attn.last_cross_attn_weights  # 从CrossModalAttention取跨模态权重

        # 5. 取[CLS] token特征，用于分类
        img_feat_cls = img_feat[:, 0]
        txt_feat_cls = txt_feat[:, 0]

        self.image_embeddings = img_feat_cls
        self.text_embeddings = txt_feat_cls

        # 6. 拼接特征并分类
        concat = torch.cat([img_feat_cls, txt_feat_cls], dim=1)
        logits = self.classifier(concat)
        return logits


# --------------------- 训练函数 ---------------------
def train(model, dataset, val_ratio=0.2, epochs=30, lr=2e-5, batch_size=16, device="cuda", logger=None):
    # 统计并输出模型参数量（模型初始化后立即执行）
    total_params, trainable_params = count_parameters(model)
    log = print if logger is None else logger.info
    log("="*50)
    log("模型参数量统计：")
    log(f"总参数量：{format_params(total_params)}（{total_params:,} 个）")
    log(f"可训练参数量：{format_params(trainable_params)}（{trainable_params:,} 个）")
    log(f"可训练参数比例：{trainable_params/total_params:.2%}")


    # print era_to_id 
    log("="*30)
    log("年代到ID的映射（era_to_id）：")
    for era, idx in dataset.era_to_id.items():
        log(f"  {era}: {idx}")
    log("="*30)

    # 划分训练集和验证集
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-5)
    scheduler = MultiStepLR(optimizer, milestones=[40, 70], gamma=0.4)
    cls_criterion = nn.CrossEntropyLoss()
    align_criterion = ContrastiveLoss(temperature=0.1)
    weight_scheduler = AdaptiveWeightScheduler()

    best_val_acc = 0.0

    # 记录总训练开始时间
    total_train_start = time.time()

    for epoch in range(epochs):
        # 记录当前epoch开始时间
        epoch_start = time.time()
        model.train()
        cls_w, align_w = weight_scheduler.get_weights(epoch)
        total_loss = total_cls = total_align = correct = total = 0

        # 记录当前epoch训练阶段开始时间
        train_phase_start = time.time()

        # 训练循环
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])
            cls_loss = cls_criterion(logits, inputs["labels"])
            align_loss = align_criterion(model.text_embeddings, model.image_embeddings)
            loss = cls_w * cls_loss + align_w * align_loss
            loss.backward()
            optimizer.step()

            batch_size = inputs["labels"].size(0)
            total += batch_size
            total_loss += loss.item() * batch_size
            total_cls += cls_loss.item() * batch_size
            total_align += align_loss.item() * batch_size
            correct += logits.argmax(1).eq(inputs["labels"]).sum().item()

        # 计算当前epoch训练阶段耗时
        train_phase_time = time.time() - train_phase_start
        train_acc = correct / total

        # 记录当前epoch验证阶段开始时间
        val_phase_start = time.time()

        # 验证循环
        model.eval()
        val_correct = val_total = 0
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                logits = model(inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])
                val_total += inputs["labels"].size(0)
                val_correct += logits.argmax(1).eq(inputs["labels"]).sum().item()
                all_true.extend(inputs["labels"].cpu().numpy())
                all_pred.extend(logits.argmax(1).cpu().numpy())

        # 计算当前epoch验证阶段耗时和总epoch耗时
        val_phase_time = time.time() - val_phase_start
        epoch_total_time = time.time() - epoch_start
        val_acc = val_correct / val_total
        precision = precision_score(all_true, all_pred, average="weighted", zero_division=0)
        recall = recall_score(all_true, all_pred, average="weighted", zero_division=0)
        f1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)

        # 日志输出时添加时间统计
        log(f"[Epoch {epoch+1}/{epochs}] | 总耗时：{epoch_total_time:.2f}s（训练：{train_phase_time:.2f}s / 验证：{val_phase_time:.2f}s）")
        log(f"          训练：Loss={total_loss/total:.4f} | ClsLoss={total_cls/total:.4f} | AlignLoss={total_align/total:.4f} | Acc={train_acc:.4f}")
        log(f"          验证：Acc={val_acc:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}")

        # 最佳模型保存逻辑
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = "/workspace/gugong/checkpoints/blip_era_finetuned_unfreeze_best.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            log(f"✅ Best model saved (acc={val_acc:.4f})")

        scheduler.step()
        torch.cuda.empty_cache()

    # 计算并输出总训练时间
    total_train_time = time.time() - total_train_start
    total_h = int(total_train_time // 3600)
    total_m = int((total_train_time % 3600) // 60)
    total_s = total_train_time % 60
    log("="*50)
    log(f"训练完成！总训练时间：{total_h}h{total_m}m{total_s:.2f}s")
    log(f"最佳验证准确率：{best_val_acc:.4f}")
    log("="*50)

    # 模型加载逻辑
    model.load_state_dict(torch.load("/workspace/gugong/checkpoints/blip_era_finetuned_unfreeze_best.pt"))
    return model