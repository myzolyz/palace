import matplotlib
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.nn.functional import interpolate
from typing import Tuple, Dict
from fine_tuner_all import FineTuner, CrossModalAttention, ContrastiveLoss
from torchvision import transforms
from transformers import BlipProcessor
import matplotlib.cm as cm
from torch.nn.functional import interpolate
import os
import cv2

def load_trained_model(
     model_path: str,
     num_classes: int = None,
     era_to_id: Dict = None,
     device: str = "cuda"
) -> Tuple[FineTuner, Dict]:
     if num_classes is None and era_to_id is not None:
          num_classes = len(era_to_id)
     else:
          raise ValueError("需提供num_classes或era_to_id以确定类别数")
     
     model = FineTuner(
          model_name=None,
          num_classes=num_classes,
          freeze_base=False,
          num_attention_heads=8
     )
     
     checkpoint = torch.load(model_path, map_location=device)
     model.load_state_dict(checkpoint)
     
     model.to(device)
     model.eval()
     
     if not hasattr(model, "processor") or model.processor is None:
          print("[Info] Model missing processor, loading from local path.")
          model.processor = BlipProcessor.from_pretrained("/workspace/MODEL/models--Salesforce--blip-image-captioning-base/snapshots/blip-image-captioning-base")

     id_to_era = {v: k for k, v in era_to_id.items()}
     return model, id_to_era


def preprocess_input(
     image_path: str,
     text: str,
     processor: object,
     image_size: int = 224
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Image.Image, Image.Image]:
     
     raw_image = Image.open(image_path).convert("RGB")
     
     val_transform = transforms.Compose([
          transforms.Resize((image_size, image_size)),
          transforms.ToTensor()
     ])
     resized_image_for_heatmap = raw_image.resize((image_size, image_size))
     
     inputs = processor(
          images=resized_image_for_heatmap,
          text=text,
          return_tensors="pt",
          padding="max_length",
          truncation=True,
          max_length=512
     )
     
     return inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"], raw_image, resized_image_for_heatmap


def generate_attention_heatmap(model, raw_image, resized_image_for_attn):
    """增强高值区域"""
    self_attn = getattr(model, "last_self_attn", None)
    cross_attn = getattr(model, "last_cross_attn", None)

    if self_attn is None and cross_attn is None:
        return raw_image

    heatmap = None
    original_size = raw_image.size  # (width, height)
    feature_size = resized_image_for_attn.size  # (224, 224)

    # 视觉自注意力处理
    if self_attn is not None and self_attn.dim() == 4:
        self_map = self_attn.mean(dim=1)[:, 0, 1:]  # 去掉CLS token
        seq_len = self_map.shape[-1]
        side = int(np.sqrt(seq_len))
        if side * side != seq_len:
            side = int(np.ceil(np.sqrt(seq_len)))
            padded = torch.zeros(side*side, device=self_map.device)
            padded[:seq_len] = self_map[0]
            self_map = padded.view(1, 1, side, side)
        else:
            self_map = self_map.view(1, 1, side, side)
        self_map = interpolate(
            self_map, size=(feature_size[1], feature_size[0]),
            mode="bilinear", align_corners=False
        ).squeeze().detach().cpu().numpy()
        heatmap = self_map

    # 交叉注意力处理
    if cross_attn is not None and cross_attn.dim() == 4:
        cross_map = cross_attn.mean(dim=1).mean(dim=1)  # 平均多头和文本token
        seq_len = cross_map.shape[-1]
        side = int(np.sqrt(seq_len))
        if side * side != seq_len:
            side = int(np.ceil(np.sqrt(seq_len)))
            padded = torch.zeros(side*side, device=cross_map.device)
            padded[:seq_len] = cross_map[0]
            cross_map = padded.view(1, 1, side, side)
        else:
            cross_map = cross_map.view(1, 1, side, side)
        cross_map = interpolate(
            cross_map, size=(feature_size[1], feature_size[0]),
            mode="bilinear", align_corners=False
        ).squeeze().detach().cpu().numpy()

        if heatmap is not None:
            heatmap = 0.7 * heatmap + 0.3 * cross_map  # 自注意力权重更高
        else:
            heatmap = cross_map

    # --------------------------
    #增强高值区域
    # --------------------------
    # 1. 基础归一化（0-1范围）
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 2. 非线性增强：压缩低值，放大高值（gamma < 1 会让高值更突出）
    gamma = 0.4  # 可调，越小红色越明显（建议0.3-0.6）
    heatmap = np.power(heatmap, gamma)
    
    # 3. 截断低值：忽略低于阈值的区域，进一步突出高值
    threshold = 0.2  # 只保留20%以上的高值区域
    heatmap[heatmap < threshold] = 0
    
    # 4. 重新归一化到0-1（确保颜色映射正确）
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 调整尺寸到原图大小
    heatmap_feature = cv2.resize(heatmap, (feature_size[0], feature_size[1]), interpolation=cv2.INTER_LINEAR)
    heatmap_original = cv2.resize(heatmap_feature, (original_size[0], original_size[1]), interpolation=cv2.INTER_LINEAR)

    # 生成更红的彩色热力图（JET色系中红色对应最高值）
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_original), cv2.COLORMAP_JET)
    raw_img_cv = np.array(raw_image)[:, :, ::-1]  # PIL RGB -> CV BGR
    overlay = cv2.addWeighted(raw_img_cv, 0.5, heatmap_colored, 0.5, 0)  # 增加热力图权重（0.5→更明显）
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def predict_era_with_heatmap(
     model_path: str,
     image_path: str,
     text: str,
     era_to_id: Dict,
     device: str = "cuda"
) -> Tuple[str, Image.Image]:
     model, id_to_era = load_trained_model(
          model_path=model_path,
          era_to_id=era_to_id,
          device=device
     )
     processor = model.processor
     
     pixel_values, input_ids, attention_mask, raw_original_image, resized_image_for_heatmap = preprocess_input(
          image_path=image_path,
          text=text,
          processor=processor,
          image_size=224
     )
     
     with torch.no_grad():
          logits = model(
               pixel_values=pixel_values.to(device),
               input_ids=input_ids.to(device),
               attention_mask=attention_mask.to(device),
          )

          pred_id = torch.argmax(logits, dim=1).item()
          pred_era = id_to_era[pred_id]

     
     heatmap_image = generate_attention_heatmap(
          model=model,
          raw_image=raw_original_image,
          resized_image_for_attn=resized_image_for_heatmap,
     )
     
     return pred_era, heatmap_image


if __name__ == "__main__":
     MODEL_PATH = "/workspace/gugong/checkpoints/blip_era_finetuned_unfreeze_best.pt"  # 训练好的模型权重路径
     IMAGE_PATH = "/workspace/gugong/data/raw/images/6a9dfc4de17a4f71bb55948d69c2be13.png"
     TEXT_DESCRIPTION = "文物名：双兽耳尊；分类：铜器；图案与纹样：；物件类型：礼器 尊（容器） 耳尊 仪式用容器 仿古彝器"
     
     ERA_TO_ID = {
          '三国两晋南北朝': 0, 
          '两汉': 1, 
          '其它': 2, 
          '商': 3, 
          '宋元明清': 4, 
          '春秋战国': 5, 
          '西周': 6, 
          '隋唐五代': 7
     }
     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
     
     print(f"[✓] 成功加载模型: {MODEL_PATH}")
     print(f"[✓] 输入图像: {IMAGE_PATH}")
     print(f"[✓] 使用设备: {DEVICE}")

     pred_era, heatmap_image = predict_era_with_heatmap(
          model_path=MODEL_PATH,
          image_path=IMAGE_PATH,
          text=TEXT_DESCRIPTION,
          era_to_id=ERA_TO_ID,
          device=DEVICE
     )
     
     print(f"预测青铜器年代：{pred_era}")
     save_dir = "/workspace/gugong/data/heatmap"
     os.makedirs(save_dir, exist_ok=True)
     original_filename = os.path.basename(IMAGE_PATH)
     save_path = os.path.join(save_dir, f"heatmap2_{pred_era}_{original_filename}")
     heatmap_image.save(save_path)
     print(f"注意力热力图已保存为：{save_path}")