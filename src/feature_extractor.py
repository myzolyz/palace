# src/feature_extractor.py 

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipModel

class FeatureExtractor:
    """
    多模态特征提取：分别提取图像和多个文本字段的嵌入，
    并保存对应的年代标签。
    """

    def __init__(self,
                 model_name: str = "/workspace/MODEL/models--Salesforce--blip-image-captioning-base/snapshots/blip-image-captioning-base",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model     = BlipModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # 指定需要提取特征的文本字段
        self.text_fields = ["文物名", "分类", "图案与纹样", "物件类型"]
        # 目标字段（年代）作为标签
        self.target_field = "年代"

    def extract_text(self, texts):
        """提取文本嵌入"""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            text_emb = text_outputs.pooler_output  # [B, D_txt]
        return text_emb.cpu().numpy()

    def extract_image(self, image_paths):
        """提取图像嵌入"""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_outputs = self.model.vision_model(**inputs)
            image_emb = image_outputs.pooler_output  # [B, D_img]
        return image_emb.cpu().numpy()

    def extract_all(self, jsonl_path: str, out_dir: str, batch_size: int = 8):
        """
        读取 dataset.jsonl，分批提取并保存到 out_dir 下：
          - image_emb.npy
          - text_emb.npy (合并所有文本字段)
          - multimodal_emb.npy (图像 + 文本)
          - labels.npy (年代标签)
          - field_embs/ 目录：每个文本字段的单独嵌入
        """
        os.makedirs(out_dir, exist_ok=True)
        field_emb_dir = os.path.join(out_dir, "field_embs")
        os.makedirs(field_emb_dir, exist_ok=True)

        # 初始化字段嵌入存储
        field_embeddings = {field: [] for field in self.text_fields}
        all_image_embs = []
        all_labels = []

        with open(jsonl_path, "r", encoding="utf-8") as fr:
            lines = fr.readlines()

        for i in tqdm(range(0, len(lines), batch_size), desc="Feature Extraction"):
            batch_lines = lines[i : i + batch_size]
            batch_data = [json.loads(line) for line in batch_lines]
    
            # 提取图像特征
            image_paths = [data["图片url"] for data in batch_data]
            image_emb = self.extract_image(image_paths)
            all_image_embs.append(image_emb)
            
            # 提取每个文本字段的特征（修改部分）
            for field in self.text_fields:
                field_values = []
                for data in batch_data:
                    # 处理缺失值和非字符串类型（如 NaN）
                    raw_value = data.get(field, "")
                    if isinstance(raw_value, float) and np.isnan(raw_value):
                        field_values.append("")  # NaN 转为空字符串
                    else:
                        field_values.append(str(raw_value))  # 强制转为字符串
                field_emb = self.extract_text(field_values)
                field_embeddings[field].append(field_emb)
            
            # 保存年代标签
            batch_labels = [str(data.get(self.target_field, "")) for data in batch_data]
            all_labels.extend(batch_labels)

        # 合并所有批次的特征
        all_image_embs = np.vstack(all_image_embs)
        
        # 保存图像特征
        np.save(os.path.join(out_dir, "image_emb.npy"), all_image_embs)
        
        # 保存每个字段的文本特征
        all_text_embs = None
        for field, embs in field_embeddings.items():
            field_embs = np.vstack(embs)
            np.save(os.path.join(field_emb_dir, f"{field}_emb.npy"), field_embs)
            
            # 合并所有文本字段的特征
            if all_text_embs is None:
                all_text_embs = field_embs
            else:
                all_text_embs = np.concatenate([all_text_embs, field_embs], axis=1)
        
        # 保存合并后的文本特征和多模态特征
        np.save(os.path.join(out_dir, "text_emb.npy"), all_text_embs)
        multimodal_emb = np.concatenate([all_image_embs, all_text_embs], axis=1)
        np.save(os.path.join(out_dir, "multimodal_emb.npy"), multimodal_emb)
        
        # 保存标签
        np.save(os.path.join(out_dir, "labels.npy"), np.array(all_labels))

        print(f"[Saved] image_emb.npy {all_image_embs.shape}")
        print(f"[Saved] text_emb.npy {all_text_embs.shape} (all fields concatenated)")
        print(f"[Saved] multimodal_emb.npy {multimodal_emb.shape}")
        print(f"[Saved] labels.npy {len(all_labels)}")
        print(f"[Saved] Individual field embeddings in {field_emb_dir}")