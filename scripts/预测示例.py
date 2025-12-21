import sys
import os
# 添加src目录到环境变量
sys.path.append(os.path.abspath("/workspace/gugong"))

import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import BlipProcessor
from src.fine_tuner_all import FineTuner  # 导入模型类
import numpy as np


class EraPredictor:
    def __init__(self, model_path, processor_path, num_attention_heads=8):
        """初始化预测器，直接内置era_to_id映射"""
        # 加载处理器
        self.processor = BlipProcessor.from_pretrained(processor_path)
        
        # 硬编码era_to_id映射（根据你的需求填写）
        self.era_to_id = {
            "三国两晋南北朝": 0,
            "两汉": 1,
            "其它": 2,
            "商": 3,
            "宋元明清": 4,
            "春秋战国": 5,
            "西周": 6,
            "隋唐五代": 7
        }
        self.id_to_era = {v: k for k, v in self.era_to_id.items()}  # 反向映射
        self.num_classes = len(self.era_to_id)  # 共8个类别
        
        # 初始化模型并加载权重
        self.model = FineTuner(
            model_name=processor_path,
            num_classes=self.num_classes,
            freeze_base=False,
            num_attention_heads=num_attention_heads
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # 评估模式
        
        # 图片预处理（与训练一致）
        self.image_transform = transforms.Resize((224, 224))

    def preprocess_input(self, image_path, text_desc):
        """预处理图片和文本"""
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)
        
        inputs = self.processor(
            images=image,
            text=text_desc,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512  # 与训练时一致
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def predict(self, image_path, text_desc):
        """预测年代概率"""
        inputs = self.preprocess_input(image_path, text_desc)
        
        with torch.no_grad():
            logits = self.model(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        
        # 计算概率并映射为年代名称
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        era_probs = {
            self.id_to_era[i]: float(probabilities[i]) 
            for i in range(self.num_classes)
        }
        # 按概率降序排列
        return dict(sorted(era_probs.items(), key=lambda x: x[1], reverse=True))


# 示例用法
if __name__ == "__main__":
    MODEL_PATH = "/workspace/gugong/checkpoints/blip_era_finetuned_unfreeze_best.pt"
    PROCESSOR_PATH = "/workspace/MODEL/models--Salesforce--blip-image-captioning-base/snapshots/blip-image-captioning-base"
    
    # 初始化预测器
    predictor = EraPredictor(
        model_path=MODEL_PATH,
        processor_path=PROCESSOR_PATH,
        num_attention_heads=8  # 与训练时一致
    )
    
    # 输入测试数据
    test_image_path = "/workspace/gugong/data/raw/images/903ff2d478e24c5fa5ef5283481e59c0.png"  # 替换为你的图片路径
    test_text = "文物名：鎏金兽纹牌饰；分类：铜器；图案与纹样：；物件类型："  # 按训练格式填写
    
    # 预测并打印结果
    probabilities = predictor.predict(test_image_path, test_text)
    print("年代预测概率：")
    for era, prob in probabilities.items():
        print(f"{era}: {prob:.4f}")