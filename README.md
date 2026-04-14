# 面镜·古 | ArcheoBLIP

> 基于多模态大模型的故宫青铜器智能断代系统

**A multimodal deep learning system for automated era classification of Palace Museum bronze artifacts.**

---

## 简介 | Introduction

**中文**

本项目以故宫数字文物库的青铜器为研究对象，构建了一套多模态智能断代系统。通过融合图像与文字描述信息，自动预测青铜器的历史朝代归属。

核心创新点：
- 在 BLIP-Base 基础上提出**显式双向跨模态注意力机制**（自注意力 + 交叉注意力），充分融合图文信息
- 引入**动态多任务权重调度器**，自适应平衡分类损失与辅助任务
- 设计 **LoRA-style 轻量化微调**方案，参数量仅 235M，训练速度较 Qwen2.5（1.5B）提升 32×
- 朝代预测 Top-1 准确率从 82.7% 提升至 **86.9%**

**English**

This project builds a multimodal intelligent era classification system for bronze artifacts from the Palace Museum digital collection. By fusing image and text description features, the system automatically predicts the historical dynasty of a given artifact.

Key contributions:
- Proposed **explicit bidirectional cross-modal attention** (self-attention + cross-attention) on top of BLIP-Base
- Introduced a **dynamic multi-task weight scheduler** for adaptive loss balancing
- Designed a **LoRA-style lightweight fine-tuning** scheme with only 235M parameters — 32× faster than Qwen2.5 (1.5B)
- Improved Top-1 dynasty prediction accuracy from 82.7% to **86.9%**

---

## 数据集 | Dataset

- 来源：故宫数字文物库（https://digicol.dpm.org.cn）
- 规模：4.2 万张青铜器图文数据
- 采集工具：Scrapy
- 图像增强：Real-ESRGAN 超分辨率处理
- 标注类别：新石器时代晚期、商代、西周、春秋、战国、秦汉、隋唐、宋元明清（共 8 个朝代）

---

## 技术栈 | Tech Stack

| 模块 | 技术 |
|------|------|
| 预训练模型 | BLIP-Base (Salesforce) |
| 轻量化微调 | LoRA-style |
| 数据采集 | Scrapy |
| 图像超分 | Real-ESRGAN |
| 训练框架 | PyTorch |
| 版本管理 | Git |

---

## 项目结构 | Project Structure

```
palace/
├── scripts/
│   ├── extract_features.py     # 多模态特征提取
│   ├── fine_tune_all.py        # 模型微调训练
│   └── 预测示例.py              # 单样本预测示例
├── src/
│   ├── feature_extractor.py    # 特征提取器（BLIP图文嵌入）
│   ├── fine_tuner_all.py       # 多模态微调模型定义
│   ├── heatmap2.py             # 注意力热力图可视化
│   └── clustering.py           # 特征聚类分析
└── README.md
```

---

## 快速开始 | Quick Start

**1. 特征提取 | Feature Extraction**

```python
python scripts/extract_features.py
```

**2. 模型训练 | Training**

```python
python scripts/fine_tune_all.py
```

**3. 朝代预测 | Prediction**

```python
python scripts/预测示例.py
```

示例输出 / Example output:

```
朝代预测概率：
商代: 0.8231
西周: 0.1124
春秋: 0.0421
...
```

---

## 实验结果 | Results

| 模型 | 参数量 | Top-1 准确率 | 相对训练速度 |
|------|--------|-------------|------------|
| ImageBind | 500M | 82.7% | 1× |
| Qwen2.5 | 1.5B | - | 0.03× |
| **ArcheoBLIP（本项目）** | **235M** | **86.9%** | **10×~32×** |

---

## 作者 | Author

**吕亦卓 Yizhuo Lv**
- 武汉大学 信息管理学院
- GitHub: [@myzolyz](https://github.com/myzolyz)

---

*本项目为学术研究项目，数据来源于故宫博物院数字文物库。*

*This project is for academic research purposes. Data sourced from the Palace Museum Digital Collection.*
