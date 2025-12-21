# /scripts/fine_tune_all.py ---------------------
import sys, os
import argparse
import torch
import logging
from torch.utils.data import DataLoader, random_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.fine_tuner_all import MultimodalEraDataset, FineTuner, train
from transformers import BlipProcessor

if __name__ == "__main__":
    # -------------------------- 配置日志 --------------------------
    log_dir = "/workspace/gugong/training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"train_log_{os.path.splitext(os.path.basename(__file__))[0]}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()

    # -------------------------- 解析参数 --------------------------
    parser = argparse.ArgumentParser("Fine-tune BLIP on era classification")
    parser.add_argument("--jsonl", default="/workspace/gugong/data/raw/10.11BronzeWare.jsonl")
    parser.add_argument("--labels", default="/workspace/gugong/data/processed/all_features/labels.npy")
    parser.add_argument("--model", default="/workspace/MODEL/models--Salesforce--blip-image-captioning-base/snapshots/blip-image-captioning-base")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    # -------------------------- 初始化数据集 --------------------------
    processor = BlipProcessor.from_pretrained(args.model)
    full_dataset = MultimodalEraDataset(
        jsonl_path=args.jsonl,
        era_labels_npy_path=args.labels,
        processor=processor,
    )



    # -------------------------- 划分训练/验证集 --------------------------
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    # -------------------------- 构建模型 --------------------------
    num_classes = full_dataset.num_classes
    model = FineTuner(
        model_name=args.model,
        num_classes=num_classes,
        freeze_base=False,  # 是否解冻主干
        num_attention_heads=8
    )

    # -------------------------- 训练 --------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"开始训练：epochs={args.epochs}, batch={args.batch}, lr={args.lr}, device={device}, val_ratio={args.val_ratio}")

    trained_model = train(
        model=model,             
        dataset=full_dataset,    
        val_ratio=args.val_ratio, 
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        device=device,
        logger=logger
    )



