# /scripts/extract_features.py

import os
from src.feature_extractor import FeatureExtractor

if __name__ == "__main__":
    JSONL_PATH = "/workspace/gugong/data/raw/10.11BronzeWare.jsonl"
    FEATURE_DIR = "/workspace/gugong/data/processed/all_features"
    BATCH_SIZE = 100

    print("Start multi-modal feature extraction ...")
    fe = FeatureExtractor()
    fe.extract_all(jsonl_path=JSONL_PATH,
                   out_dir=FEATURE_DIR,
                   batch_size=BATCH_SIZE)
    print("Feature extraction completed.")
