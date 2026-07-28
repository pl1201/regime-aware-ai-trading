"""
Script Train Sentiment Model cho phân tích tâm lý thị trường.

Hỗ trợ:
- FinBERT (tiếng Anh, tin tài chính)
- PhoBERT (tiếng Việt)
- Fine-tune trên dữ liệu custom

Data format: CSV với cột text, label (0=negative, 1=neutral, 2=positive)

Usage:
    python scripts/train_sentiment_model.py --data data/sentiment/train.csv --model finbert --epochs 3
    python scripts/train_sentiment_model.py --data data/sentiment/train.csv --model phobert --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Optional: transformers
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset
    HAS_TRANSFORMERS = True
    try:
        from transformers import EarlyStoppingCallback
        HAS_EARLY_STOPPING = True
    except ImportError:
        HAS_EARLY_STOPPING = False
except ImportError:
    HAS_TRANSFORMERS = False
    HAS_EARLY_STOPPING = False


# Model configs
MODEL_CONFIGS = {
    "finbert": {
        "model_name": "ProsusAI/finbert",
        "num_labels": 3,
        "id2label": {0: "negative", 1: "neutral", 2: "positive"},
        "label2id": {"negative": 0, "neutral": 1, "positive": 2},
    },
    "phobert": {
        "model_name": "vinai/phobert-base",
        "num_labels": 3,
        "id2label": {0: "negative", 1: "neutral", 2: "positive"},
        "label2id": {"negative": 0, "neutral": 1, "positive": 2},
    },
}


def load_sentiment_data(
    data_path: str,
    text_col: str = "text",
    label_col: str = "label",
    max_samples: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load dữ liệu sentiment từ CSV.
    
    Args:
        data_path: Đường dẫn file CSV
        text_col: Tên cột chứa text
        label_col: Tên cột chứa label
        max_samples: Giới hạn số mẫu (None = không giới hạn)
    
    Returns:
        (texts, labels)
    """
    df = pd.read_csv(data_path)
    
    if text_col not in df.columns:
        raise ValueError(f"Cột '{text_col}' không tồn tại. Các cột có: {list(df.columns)}")
    
    # Xử lý label: có thể là string (positive/negative/neutral) hoặc int (0,1,2)
    if label_col not in df.columns:
        raise ValueError(f"Cột '{label_col}' không tồn tại.")
    
    labels = df[label_col].copy()
    
    # Convert string labels to int
    label_map = {"negative": 0, "neutral": 1, "positive": 2, -1: 0, 0: 1, 1: 2}
    if labels.dtype == object or labels.dtype.name == "string":
        labels = labels.str.lower().map(label_map)
    elif labels.dtype in [np.int64, np.int32]:
        # Nếu đã là int nhưng -1,0,1 -> convert sang 0,1,2
        labels = labels.map(lambda x: label_map.get(x, x) if x in [-1, 0, 1] else x)
    
    # Drop NaN
    valid_mask = labels.notna()
    df = df[valid_mask].copy()
    labels = labels[valid_mask].astype(int)
    
    texts = df[text_col].astype(str).tolist()
    
    if max_samples:
        texts = texts[:max_samples]
        labels = labels.iloc[:max_samples]
    
    return pd.Series(texts), labels


def preprocess_text(texts: List[str], max_length: int = 256) -> List[str]:
    """
    Tiền xử lý text: strip, loại bỏ empty.
    """
    cleaned = []
    for t in texts:
        if isinstance(t, str):
            t = t.strip()
            if len(t) > 0:
                cleaned.append(t)
        else:
            cleaned.append(str(t).strip() if t else "")
    return cleaned


def train_sentiment_model(
    data_path: str,
    model_type: str = "finbert",
    output_dir: str = "models/sentiment",
    text_col: str = "text",
    label_col: str = "label",
    val_split: float = 0.2,
    max_length: int = 256,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> Dict:
    """
    Train sentiment model.
    
    Returns:
        Dict với metrics và path model đã lưu.
    """
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "Cần cài đặt: pip install transformers datasets"
        )
    
    config = MODEL_CONFIGS.get(model_type)
    if not config:
        raise ValueError(f"model_type phải là một trong {list(MODEL_CONFIGS.keys())}")
    
    # Load data
    texts_series, labels = load_sentiment_data(data_path, text_col, label_col)
    texts = preprocess_text(texts_series.tolist(), max_length)
    labels = labels.tolist()
    
    if len(texts) < 50:
        raise ValueError(
            f"Cần ít nhất 50 mẫu để train. Hiện có {len(texts)} mẫu. "
            "Xem docs/SENTIMENT_ANALYSIS_GUIDE.md để biết cách chuẩn bị data."
        )
    
    # Train/Val split (stratified)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=val_split, stratify=labels, random_state=seed
    )
    
    # Load tokenizer và model
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        id2label=config["id2label"],
        label2id=config["label2id"],
    )
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": X_train, "label": y_train})
    val_dataset = Dataset.from_dict({"text": X_val, "label": y_val})
    
    train_dataset = train_dataset.map(
        lambda x: tokenize({"text": x["text"]}),
        batched=True,
        remove_columns=["text"],
    )
    train_dataset = train_dataset.rename_column("label", "labels")
    
    val_dataset = val_dataset.map(
        lambda x: tokenize({"text": x["text"]}),
        batched=True,
        remove_columns=["text"],
    )
    val_dataset = val_dataset.rename_column("label", "labels")
    
    # Training args
    output_path = Path(output_dir) / f"{model_type}_finetuned"
    output_path.mkdir(parents=True, exist_ok=True)
    logs_dir = output_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=str(logs_dir),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, preds, average="macro")
        return {"f1_macro": f1}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if (epochs > 2 and HAS_EARLY_STOPPING) else [],
    )
    
    trainer.train()
    eval_result = trainer.evaluate()
    
    # Save
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
    
    # Save config
    with open(output_path / "training_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_type": model_type,
            "base_model": config["model_name"],
            "num_labels": config["num_labels"],
            "id2label": config["id2label"],
            "max_length": max_length,
            "eval_f1_macro": eval_result.get("eval_f1_macro", 0),
        }, f, indent=2, ensure_ascii=False)
    
    # Classification report
    preds = trainer.predict(val_dataset)
    y_pred = np.argmax(preds.predictions, axis=-1)
    report = classification_report(y_val, y_pred, target_names=["negative", "neutral", "positive"])
    print("\n" + report)
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_pred))
    
    return {
        "model_path": str(output_path),
        "eval_f1_macro": eval_result.get("eval_f1_macro", 0),
        "classification_report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Sentiment Model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/sentiment/sample_train.csv",
        help="Đường dẫn file CSV training",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["finbert", "phobert"],
        default="finbert",
        help="Loại model: finbert (EN) hoặc phobert (VI)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/sentiment",
        help="Thư mục lưu model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Số epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Tỷ lệ validation (0.0-1.0)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max sequence length",
    )
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"❌ File không tồn tại: {args.data}")
        print("Tạo file mẫu tại data/sentiment/sample_train.csv và thêm dữ liệu của bạn.")
        return 1
    
    print(f"📂 Load data từ {args.data}")
    print(f"🤖 Model: {args.model}")
    print(f"📤 Output: {args.output}")
    
    result = train_sentiment_model(
        data_path=args.data,
        model_type=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        max_length=args.max_length,
    )
    
    print(f"\n✅ Training xong! Model lưu tại: {result['model_path']}")
    print(f"   F1-macro: {result['eval_f1_macro']:.4f}")
    return 0


if __name__ == "__main__":
    exit(main())
