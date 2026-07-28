"""
Sentiment Model - Load và inference sentiment từ text.

Hỗ trợ:
- FinBERT (ProsusAI/finbert)
- PhoBERT fine-tuned
- Custom model từ thư mục local
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional: transformers
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class SentimentModel:
    """
    Wrapper cho sentiment model (FinBERT, PhoBERT, etc.).
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None
    ):
        """
        Args:
            model_path: Đường dẫn model đã fine-tune (local). Nếu None thì dùng model_name.
            model_name: Tên model trên HuggingFace (dùng khi model_path=None).
            device: 'cuda', 'cpu', hoặc None (auto).
        """
        if not HAS_TORCH:
            raise ImportError("Cần cài đặt: pip install torch transformers")
        
        self.model_path = model_path
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        load_path = model_path or model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping (FinBERT/PhoBERT thường dùng 0,1,2)
        self.id2label = getattr(self.model.config, "id2label", {0: "negative", 1: "neutral", 2: "positive"})
        if isinstance(self.id2label, dict) and any(isinstance(k, str) for k in self.id2label.keys()):
            self.id2label = {int(k): v for k, v in self.id2label.items()}
    
    def predict(
        self,
        texts: Union[str, List[str]],
        return_probs: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dự đoán sentiment cho text(s).
        
        Args:
            texts: Một string hoặc list strings.
            return_probs: Có trả về probabilities không.
        
        Returns:
            (labels, probs) - labels là class (0,1,2), probs là [P(neg), P(neutral), P(pos)]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits.cpu().numpy()
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        labels = np.argmax(probs, axis=1)
        
        if return_probs:
            return labels, probs
        return labels, probs
    
    def predict_score(self, texts: Union[str, List[str]]) -> Union[float, np.ndarray]:
        """
        Trả về sentiment score chuẩn hóa [-1, 1].
        score = P(positive) - P(negative)
        """
        _, probs = self.predict(texts, return_probs=True)
        # probs: [P(neg), P(neutral), P(pos)]
        if len(probs.shape) == 1:
            probs = probs.reshape(1, -1)
        scores = probs[:, 2] - probs[:, 0]  # positive - negative
        if len(scores) == 1:
            return float(scores[0])
        return scores


# Global model instance (lazy load)
_sentiment_model: Optional[SentimentModel] = None


def predict_sentiment(
    texts: Union[str, List[str]],
    model_path: Optional[str] = None,
    return_score: bool = True
) -> Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Hàm tiện ích để predict sentiment.
    
    Args:
        texts: Text hoặc list texts.
        model_path: Đường dẫn model. None = dùng FinBERT mặc định.
        return_score: True = trả về score [-1,1], False = trả về (labels, probs).
    
    Returns:
        Nếu return_score=True: float hoặc ndarray score.
        Nếu return_score=False: (labels, probs).
    """
    global _sentiment_model
    
    if _sentiment_model is None:
        _sentiment_model = SentimentModel(model_path=model_path)
    
    if return_score:
        return _sentiment_model.predict_score(texts)
    return _sentiment_model.predict(texts)
