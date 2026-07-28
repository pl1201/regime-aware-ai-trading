"""
Sentiment Analysis Module - Phân tích tâm lý thị trường từ văn bản.

Tính năng:
- Load sentiment model (FinBERT/PhoBERT)
- Inference sentiment từ text
- Aggregate sentiment theo time window để tích hợp vào trading
"""

try:
    from algo_trading.sentiment.model import SentimentModel, predict_sentiment
except ImportError:
    SentimentModel = None
    predict_sentiment = None

try:
    from algo_trading.sentiment.aggregator import SentimentAggregator
except ImportError:
    SentimentAggregator = None

__all__ = [
    "SentimentModel",
    "predict_sentiment",
    "SentimentAggregator",
]
