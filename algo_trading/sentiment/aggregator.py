"""
Sentiment Aggregator - Tổng hợp sentiment theo time window để tích hợp với OHLCV.

Output: Các features sentiment có thể merge với DataFrame giá theo timestamp.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from algo_trading.sentiment.model import SentimentModel
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    SentimentModel = None


class SentimentAggregator:
    """
    Aggregate sentiment theo time window (1h, 4h, 1d) để align với OHLCV.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        symbol: str = "BTCUSDT"
    ):
        """
        Args:
            model_path: Đường dẫn sentiment model. None = dùng FinBERT mặc định.
            symbol: Symbol mặc định.
        """
        self.symbol = symbol
        self.model = None
        if HAS_MODEL and SentimentModel:
            try:
                self.model = SentimentModel(model_path=model_path) if model_path else SentimentModel()
            except Exception as e:
                logger.warning(f"Không load được SentimentModel: {e}. Sẽ cần pre-computed sentiment data.")
        else:
            logger.warning("Transformers chưa cài. SentimentAggregator cần pre-computed data.")
    
    def aggregate_from_raw_news(
        self,
        news_df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        text_col: str = "text",
        symbol_col: Optional[str] = "symbol",
        target_timestamps: Optional[pd.DatetimeIndex] = None,
        windows: List[str] = ["1h", "4h", "1d"]
    ) -> pd.DataFrame:
        """
        Từ DataFrame tin tức raw, infer sentiment và aggregate theo time window.
        
        Args:
            news_df: DataFrame với cột timestamp, text, (symbol).
            timestamp_col: Tên cột timestamp.
            text_col: Tên cột text.
            symbol_col: Tên cột symbol (None nếu không filter).
            target_timestamps: Các timestamp cần aggregate (thường là index của OHLCV).
            windows: Các window ['1h', '4h', '1d'].
        
        Returns:
            DataFrame với index = target_timestamps, columns = sentiment features.
        """
        if self.model is None:
            raise ValueError("Cần load SentimentModel. Cài: pip install torch transformers")
        
        news_df = news_df.copy()
        news_df[timestamp_col] = pd.to_datetime(news_df[timestamp_col])
        
        if symbol_col and symbol_col in news_df.columns:
            news_df = news_df[news_df[symbol_col] == self.symbol]
        
        # Infer sentiment cho mỗi tin
        texts = news_df[text_col].astype(str).tolist()
        if len(texts) == 0:
            logger.warning("Không có tin nào để aggregate.")
            return pd.DataFrame()
        
        scores = self.model.predict_score(texts)
        if isinstance(scores, (int, float)):
            scores = [scores]
        news_df["sentiment_score"] = scores
        
        if target_timestamps is None:
            # Tạo target từ min đến max timestamp, resample 1h
            ts_min = news_df[timestamp_col].min().floor("h")
            ts_max = news_df[timestamp_col].max().ceil("h")
            target_timestamps = pd.date_range(ts_min, ts_max, freq="1h")
        
        results = []
        for ts in target_timestamps:
            row = {"timestamp": ts}
            for w in windows:
                if w == "1h":
                    start = ts - timedelta(hours=1)
                elif w == "4h":
                    start = ts - timedelta(hours=4)
                elif w == "1d":
                    start = ts - timedelta(days=1)
                else:
                    continue
                
                mask = (news_df[timestamp_col] >= start) & (news_df[timestamp_col] < ts)
                subset = news_df.loc[mask, "sentiment_score"]
                
                if len(subset) > 0:
                    row[f"sentiment_mean_{w}"] = subset.mean()
                    row[f"sentiment_std_{w}"] = subset.std()
                    row[f"sentiment_count_{w}"] = len(subset)
                    row[f"sentiment_positive_ratio_{w}"] = (subset > 0.2).sum() / len(subset)
                    row[f"sentiment_negative_ratio_{w}"] = (subset < -0.2).sum() / len(subset)
                else:
                    row[f"sentiment_mean_{w}"] = np.nan
                    row[f"sentiment_std_{w}"] = np.nan
                    row[f"sentiment_count_{w}"] = 0
                    row[f"sentiment_positive_ratio_{w}"] = np.nan
                    row[f"sentiment_negative_ratio_{w}"] = np.nan
            
            results.append(row)
        
        out = pd.DataFrame(results).set_index("timestamp")
        return out
    
    def merge_with_ohlcv(
        self,
        ohlcv_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        how: str = "ffill"
    ) -> pd.DataFrame:
        """
        Merge sentiment features vào OHLCV DataFrame theo index (timestamp).
        
        Args:
            ohlcv_df: DataFrame OHLCV với DatetimeIndex.
            sentiment_df: DataFrame sentiment với DatetimeIndex.
            how: 'ffill' (forward fill), 'bfill', hoặc 'inner'.
        
        Returns:
            ohlcv_df với thêm các cột sentiment_*.
        """
        common_cols = [c for c in sentiment_df.columns if c in ohlcv_df.columns]
        sentiment_merge = sentiment_df.drop(columns=common_cols, errors="ignore")
        
        merged = ohlcv_df.join(sentiment_merge, how="left")
        
        if how == "ffill":
            merged = merged.ffill()
        elif how == "bfill":
            merged = merged.bfill()
        
        return merged
