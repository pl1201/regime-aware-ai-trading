"""
Volatility indicators: True Range, ATR
"""
import pandas as pd
from .utils import ensure_datetime_index


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range"""
    df = ensure_datetime_index(df.copy())
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range"""
    return true_range(df).rolling(window=window, min_periods=window).mean()

