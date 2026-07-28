"""
Moving Average indicators: SMA, EMA, WMA
"""
import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()


def wma(series: pd.Series, window: int) -> pd.Series:
    """Weighted Moving Average"""
    weights = np.arange(1, window + 1)
    def f(x):
        return np.dot(x, weights) / weights.sum()
    return series.rolling(window).apply(f, raw=True)



