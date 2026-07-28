"""
Z-score indicator
"""
import pandas as pd


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Z-score normalized indicator"""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / (std + 1e-12)
