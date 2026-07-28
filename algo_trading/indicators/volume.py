"""
Volume-based indicators: VWAP
"""
import numpy as np
import pandas as pd
from .utils import ensure_datetime_index


def vwap(df: pd.DataFrame, window: int | None = None) -> pd.Series:
    """
    Volume Weighted Average Price
    
    If window is None, calculates cumulative VWAP.
    Otherwise, calculates rolling VWAP over the specified window.
    """
    df = ensure_datetime_index(df.copy())
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    vol = df.get('volume', pd.Series(index=df.index, data=np.nan)).fillna(0)
    tpv = typical_price * vol
    if window is None:
        cum_tpv = tpv.cumsum()
        cum_vol = vol.cumsum().replace(0, np.nan)
        return cum_tpv / cum_vol
    else:
        return (tpv.rolling(window=window, min_periods=window).sum() /
                vol.rolling(window=window, min_periods=window).sum().replace(0, np.nan))

