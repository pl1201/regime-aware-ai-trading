"""VWAP Mean Reversion Strategy"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import BaseStrategy, StrategyResult
from algo_trading.indicators import vwap, atr


class VWAPMeanReversionStrategy(BaseStrategy):
    """
    VWAP Mean Reversion
    Nguyên lý: giá lệch xa VWAP -> kỳ vọng hồi về VWAP.
    Tham số: thr=1.5 (đơn vị ATR), atr_window=14
    """
    name = "VWAP Mean Reversion"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        v = vwap(df)
        a = atr(df, 14)
        thr = float(self.params.get('thr', 1.5))
        close = df['close']
        dist = close - v
        sig = pd.Series(0, index=df.index)
        sig[dist < -thr * a] = 1  # dưới xa -> mua
        sig[dist > thr * a] = -1  # trên xa -> bán
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'thr_atr': thr})

