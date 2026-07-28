"""Volume Profile Imbalance Strategy"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import BaseStrategy, StrategyResult


class VolumeProfileImbalanceStrategy(BaseStrategy):
    """
    Volume Profile Imbalance (xấp xỉ)
    Nguyên lý: trong cửa sổ N, tạo histogram giá theo bins và tổng hợp volume; khi giá vượt ra khỏi vùng giá trị cao (HVN) -> breakout theo hướng đó.
    Tham số: window=200, bins=20
    """
    name = "Volume Profile Imbalance"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        vol = df.get('volume', pd.Series(0, index=df.index))
        window = int(self.params.get('window', 200))
        bins = int(self.params.get('bins', 20))
        hvn_low = pd.Series(np.nan, index=df.index)
        hvn_high = pd.Series(np.nan, index=df.index)
        for i in range(window, len(df)):
            c = close.iloc[i-window:i]
            v = vol.iloc[i-window:i]
            counts, edges = np.histogram(c, bins=bins, weights=v)
            idx = counts.argmax()
            hvn_low.iloc[i] = edges[max(0, idx-1)]
            hvn_high.iloc[i] = edges[min(len(edges)-1, idx+1)]
        sig = pd.Series(0, index=df.index)
        sig[close > hvn_high] = 1
        sig[close < hvn_low] = -1
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'window': window, 'bins': bins})

