"""MACD Momentum Strategy"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import BaseStrategy, StrategyResult
from algo_trading.indicators import macd


class MACDMomentumStrategy(BaseStrategy):
    """
    MACD Momentum
    Nguyên lý: giao dịch theo hướng MACD trên signal, và histogram mở rộng thu hẹp.
    Tham số: fast=12, slow=26, signal=9
    """
    name = "MACD Momentum"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        fast = int(self.params.get('fast', 12))
        slow = int(self.params.get('slow', 26))
        sigp = int(self.params.get('signal', 9))
        macd_line, signal_line, hist = macd(close, fast, slow, sigp)
        sig = pd.Series(0, index=df.index)
        sig[macd_line > signal_line] = 1
        sig[macd_line < signal_line] = -1
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'fast': fast, 'slow': slow, 'signal': sigp})

