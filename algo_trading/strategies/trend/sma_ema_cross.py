"""SMA/EMA Crossover Strategy"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import BaseStrategy, StrategyResult, cross_over
from algo_trading.indicators import sma, ema


class SMAEMACrossStrategy(BaseStrategy):
    """
    SMA/EMA Crossover
    Nguyên lý: mua khi đường trung bình ngắn hạn cắt lên dài hạn, bán khi cắt xuống.
    Tham số: fast=20, slow=50, ma_type in {sma, ema}
    """
    name = "SMA/EMA Crossover"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        fast = int(self.params.get('fast', 20))
        slow = int(self.params.get('slow', 50))
        ma_type = self.params.get('ma_type', 'ema')
        ma_fast = ema(close, fast) if ma_type == 'ema' else sma(close, fast)
        ma_slow = ema(close, slow) if ma_type == 'ema' else sma(close, slow)
        sig = cross_over(ma_fast, ma_slow)
        # giữ vị thế theo tín hiệu cắt
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'fast': fast, 'slow': slow, 'ma_type': ma_type})


































































