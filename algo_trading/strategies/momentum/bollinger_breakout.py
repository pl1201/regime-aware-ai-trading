"""Bollinger Bands Breakout Strategy"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import BaseStrategy, StrategyResult
from algo_trading.indicators import bollinger_bands


class BollingerBreakoutStrategy(BaseStrategy):
    """
    Bollinger Bands Breakout
    Nguyên lý: breakout trên dải trên mua, dưới dải dưới bán; thoát khi trở lại dải giữa.
    Tham số: window=20, k=2
    """
    name = "Bollinger Breakout"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        window = int(self.params.get('window', 20))
        k = float(self.params.get('k', 2.0))
        m, u, l = bollinger_bands(close, window, k)
        sig = pd.Series(0, index=df.index)
        sig[close > u] = 1
        sig[close < l] = -1
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        # exit rule: when price crosses middle back
        exit_long = (pos.shift(1) > 0) & (close < m)
        exit_short = (pos.shift(1) < 0) & (close > m)
        pos[exit_long | exit_short] = 0
        pos = pos.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'window': window, 'k': k})

