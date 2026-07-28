"""Renko Trend Following Strategy"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import BaseStrategy, StrategyResult
from algo_trading.indicators import atr


class RenkoTrendStrategy(BaseStrategy):
    """
    Renko Trend Following (xấp xỉ)
    Nguyên lý: chuyển chuỗi giá sang brick kích thước theo ATR; theo dõi hướng bricks.
    Tham số: brick_atr=14, brick_k=1.0
    """
    name = "Renko Trend"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        a = atr(df, int(self.params.get('brick_atr', 14)))
        k = float(self.params.get('brick_k', 1.0))
        brick = k * a
        direction = pd.Series(0, index=df.index)
        ref = close.iloc[0]
        dir_ = 0
        for i in range(1, len(close)):
            c = close.iloc[i]
            if not np.isnan(brick.iloc[i]):
                b = brick.iloc[i]
            elif i > 1:
                # Try to get the last valid brick value
                valid_bricks = brick.iloc[:i].dropna()
                if len(valid_bricks) > 0:
                    b = valid_bricks.iloc[-1]
                else:
                    b = np.nan
            else:
                b = np.nan
            if np.isnan(b):
                direction.iloc[i] = 0
                continue
            if c - ref >= b:
                dir_ = 1
                ref = c
            elif ref - c >= b:
                dir_ = -1
                ref = c
            direction.iloc[i] = dir_
        pos = direction.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'brick_k': k})


























