"""RSI Divergence Strategy"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import BaseStrategy, StrategyResult
from algo_trading.indicators import rsi


class RSIDivergenceStrategy(BaseStrategy):
    """
    RSI + Divergence (đơn giản)
    Nguyên lý: quá mua/quá bán theo RSI, cộng thêm kiểm tra phân kỳ đơn giản: giá tạo đỉnh cao hơn nhưng RSI tạo đỉnh thấp hơn (bearish), ngược lại là bullish.
    Tham số: period=14, ob=70, os=30, lookback=5
    """
    name = "RSI Divergence"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        period = int(self.params.get('period', 14))
        ob = float(self.params.get('overbought', 70))
        os_ = float(self.params.get('oversold', 30))
        look = int(self.params.get('lookback', 5))
        r = rsi(close, period)
        # tín hiệu nền tảng
        base = pd.Series(0, index=df.index)
        base[r < os_] = 1
        base[r > ob] = -1
        # phân kỳ đơn giản: so sánh hai pivot gần nhất trong lookback
        price_high = close.rolling(look).max()
        price_low = close.rolling(look).min()
        rsi_high = r.rolling(look).max()
        rsi_low = r.rolling(look).min()
        bearish = ((price_high > price_high.shift(look)) & (rsi_high < rsi_high.shift(look))).astype(int) * -1
        bullish = ((price_low < price_low.shift(look)) & (rsi_low > rsi_low.shift(look))).astype(int) * 1
        div = bullish + bearish
        sig = base.where(div == 0, div)
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'period': period, 'overbought': ob, 'oversold': os_, 'lookback': look})

