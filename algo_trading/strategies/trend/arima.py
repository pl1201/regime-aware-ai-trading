"""ARIMA/SARIMA Forecast Strategy"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import BaseStrategy, StrategyResult, cross_over
from algo_trading.indicators import ema


class ARIMAStrategy(BaseStrategy):
    """
    ARIMA/SARIMA Forecast
    Nguyên lý: dự báo bước kế tiếp bằng ARIMA. Nếu dự báo tăng -> mua, giảm -> bán.
    Tham số: order=(1,1,1) (SARIMA có seasonal_order)
    """
    name = "ARIMA/SARIMA"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        order = self.params.get('order', (1,1,1))
        seasonal_order = self.params.get('seasonal_order', None)
        try:
            import statsmodels.api as sm
        except Exception:
            # fallback: dùng EMA crossover nhẹ
            ema_fast = ema(close, 10)
            ema_slow = ema(close, 30)
            sig = cross_over(ema_fast, ema_slow)
            pos = sig.replace(0, np.nan).ffill().fillna(0)
            return StrategyResult(signals=pos, meta={'fallback':'ema'})
        # rolling one-step forecast
        sig = pd.Series(0, index=df.index)
        window = int(self.params.get('window', 200))
        for i in range(window, len(close)):
            y = close.iloc[i-window:i]
            try:
                if seasonal_order is None:
                    model = sm.tsa.ARIMA(y, order=order)
                else:
                    model = sm.tsa.SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                f = res.forecast(1)
                sig.iloc[i] = 1 if f.iloc[-1] > y.iloc[-1] else -1
            except Exception:
                continue
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'order': order, 'seasonal_order': seasonal_order})


























