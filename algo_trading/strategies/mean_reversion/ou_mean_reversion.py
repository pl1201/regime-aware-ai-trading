"""Ornstein-Uhlenbeck Mean Reversion Strategy"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import BaseStrategy, StrategyResult


class OUProcessMeanReversionStrategy(BaseStrategy):
    """
    Ornstein–Uhlenbeck Mean Reversion
    Nguyên lý: coi giá/chuỗi spread là quy trình OU; giao dịch khi z-score lệch xa mức cân bằng.
    Tham số: lookback=100, z=1.5
    """
    name = "OU Mean Reversion"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        x = df['close']
        look = int(self.params.get('lookback', 100))
        zthr = float(self.params.get('z', 1.5))
        # ước lượng OU qua AR(1): x_t = a + b x_{t-1} + e
        x1 = x.shift(1)
        # rolling regression
        zscores = pd.Series(np.nan, index=x.index)
        for i in range(look, len(x)):
            y = x.iloc[i-look+1:i+1]
            X = np.vstack([np.ones(len(y)-1), y.shift(1).iloc[1:].values]).T
            yy = y.iloc[1:].values
            try:
                beta = np.linalg.lstsq(X, yy, rcond=None)[0]
            except Exception:
                continue
            a, b = beta
            mu = a / (1 - b) if (1-b)!=0 else y.mean()
            resid = y - mu
            zscores.iloc[i] = (y.iloc[-1] - mu) / (resid.std(ddof=1) + 1e-12)
        sig = pd.Series(0, index=df.index)
        sig[zscores < -zthr] = 1
        sig[zscores > zthr] = -1
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'lookback': look, 'z': zthr})

