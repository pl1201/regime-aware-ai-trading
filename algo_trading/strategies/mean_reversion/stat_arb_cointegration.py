"""Statistical Arbitrage Cointegration Strategy"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import BaseStrategy, StrategyResult


class StatArbCointegrationStrategy(BaseStrategy):
    """
    Statistical Arbitrage (Cointegration) – Pairs Trading
    Cần 2 chuỗi giá: close_X và close_Y trong df (cột 'close_Y' tồn tại) hoặc truyền vào params 'other'.
    Nguyên lý: kiểm định đồng liên kết; tạo spread = X - beta*Y; trade mean-reversion theo z-score.
    Tham số: lookback=250, z=2
    """
    name = "StatArb Cointegration"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        x = df['close']
        if 'close_Y' in df.columns:
            y = df['close_Y']
        elif 'other' in self.params:
            y = self.params['other'].reindex(df.index)['close']
        else:
            # không đủ dữ liệu -> flat
            return StrategyResult(signals=pd.Series(0, index=df.index), meta={'error':'need second series'})
        look = int(self.params.get('lookback', 250))
        zthr = float(self.params.get('z', 2.0))
        try:
            import statsmodels.api as sm
            from statsmodels.tsa.stattools import coint
        except Exception:
            # fallback: OLS beta rolling
            pass
        spread = pd.Series(np.nan, index=df.index)
        zscores = pd.Series(np.nan, index=df.index)
        for i in range(look, len(df)):
            X = y.iloc[i-look:i]
            Y = x.iloc[i-look:i]
            X1 = np.vstack([X.values, np.ones(len(X))]).T
            beta, alpha = np.linalg.lstsq(X1, Y.values, rcond=None)[0]
            sp = Y - (alpha + beta*X)
            spread.iloc[i] = sp.iloc[-1]
            zscores.iloc[i] = (sp.iloc[-1] - sp.mean())/(sp.std(ddof=1)+1e-12)
        sig = pd.Series(0, index=df.index)
        sig[zscores > zthr] = -1
        sig[zscores < -zthr] = 1
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'lookback': look, 'z': zthr})

