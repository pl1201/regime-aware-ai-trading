"""GARCH Volatility Strategy"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import BaseStrategy, StrategyResult


class GARCHVolatilityStrategy(BaseStrategy):
    """
    GARCH Volatility (quy mô vị thế theo vol)
    Nguyên lý: ước lượng volatility bằng GARCH(1,1); khi vol thấp và động lượng dương -> tăng vị thế; vol cao -> giảm.
    Tham số: window=250
    """
    name = "GARCH Volatility"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        ret = np.log(close).diff().fillna(0)
        vol = pd.Series(np.nan, index=df.index)
        try:
            from arch import arch_model
            window = int(self.params.get('window', 500))
            for i in range(window, len(ret)):
                r = ret.iloc[i-window:i]
                try:
                    am = arch_model(r*100, vol='Garch', p=1, o=0, q=1, dist='normal')
                    res = am.fit(disp='off')
                    f = res.forecast(horizon=1).variance.iloc[-1,0]
                    vol.iloc[i] = np.sqrt(f)/100
                except Exception:
                    continue
        except Exception:
            # fallback: rolling std
            window = int(self.params.get('window', 250))
            vol = ret.rolling(window).std().reindex(df.index)
        mom = close.diff(5)
        raw = np.sign(mom)
        # scale by inverse vol
        v = (vol - vol.min())/(vol.max()-vol.min()+1e-12)
        scale = (1 - v).fillna(0.5)
        pos = (raw * scale).fillna(0).clip(-1,1)
        return StrategyResult(signals=pos, meta={'window': window})


































































