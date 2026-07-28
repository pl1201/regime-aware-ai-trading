from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import BaseStrategy, StrategyResult, cross_over
from algo_trading.indicators import ema


class LSTMTransformerStrategy(BaseStrategy):
    name = "LSTM/Transformer"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close'].astype(float)
        look = int(self.params.get('lookback', 50))
        try:
            import torch
            import torch.nn as nn
        except Exception:
            # fallback ema
            ema_fast = ema(close, 10)
            ema_slow = ema(close, 30)
            sig = cross_over(ema_fast, ema_slow)
            pos = sig.replace(0, np.nan).ffill().fillna(0)
            return StrategyResult(signals=pos, meta={'fallback':'ema'})

        class TinyLSTM(nn.Module):
            def __init__(self, input_size=1, hidden=16):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
                self.fc = nn.Linear(hidden, 1)
            def forward(self, x):
                o,_ = self.lstm(x)
                return self.fc(o[:,-1,:])
        # prepare data (quick train)
        x = close.values
        X, Y = [], []
        for i in range(look, len(x)):
            X.append(x[i-look:i])
            Y.append(x[i])
        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)
        model = TinyLSTM()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(10):  # few epochs for speed
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(look, len(x)):
                seq = torch.tensor(x[i-look:i], dtype=torch.float32).view(1, look, 1)
                p = model(seq).item()
                preds.append(p)
        sig = pd.Series(0, index=df.index)
        sig.iloc[look:] = np.where(np.array(preds) > x[look-1:-1], 1, -1)
        pos = sig.replace(0, pd.NA).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'lookback': look})

