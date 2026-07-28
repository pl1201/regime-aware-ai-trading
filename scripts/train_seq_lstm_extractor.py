
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo_trading.data_loader.loader import load_data
from algo_trading.ml.sequence_extractor import _TinyLSTM  # internal, ok for script use


class SeqDataset(Dataset):
    def __init__(self, rets: np.ndarray, seq_len: int, horizon: int):
        self.rets = rets.astype(np.float32)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)

        # valid t: need [t-seq_len, t) as input, and future return over [t, t+horizon]
        self.idxs = np.arange(self.seq_len, len(self.rets) - self.horizon, dtype=int)

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = int(self.idxs[i])
        x = self.rets[t - self.seq_len : t]  # (L,)
        y = self.rets[t : t + self.horizon].sum()  # horizon log-return approx
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/CRYPTO_BTCUSD, 1D.csv")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="models/seq_lstm_extractor.pt")
    args = ap.parse_args()

    device = torch.device(args.device)

    df = load_data("csv", path=args.csv, add_features=False, dropna=True)
    if df.empty or "close" not in df.columns:
        raise SystemExit("CSV missing 'close' or empty.")

    close = df["close"].astype(float)
    rets = np.log(close.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    # Split time-wise: last 15% as val
    n = len(rets)
    n_val = max(int(n * 0.15), args.seq_len + args.horizon + 10)
    rets_train = rets[: n - n_val]
    rets_val = rets[n - n_val :]

    ds_train = SeqDataset(rets_train, seq_len=args.seq_len, horizon=args.horizon)
    ds_val = SeqDataset(rets_val, seq_len=args.seq_len, horizon=args.horizon)
    if len(ds_train) < 200:
        raise SystemExit(f"Not enough samples for training: {len(ds_train)}")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = _TinyLSTM(
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 5
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in dl_train:
            xb = xb.to(device)  # (B,L,1)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(float(loss.detach().cpu().item()))

        tr = float(np.mean(train_losses)) if train_losses else math.nan
        va = float(np.mean(val_losses)) if val_losses else math.nan
        print(f"epoch={epoch:03d} train_mse={tr:.6f} val_mse={va:.6f}")

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("early_stop")
                break

    if best_state is None:
        best_state = model.state_dict()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": "lstm",
            "seq_len": int(args.seq_len),
            "hidden_size": int(args.hidden_size),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "horizon": int(args.horizon),
            "use_log_returns": True,
            "state_dict": best_state,
        },
        str(out_path),
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

