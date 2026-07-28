

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class _TinyTCN(nn.Module):

    def __init__(self, seq_len: int = 64, channels: int = 16, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=kernel_size, padding=kernel_size - 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size - 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L) – we rely on padding then slice to keep it causal-ish
        h = self.net(x)
        h = h.squeeze(-1)
        return self.head(h).squeeze(-1)


class _TinyLSTM(nn.Module):


    def __init__(self, seq_len: int = 64, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.head = nn.Linear(int(hidden_size), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1)
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.head(h_last).squeeze(-1)


@dataclass
class SequenceExtractorConfig:
    enabled: bool = True
    model_path: str = "models/seq_lstm_extractor.pt"
    seq_len: int = 64
    device: str = "cpu"
    # Input construction
    use_log_returns: bool = True
    # Fallback features if checkpoint unavailable
    fallback: bool = True


class SequenceFeatureExtractor:


    def __init__(self, cfg: Optional[SequenceExtractorConfig] = None):
        self.cfg = cfg or SequenceExtractorConfig()
        self._model: Optional[nn.Module] = None
        self._loaded_meta: Dict[str, Any] = {}

    @property
    def loaded_arch(self) -> Optional[str]:
        arch = self._loaded_meta.get("arch", None)
        return str(arch).lower() if arch is not None else None

    def _returns(self, close: pd.Series) -> pd.Series:
        if self.cfg.use_log_returns:
            x = np.log(close.replace(0, np.nan)).diff()
        else:
            x = close.pct_change()
        return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _load_model_if_available(self) -> None:
        if self._model is not None:
            return
        if not TORCH_AVAILABLE:
            return

        p = Path(self.cfg.model_path)
        if not p.exists():
            return

        try:
            ckpt = torch.load(str(p), map_location=self.cfg.device)
            arch = str(ckpt.get("arch", "tcn")).lower()
            seq_len = int(ckpt.get("seq_len", self.cfg.seq_len))
            dropout = float(ckpt.get("dropout", 0.1))

            if arch in ("lstm", "gru"):
                hidden_size = int(ckpt.get("hidden_size", 32))
                num_layers = int(ckpt.get("num_layers", 1))
                # (we treat 'gru' as lstm here unless you later add a GRU class)
                model = _TinyLSTM(seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
                meta = {
                    "arch": "lstm",
                    "seq_len": seq_len,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                }
            else:
                channels = int(ckpt.get("channels", 16))
                kernel_size = int(ckpt.get("kernel_size", 3))
                model = _TinyTCN(seq_len=seq_len, channels=channels, kernel_size=kernel_size, dropout=dropout)
                meta = {
                    "arch": "tcn",
                    "seq_len": seq_len,
                    "channels": channels,
                    "kernel_size": kernel_size,
                    "dropout": dropout,
                }

            state = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            model.eval()
            model.to(self.cfg.device)

            self._model = model
            meta["source"] = str(p)
            self._loaded_meta = meta
        except Exception as e:
            warnings.warn(f"⚠️ Could not load seq extractor checkpoint '{p}': {e}")
            self._model = None
            self._loaded_meta = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.enabled:
            return pd.DataFrame(index=df.index)
        if df is None or df.empty or "close" not in df.columns:
            return pd.DataFrame(index=df.index)

        close = df["close"].astype(float)
        rets = self._returns(close)
        L = int(self.cfg.seq_len)

        # Deterministic features (always available)
        seq_vol = rets.rolling(L).std().fillna(0.0)
        seq_trend = rets.rolling(L).mean().fillna(0.0)

        seq_vol = seq_vol.shift(1).fillna(0.0)
        seq_trend = seq_trend.shift(1).fillna(0.0)

        seq_score = pd.Series(0.0, index=df.index)

        # Deep score if checkpoint + torch available
        self._load_model_if_available()
        if (self._model is None or not TORCH_AVAILABLE) and not self.cfg.fallback:
            raise RuntimeError(
                "Strict mode yêu cầu sequence checkpoint + torch, nhưng model chưa được load."
            )

        if self._model is not None and TORCH_AVAILABLE:
            try:
                arr = rets.values.astype(np.float32)
                if len(arr) >= L + 1:
                    # windows ending at t-1 (avoid lookahead): indices [t-L, t)
                    n = len(arr)
                    X = np.zeros((n, L), dtype=np.float32)
                    for t in range(L + 1, n):
                        X[t, :] = arr[t - L : t]

                    arch = str(self._loaded_meta.get("arch", "tcn")).lower()
                    if arch == "lstm":
                        # (n, L, 1)
                        x_t = torch.from_numpy(X).unsqueeze(-1).to(self.cfg.device)
                    else:
                        # (n, 1, L)
                        x_t = torch.from_numpy(X).unsqueeze(1).to(self.cfg.device)

                    with torch.no_grad():
                        out = self._model(x_t).detach().cpu().numpy().astype(np.float32)
                    seq_score = pd.Series(out, index=df.index).fillna(0.0)
                else:
                    seq_score = pd.Series(0.0, index=df.index)
            except Exception as e:
                if self.cfg.fallback:
                    warnings.warn(f"⚠️ seq_score inference failed; fallback to deterministic: {e}")
                    seq_score = pd.Series(0.0, index=df.index)
                else:
                    raise RuntimeError(f"Strict mode: seq_score inference failed: {e}") from e
        else:
            if self.cfg.fallback:
                seq_score = (seq_trend / (seq_vol + 1e-8)).clip(-5, 5).fillna(0.0)

        out = pd.DataFrame(
            {
                "seq_score": pd.to_numeric(seq_score, errors="coerce").fillna(0.0),
                "seq_vol": pd.to_numeric(seq_vol, errors="coerce").fillna(0.0),
                "seq_trend": pd.to_numeric(seq_trend, errors="coerce").fillna(0.0),
            },
            index=df.index,
        )
        return out

