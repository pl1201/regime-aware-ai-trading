"""
Triple-Barrier Labeling cho ML models.

Tách từ train_regime_ensemble_models_advanced.py

Includes:
- generate_labels_triple_barrier: TP/SL/time barriers + neutral zone
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from algo_trading.indicators import atr


def generate_labels_triple_barrier(
    df: pd.DataFrame,
    horizon: int = 5,
    neutral_quantile: float = 0.35,
    tp_atr_mult: float = 1.5,
    sl_atr_mult: float = 1.2,
    dynamic_horizon: bool = True,
    horizon_min: int = 3,
    horizon_max: int = 12,
    target_atr_ratio: Optional[float] = None,
    return_horizons: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Tạo nhãn theo triple-barrier (TP/SL/time) + neutral threshold.

    Returns:
        y_raw: {-1, 0, 1}
        forward_returns: Return tại time-barrier
        used_horizons: (optional) Horizon thực tế cho mỗi bar
    """
    close = df["close"].astype(float).values
    high = df.get("high", df["close"]).astype(float).values
    low = df.get("low", df["close"]).astype(float).values

    if {"high", "low", "close"}.issubset(set(df.columns)):
        atr_series = atr(df, 14).astype(float)
        atr_ratio = (atr_series / df["close"].astype(float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        atr_ratio = df["close"].astype(float).pct_change().rolling(20).std().fillna(0.0)

    n = len(df)
    y_raw = np.zeros(n, dtype=int)
    forward_returns = np.zeros(n, dtype=float)
    used_horizons = np.full(n, horizon, dtype=int)

    barrier_floor = 0.0025

    if target_atr_ratio is None:
        target_atr_ratio = float(np.median(atr_ratio.values[np.isfinite(atr_ratio.values)])) if len(atr_ratio) else 0.01
        if not np.isfinite(target_atr_ratio) or target_atr_ratio <= 0:
            target_atr_ratio = 0.01

    for i in range(n):
        if dynamic_horizon:
            ar_i = float(max(atr_ratio.iloc[i], barrier_floor))
            h_i = int(round(horizon * (target_atr_ratio / ar_i)))
            h_i = int(np.clip(h_i, horizon_min, horizon_max))
        else:
            h_i = int(horizon)

        used_horizons[i] = h_i

        if i + h_i >= n:
            continue

        entry = close[i]
        ar = float(max(atr_ratio.iloc[i], barrier_floor))
        tp = tp_atr_mult * ar
        sl = sl_atr_mult * ar

        upper = entry * (1.0 + tp)
        lower = entry * (1.0 - sl)

        label = 0
        for j in range(1, h_i + 1):
            if high[i + j] >= upper:
                label = 1
                break
            if low[i + j] <= lower:
                label = -1
                break

        if label == 0:
            time_ret = (close[i + h_i] / entry) - 1.0
            label = 1 if time_ret > 0 else (-1 if time_ret < 0 else 0)

        fwd_ret = (close[i + h_i] / entry) - 1.0
        forward_returns[i] = fwd_ret
        y_raw[i] = label

    valid_abs_ret = np.abs(forward_returns[np.isfinite(forward_returns)])
    if len(valid_abs_ret) > 0:
        thr = float(np.quantile(valid_abs_ret, neutral_quantile))
        y_raw[np.abs(forward_returns) < thr] = 0

    if return_horizons:
        return y_raw, forward_returns, used_horizons
    return y_raw, forward_returns
