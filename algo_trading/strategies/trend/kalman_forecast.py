"""Kalman Filter Forecast Strategy"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import BaseStrategy, StrategyResult


class KalmanFilterForecastStrategy(BaseStrategy):
    """
    Kalman Filter Price Forecasting (mô hình 2 trạng thái: level + trend)
    Nguyên lý: bộ lọc Kalman ẩn 2D (local linear trend). Mua khi xu hướng (trend) > 0, bán khi < 0.
    Tham số: q=0.0001, r=0.001
    """
    name = "Kalman Forecast"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        y = df['close'].values.astype(float)
        n = len(y)
        q = float(self.params.get('q', 1e-4))  # process noise
        r = float(self.params.get('r', 1e-3))  # observation noise
        # state: [level, trend]
        F = np.array([[1, 1], [0, 1]], dtype=float)
        H = np.array([[1, 0]], dtype=float)
        Q = q * np.eye(2)
        R = np.array([[r]], dtype=float)
        x = np.array([[y[0]], [0.0]])
        P = np.eye(2)
        level = np.zeros(n)
        trend = np.zeros(n)
        for i in range(n):
            # predict
            x = F @ x
            P = F @ P @ F.T + Q
            # update
            z = np.array([[y[i]]])
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z - H @ x)
            P = (np.eye(2) - K @ H) @ P
            level[i], trend[i] = x.flatten()
        sig = pd.Series(0, index=df.index)
        sig[trend > 0] = 1
        sig[trend < 0] = -1
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'q': q, 'r': r})


































































