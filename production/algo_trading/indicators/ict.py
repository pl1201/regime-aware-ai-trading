"""
ICT-style utilities: Order Block, swing detection, Fibonacci confluence.

Mục tiêu:
- Cung cấp các hàm thuần numpy/pandas dễ tích hợp vào cả:
  - Feature engineering (training ML)
  - Entry filter (lọc signals trong strategy)
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


FIB_LEVELS = [0.382, 0.5, 0.618, 0.786]


def detect_swing_high_low(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    Phát hiện swing high / swing low đơn giản bằng rolling window.

    lookback=3 → swing được xác định trên 7 nến (3 trái, 1 giữa, 3 phải).
    """
    high = df["high"]
    low = df["low"]

    roll_max = high.rolling(2 * lookback + 1, center=True).max()
    roll_min = low.rolling(2 * lookback + 1, center=True).min()

    swing_high = high.eq(roll_max)
    swing_low = low.eq(roll_min)

    return pd.DataFrame(
        {
            "swing_high": swing_high.fillna(False),
            "swing_low": swing_low.fillna(False),
        },
        index=df.index,
    )


def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 20,
    min_body_pct: float = 0.005,
) -> Dict[str, pd.Series]:

    close = df["close"]
    open_ = df.get("open", close)
    high = df["high"]
    low = df["low"]

    swings = detect_swing_high_low(df, lookback=2)
    swing_high = swings["swing_high"]
    swing_low = swings["swing_low"]

    body = (close - open_).abs()
    body_pct = body / close.replace(0, np.nan)

    ob_bull = pd.Series(np.nan, index=df.index)
    ob_bear = pd.Series(np.nan, index=df.index)

    for i in range(lookback, len(df)):
        if swing_high.iloc[i]:
            window = df.iloc[i - lookback : i]
            bears = window[window["close"] < window["open"]]
            bears = bears[body_pct.loc[bears.index] > min_body_pct]
            if not bears.empty:
                last_bear = bears.iloc[-1]
                ob_bull.iloc[i] = last_bear["open"]

        if swing_low.iloc[i]:
            window = df.iloc[i - lookback : i]
            bulls = window[window["close"] > window["open"]]
            bulls = bulls[body_pct.loc[bulls.index] > min_body_pct]
            if not bulls.empty:
                last_bull = bulls.iloc[-1]
                ob_bear.iloc[i] = last_bull["open"]

    return {
        "ob_bull_level": ob_bull.ffill(),
        "ob_bear_level": ob_bear.ffill(),
    }


def ob_confluence_signal(
    df: pd.DataFrame,
    ob_bull: pd.Series,
    ob_bear: pd.Series,
    tolerance_pct: float = 0.002,
) -> Dict[str, pd.Series]:
    """
    Tạo tín hiệu:
        - ob_long_zone: giá nằm gần vùng OB bullish (demand)
        - ob_short_zone: giá nằm gần vùng OB bearish (supply)
    """
    close = df["close"]

    ob_long_zone = ((close - ob_bull).abs() / close < tolerance_pct).astype(float)
    ob_short_zone = ((close - ob_bear).abs() / close < tolerance_pct).astype(float)

    return {
        "ob_long_zone": ob_long_zone,
        "ob_short_zone": ob_short_zone,
    }


def fib_levels_from_swing(
    swing_low: float,
    swing_high: float,
    direction: str = "up",
) -> Dict[str, float]:
    levels: Dict[str, float] = {}
    if direction == "up":
        for lvl in FIB_LEVELS:
            price = swing_high - (swing_high - swing_low) * lvl
            levels[f"fib_{int(lvl * 100)}"] = price
    else:
        for lvl in FIB_LEVELS:
            price = swing_low + (swing_high - swing_low) * lvl
            levels[f"fib_{int(lvl * 100)}"] = price
    return levels


def fib_features(
    df: pd.DataFrame,
    lookback: int = 100,
) -> pd.DataFrame:
    """
    Tạo features Fibonacci dựa trên swing gần nhất trong window lookback.

    Trả về:
        - fib_dist_nearest: khoảng cách (tương đối) tới mức fib gần nhất
        - fib_zone: tên mức fib gần nhất (fib_38, fib_50, fib_61, fib_78)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    swing_high = high.rolling(lookback).max()
    swing_low = low.rolling(lookback).min()

    mid = swing_low + (swing_high - swing_low) / 2.0
    direction = np.where(close >= mid, "up", "down")

    fib_dist_nearest = []
    fib_zone = []

    for i in range(len(df)):
        if pd.isna(swing_high.iloc[i]) or pd.isna(swing_low.iloc[i]):
            fib_dist_nearest.append(np.nan)
            fib_zone.append("none")
            continue

        if direction[i] == "up":
            levels = fib_levels_from_swing(swing_low.iloc[i], swing_high.iloc[i], "up")
        else:
            levels = fib_levels_from_swing(swing_low.iloc[i], swing_high.iloc[i], "down")

        dists = {
            name: abs(close.iloc[i] - price) / close.iloc[i]
            for name, price in levels.items()
        }
        nearest = min(dists, key=dists.get)
        fib_dist_nearest.append(dists[nearest])
        fib_zone.append(nearest)

    return pd.DataFrame(
        {
            "fib_dist_nearest": fib_dist_nearest,
            "fib_zone": pd.Series(fib_zone, index=df.index),
        },
        index=df.index,
    )


