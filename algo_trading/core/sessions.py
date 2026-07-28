from __future__ import annotations

"""
Phân tích hành vi theo khung giờ / phiên (Asia, Europe, US).

Được tách ra từ algo_trading.utils.session_analysis để dùng chung cho CLI và UI.
"""

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd

from .metrics import to_returns


SessionName = Literal["asia", "europe", "us", "other"]


@dataclass
class SessionConfig:
    """
    Cấu hình khung giờ cho các phiên, theo giờ trong ngày (0–24) sau khi chuyển sang timezone chung.

    Mặc định (theo UTC, tương đối gần thực tế FX/Index):
    - Asia:   00:00–08:00
    - Europe: 07:00–16:00
    - US:     13:00–21:00

    Có overlap giữa các phiên, nên ta ưu tiên:
    Asia -> Europe -> US (thằng sau sẽ override nếu trùng).
    """

    asia_start: int = 0
    asia_end: int = 8
    europe_start: int = 7
    europe_end: int = 16
    us_start: int = 13
    us_end: int = 21


def _ensure_timezone(idx: pd.DatetimeIndex, tz: str | None) -> pd.DatetimeIndex:
    if tz:
        if idx.tz is None:
            return idx.tz_localize(tz)
        return idx.tz_convert(tz)
    # không chỉ định tz -> giữ nguyên, nếu không có tz thì coi là UTC
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx


def label_sessions(
    df: pd.DataFrame,
    tz: Optional[str] = "UTC",
    cfg: Optional[SessionConfig] = None,
) -> pd.Series:
    """
    Gán nhãn phiên (asia/europe/us/other) cho từng bar.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index phải là DatetimeIndex để phân tích phiên.")

    cfg = cfg or SessionConfig()
    idx = _ensure_timezone(df.index, tz)
    hour = idx.hour

    session = pd.Series("other", index=idx, dtype="object")

    # Asia
    mask_asia = (hour >= cfg.asia_start) & (hour < cfg.asia_end)
    session[mask_asia] = "asia"

    # Europe
    mask_eu = (hour >= cfg.europe_start) & (hour < cfg.europe_end)
    session[mask_eu] = "europe"

    # US
    mask_us = (hour >= cfg.us_start) & (hour < cfg.us_end)
    session[mask_us] = "us"

    return session.astype("category")


def session_return_stats(
    df: pd.DataFrame,
    price_col: str = "close",
    tz: Optional[str] = "UTC",
    cfg: Optional[SessionConfig] = None,
) -> pd.DataFrame:
    """
    Tính thống kê return theo từng phiên (Asia/Europe/US/Other).
    """
    if price_col not in df.columns:
        raise ValueError(f"Thiếu cột '{price_col}' trong DataFrame.")

    # Tính return
    ret = to_returns(df[price_col])
    # align lại với df (bỏ bar đầu)
    df = df.loc[ret.index]

    sessions = label_sessions(df, tz=tz, cfg=cfg)

    out_rows = []
    for sess, g in ret.groupby(sessions):
        if g.empty:
            continue
        mu = g.mean()
        sigma = g.std(ddof=1)
        n = g.count()
        sharpe_like = (mu / sigma * np.sqrt(n)) if sigma > 0 else np.nan
        pos_ratio = (g > 0).mean()
        out_rows.append(
            {
                "session": sess,
                "count": int(n),
                "mean_ret": float(mu),
                "std_ret": float(sigma),
                "sharpe_like": float(sharpe_like) if sharpe_like == sharpe_like else np.nan,
                "pos_ratio": float(pos_ratio),
            }
        )

    if not out_rows:
        return pd.DataFrame(columns=["count", "mean_ret", "std_ret", "sharpe_like", "pos_ratio"])

    res = pd.DataFrame(out_rows).set_index("session").sort_index()
    return res


def hour_of_day_return_stats(
    df: pd.DataFrame,
    price_col: str = "close",
    tz: Optional[str] = "UTC",
) -> pd.DataFrame:
    """
    Phân tích return theo từng giờ trong ngày (0–23).
    """
    if price_col not in df.columns:
        raise ValueError(f"Thiếu cột '{price_col}' trong DataFrame.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index phải là DatetimeIndex.")

    idx = _ensure_timezone(df.index, tz)
    ret = to_returns(df[price_col])
    ret = ret.reindex(idx).dropna()

    hours = idx.reindex(ret.index).hour

    rows = []
    for h, g in ret.groupby(hours):
        if g.empty:
            continue
        mu = g.mean()
        sigma = g.std(ddof=1)
        n = g.count()
        sharpe_like = (mu / sigma * np.sqrt(n)) if sigma > 0 else np.nan
        pos_ratio = (g > 0).mean()
        rows.append(
            {
                "hour": int(h),
                "count": int(n),
                "mean_ret": float(mu),
                "std_ret": float(sigma),
                "sharpe_like": float(sharpe_like) if sharpe_like == sharpe_like else np.nan,
                "pos_ratio": float(pos_ratio),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["count", "mean_ret", "std_ret", "sharpe_like", "pos_ratio"])

    res = pd.DataFrame(rows).set_index("hour").sort_index()
    return res






