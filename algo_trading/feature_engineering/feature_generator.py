"""Backward-compatible feature generator for live trading scripts.

This module adapts legacy imports to the current multi-timeframe feature
implementation in `algo_trading.features.multi_timeframe`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from algo_trading.features.multi_timeframe import add_multi_timeframe_features as _add_mt_features


def _normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the input has a DatetimeIndex and required OHLCV columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
            out = out.set_index("timestamp")
        else:
            out.index = pd.to_datetime(out.index, errors="coerce")

    out = out.sort_index()

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    if "volume" not in out.columns:
        out["volume"] = 0.0

    return out


def _infer_base_timeframe(df: pd.DataFrame) -> str:
    """Infer a pandas-style timeframe string from DatetimeIndex."""
    if len(df.index) < 3:
        return "1H"

    freq = pd.infer_freq(df.index)
    if freq:
        freq_upper = str(freq).upper()
        return "1H" if freq_upper == "H" else freq_upper

    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        return "1H"

    minutes = int(diffs.median().total_seconds() // 60)
    if minutes <= 5:
        return "5T"
    if minutes <= 15:
        return "15T"
    if minutes <= 30:
        return "30T"
    if minutes <= 60:
        return "1H"
    if minutes <= 240:
        return "4H"
    return "1D"


def _default_multi_timeframes(base_timeframe: str) -> List[str]:
    """Pick higher-timeframe confirmations based on base timeframe."""
    base = str(base_timeframe).upper()
    if base in {"1T", "3T", "5T"}:
        return ["15T", "30T", "1H", "4H"]
    if base in {"10T", "15T", "30T"}:
        return ["1H", "4H", "1D"]
    if base in {"1H", "2H", "4H"}:
        return ["4H", "1D", "1W"]
    if base in {"6H", "12H", "1D"}:
        return ["1D", "1W"]
    return ["1H", "4H", "1D"]


@dataclass
class FeatureGenerator:
    """Legacy-compatible feature generator class."""

    base_timeframe: Optional[str] = None
    multi_timeframes: Optional[List[str]] = None

    def add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate multi-timeframe features with sensible defaults."""
        normalized = _normalize_input_df(df)
        base_tf = self.base_timeframe or _infer_base_timeframe(normalized)
        mt_tfs = self.multi_timeframes or _default_multi_timeframes(base_tf)

        return _add_mt_features(
            normalized,
            base_timeframe=base_tf,
            multi_timeframes=mt_tfs,
        )


def add_multi_timeframe_features(
    df: pd.DataFrame,
    base_timeframe: Optional[str] = None,
    multi_timeframes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Backward-compatible function API used by existing live scripts."""
    generator = FeatureGenerator(
        base_timeframe=base_timeframe,
        multi_timeframes=multi_timeframes,
    )
    return generator.add_multi_timeframe_features(df)
