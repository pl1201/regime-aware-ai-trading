"""Backward-compatible data loader module.

Some legacy scripts import from `algo_trading.data_loader.data_loader`.
This module bridges those imports to the current implementation in
`algo_trading.data_loader.loader`.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .loader import load_binance, load_csv, load_data, load_parquet, load_yfinance


def load_multi_timeframe_data(
    source: str = "binance",
    symbol: str = "BTCUSDT",
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: str = "spot",
    intervals: Optional[Dict[str, str]] = None,
    add_features: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load multiple timeframes and return a dict of DataFrames.

    Args:
        source: Data source, e.g. ``binance`` / ``yfinance`` / ``csv`` / ``parquet``.
        symbol: Trading symbol for API-based sources.
        start: Start date string.
        end: End date string.
        market: Market type for Binance source.
        intervals: Mapping name -> interval. Defaults to 1h/4h/1d.
        add_features: Whether to add basic indicators in the base loader.

    Returns:
        Dict mapping timeframe alias to DataFrame.
    """
    tf_map = intervals or {
        "primary": "1h",
        "confirmation": "4h",
        "trend": "1d",
    }

    out: Dict[str, pd.DataFrame] = {}
    for name, interval in tf_map.items():
        kwargs = {
            "source": source,
            "add_features": add_features,
            "start": start,
            "end": end,
        }

        if source.lower() == "binance":
            kwargs.update({"symbol": symbol, "interval": interval, "market": market})
        elif source.lower() == "yfinance":
            kwargs.update({"ticker": symbol, "interval": interval})
        else:
            # For file sources, caller should pass specific params directly via load_data.
            kwargs.update({"symbol": symbol, "interval": interval, "market": market})

        out[name] = load_data(**kwargs)

    return out


__all__ = [
    "load_data",
    "load_csv",
    "load_parquet",
    "load_yfinance",
    "load_binance",
    "load_multi_timeframe_data",
]
