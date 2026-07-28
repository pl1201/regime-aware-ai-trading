from __future__ import annotations
"""
Giữ lại file này như một lớp bọc mỏng để tương thích ngược.
Logic chính đã được chuyển sang algo_trading.core.backtest_vectorized.
"""

from dataclasses import dataclass  # re-export để không phá type hints cũ
from typing import Optional, Dict, Any, Tuple

import pandas as pd

from algo_trading.core.backtest_vectorized import (  # type: ignore
    BacktestConfig,
    RiskConfig,
    vectorized_pnl,
    barwise_with_stops,
    run_backtest,
)

__all__ = [
    "BacktestConfig",
    "RiskConfig",
    "vectorized_pnl",
    "barwise_with_stops",
    "run_backtest",
]

