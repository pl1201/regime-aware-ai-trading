from __future__ import annotations
"""
Giữ lại file này như một lớp bọc mỏng để tương thích ngược.
Logic chính đã được chuyển sang algo_trading.core.backtest_event.
"""

from dataclasses import dataclass  # re-export để không phá type hints cũ
from typing import Optional, Dict, Any, List

import pandas as pd

from algo_trading.core.backtest_event import (  # type: ignore
    EventConfig,
    Broker,
    run_event_backtest,
)
from algo_trading.core.backtest_vectorized import RiskConfig  # tái sử dụng cấu hình rủi ro

__all__ = [
    "EventConfig",
    "Broker",
    "RiskConfig",
    "run_event_backtest",
]

