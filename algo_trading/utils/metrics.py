"""
Compat layer: re-export metrics từ algo_trading.core.metrics để giữ tương thích ngược.
"""

from algo_trading.core.metrics import (  # type: ignore
    to_returns,
    cum_returns,
    drawdown_series,
    max_drawdown,
    volatility,
    sharpe_ratio,
    downside_deviation,
    sortino_ratio,
    calmar_ratio,
    compound_annual_growth_rate,
    performance_summary,
)

__all__ = [
    "to_returns",
    "cum_returns",
    "drawdown_series",
    "max_drawdown",
    "volatility",
    "sharpe_ratio",
    "downside_deviation",
    "sortino_ratio",
    "calmar_ratio",
    "compound_annual_growth_rate",
    "performance_summary",
]