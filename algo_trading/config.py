"""Typed configuration shared by application and infrastructure layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BotConfig:
    """Runtime configuration for the trading application.

    Keeping this model outside the bot orchestrator prevents exchange adapters
    from depending on application entry points.
    """

    mode: str = "paper"
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    strategy_name: str = "sma_ema"
    strategy_params: dict[str, Any] = field(default_factory=dict)
    risk_per_trade: float = 0.1
    sl_pct: float | None = None
    tp_pct: float | None = None
    trailing_pct: float | None = None
    sl_atr_k: float | None = None
    tp_atr_k: float | None = None
    trailing_atr_k: float | None = None
    atr_col: str = "ATR14"
    history_limit: int = 200
    cool_down_sec: int = 60
    check_interval_sec: int = 30
    max_position_size: float | None = None
    max_dca_orders: int = 1

    def __post_init__(self) -> None:
        self.mode = self.mode.lower()
        self.exchange = self.exchange.lower()

        if self.mode not in {"paper", "testnet", "live"}:
            raise ValueError("mode must be one of: paper, testnet, live")
        if self.exchange not in {"binance", "okx"}:
            raise ValueError("exchange must be one of: binance, okx")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be in the interval (0, 1]")
        if self.history_limit < 2:
            raise ValueError("history_limit must be at least 2")
        if self.max_dca_orders < 1:
            raise ValueError("max_dca_orders must be at least 1")
