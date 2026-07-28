"""Live trading module"""
from .binance_sma_bot import main
from typing import Literal
market_type: Literal["spot", "swap"] = "swap"

__all__ = ['main']

