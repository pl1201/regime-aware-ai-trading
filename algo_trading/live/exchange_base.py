"""
Abstract base class cho exchange clients.
Hỗ trợ nhiều exchange (Binance, OKX, ...)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd


@dataclass
class SymbolFilters:
    """Filters từ exchange info."""
    step_size: float
    min_qty: float
    min_notional: float
    tick_size: float


class ExchangeClient(ABC):
    """Abstract base class cho exchange clients."""
    
    @abstractmethod
    def get_klines_df(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Lấy dữ liệu kline và trả về DataFrame."""
        pass
    
    @abstractmethod
    def get_last_price(self, symbol: str) -> float:
        """Lấy giá hiện tại."""
        pass
    
    @abstractmethod
    def get_asset_balance(self, asset: str) -> float:
        """Lấy số dư asset."""
        pass
    
    @abstractmethod
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Đặt lệnh market."""
        pass
    
    @abstractmethod
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """Đặt lệnh limit."""
        pass
    
    @abstractmethod
    def _fetch_symbol_filters(self, symbol: str) -> SymbolFilters:
        """Lấy filters từ exchange info."""
        pass
