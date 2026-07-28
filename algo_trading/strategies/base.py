"""
BaseStrategy và interface generate_signals cho hệ thống chiến lược.

Cách triển khai một chiến lược mới:
- Tạo class kế thừa BaseStrategy
- Ghi đè hàm generate_signals(self, df: pd.DataFrame) -> StrategyResult
- Trả về:
    - signals: pd.Series với các giá trị trong {-1, 0, +1}, index là DatetimeIndex khớp với df.index
    - meta (tùy chọn): dict chứa tham số, thiết lập SL/TP/trailing, sizing, vv.

Ví dụ khởi tạo và chạy:
    strat = MyStrategy(fast=20, slow=50)
    result = strat.generate_signals(df)
    pos = result.signals
"""
from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Dict, Any
import pandas as pd
import numpy as np


@dataclass
class StrategyResult:
    signals: pd.Series  # -1, 0, +1 theo thời gian
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(abc.ABC):
    """
    Lớp cơ sở cho mọi chiến lược.

    Tham số khởi tạo (tùy chọn): truyền dưới dạng keyword arguments.
    Ví dụ: BaseStrategy(fast=12, slow=26, risk={"sl":0.01, "tp":0.02})
    """
    name: str = "BaseStrategy"

    def __init__(self, **params):
        self.params: Dict[str, Any] = params or {}

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.params

    def set_params(self, **params) -> "BaseStrategy":
        self.params.update(params)
        return self

    @abc.abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """
        Tạo tín hiệu giao dịch cho DataFrame df (cần có cột 'close' và index thời gian).
        Trả về StrategyResult(signals, meta).
        """
        raise NotImplementedError

    # Alias để tương thích: các nơi khác có thể gọi generate
    def generate(self, df: pd.DataFrame) -> StrategyResult:
        return self.generate_signals(df)

    def description(self) -> str:
        return (self.__doc__ or self.name or "").strip()

    def info(self) -> str:
        return f"{self.__class__.__name__} params={self.params}"

    # --------- Tiện ích chuẩn hóa tín hiệu ---------
    @staticmethod
    def validate_signals(signals: pd.Series, index: pd.Index) -> pd.Series:
        """
        Bảo đảm tín hiệu là Series với index khớp, chỉ nhận giá trị trong [-1, 1].
        """
        if not isinstance(signals, pd.Series):
            signals = pd.Series(signals, index=index)
        if not signals.index.equals(index):
            signals = signals.reindex(index)
        signals = signals.clip(lower=-1, upper=1).fillna(0)
        return signals


# --------- Tiện ích chung cho nhiều chiến lược ---------

def cross_over(a: pd.Series | pd.DataFrame, b: pd.Series | pd.DataFrame) -> pd.Series:
    """Trả về tín hiệu cắt nhau giữa hai chuỗi a và b.
    1 khi a cắt lên b; -1 khi a cắt xuống b; 0 nếu không có sự kiện.
    Bền vững với trường hợp a/b là DataFrame 1 cột (yfinance nhiều khi tạo dạng này).
    """
    # ép về Series nếu là DataFrame 1 cột
    if isinstance(a, pd.DataFrame):
        a = a.iloc[:, 0]
    if isinstance(b, pd.DataFrame):
        b = b.iloc[:, 0]
    a_aligned, b_aligned = a.align(b, join='inner')
    sign_now = np.sign(a_aligned - b_aligned)
    sign_prev = sign_now.shift(1)
    cross_up = (sign_now > 0) & (sign_prev <= 0)
    cross_dn = (sign_now < 0) & (sign_prev >= 0)
    out = pd.Series(0.0, index=a_aligned.index)
    out[cross_up.values] = 1.0
    out[cross_dn.values] = -1.0
    return out
