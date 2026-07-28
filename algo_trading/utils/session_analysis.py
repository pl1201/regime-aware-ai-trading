from __future__ import annotations

"""
Phân tích hành vi theo khung giờ / phiên (Asia, Europe, US).

Ý tưởng:
- Nhận DataFrame giá với DatetimeIndex (UTC hoặc timezone khác)
- Tính return theo bar
- Gán nhãn phiên (asia / europe / us / other) dựa trên giờ trong ngày
- Tính thống kê theo từng phiên: mean return, std, Sharpe xấp xỉ, tần suất

Lưu ý:
- Mặc định coi index đang ở UTC. Nếu dữ liệu ở timezone khác, truyền tz tương ứng.
"""

from algo_trading.core.sessions import (  # type: ignore
    SessionConfig,
    label_sessions,
    session_return_stats,
    hour_of_day_return_stats,
)

__all__ = [
    "SessionConfig",
    "label_sessions",
    "session_return_stats",
    "hour_of_day_return_stats",
]