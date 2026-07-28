"""
Circuit Breaker Module - Tự động dừng trading khi loss vượt ngưỡng.

Chức năng:
- Theo dõi daily P&L (realized + unrealized)
- Dừng trading khi daily loss > max_daily_loss_pct
- Dừng trading khi N lệnh thua liên tiếp
- Auto-reset hàng ngày (UTC midnight)
- Cooldown period sau khi trigger

Cách dùng:
    cb = CircuitBreaker(max_daily_loss_pct=5.0, max_consecutive_losses=5)
    
    # Trước khi vào lệnh:
    if not cb.check_can_trade():
        logger.warning("Circuit breaker triggered!")
        return
    
    # Sau khi đóng lệnh:
    cb.record_trade(pnl=profit_or_loss)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Cấu hình cho circuit breaker."""
    max_daily_loss_pct: float = 5.0        # % loss tối đa trong ngày
    max_consecutive_losses: int = 5         # Số lệnh thua liên tiếp tối đa
    cooldown_after_trigger_sec: int = 3600  # Thời gian chờ sau khi trigger (giây)
    initial_capital: float = 10000.0        # Vốn ban đầu để tính %


class CircuitBreaker:
    """
    Circuit Breaker để bảo vệ tài khoản khỏi drawdown lớn.
    
    Tự động dừng trading khi:
    1. Tổng loss trong ngày vượt max_daily_loss_pct
    2. Thua liên tiếp max_consecutive_losses lệnh
    
    Auto-reset mỗi ngày (UTC midnight).
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None, **kwargs):
        """
        Args:
            config: CircuitBreakerConfig object
            **kwargs: Hoặc truyền trực tiếp params (max_daily_loss_pct, etc.)
        """
        if config is not None:
            self.config = config
        else:
            self.config = CircuitBreakerConfig(**kwargs)
        
        # State tracking
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._trade_history_today: List[float] = []
        self._current_date: Optional[datetime] = None
        self._triggered: bool = False
        self._trigger_reason: Optional[str] = None
        self._trigger_time: Optional[datetime] = None
        self._total_trades_today: int = 0
        self._winning_trades_today: int = 0
        self._losing_trades_today: int = 0
        
        # Initialize date
        self._check_date_reset()
    
    def _check_date_reset(self) -> None:
        """Auto-reset khi sang ngày mới (UTC)."""
        now = datetime.now(timezone.utc)
        today = now.date()
        
        if self._current_date is None or self._current_date != today:
            self._reset_daily_stats()
            self._current_date = today
            logger.info(f"🔄 Circuit Breaker: Reset daily stats cho {today}")
    
    def _reset_daily_stats(self) -> None:
        """Reset tất cả stats hàng ngày."""
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._trade_history_today = []
        self._triggered = False
        self._trigger_reason = None
        self._trigger_time = None
        self._total_trades_today = 0
        self._winning_trades_today = 0
        self._losing_trades_today = 0
    
    def reset_daily(self) -> None:
        """Public method để reset thủ công (cho testing hoặc admin reset)."""
        self._reset_daily_stats()
        self._current_date = datetime.now(timezone.utc).date()
        logger.info("🔄 Circuit Breaker: Manual reset")
    
    def record_trade(self, pnl: float) -> None:
        """
        Ghi nhận kết quả 1 trade.
        
        Args:
            pnl: Profit/Loss của trade (dương = lãi, âm = lỗ)
        """
        self._check_date_reset()
        
        self._daily_pnl += pnl
        self._trade_history_today.append(pnl)
        self._total_trades_today += 1
        
        if pnl >= 0:
            self._winning_trades_today += 1
            self._consecutive_losses = 0
        else:
            self._losing_trades_today += 1
            self._consecutive_losses += 1
        
        # Check triggers
        self._evaluate_triggers()
        
        logger.info(
            f"📝 Circuit Breaker: Trade PnL={pnl:+.2f} | "
            f"Daily PnL={self._daily_pnl:+.2f} | "
            f"Consecutive Losses={self._consecutive_losses} | "
            f"Triggered={self._triggered}"
        )
    
    def _evaluate_triggers(self) -> None:
        """Kiểm tra và trigger circuit breaker nếu cần."""
        if self._triggered:
            return  # Đã trigger rồi
        
        # Check 1: Daily loss threshold
        if self.config.initial_capital > 0:
            daily_loss_pct = abs(self._daily_pnl) / self.config.initial_capital * 100
            if self._daily_pnl < 0 and daily_loss_pct >= self.config.max_daily_loss_pct:
                self._trigger(
                    f"Daily loss {daily_loss_pct:.1f}% >= "
                    f"threshold {self.config.max_daily_loss_pct:.1f}%"
                )
                return
        
        # Check 2: Consecutive losses
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self._trigger(
                f"Consecutive losses {self._consecutive_losses} >= "
                f"threshold {self.config.max_consecutive_losses}"
            )
            return
    
    def _trigger(self, reason: str) -> None:
        """Trigger circuit breaker."""
        self._triggered = True
        self._trigger_reason = reason
        self._trigger_time = datetime.now(timezone.utc)
        logger.warning(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")
    
    def check_can_trade(self, unrealized_pnl: float = 0.0) -> bool:
        """
        Kiểm tra xem có được phép trade không.
        
        Args:
            unrealized_pnl: P&L chưa thực hiện hiện tại (optional)
        
        Returns:
            True nếu được phép trade, False nếu bị block
        """
        self._check_date_reset()
        
        # Nếu đã triggered, check cooldown
        if self._triggered:
            if self._trigger_time is not None:
                elapsed = (datetime.now(timezone.utc) - self._trigger_time).total_seconds()
                if elapsed < self.config.cooldown_after_trigger_sec:
                    remaining = self.config.cooldown_after_trigger_sec - elapsed
                    logger.info(
                        f"⏳ Circuit Breaker: Cooldown {remaining:.0f}s remaining | "
                        f"Reason: {self._trigger_reason}"
                    )
                    return False
                else:
                    # Cooldown hết, nhưng vẫn blocked cho đến hết ngày
                    # (chỉ reset khi sang ngày mới)
                    logger.info(
                        f"🔒 Circuit Breaker: Blocked until end of day | "
                        f"Reason: {self._trigger_reason}"
                    )
                    return False
            return False
        
        # Check unrealized PnL + realized PnL
        total_pnl = self._daily_pnl + unrealized_pnl
        if self.config.initial_capital > 0 and total_pnl < 0:
            total_loss_pct = abs(total_pnl) / self.config.initial_capital * 100
            if total_loss_pct >= self.config.max_daily_loss_pct:
                self._trigger(
                    f"Combined loss (realized+unrealized) {total_loss_pct:.1f}% >= "
                    f"threshold {self.config.max_daily_loss_pct:.1f}%"
                )
                return False
        
        return True
    
    def is_triggered(self) -> bool:
        """Trả về True nếu circuit breaker đang active."""
        self._check_date_reset()
        return self._triggered
    
    @property
    def trigger_reason(self) -> Optional[str]:
        """Lý do trigger (None nếu chưa trigger)."""
        return self._trigger_reason
    
    @property
    def daily_pnl(self) -> float:
        """P&L trong ngày."""
        return self._daily_pnl
    
    @property
    def consecutive_losses(self) -> int:
        """Số lệnh thua liên tiếp."""
        return self._consecutive_losses
    
    @property
    def total_trades_today(self) -> int:
        """Tổng số trades trong ngày."""
        return self._total_trades_today
    
    def get_status(self) -> dict:
        """Trả về dict trạng thái cho UI/Telegram."""
        self._check_date_reset()
        return {
            "triggered": self._triggered,
            "trigger_reason": self._trigger_reason,
            "daily_pnl": self._daily_pnl,
            "consecutive_losses": self._consecutive_losses,
            "total_trades_today": self._total_trades_today,
            "winning_trades_today": self._winning_trades_today,
            "losing_trades_today": self._losing_trades_today,
            "max_daily_loss_pct": self.config.max_daily_loss_pct,
            "max_consecutive_losses": self.config.max_consecutive_losses,
        }
