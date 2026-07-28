"""
Unit tests cho circuits breaker module.

Test coverage:
- No trigger under limits
- Trigger on daily loss
- Trigger on consecutive losses
- Daily reset
- Cooldown behavior
- get_status
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from algo_trading.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


class TestCircuitBreakerNoTrigger:
    """Test that circuit breaker does NOT trigger under limits."""

    def test_small_losses_no_trigger(self):
        """Multiple small losses should not trigger if under threshold."""
        cb = CircuitBreaker(
            max_daily_loss_pct=5.0,
            max_consecutive_losses=5,
            initial_capital=10000.0,
        )
        
        # 3 small losses = 3 * 50 = 150 = 1.5% of 10000
        for _ in range(3):
            cb.record_trade(-50.0)
        
        assert not cb.is_triggered()
        assert cb.check_can_trade()

    def test_wins_reset_consecutive(self):
        """Winning trades should reset consecutive loss counter."""
        cb = CircuitBreaker(
            max_daily_loss_pct=10.0,
            max_consecutive_losses=3,
            initial_capital=10000.0,
        )
        
        cb.record_trade(-10.0)
        cb.record_trade(-10.0)
        assert cb.consecutive_losses == 2
        
        cb.record_trade(100.0)  # Win resets consecutive
        assert cb.consecutive_losses == 0
        assert not cb.is_triggered()


class TestCircuitBreakerDailyLoss:
    """Test trigger on daily loss threshold."""

    def test_trigger_on_daily_loss(self):
        """Should trigger when daily loss exceeds threshold."""
        cb = CircuitBreaker(
            max_daily_loss_pct=5.0,
            max_consecutive_losses=100,  # High so it won't trigger
            initial_capital=10000.0,
        )
        
        # Loss = 500 = 5% of 10000 => should trigger
        cb.record_trade(-500.0)
        
        assert cb.is_triggered()
        assert not cb.check_can_trade()
        assert "Daily loss" in cb.trigger_reason

    def test_trigger_on_cumulative_loss(self):
        """Multiple losses adding up to threshold should trigger."""
        cb = CircuitBreaker(
            max_daily_loss_pct=5.0,
            max_consecutive_losses=100,
            initial_capital=10000.0,
        )
        
        cb.record_trade(-200.0)
        assert not cb.is_triggered()  # 2% < 5%
        
        cb.record_trade(-300.0)
        assert cb.is_triggered()  # 5% == 5%

    def test_unrealized_pnl_check(self):
        """check_can_trade with unrealized PnL should trigger."""
        cb = CircuitBreaker(
            max_daily_loss_pct=5.0,
            max_consecutive_losses=100,
            initial_capital=10000.0,
        )
        
        cb.record_trade(-200.0)  # Realized = -200 (2%)
        
        # Unrealized = -400 => total = -600 (6%) > 5%
        can_trade = cb.check_can_trade(unrealized_pnl=-400.0)
        assert not can_trade
        assert cb.is_triggered()


class TestCircuitBreakerConsecutiveLosses:
    """Test trigger on consecutive losing trades."""

    def test_trigger_on_consecutive_losses(self):
        """Should trigger after N consecutive losses."""
        cb = CircuitBreaker(
            max_daily_loss_pct=99.0,  # High so it won't trigger on pct
            max_consecutive_losses=3,
            initial_capital=10000.0,
        )
        
        cb.record_trade(-1.0)
        cb.record_trade(-1.0)
        assert not cb.is_triggered()  # 2 < 3
        
        cb.record_trade(-1.0)
        assert cb.is_triggered()  # 3 >= 3
        assert "Consecutive" in cb.trigger_reason


class TestCircuitBreakerReset:
    """Test daily reset behavior."""

    def test_manual_reset(self):
        """reset_daily should clear all state."""
        cb = CircuitBreaker(
            max_daily_loss_pct=5.0,
            max_consecutive_losses=3,
            initial_capital=10000.0,
        )
        
        cb.record_trade(-500.0)
        assert cb.is_triggered()
        
        cb.reset_daily()
        assert not cb.is_triggered()
        assert cb.daily_pnl == 0.0
        assert cb.consecutive_losses == 0
        assert cb.check_can_trade()

    def test_auto_reset_on_new_day(self):
        """Should auto-reset when date changes."""
        cb = CircuitBreaker(
            max_daily_loss_pct=5.0,
            max_consecutive_losses=3,
            initial_capital=10000.0,
        )
        
        cb.record_trade(-500.0)
        assert cb.is_triggered()
        
        # Simulate date change
        tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
        cb._current_date = datetime.now(timezone.utc).date() - timedelta(days=1)
        
        # Next check should trigger reset
        assert cb.check_can_trade()  # Auto-resets because date differs


class TestCircuitBreakerCooldown:
    """Test cooldown behavior after trigger."""

    def test_blocked_during_cooldown(self):
        """Should block trading during cooldown period."""
        cb = CircuitBreaker(
            max_daily_loss_pct=5.0,
            max_consecutive_losses=3,
            cooldown_after_trigger_sec=3600,
            initial_capital=10000.0,
        )
        
        cb.record_trade(-500.0)
        assert cb.is_triggered()
        
        # Should be blocked
        assert not cb.check_can_trade()


class TestCircuitBreakerStatus:
    """Test get_status method."""

    def test_status_dict_keys(self):
        """get_status should return all expected keys."""
        cb = CircuitBreaker(
            max_daily_loss_pct=5.0,
            max_consecutive_losses=3,
            initial_capital=10000.0,
        )
        
        status = cb.get_status()
        expected_keys = {
            "triggered", "trigger_reason", "daily_pnl",
            "consecutive_losses", "total_trades_today",
            "winning_trades_today", "losing_trades_today",
            "max_daily_loss_pct", "max_consecutive_losses",
        }
        assert expected_keys == set(status.keys())

    def test_status_reflects_state(self):
        """Status should reflect current state."""
        cb = CircuitBreaker(
            max_daily_loss_pct=5.0,
            max_consecutive_losses=5,
            initial_capital=10000.0,
        )
        
        cb.record_trade(100.0)
        cb.record_trade(-50.0)
        
        status = cb.get_status()
        assert status["daily_pnl"] == 50.0
        assert status["total_trades_today"] == 2
        assert status["winning_trades_today"] == 1
        assert status["losing_trades_today"] == 1
        assert status["consecutive_losses"] == 1
        assert not status["triggered"]
