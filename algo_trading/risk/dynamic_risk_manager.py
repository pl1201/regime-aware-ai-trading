"""
Dynamic Risk Manager - Quản lý rủi ro thích ứng

Mục tiêu:
- Giảm drawdown từ 44% → < 25%
- Dynamic position sizing dựa trên confidence, volatility, drawdown
- Adaptive SL/TP dựa trên ATR và market conditions
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class RiskConfig:
    """Cấu hình quản lý rủi ro"""
    # Base risk parameters
    max_risk_per_trade: float = 0.02  # Tăng từ 1.5% lên 2%
    max_daily_risk: float = 0.06       # Tăng từ 4% lên 6%
    max_drawdown_limit: float = 0.25   # Tăng từ 20% lên 25%

    # Position sizing parameters
    confidence_multiplier_range: Tuple[float, float] = (0.3, 1.5)  # Tăng từ (0.2, 1.2) lên (0.3, 1.5)
    volatility_multiplier_range: Tuple[float, float] = (0.2, 1.8)  # Tăng từ (0.3, 1.5) lên (0.2, 1.8)
    drawdown_multiplier_range: Tuple[float, float] = (0.05, 1.0)    # Tăng từ (0.1, 0.8) lên (0.05, 1.0)

    # SL/TP parameters
    sl_atr_multiplier: float = 2.5     # Tăng từ 2.0 lên 2.5 ATR
    tp_sl_ratio: float = 3.0          # Tăng từ 2.5 lên 3.0
    min_risk_reward: float = 1.8       # Giảm từ 2.0 xuống 1.8
    max_risk_reward: float = 8.0       # Tăng từ 6.0 lên 8.0

    # Volatility parameters
    volatility_target: float = 0.015   # Tăng từ 1.2% lên 1.5%
    volatility_window: int = 14       # ATR window

    # Trailing stop parameters (mới thêm)
    enable_trailing_stop: bool = True
    trailing_stop_percent: float = 0.015  # Tăng từ 1% lên 1.5%
    trailing_activation_percent: float = 0.015  # Tăng từ 2% xuống 1.5%

    # Correlation filter (mới thêm)
    max_correlation_risk: float = 0.4  # Tăng từ 0.3 lên 0.4

    # Enable/disable features
    enable_dynamic_sizing: bool = True
    enable_adaptive_sltp: bool = True
    enable_risk_limits: bool = True
    enable_trailing_stop: bool = True


class DynamicRiskManager:
    """
    Quản lý rủi ro động dựa trên:
    1. Confidence score (tín hiệu mạnh hơn → vị thế lớn hơn)
    2. Market volatility (biến động cao → vị thế nhỏ hơn)
    3. Current drawdown (drawdown cao → vị thế nhỏ hơn)
    4. Market regime (trending → vị thế lớn hơn)
    5. Correlation với các vị thế đang mở (tránh quá tập trung)
    6. Trailing stop để bảo vệ lợi nhuận
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.trade_history: list[Dict] = []
        self.daily_risk_used: float = 0.0
        self.last_trade_date: Optional[datetime] = None

    def calculate_position_size(
        self,
        account_balance: float,
        confidence_score: float,
        volatility: float,
        current_drawdown: float = 0.0,
        regime_type: str = "trending",  # trending, ranging, volatile, calm
        atr: Optional[float] = None,
        market_regime_confidence: float = 1.0,
        correlated_positions: int = 0,  # Số vị thế correlated đang mở
        open_positions: int = 0  # Tổng số vị thế đang mở
    ) -> float:
        """
        Tính kích thước vị thế động

        Args:
            account_balance: Số dư tài khoản
            confidence_score: Độ tin cậy tín hiệu (0-1)
            volatility: Volatility hiện tại
            current_drawdown: Drawdown hiện tại (0-1)
            regime_type: Loại chế độ thị trường
            atr: Average True Range
            market_regime_confidence: Độ tin cậy regime (0-1)
            correlated_positions: Số vị thế correlated đang mở
            open_positions: Tổng số vị thế đang mở

        Returns:
            Position size (số lượng contract/coins)
        """
        if not self.config.enable_dynamic_sizing:
            return account_balance * self.config.max_risk_per_trade

        # Base risk
        base_risk = account_balance * self.config.max_risk_per_trade

        # 1. Confidence multiplier (0.2 - 1.2)
        conf_min, conf_max = self.config.confidence_multiplier_range
        confidence_mult = conf_min + (confidence_score * (conf_max - conf_min))

        # 2. Volatility multiplier (0.3 - 1.5)
        vol_min, vol_max = self.config.volatility_multiplier_range
        if volatility > 0:
            vol_target = self.config.volatility_target
            # Volatility thấp → multiplier cao (cơ hội tốt)
            # Volatility cao → multiplier thấp (rủi ro cao)
            vol_mult = vol_target / (volatility + 1e-8)
            vol_mult = np.clip(vol_mult, vol_min, vol_max)
        else:
            vol_mult = 1.0

        # 3. Drawdown multiplier (0.1 - 0.8)
        dd_min, dd_max = self.config.drawdown_multiplier_range
        if current_drawdown > 0:
            # Drawdown cao → multiplier thấp
            dd_mult = dd_max - (current_drawdown * (dd_max - dd_min))
            dd_mult = np.clip(dd_mult, dd_min, dd_max)
        else:
            dd_mult = dd_max

        # 4. Regime multiplier
        regime_mult = self._get_regime_multiplier(regime_type)

        # 5. Market confidence multiplier
        market_conf_mult = 0.5 + (market_regime_confidence * 0.5)  # 0.5-1.0

        # 6. Correlation penalty (nếu có nhiều vị thế correlated)
        correlation_penalty = 1.0
        if correlated_positions > 0:
            # Giảm position size theo số vị thế correlated
            correlation_penalty = max(0.3, 1.0 - (correlated_positions * 0.15))

        # 7. Open positions penalty (tránh quá nhiều vị thế cùng lúc)
        open_positions_penalty = 1.0
        if open_positions > 3:  # Nếu có hơn 3 vị thế đang mở
            open_positions_penalty = max(0.5, 1.0 - ((open_positions - 3) * 0.1))

        # Calculate final position size
        position_size = base_risk * confidence_mult * vol_mult * dd_mult * regime_mult * market_conf_mult * correlation_penalty * open_positions_penalty

        # Apply daily risk limit
        if self.config.enable_risk_limits:
            today = datetime.now().date()
            if self.last_trade_date and self.last_trade_date.date() != today:
                self.daily_risk_used = 0.0
                self.last_trade_date = datetime.now()

            if self.daily_risk_used + position_size > account_balance * self.config.max_daily_risk:
                # Reduce position size to stay within daily limit
                remaining_risk = account_balance * self.config.max_daily_risk - self.daily_risk_used
                position_size = min(position_size, remaining_risk)

        # Update daily risk used
        self.daily_risk_used += position_size
        self.last_trade_date = datetime.now()

        return position_size

    def calculate_sl_tp(
        self,
        entry_price: float,
        direction: int,  # 1 for long, -1 for short
        volatility: float,
        atr: float,
        current_price: Optional[float] = None,
        regime_type: str = "trending",
        enable_trailing_stop: bool = True  # Thêm tham số trailing stop
    ) -> Tuple[float, float, Dict]:
        """
        Tính Stop Loss và Take Profit động

        Args:
            entry_price: Giá vào lệnh
            direction: Hướng giao dịch (1: long, -1: short)
            volatility: Volatility hiện tại
            atr: Average True Range
            current_price: Giá hiện tại (nếu có)
            regime_type: Loại chế độ thị trường
            enable_trailing_stop: Có bật trailing stop không

        Returns:
            Tuple of (stop_loss, take_profit, sl_tp_info)
        """
        if not self.config.enable_adaptive_sltp:
            # Default calculation
            sl_distance = entry_price * 0.01  # 1% default
            tp_distance = sl_distance * self.config.tp_sl_ratio

            if direction == 1:  # Long
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
            else:  # Short
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance

            return stop_loss, take_profit, {
                'sl_distance': sl_distance,
                'tp_distance': tp_distance,
                'risk_reward': self.config.tp_sl_ratio,
                'method': 'default'
            }

        # Adaptive calculation based on ATR
        sl_atr_mult = self.config.sl_atr_multiplier
        sl_distance = atr * sl_atr_mult

        # Adjust based on regime
        regime_mult = self._get_regime_sl_multiplier(regime_type)
        sl_distance *= regime_mult

        # Minimum SL distance
        min_sl_distance = entry_price * 0.005  # 0.5%
        sl_distance = max(sl_distance, min_sl_distance)

        # Calculate TP distance based on risk/reward ratio
        tp_distance = sl_distance * self.config.tp_sl_ratio

        # Ensure minimum R:R ratio
        min_rr = self.config.min_risk_reward
        max_rr = self.config.max_risk_reward
        current_rr = tp_distance / (sl_distance + 1e-8)

        if current_rr < min_rr:
            tp_distance = sl_distance * min_rr
        elif current_rr > max_rr:
            tp_distance = sl_distance * max_rr

        if direction == 1:  # Long
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # Short
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        # Ensure SL/TP are reasonable
        if current_price is not None:
            # Don't place SL too close to current price
            min_distance = entry_price * 0.001  # 0.1%
            if direction == 1:  # Long
                stop_loss = min(stop_loss, entry_price - min_distance)
                take_profit = max(take_profit, entry_price + min_distance * 2)
            else:  # Short
                stop_loss = max(stop_loss, entry_price + min_distance)
                take_profit = min(take_profit, entry_price - min_distance * 2)

        sl_tp_info = {
            'sl_distance': sl_distance,
            'tp_distance': tp_distance,
            'risk_reward': tp_distance / (sl_distance + 1e-8),
            'atr': atr,
            'volatility': volatility,
            'regime_type': regime_type,
            'method': 'adaptive_atr',
            'trailing_stop_enabled': enable_trailing_stop,
            'trailing_stop_percent': self.config.trailing_stop_percent if enable_trailing_stop else None,
            'trailing_activation_percent': self.config.trailing_activation_percent if enable_trailing_stop else None
        }

        return stop_loss, take_profit, sl_tp_info

    def _get_regime_multiplier(self, regime_type: str) -> float:
        """Get position size multiplier based on regime type"""
        regime_multipliers = {
            'trending': 1.3,   # Tăng vị thế trong xu hướng (từ 1.2 lên 1.3)
            'volatile': 0.7,   # Giảm vị thế trong biến động cao (từ 0.8 xuống 0.7)
            'ranging': 0.5,    # Giảm vị thế trong đi ngang (từ 0.6 xuống 0.5)
            'calm': 0.9,       # Vị thế trung bình trong yên tĩnh
        }
        return regime_multipliers.get(regime_type, 1.0)

    def _get_regime_sl_multiplier(self, regime_type: str) -> float:
        """Get SL distance multiplier based on regime type"""
        regime_sl_multipliers = {
            'trending': 1.0,   # SL bình thường trong xu hướng
            'volatile': 1.7,   # SL rộng hơn trong biến động cao (từ 1.5 lên 1.7)
            'ranging': 1.3,    # SL rộng hơn trong đi ngang (từ 1.2 lên 1.3)
            'calm': 0.8,       # SL hẹp hơn trong yên tĩnh
        }
        return regime_sl_multipliers.get(regime_type, 1.0)

    def check_risk_limits(
        self,
        account_balance: float,
        current_drawdown: float,
        open_positions: int = 0,  # Thêm số vị thế đang mở
        correlation_risk: float = 0.0  # Thêm rủi ro từ correlation
    ) -> Tuple[bool, Dict]:
        """
        Kiểm tra giới hạn rủi ro

        Returns:
            Tuple of (can_trade, risk_info)
        """
        if not self.config.enable_risk_limits:
            return True, {'can_trade': True, 'reason': 'limits_disabled'}

        # Check drawdown limit
        if current_drawdown > self.config.max_drawdown_limit:
            return False, {
                'can_trade': False,
                'reason': 'drawdown_limit_exceeded',
                'current_drawdown': current_drawdown,
                'max_drawdown_limit': self.config.max_drawdown_limit
            }

        # Check daily risk limit
        today = datetime.now().date()
        if self.last_trade_date and self.last_trade_date.date() == today:
            daily_risk_ratio = self.daily_risk_used / (account_balance * self.config.max_daily_risk)
            if daily_risk_ratio > 0.95:  # 95% limit
                return False, {
                    'can_trade': False,
                    'reason': 'daily_risk_limit_approaching',
                    'daily_risk_used': self.daily_risk_used,
                    'daily_risk_limit': account_balance * self.config.max_daily_risk,
                    'daily_risk_ratio': daily_risk_ratio
                }

        # Check correlation risk limit
        if correlation_risk > self.config.max_correlation_risk:
            return False, {
                'can_trade': False,
                'reason': 'correlation_risk_exceeded',
                'correlation_risk': correlation_risk,
                'max_correlation_risk': self.config.max_correlation_risk
            }

        # Check maximum open positions
        if open_positions > 8:  # Max 8 vị thế cùng lúc
            return False, {
                'can_trade': False,
                'reason': 'max_positions_exceeded',
                'open_positions': open_positions,
                'max_positions': 8
            }

        return True, {
            'can_trade': True,
            'reason': 'within_limits',
            'current_drawdown': current_drawdown,
            'daily_risk_used': self.daily_risk_used,
            'daily_risk_limit': account_balance * self.config.max_daily_risk,
            'open_positions': open_positions,
            'correlation_risk': correlation_risk
        }

    def log_trade(
        self,
        entry_price: float,
        exit_price: float,
        direction: int,
        position_size: float,
        pnl: float,
        trade_time: datetime
    ):
        """Log trade for risk analysis"""
        self.trade_history.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'position_size': position_size,
            'pnl': pnl,
            'trade_time': trade_time,
            'timestamp': datetime.now()
        })

    def get_risk_report(self) -> str:
        """Tạo báo cáo rủi ro"""
        if not self.trade_history:
            return "No trade history available."

        df = pd.DataFrame(self.trade_history)
        total_trades = len(df)
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]

        total_pnl = df['pnl'].sum()
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        winrate = len(winning_trades) / total_trades if total_trades > 0 else 0

        # Calculate drawdown
        cumulative_pnl = df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        report = f"""
🛡️ DYNAMIC RISK MANAGEMENT REPORT
{'='*50}
📊 TRADE STATISTICS:
  • Total trades:     {total_trades}
  • Winrate:          {winrate:.2%}
  • Total PnL:        {total_pnl:.4f}
  • Max Drawdown:     {max_drawdown:.4f}
  • Avg Win:          {avg_win:.4f}
  • Avg Loss:         {avg_loss:.4f}
  • Profit Factor:    {abs(avg_win/avg_loss) if avg_loss > 0 else 0:.2f}

📈 RISK METRICS:
  • Daily risk used:  {self.daily_risk_used:.4f}
  • Last trade date:  {self.last_trade_date}

📈 RECENT TRADES:
"""
        recent_trades = df.tail(10)
        for _, trade in recent_trades.iterrows():
            pnl_sign = "+" if trade['pnl'] > 0 else ""
            report += f"  • {trade['trade_time'].strftime('%Y-%m-%d %H:%M')} | {pnl_sign}{trade['pnl']:.4f}\n"

        return report

    def reset_daily_risk(self):
        """Reset daily risk counter"""
        self.daily_risk_used = 0.0
        self.last_trade_date = None


# Quick test
if __name__ == "__main__":
    # Create risk manager
    config = RiskConfig(
        max_risk_per_trade=0.02,
        max_daily_risk=0.06,
        enable_dynamic_sizing=True,
        enable_adaptive_sltp=True,
        enable_risk_limits=True
    )
    risk_manager = DynamicRiskManager(config)

    # Test position sizing
    account_balance = 10000  # $10,000
    confidence_score = 0.8
    volatility = 0.015  # 1.5%
    current_drawdown = 0.1  # 10%
    atr = 150  # $150

    position_size = risk_manager.calculate_position_size(
        account_balance=account_balance,
        confidence_score=confidence_score,
        volatility=volatility,
        current_drawdown=current_drawdown,
        regime_type="trending",
        atr=atr
    )

    print(f"Position size: ${position_size:.2f}")

    # Test SL/TP calculation
    entry_price = 50000  # BTC price
    stop_loss, take_profit, sl_tp_info = risk_manager.calculate_sl_tp(
        entry_price=entry_price,
        direction=1,  # Long
        volatility=volatility,
        atr=atr,
        regime_type="trending"
    )

    print(f"Entry: ${entry_price:.2f}")
    print(f"Stop Loss: ${stop_loss:.2f}")
    print(f"Take Profit: ${take_profit:.2f}")
    print(f"Risk/Reward: {sl_tp_info['risk_reward']:.2f}")