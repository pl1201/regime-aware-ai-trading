

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

# ============================================================================
# 1. DATA VALIDATION & CLEANING
# ============================================================================

class DataQuality(Enum):
    """Data quality flags"""
    VALID = "valid"
    MISSING_VALUES = "missing_values"
    DUPLICATE_TIMESTAMPS = "duplicate_timestamps"
    NEGATIVE_VOLUME = "negative_volume"
    SUSPICIOUS_PRICES = "suspicious_prices"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class DataValidationResult:
    """Result of data validation"""
    is_valid: bool
    quality_flags: List[DataQuality] = field(default_factory=list)
    cleaned_df: Optional[pd.DataFrame] = None
    issues: List[str] = field(default_factory=list)


def validate_market_data(
    df: pd.DataFrame,
    required_columns: List[str] = None,
    min_bars: int = 100,
    check_ohlc: bool = True,
    check_volume: bool = True,
) -> DataValidationResult:
    """
    Validate và clean market data.
    
    Args:
        df: DataFrame với OHLCV data
        required_columns: Danh sách cột bắt buộc
        min_bars: Số lượng bars tối thiểu
        check_ohlc: Kiểm tra logic OHLC (high >= low, etc.)
        check_volume: Kiểm tra volume >= 0
        
    Returns:
        DataValidationResult với cleaned data và flags
    """
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close']
    
    flags = []
    issues = []
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return DataValidationResult(
            is_valid=False,
            quality_flags=[DataQuality.INSUFFICIENT_DATA],
            issues=[f"Missing columns: {missing_cols}"]
        )
    
    # Check minimum bars
    if len(df) < min_bars:
        flags.append(DataQuality.INSUFFICIENT_DATA)
        issues.append(f"Insufficient data: {len(df)} bars < {min_bars} required")
    
    # Check for missing values
    cleaned_df = df.copy()
    if cleaned_df[required_columns].isnull().any().any():
        flags.append(DataQuality.MISSING_VALUES)
        # Forward fill then backward fill
        cleaned_df[required_columns] = cleaned_df[required_columns].fillna(method='ffill').fillna(method='bfill')
        issues.append("Found missing values, filled with forward/backward fill")
    
    # Check duplicate timestamps
    if cleaned_df.index.duplicated().any():
        flags.append(DataQuality.DUPLICATE_TIMESTAMPS)
        cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='last')]
        issues.append("Found duplicate timestamps, kept last")
    
    # Check OHLC logic
    if check_ohlc:
        invalid_ohlc = (
            (cleaned_df['high'] < cleaned_df['low']) |
            (cleaned_df['high'] < cleaned_df['open']) |
            (cleaned_df['high'] < cleaned_df['close']) |
            (cleaned_df['low'] > cleaned_df['open']) |
            (cleaned_df['low'] > cleaned_df['close'])
        )
        if invalid_ohlc.any():
            flags.append(DataQuality.SUSPICIOUS_PRICES)
            issues.append(f"Found {invalid_ohlc.sum()} bars with invalid OHLC logic")
            # Fix: ensure high >= max(open, close) and low <= min(open, close)
            cleaned_df['high'] = cleaned_df[['high', 'open', 'close']].max(axis=1)
            cleaned_df['low'] = cleaned_df[['low', 'open', 'close']].min(axis=1)
    
    # Check volume
    if check_volume and 'volume' in cleaned_df.columns:
        negative_volume = cleaned_df['volume'] < 0
        if negative_volume.any():
            flags.append(DataQuality.NEGATIVE_VOLUME)
            cleaned_df.loc[negative_volume, 'volume'] = 0
            issues.append(f"Found {negative_volume.sum()} bars with negative volume, set to 0")
    
    is_valid = len(flags) == 0 or DataQuality.INSUFFICIENT_DATA not in flags
    
    return DataValidationResult(
        is_valid=is_valid,
        quality_flags=flags,
        cleaned_df=cleaned_df,
        issues=issues
    )


# ============================================================================
# 2. EXECUTION SIMULATOR
# ============================================================================

@dataclass
class ExecutionConfig:
    """
    Execution simulation configuration.
    
    CRITICAL: Execution costs phải được mô phỏng đúng để metrics có ý nghĩa.
    Bỏ qua execution costs sẽ làm metrics bị ảo (look-ahead bias, survivorship bias).
    """
    # Trading fees
    maker_fee_bps: float = 2.0  # 0.02% maker fee
    taker_fee_bps: float = 4.0  # 0.04% taker fee
    use_maker_taker: bool = False  # Nếu False, dùng taker fee cho tất cả
    
    # Slippage (có thể fixed hoặc volatility-based)
    slippage_bps: float = 5.0  # 0.05% fixed slippage
    use_volatility_slippage: bool = False  # Nếu True, slippage = volatility * multiplier
    volatility_slippage_multiplier: float = 0.5
    
    # Latency
    execution_delay_bars: int = 1  # Delay 1 bar (realistic)
    use_stochastic_latency: bool = False  # Stochastic delay
    latency_distribution: str = "exponential"  # "exponential", "normal"
    
    # Partial fills (advanced)
    enable_partial_fills: bool = False
    fill_probability: float = 1.0  # Probability of full fill
    
    def calculate_fee(self, notional: float, is_maker: bool = False) -> float:
        """Calculate trading fee"""
        fee_bps = self.maker_fee_bps if (self.use_maker_taker and is_maker) else self.taker_fee_bps
        return notional * (fee_bps / 10000)
    
    def calculate_slippage(self, notional: float, price: float, volatility: float = None) -> float:
        """Calculate slippage cost"""
        if self.use_volatility_slippage and volatility is not None:
            slippage_pct = volatility * self.volatility_slippage_multiplier
        else:
            slippage_pct = self.slippage_bps / 10000
        
        return notional * slippage_pct


class ExecutionSimulator:
    """
    Simulate realistic execution với fees, slippage, latency.
    
    Vì sao quan trọng:
    - Fees và slippage có thể ăn mất 0.1-0.5% mỗi trade
    - Latency có thể làm bạn vào lệnh ở giá xấu hơn
    - Bỏ qua execution = overestimate returns 20-50%
    """
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
    
    def simulate_execution(
        self,
        target_position: float,
        current_position: float,
        price: float,
        volatility: float = None,
        timestamp: pd.Timestamp = None,
    ) -> Dict[str, Any]:
        """
        Simulate execution của một order.
        
        Returns:
            Dict với:
            - executed_position: Position thực tế sau execution
            - execution_price: Giá thực tế (có slippage)
            - total_cost: Tổng cost (fees + slippage)
            - execution_delay: Delay trong bars
        """
        # Calculate position change
        position_change = target_position - current_position
        
        if abs(position_change) < 1e-8:  # No change
            return {
                'executed_position': current_position,
                'execution_price': price,
                'total_cost': 0.0,
                'execution_delay': 0,
            }
        
        # Apply latency
        execution_delay = self.config.execution_delay_bars
        if self.config.use_stochastic_latency:
            if self.config.latency_distribution == "exponential":
                execution_delay = int(np.random.exponential(1.0)) + 1
            elif self.config.latency_distribution == "normal":
                execution_delay = max(1, int(np.random.normal(1.0, 0.5)))
        
        # Apply slippage
        slippage_cost = self.config.calculate_slippage(
            abs(position_change) * price,
            price,
            volatility
        )
        execution_price = price * (1 + np.sign(position_change) * slippage_cost / (abs(position_change) * price))
        
        # Apply fees
        notional = abs(position_change) * execution_price
        fee = self.config.calculate_fee(notional, is_maker=False)
        total_cost = fee + slippage_cost
        
        # Partial fills (simplified)
        if self.config.enable_partial_fills:
            fill_ratio = np.random.binomial(1, self.config.fill_probability)
            executed_position = current_position + position_change * fill_ratio
        else:
            executed_position = target_position
        
        return {
            'executed_position': executed_position,
            'execution_price': execution_price,
            'total_cost': total_cost,
            'execution_delay': execution_delay,
        }


# ============================================================================
# 3. RISK ENGINE (ĐỘC LẬP VỚI STRATEGY)
# ============================================================================

@dataclass
class RiskConfig:
    """
    Risk management configuration.
    
    CRITICAL: Risk engine PHẢI độc lập với strategy.
    Strategy không được tự kiểm soát risk (single responsibility).
    """
    # Position sizing
    max_position_size_pct: float = 0.1  # Max 10% equity per position
    max_leverage: float = 1.0  # Max leverage (1.0 = no leverage)
    max_concurrent_trades: int = 10  # Max số positions cùng lúc
    
    # Risk limits
    max_portfolio_risk_pct: float = 0.2  # Max 20% portfolio risk
    max_drawdown_limit: float = 0.25  # Kill switch at 25% drawdown
    max_daily_loss_pct: float = 0.05  # Max 5% daily loss
    
    # Position sizing method
    sizing_method: str = "fixed"  # "fixed", "volatility_adjusted", "kelly_capped"
    volatility_lookback: int = 20  # For volatility-adjusted sizing
    kelly_fraction: float = 0.25  # Fraction of Kelly criterion (conservative)
    
    # Stop loss / Take profit (risk layer, not strategy)
    use_stop_loss: bool = False
    stop_loss_pct: float = 0.02  # 2% stop loss
    use_take_profit: bool = False
    take_profit_pct: float = 0.04  # 4% take profit
    
    def calculate_position_size(
        self,
        equity: float,
        signal_strength: float = 1.0,
        volatility: float = None,
        win_rate: float = None,
        avg_win: float = None,
        avg_loss: float = None,
    ) -> float:
        """
        Calculate position size theo sizing method.
        
        Args:
            equity: Current equity
            signal_strength: Signal strength (-1 to 1)
            volatility: Current volatility (for volatility-adjusted)
            win_rate: Historical win rate (for Kelly)
            avg_win: Average win (for Kelly)
            avg_loss: Average loss (for Kelly)
        """
        if self.sizing_method == "fixed":
            size = equity * self.max_position_size_pct * abs(signal_strength)
        
        elif self.sizing_method == "volatility_adjusted":
            if volatility is None or volatility == 0:
                size = equity * self.max_position_size_pct * abs(signal_strength)
            else:
                # Inverse volatility scaling
                target_vol = 0.02  # Target 2% volatility
                vol_adj = min(target_vol / volatility, 2.0)  # Cap at 2x
                size = equity * self.max_position_size_pct * abs(signal_strength) * vol_adj
        
        elif self.sizing_method == "kelly_capped":
            if win_rate is None or avg_win is None or avg_loss is None or avg_loss == 0:
                size = equity * self.max_position_size_pct * abs(signal_strength)
            else:
                # Kelly criterion: f = (p * b - q) / b
                # where p = win_rate, q = 1 - p, b = avg_win / avg_loss
                b = avg_win / abs(avg_loss)
                kelly = (win_rate * b - (1 - win_rate)) / b
                kelly = max(0, min(kelly, 0.25))  # Cap at 25%
                size = equity * kelly * self.kelly_fraction * abs(signal_strength)
        else:
            size = equity * self.max_position_size_pct * abs(signal_strength)
        
        # Apply max position size limit
        size = min(size, equity * self.max_position_size_pct)
        
        return size


class RiskEngine:
    """
    Risk management engine độc lập với strategy.
    
    Vì sao quan trọng:
    - Strategy chỉ nên focus vào signal generation
    - Risk management là một concern riêng (separation of concerns)
    - Cho phép backtest với nhiều risk profiles khác nhau
    """
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.current_positions: Dict[str, float] = {}  # symbol -> position
        self.daily_pnl: float = 0.0
        self.last_reset_date: Optional[pd.Timestamp] = None
    
    def check_risk_limits(
        self,
        equity: float,
        current_drawdown: float,
        timestamp: pd.Timestamp,
    ) -> Tuple[bool, str]:
        """
        Check risk limits và kill switch.
        
        Returns:
            (is_allowed, reason)
        """
        # Check max drawdown
        if abs(current_drawdown) > self.config.max_drawdown_limit:
            return False, f"Max drawdown limit exceeded: {current_drawdown:.2%}"
        
        # Check daily loss
        if timestamp is not None and self.last_reset_date is not None:
            if timestamp.date() != self.last_reset_date.date():
                self.daily_pnl = 0.0
                self.last_reset_date = timestamp
        
        if self.daily_pnl < -equity * self.config.max_daily_loss_pct:
            return False, f"Daily loss limit exceeded: {self.daily_pnl:.2f}"
        
        # Check max concurrent trades
        active_positions = sum(1 for pos in self.current_positions.values() if abs(pos) > 1e-8)
        if active_positions >= self.config.max_concurrent_trades:
            return False, f"Max concurrent trades exceeded: {active_positions}"
        
        return True, "OK"
    
    def apply_position_limits(
        self,
        target_position: float,
        equity: float,
        signal_strength: float = 1.0,
        volatility: float = None,
    ) -> float:
        """
        Apply position sizing và leverage limits.
        
        Returns:
            Adjusted position size
        """
        # Calculate position size
        position_size = self.config.calculate_position_size(
            equity=equity,
            signal_strength=signal_strength,
            volatility=volatility,
        )
        
        # Apply leverage limit
        max_notional = equity * self.config.max_leverage
        if abs(target_position) * position_size > max_notional:
            position_size = max_notional / abs(target_position) if target_position != 0 else 0
        
        return np.sign(target_position) * min(abs(target_position), position_size / abs(target_position) if target_position != 0 else 0)


# ============================================================================
# 4. EQUITY CURVE ENGINE (MARK-TO-MARKET)
# ============================================================================

class EquityCurveEngine:
    """
    Equity curve engine với mark-to-market đúng cách.
    
    CRITICAL RULES:
    1. Equity phải được mark-to-market theo từng bar
    2. Mọi metric đều tính trên equity curve, không tính trên PnL rời rạc
    3. Equity = Previous Equity + Unrealized PnL + Realized PnL - Costs
    
    ANTI-PATTERNS (SAI):
    - Reset equity về initial capital
    - Dùng PnL / initial capital thay vì equity curve
    - Không mark-to-market unrealized positions
    """
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.equity_curve: List[float] = [initial_capital]
        self.cash: float = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> position size
        self.entry_prices: Dict[str, float] = {}  # symbol -> entry price
    
    def update_equity(
        self,
        current_prices: Dict[str, float],
        realized_pnl: float = 0.0,
        costs: float = 0.0,
    ) -> float:
        """
        Update equity với mark-to-market.
        
        Formula:
            Equity(t) = Cash(t) + Sum(Unrealized PnL(t))
            Cash(t) = Cash(t-1) + Realized PnL - Costs
        
        Args:
            current_prices: Dict symbol -> current price
            realized_pnl: Realized PnL từ closed positions
            costs: Execution costs (fees + slippage)
        
        Returns:
            Current equity
        """
        # Update cash
        self.cash += realized_pnl - costs
        
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        for symbol, position in self.positions.items():
            if abs(position) > 1e-8 and symbol in current_prices:
                entry_price = self.entry_prices.get(symbol, current_prices[symbol])
                unrealized_pnl += (current_prices[symbol] - entry_price) * position
        
        # Total equity
        equity = self.cash + unrealized_pnl
        self.equity_curve.append(equity)
        
        return equity
    
    def open_position(self, symbol: str, size: float, price: float):
        """Open a new position"""
        if symbol in self.positions:
            # Average entry price
            total_size = self.positions[symbol] + size
            if total_size != 0:
                self.entry_prices[symbol] = (
                    (self.entry_prices[symbol] * self.positions[symbol] + price * size) / total_size
                )
            self.positions[symbol] = total_size
        else:
            self.positions[symbol] = size
            self.entry_prices[symbol] = price
    
    def close_position(self, symbol: str, price: float) -> float:
        """Close a position and return realized PnL"""
        if symbol not in self.positions or abs(self.positions[symbol]) < 1e-8:
            return 0.0
        
        entry_price = self.entry_prices[symbol]
        position = self.positions[symbol]
        realized_pnl = (price - entry_price) * position
        
        # Remove position
        del self.positions[symbol]
        del self.entry_prices[symbol]
        
        return realized_pnl
    
    def get_equity_series(self, index: pd.DatetimeIndex) -> pd.Series:
        """Get equity as pandas Series"""
        return pd.Series(self.equity_curve[:len(index)], index=index[:len(self.equity_curve)])


# ============================================================================
# 5. METRICS ENGINE (INDUSTRY STANDARD)
# ============================================================================

# Sẽ được implement trong file tiếp theo để tránh quá dài



