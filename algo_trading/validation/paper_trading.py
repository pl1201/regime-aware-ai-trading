"""
Paper Trading Framework for Strategy Validation

This module implements a comprehensive paper trading system to validate
trading strategies in real-time with simulated money.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
import warnings
from datetime import datetime, timedelta
import json
import time
import threading
from dataclasses import dataclass, asdict
from collections import deque
import matplotlib.pyplot as plt

@dataclass
class Trade:
    """Trade data structure"""
    timestamp: datetime
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: Optional[float] = None
    size: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str = "open"  # open, closed, stopped, taken
    pnl: float = 0.0
    duration: Optional[timedelta] = None

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    direction: int
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime

class PaperTradingSimulator:
    """
    Paper Trading Simulator for strategy validation
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_positions: int = 10,
        slippage: float = 0.001,  # 0.1% slippage
        commission: float = 0.001  # 0.1% commission
    ):
        """
        Initialize Paper Trading Simulator

        Args:
            initial_balance: Starting account balance
            max_positions: Maximum concurrent positions
            slippage: Slippage percentage
            commission: Commission percentage
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_positions = max_positions
        self.slippage = slippage
        self.commission = commission

        # Trading state
        self.positions: List[Position] = []
        self.trade_history: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self.open_trades: Dict[str, Trade] = {}  # symbol -> Trade

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = initial_balance

        # Market data simulation
        self.current_prices = {}
        self.price_history = {}
        self.timestamp = datetime.now()

        print(f"Paper Trading Simulator Initialized")
        print(f"   Initial Balance: ${initial_balance:,.2f}")
        print(f"   Max Positions: {max_positions}")
        print(f"   Slippage: {slippage*100:.2f}%")
        print(f"   Commission: {commission*100:.2f}%")

    def update_market_data(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Update market data for a symbol

        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Timestamp (optional)
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.current_prices[symbol] = price
        self.timestamp = timestamp

        # Store price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append((timestamp, price))

        # Update equity history
        self._update_equity_history()

    def execute_trade(
        self,
        symbol: str,
        direction: int,  # 1 for long, -1 for short
        size: float,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Execute a trade

        Args:
            symbol: Trading symbol
            direction: Trade direction (1 for long, -1 for short)
            size: Position size
            entry_price: Entry price (optional, uses current price if None)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Trade object if successful, None if failed
        """
        # Check if we can open position
        if len(self.positions) >= self.max_positions:
            warnings.warn("Maximum positions reached")
            return None

        # Get current price if not provided
        if entry_price is None:
            if symbol not in self.current_prices:
                warnings.warn(f"No price data for {symbol}")
                return None
            entry_price = self.current_prices[symbol]

        # Apply slippage
        if direction == 1:  # Long
            entry_price = entry_price * (1 + self.slippage)
        else:  # Short
            entry_price = entry_price * (1 - self.slippage)

        # Check if we have enough balance
        required_margin = size * entry_price
        commission_cost = required_margin * self.commission
        total_cost = required_margin + commission_cost

        if self.balance < total_cost:
            warnings.warn("Insufficient balance")
            return None

        # Create position
        position = Position(
            symbol=symbol,
            direction=direction,
            size=size,
            entry_price=entry_price,
            stop_loss=stop_loss or entry_price,
            take_profit=take_profit or entry_price,
            timestamp=self.timestamp
        )

        self.positions.append(position)

        # Create trade record
        trade = Trade(
            timestamp=self.timestamp,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status="open"
        )

        self.open_trades[symbol] = trade
        self.trade_history.append(trade)

        # Deduct from balance
        self.balance -= total_cost

        self.total_trades += 1
        print(f"Trade executed: {symbol} {'LONG' if direction == 1 else 'SHORT'} "
              f"Size: {size:.4f} Price: {entry_price:.2f}")

        return trade

    def close_position(
        self,
        symbol: str,
        exit_price: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Close an open position

        Args:
            symbol: Trading symbol
            exit_price: Exit price (optional, uses current price if None)

        Returns:
            Closed trade object if successful, None if failed
        """
        # Find position
        position = None
        position_idx = None
        for i, pos in enumerate(self.positions):
            if pos.symbol == symbol:
                position = pos
                position_idx = i
                break

        if position is None:
            warnings.warn(f"No open position for {symbol}")
            return None

        # Get exit price
        if exit_price is None:
            if symbol not in self.current_prices:
                warnings.warn(f"No price data for {symbol}")
                return None
            exit_price = self.current_prices[symbol]

        # Apply slippage
        if position.direction == 1:  # Long
            exit_price = exit_price * (1 - self.slippage)
        else:  # Short
            exit_price = exit_price * (1 + self.slippage)

        # Calculate PnL
        if position.direction == 1:  # Long
            pnl = (exit_price - position.entry_price) * position.size
        else:  # Short
            pnl = (position.entry_price - exit_price) * position.size

        # Apply commission
        commission_cost = (position.entry_price * position.size * self.commission) + \
                         (exit_price * position.size * self.commission)
        pnl -= commission_cost

        # Update trade
        trade = self.open_trades.get(symbol)
        if trade:
            trade.exit_price = exit_price
            trade.status = "closed"
            trade.pnl = pnl
            trade.duration = self.timestamp - trade.timestamp
            del self.open_trades[symbol]

        # Remove position
        self.positions.pop(position_idx)

        # Update balance
        self.balance += position.entry_price * position.size + pnl

        # Update statistics
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1

        print(f"Position closed: {symbol} PnL: ${pnl:.2f}")

        return trade

    def check_stop_loss_take_profit(self):
        """
        Check and execute stop loss/take profit orders
        """
        closed_symbols = []

        for symbol, trade in self.open_trades.items():
            if symbol not in self.current_prices:
                continue

            current_price = self.current_prices[symbol]
            position = None
            for pos in self.positions:
                if pos.symbol == symbol:
                    position = pos
                    break

            if position is None:
                continue

            # Check stop loss
            if position.direction == 1:  # Long
                if current_price <= position.stop_loss:
                    print(f"Stop loss triggered: {symbol}")
                    self.close_position(symbol, position.stop_loss)
                    closed_symbols.append(symbol)
            else:  # Short
                if current_price >= position.stop_loss:
                    print(f"Stop loss triggered: {symbol}")
                    self.close_position(symbol, position.stop_loss)
                    closed_symbols.append(symbol)

            # Check take profit
            if position.direction == 1:  # Long
                if current_price >= position.take_profit:
                    print(f"Take profit triggered: {symbol}")
                    self.close_position(symbol, position.take_profit)
                    closed_symbols.append(symbol)
            else:  # Short
                if current_price <= position.take_profit:
                    print(f"Take profit triggered: {symbol}")
                    self.close_position(symbol, position.take_profit)
                    closed_symbols.append(symbol)

        # Clean up closed trades
        for symbol in closed_symbols:
            if symbol in self.open_trades:
                del self.open_trades[symbol]

    def _update_equity_history(self):
        """
        Update equity history with current balance and positions
        """
        total_equity = self.balance

        # Add value of open positions
        for position in self.positions:
            if position.symbol in self.current_prices:
                current_price = self.current_prices[position.symbol]
                if position.direction == 1:  # Long
                    position_value = (current_price - position.entry_price) * position.size
                else:  # Short
                    position_value = (position.entry_price - current_price) * position.size
                total_equity += position_value

        self.equity_history.append((self.timestamp, total_equity))

        # Update drawdown
        self.peak_equity = max(self.peak_equity, total_equity)
        current_drawdown = (self.peak_equity - total_equity) / self.peak_equity if self.peak_equity > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics

        Returns:
            Performance metrics dictionary
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'current_balance': self.balance,
                'current_equity': self._get_current_equity()
            }

        # Calculate metrics
        closed_trades = [t for t in self.trade_history if t.status == "closed"]
        winning_trades = [t for t in closed_trades if t.pnl > 0]

        total_pnl = sum(t.pnl for t in closed_trades)
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0.0
        total_return = (self.balance - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0.0

        # Sharpe ratio (simplified)
        if closed_trades:
            daily_returns = []
            for trade in closed_trades:
                if trade.pnl != 0:
                    daily_returns.append(trade.pnl / self.initial_balance)

            if daily_returns:
                avg_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'current_balance': self.balance,
            'current_equity': self._get_current_equity()
        }

    def _get_current_equity(self) -> float:
        """
        Calculate current total equity

        Returns:
            Current equity value
        """
        equity = self.balance

        # Add open positions value
        for position in self.positions:
            if position.symbol in self.current_prices:
                current_price = self.current_prices[position.symbol]
                if position.direction == 1:  # Long
                    position_value = (current_price - position.entry_price) * position.size
                else:  # Short
                    position_value = (position.entry_price - current_price) * position.size
                equity += position_value

        return equity

    def plot_performance(self):
        """
        Plot performance charts
        """
        if not self.equity_history:
            print("No performance data to plot")
            return

        # Extract data
        timestamps = [t[0] for t in self.equity_history]
        equity_values = [t[1] for t in self.equity_history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Equity curve
        ax1.plot(timestamps, equity_values, 'b-', linewidth=2)
        ax1.set_title('Paper Trading Performance - Equity Curve')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)

        # Drawdown
        peak_values = []
        peak = equity_values[0]
        for value in equity_values:
            peak = max(peak, value)
            peak_values.append(peak)

        drawdowns = [100 * (peak - value) / peak if peak > 0 else 0
                    for peak, value in zip(peak_values, equity_values)]

        ax2.plot(timestamps, drawdowns, 'r-', linewidth=2)
        ax2.set_title('Drawdown (%)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_detailed_report(self) -> str:
        """
        Generate detailed performance report

        Returns:
            Formatted report string
        """
        metrics = self.get_performance_metrics()

        report = []
        report.append("PAPER TRADING DETAILED REPORT")
        report.append("=" * 50)
        report.append(f"Current Balance:     ${metrics['current_balance']:,.2f}")
        report.append(f"Current Equity:      ${metrics['current_equity']:,.2f}")
        report.append(f"Total PnL:           ${metrics['total_pnl']:,.2f}")
        report.append(f"Total Return:        {metrics['total_return']*100:.2f}%")
        report.append(f"Total Trades:        {metrics['total_trades']}")
        report.append(f"Winning Trades:      {metrics['winning_trades']}")
        report.append(f"Win Rate:            {metrics['win_rate']*100:.2f}%")
        report.append(f"Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")
        report.append(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        report.append("")
        report.append("OPEN POSITIONS:")
        report.append("-" * 30)
        if self.positions:
            for position in self.positions:
                report.append(f"{position.symbol:<10} {position.direction:>5} "
                            f"Size: {position.size:>8.4f} "
                            f"Entry: {position.entry_price:>8.2f}")
        else:
            report.append("No open positions")

        return "\n".join(report)

    def save_state(self, filepath: str):
        """
        Save current state to file

        Args:
            filepath: Path to save file
        """
        state = {
            'balance': self.balance,
            'positions': [asdict(pos) for pos in self.positions],
            'trade_history': [asdict(trade) for trade in self.trade_history],
            'equity_history': [(t.isoformat(), v) for t, v in self.equity_history],
            'open_trades': {symbol: asdict(trade) for symbol, trade in self.open_trades.items()},
            'current_prices': self.current_prices,
            'timestamp': self.timestamp.isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"State saved to {filepath}")

    def load_state(self, filepath: str):
        """
        Load state from file

        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.balance = state['balance']
        self.positions = [Position(**pos) for pos in state['positions']]
        self.trade_history = [Trade(**trade) for trade in state['trade_history']]
        self.equity_history = [(datetime.fromisoformat(t), v) for t, v in state['equity_history']]
        self.open_trades = {symbol: Trade(**trade) for symbol, trade in state['open_trades'].items()}
        self.current_prices = state['current_prices']
        self.timestamp = datetime.fromisoformat(state['timestamp'])

        print(f"State loaded from {filepath}")

    def run_backtest(
        self,
        model,
        data: pd.DataFrame,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h"
    ):
        """
        Run backtest with historical data

        Args:
            model: Trading model
            data: Historical data DataFrame with columns: timestamp, open, high, low, close, volume
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
        """
        print(f"Running backtest for {symbol} on {timeframe} timeframe...")
        print(f"Data points: {len(data)}")

        # Sort data by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)

        for idx, row in data.iterrows():
            # Update market data
            self.update_market_data(symbol, row['close'], row['timestamp'])

            # Check stop loss/take profit
            self.check_stop_loss_take_profit()

            # Get model prediction (every hour for example)
            if idx % 60 == 0:  # Every 60 minutes
                try:
                    # Prepare features (simplified)
                    features = row[['open', 'high', 'low', 'close', 'volume']].values.reshape(1, -1)

                    # Get prediction
                    prediction = model.predict(features)[0] if hasattr(model, 'predict') else 0

                    # Execute trade based on prediction
                    if prediction == 1 and len(self.positions) < self.max_positions:
                        # Buy signal
                        size = self.balance * 0.02 / row['close']  # 2% of balance
                        sl = row['close'] * 0.98  # 2% stop loss
                        tp = row['close'] * 1.04  # 4% take profit
                        self.execute_trade(symbol, 1, size, row['close'], sl, tp)

                    elif prediction == -1 and len(self.positions) < self.max_positions:
                        # Sell signal
                        size = self.balance * 0.02 / row['close']  # 2% of balance
                        sl = row['close'] * 1.02  # 2% stop loss
                        tp = row['close'] * 0.96  # 4% take profit
                        self.execute_trade(symbol, -1, size, row['close'], sl, tp)

                    elif prediction == 0 and self.positions:
                        # Close all positions
                        for pos in self.positions[:]:
                            self.close_position(pos.symbol, row['close'])

                except Exception as e:
                    warnings.warn(f"Prediction failed at index {idx}: {e}")

            # Progress indicator
            if idx % 1000 == 0:
                print(f"   Processed {idx}/{len(data)} data points...")

        print("Backtest completed!")
        print(self.get_detailed_report())


# Quick test
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_points = 1000

    # Generate sample price data
    timestamps = pd.date_range('2023-01-01', periods=n_points, freq='1H')
    prices = 40000 + np.cumsum(np.random.randn(n_points) * 100)  # Random walk around 40000

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.random.rand(n_points) * 200,
        'low': prices - np.random.rand(n_points) * 200,
        'close': prices + np.random.randn(n_points) * 50,
        'volume': np.random.rand(n_points) * 1000
    })

    # Create dummy model (replace with actual model)
    class DummyModel:
        def predict(self, X):
            # Random predictions
            return np.random.choice([-1, 0, 1], len(X))

    model = DummyModel()

    # Create paper trading simulator
    simulator = PaperTradingSimulator(
        initial_balance=10000.0,
        max_positions=5,
        slippage=0.001,
        commission=0.001
    )

    # Run backtest
    print("Running Paper Trading Backtest...")
    simulator.run_backtest(model, data, "BTC/USDT", "1h")

    # Show final report
    print("\n" + "="*60)
    print("FINAL PAPER TRADING REPORT")
    print("="*60)
    print(simulator.get_detailed_report())