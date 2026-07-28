from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

from algo_trading.core.metrics import performance_summary, to_returns, infer_freq_label_from_index, has_min_bars_for_freq


@dataclass
class BacktestConfig:
    """Configuration for vectorized backtest."""
    initial_capital: float = 1.0
    leverage: float = 1.0
    allow_short: bool = True
    commission: float = 0.0005
    slippage_bps: float = 1.0
    use_next_open: bool = True
    freq: Optional[str] = None
    risk_per_trade: Optional[float] = None
    fixed_size: Optional[float] = None


@dataclass
class RiskConfig:
    """Risk management configuration for stop loss, take profit, and trailing stops."""
    sl_pct: Optional[float] = None
    tp_pct: Optional[float] = None
    trailing_pct: Optional[float] = None
    sl_atr_k: Optional[float] = None
    tp_atr_k: Optional[float] = None
    trailing_atr_k: Optional[float] = None
    atr_col: str = 'ATR14'


def vectorized_pnl(
    df: pd.DataFrame,
    signals: pd.Series,
    cfg: BacktestConfig,
) -> Tuple[pd.Series, pd.Series]:

    if df.empty or signals.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Align signals with dataframe
    if isinstance(signals, pd.Series):
        signals = signals.reindex(df.index, method='ffill').fillna(0)
    else:
        signals = pd.Series(signals, index=df.index[:len(signals)]).reindex(df.index, method='ffill').fillna(0)
    
    # Get price column (use next open if configured)
    if cfg.use_next_open and 'open' in df.columns:
        prices = df['open'].shift(-1).fillna(df['close'])
    else:
        prices = df['close']
    
    # Calculate returns
    returns = prices.pct_change().fillna(0)
    
    # Position is lagged by 1 (enter at next bar)
    positions = signals.shift(1).fillna(0)
    
    if cfg.leverage != 1.0:
        positions = positions * cfg.leverage
    

    strategy_returns = positions * returns
    
    position_changes = positions.diff().abs()
    
    # Commission: tính trên notional value của position change
    # Nếu position change = 1.0 và price = P, notional = P
    # Commission = P * commission_rate
    # Nhưng vì chúng ta đang dùng returns (fraction), cần normalize
    # Commission cost as fraction of capital = position_change * commission_rate
    commission_cost = position_changes * cfg.commission
    
    # Slippage: tương tự
    slippage_cost = position_changes * (cfg.slippage_bps / 10000)
    
    net_returns = strategy_returns - commission_cost - slippage_cost
    
    net_returns = np.clip(net_returns, -0.5, 0.5)
    
    # Calculate equity curve
    # CRITICAL: Phải dùng net_returns, không phải strategy_returns
    equity = cfg.initial_capital * (1 + net_returns).cumprod()
    
    # Sanity check: Equity không được quá lớn
    max_reasonable_equity = cfg.initial_capital * 1e6  # 1 million times
    if equity.iloc[-1] > max_reasonable_equity:
        import warnings
        warnings.warn(
            f"⚠️ Equity curve quá lớn ({equity.iloc[-1]:.2e}), có thể do:\n"
            f"  1. Returns quá cao (mean: {net_returns.mean():.6f})\n"
            f"  2. Commission/slippage không đủ\n"
            f"  3. Look-ahead bias trong signals\n"
            f"  → Capping equity để tránh numerical issues"
        )
        # Cap equity và recalculate
        equity = np.minimum(equity, max_reasonable_equity)
    
    return equity, net_returns


def barwise_with_stops(
    df: pd.DataFrame,
    signals: pd.Series,
    cfg: BacktestConfig,
    risk: Optional[RiskConfig] = None,
    max_trades: int = 100,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    if df.empty or signals.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()
    
    # Align signals
    if isinstance(signals, pd.Series):
        signals = signals.reindex(df.index, method='ffill').fillna(0)
    else:
        signals = pd.Series(signals, index=df.index[:len(signals)]).reindex(df.index, method='ffill').fillna(0)
    
    # Initialize arrays
    n = len(df)
    equity = np.zeros(n)
    equity[0] = cfg.initial_capital
    position = 0.0
    entry_price = 0.0
    sl_price = None
    tp_price = None
    trailing_high = None
    trailing_low = None
    
    # Get ATR if needed
    atr_series = None
    if risk and (risk.sl_atr_k or risk.tp_atr_k or risk.trailing_atr_k):
        if risk.atr_col in df.columns:
            atr_series = df[risk.atr_col].values
        else:
            # Fallback: calculate ATR if not present
            atr_series = _calculate_atr(df, period=14)
    
    trades = []
    current_trade = None
    trade_count = 0
    
    for i in range(1, n):
        if trade_count >= max_trades:
            break
        equity[i] = equity[i-1]
        signal = signals.iloc[i] if i < len(signals) else 0
        
        # Get prices
        open_price = df['open'].iloc[i] if 'open' in df.columns else df['close'].iloc[i]
        high_price = df['high'].iloc[i] if 'high' in df.columns else df['close'].iloc[i]
        low_price = df['low'].iloc[i] if 'low' in df.columns else df['close'].iloc[i]
        close_price = df['close'].iloc[i]
        if position != 0 and risk:
            # Check SL/TP
            if position > 0:
                if sl_price and low_price <= sl_price:
                    # Stop loss hit
                    exit_price = sl_price
                    pnl_pct = (exit_price / entry_price - 1) * cfg.leverage
                    equity[i] = equity[i-1] * (1 + pnl_pct)
                    if current_trade:
                        current_trade['exit_idx'] = i
                        current_trade['exit_price'] = exit_price
                        current_trade['pnl'] = (exit_price - entry_price) * position
                        trades.append(current_trade)
                        trade_count += 1
                    position = 0
                    current_trade = None
                    sl_price = None
                    tp_price = None
                    trailing_high = None
                    if trade_count >= max_trades:
                        break
                    continue
                elif tp_price and high_price >= tp_price:
                    # Take profit hit
                    exit_price = tp_price
                    pnl_pct = (exit_price / entry_price - 1) * cfg.leverage
                    equity[i] = equity[i-1] * (1 + pnl_pct)
                    if current_trade:
                        current_trade['exit_idx'] = i
                        current_trade['exit_price'] = exit_price
                        current_trade['pnl'] = (exit_price - entry_price) * position
                        trades.append(current_trade)
                        trade_count += 1
                    position = 0
                    current_trade = None
                    sl_price = None
                    tp_price = None
                    trailing_high = None
                    if trade_count >= max_trades:
                        break
                    continue
                elif trailing_high is not None:
                    # Update trailing stop
                    if high_price > trailing_high:
                        trailing_high = high_price
                        if risk.trailing_pct:
                            sl_price = trailing_high * (1 - risk.trailing_pct)
                        elif risk.trailing_atr_k and atr_series is not None:
                            sl_price = trailing_high - risk.trailing_atr_k * atr_series[i]
                    if low_price <= sl_price:
                        exit_price = sl_price
                        pnl_pct = (exit_price / entry_price - 1) * cfg.leverage
                        equity[i] = equity[i-1] * (1 + pnl_pct)
                        if current_trade:
                            current_trade['exit_idx'] = i
                            current_trade['exit_price'] = exit_price
                            current_trade['pnl'] = (exit_price - entry_price) * position
                            trades.append(current_trade)
                            trade_count += 1
                        position = 0
                        current_trade = None
                        sl_price = None
                        tp_price = None
                        trailing_high = None
                        if trade_count >= max_trades:
                            break
                        continue
            elif position < 0:  # Short position
                if sl_price and high_price >= sl_price:
                    # Stop loss hit
                    exit_price = sl_price
                    pnl_pct = (1 - exit_price / entry_price) * cfg.leverage
                    equity[i] = equity[i-1] * (1 + pnl_pct)
                    if current_trade:
                        current_trade['exit_idx'] = i
                        current_trade['exit_price'] = exit_price
                        current_trade['pnl'] = (entry_price - exit_price) * abs(position)
                        trades.append(current_trade)
                        trade_count += 1
                    position = 0
                    current_trade = None
                    sl_price = None
                    tp_price = None
                    trailing_low = None
                    if trade_count >= max_trades:
                        break
                    continue
                elif tp_price and low_price <= tp_price:
                    # Take profit hit
                    exit_price = tp_price
                    pnl_pct = (1 - exit_price / entry_price) * cfg.leverage
                    equity[i] = equity[i-1] * (1 + pnl_pct)
                    if current_trade:
                        current_trade['exit_idx'] = i
                        current_trade['exit_price'] = exit_price
                        current_trade['pnl'] = (entry_price - exit_price) * abs(position)
                        trades.append(current_trade)
                        trade_count += 1
                    position = 0
                    current_trade = None
                    sl_price = None
                    tp_price = None
                    trailing_low = None
                    if trade_count >= max_trades:
                        break
                    continue
                elif trailing_low is not None:
                    # Update trailing stop
                    if low_price < trailing_low:
                        trailing_low = low_price
                        if risk.trailing_pct:
                            sl_price = trailing_low * (1 + risk.trailing_pct)
                        elif risk.trailing_atr_k and atr_series is not None:
                            sl_price = trailing_low + risk.trailing_atr_k * atr_series[i]
                    if high_price >= sl_price:
                        exit_price = sl_price
                        pnl_pct = (1 - exit_price / entry_price) * cfg.leverage
                        equity[i] = equity[i-1] * (1 + pnl_pct)
                        if current_trade:
                            current_trade['exit_idx'] = i
                            current_trade['exit_price'] = exit_price
                            current_trade['pnl'] = (entry_price - exit_price) * abs(position)
                            trades.append(current_trade)
                            trade_count += 1
                        position = 0
                        current_trade = None
                        sl_price = None
                        tp_price = None
                        trailing_low = None
                        if trade_count >= max_trades:
                            break
                        continue
        
        # Enter new position based on signal (chỉ nếu chưa đạt giới hạn)
        if signal != 0 and position == 0 and trade_count < max_trades:
            entry_price = open_price if cfg.use_next_open else close_price
            position = signal
            
            # Calculate position size
            if cfg.fixed_size:
                position = position * cfg.fixed_size
            elif cfg.risk_per_trade and risk and (risk.sl_pct or (risk.sl_atr_k and atr_series is not None)):
                # Position sizing based on risk
                if risk.sl_pct:
                    risk_amount = equity[i-1] * cfg.risk_per_trade
                    sl_distance = entry_price * risk.sl_pct
                elif risk.sl_atr_k and atr_series is not None:
                    risk_amount = equity[i-1] * cfg.risk_per_trade
                    sl_distance = risk.sl_atr_k * atr_series[i]
                else:
                    sl_distance = entry_price * 0.01  # Default 1%
                position_size = risk_amount / sl_distance if sl_distance > 0 else 1.0
                position = position * position_size
            
            # Set stop loss and take profit
            if risk:
                if position > 0:  # Long
                    if risk.sl_pct:
                        sl_price = entry_price * (1 - risk.sl_pct)
                    elif risk.sl_atr_k and atr_series is not None:
                        sl_price = entry_price - risk.sl_atr_k * atr_series[i]
                    
                    if risk.tp_pct:
                        tp_price = entry_price * (1 + risk.tp_pct)
                    elif risk.tp_atr_k and atr_series is not None:
                        tp_price = entry_price + risk.tp_atr_k * atr_series[i]
                    
                    if risk.trailing_pct or risk.trailing_atr_k:
                        trailing_high = high_price
                else:  # Short
                    if risk.sl_pct:
                        sl_price = entry_price * (1 + risk.sl_pct)
                    elif risk.sl_atr_k and atr_series is not None:
                        sl_price = entry_price + risk.sl_atr_k * atr_series[i]
                    
                    if risk.tp_pct:
                        tp_price = entry_price * (1 - risk.tp_pct)
                    elif risk.tp_atr_k and atr_series is not None:
                        tp_price = entry_price - risk.tp_atr_k * atr_series[i]
                    
                    if risk.trailing_pct or risk.trailing_atr_k:
                        trailing_low = low_price
            
            # Calculate risk percent
            risk_percent = None
            if risk:
                if risk.sl_pct:
                    risk_percent = risk.sl_pct
                elif risk.sl_atr_k and atr_series is not None:
                    risk_percent = (risk.sl_atr_k * atr_series[i]) / entry_price
            
            # Record trade entry
            current_trade = {
                'entry_idx': i,
                'entry_time': df.index[i],
                'entry_price': entry_price,
                'position': position,
                'position_size': abs(position),
                'signal': signal,
                'stop_loss': sl_price if risk else None,
                'take_profit': tp_price if risk else None,
                'risk_percent': risk_percent,
            }
        
        # Update equity based on current position
        if position != 0:
            if cfg.use_next_open:
                current_price = open_price
            else:
                current_price = close_price
            
            if position > 0:
                pnl_pct = (current_price / entry_price - 1) * cfg.leverage
            else:
                pnl_pct = (1 - current_price / entry_price) * cfg.leverage
            
            equity[i] = equity[i-1] * (1 + pnl_pct)
            
            # Apply commission and slippage on position changes
            # (simplified: apply on entry/exit only)
    
    # Close any open position at the end (nếu chưa đạt giới hạn)
    if position != 0 and current_trade and trade_count < max_trades:
        exit_price = df['close'].iloc[-1]
        if position > 0:
            pnl_pct = (exit_price / entry_price - 1) * cfg.leverage
        else:
            pnl_pct = (1 - exit_price / entry_price) * cfg.leverage
        equity[-1] = equity[-2] * (1 + pnl_pct)
        current_trade['exit_idx'] = n - 1
        current_trade['exit_price'] = exit_price
        if position > 0:
            current_trade['pnl'] = (exit_price - entry_price) * position
        else:
            current_trade['pnl'] = (entry_price - exit_price) * abs(position)
        trades.append(current_trade)
    
    # Convert to Series
    equity_series = pd.Series(equity, index=df.index)
    returns = equity_series.pct_change().fillna(0)
    
    if trades:
        trades_df = pd.DataFrame(trades)
        # Ensure entry_time exists - sử dụng cách an toàn hơn
        if 'entry_time' not in trades_df.columns and 'entry_idx' in trades_df.columns:
            entry_times = []
            for idx_val in trades_df['entry_idx']:
                try:
                    if pd.notna(idx_val):
                        idx_int = int(idx_val)
                        if 0 <= idx_int < len(df):
                            entry_times.append(df.index[idx_int])
                        else:
                            entry_times.append(None)
                    else:
                        entry_times.append(None)
                except (ValueError, TypeError, IndexError):
                    entry_times.append(None)
            trades_df['entry_time'] = entry_times
        
        if 'exit_idx' in trades_df.columns and 'exit_time' not in trades_df.columns:
            exit_times = []
            for idx_val in trades_df['exit_idx']:
                try:
                    if pd.notna(idx_val):
                        idx_int = int(idx_val)
                        if 0 <= idx_int < len(df):
                            exit_times.append(df.index[idx_int])
                        else:
                            exit_times.append(None)
                    else:
                        exit_times.append(None)
                except (ValueError, TypeError, IndexError):
                    exit_times.append(None)
            trades_df['exit_time'] = exit_times
        
        # Giới hạn 100 lệnh
        if len(trades_df) > 100:
            trades_df = trades_df.head(100).reset_index(drop=True)
        else:
            trades_df = trades_df.reset_index(drop=True)
    else:
        trades_df = pd.DataFrame()
    
    return equity_series, returns, trades_df


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index)
    
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().fillna(tr)
    
    return atr


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    cfg: Optional[BacktestConfig] = None,
    risk: Optional[RiskConfig] = None,
    max_trades: int = 100,
) -> Dict[str, Any]:
    """
    Main entry point for vectorized backtest.
    
    Args:
        df: DataFrame with OHLCV data
        signals: Series with signals (-1, 0, 1)
        cfg: BacktestConfig (defaults to BacktestConfig())
        risk: Optional RiskConfig for SL/TP/Trailing
        max_trades: Maximum number of trades to execute (default: 100)
        
    Returns:
        Dictionary with 'summary', 'equity', 'returns', and 'trades'
    """
    if cfg is None:
        cfg = BacktestConfig()
    
    # Infer frequency if not provided
    freq_label = cfg.freq or infer_freq_label_from_index(df.index)

    # Use barwise_with_stops if risk management is enabled
    if risk and (risk.sl_pct or risk.tp_pct or risk.trailing_pct or 
                 risk.sl_atr_k or risk.tp_atr_k or risk.trailing_atr_k):
        equity, returns, trades = barwise_with_stops(df, signals, cfg, risk, max_trades)
    else:
        equity, returns = vectorized_pnl(df, signals, cfg)
        trades = pd.DataFrame()
    
    # Calculate summary metrics
    summary = performance_summary(equity, returns, freq_label)
    summary['freq'] = freq_label
    summary['has_sufficient_data'] = has_min_bars_for_freq(len(df), freq_label)
    
    # Add additional metrics
    if not trades.empty and 'pnl' in trades.columns:
        summary['TotalTrades'] = len(trades)
        summary['WinningTrades'] = len(trades[trades['pnl'] > 0])
        summary['LosingTrades'] = len(trades[trades['pnl'] < 0])
        if summary['TotalTrades'] > 0:
            summary['WinRate'] = summary['WinningTrades'] / summary['TotalTrades']
        else:
            summary['WinRate'] = 0.0
        if summary['LosingTrades'] > 0:
            avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if summary['WinningTrades'] > 0 else 0
            avg_loss = abs(trades[trades['pnl'] < 0]['pnl'].mean())
            summary['ProfitFactor'] = (avg_win * summary['WinningTrades']) / (avg_loss * summary['LosingTrades']) if avg_loss > 0 else np.inf
        else:
            summary['ProfitFactor'] = np.inf
    else:
        summary['TotalTrades'] = 0
        summary['WinningTrades'] = 0
        summary['LosingTrades'] = 0
        summary['WinRate'] = 0.0
        summary['ProfitFactor'] = 0.0
    
    return {
        'summary': summary,
        'equity': equity,
        'returns': returns,
        'trades': trades,
    }
