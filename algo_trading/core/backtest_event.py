from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from algo_trading.core.metrics import (
    performance_summary,
    to_returns,
    infer_freq_label_from_index,
    has_min_bars_for_freq,
)
from algo_trading.core.backtest_vectorized import RiskConfig
from algo_trading.core.risk_exit_engine import (
    RiskExitEngineConfig,
    ensure_atr14,
    risk_exit_check_intrabar,
)


@dataclass
class EventConfig:
    """Configuration for event-driven backtest."""
    initial_cash: float = 10000.0
    leverage: float = 1.0
    allow_short: bool = True
    commission: float = 0.0005
    slippage_bps: float = 1.0
    use_next_open: bool = True
    price_col: str = 'close'
    open_col: str = 'open'
    high_col: str = 'high'
    low_col: str = 'low'
    freq: Optional[str] = None
    # Optional pre-trade cost budget check.
    # If enabled, entry requires expected edge (in bps) >= roundtrip_cost_bps + edge_buffer_bps.
    enable_cost_budget_check: bool = False
    expected_edge_col: Optional[str] = None
    edge_buffer_bps: float = 0.0


class Broker:
    """Simple broker that executes trades and manages portfolio."""
    
    def __init__(self, config: EventConfig):
        self.config = config
        self.cash = config.initial_cash
        self.position = 0.0  # Current position size
        self.entry_price = 0.0
        self.equity_history: List[float] = [config.initial_cash]
        self.trades: List[Dict[str, Any]] = []
        self.current_trade: Optional[Dict[str, Any]] = None
        
    def get_equity(self, current_price: float) -> float:
        if self.position == 0:
            return self.cash
        unrealized_pnl = (current_price - self.entry_price) * self.position * self.config.leverage
        return self.cash + unrealized_pnl
    
    def enter_position(self, price: float, size: float, signal: int, idx: int, timestamp: Any) -> bool:
        if self.position != 0:
            return False 
        
        # Calculate cost including commission and slippage
        cost = abs(size) * price
        commission_cost = cost * self.config.commission
        slippage_cost = cost * (self.config.slippage_bps / 10000)
        total_cost = cost + commission_cost + slippage_cost
        
        if total_cost > self.cash:
            return False  # Not enough cash
        
        self.cash -= total_cost
        self.position = size
        self.entry_price = price
        
        self.current_trade = {
            'entry_idx': idx,
            'entry_time': timestamp,
            'entry_price': price,
            'position': size,
            'position_size': abs(size),
            'signal': signal,
            'entry_notional': float(cost),
            'entry_commission': float(commission_cost),
            'entry_slippage': float(slippage_cost),
            'entry_cost_total': float(commission_cost + slippage_cost),
            'stop_loss': None,  # Will be set by risk_check_intrabar
            'take_profit': None,  # Will be set by risk_check_intrabar
            'risk_percent': None,  # Will be 
        }
        return True
    
    def exit_position(self, price: float, idx: int, timestamp: Any, reason: str = 'signal') -> bool:
        """Exit current position."""
        if self.position == 0:
            return False
        
        # Calculate proceeds
        proceeds = abs(self.position) * price
        commission_cost = proceeds * self.config.commission
        slippage_cost = proceeds * (self.config.slippage_bps / 10000)
        net_proceeds = proceeds - commission_cost - slippage_cost
        
        # Calculate gross PnL (before transaction costs)
        if self.position > 0:
            gross_pnl = (price - self.entry_price) * self.position * self.config.leverage
        else:
            gross_pnl = (self.entry_price - price) * abs(self.position) * self.config.leverage
        
        self.cash += net_proceeds
        
        # Record trade
        if self.current_trade:
            entry_cost_total = float(self.current_trade.get('entry_cost_total', 0.0))
            exit_cost_total = float(commission_cost + slippage_cost)
            total_cost = float(entry_cost_total + exit_cost_total)
            net_pnl = float(gross_pnl - total_cost)

            self.current_trade['exit_idx'] = idx
            self.current_trade['exit_time'] = timestamp
            self.current_trade['exit_price'] = price
            self.current_trade['exit_notional'] = float(proceeds)
            self.current_trade['exit_commission'] = float(commission_cost)
            self.current_trade['exit_slippage'] = float(slippage_cost)
            self.current_trade['exit_cost_total'] = exit_cost_total
            self.current_trade['total_cost'] = total_cost
            self.current_trade['pnl_gross'] = float(gross_pnl)
            self.current_trade['pnl'] = float(net_pnl)  # Backward-compatible alias now meaning net pnl.
            self.current_trade['pnl_net'] = float(net_pnl)
            self.current_trade['reason'] = reason
            self.trades.append(self.current_trade)
        
        self.position = 0.0
        self.entry_price = 0.0
        self.current_trade = None
        return True
    
    def update_equity(self, current_price: float):
        """Update equity history."""
        equity = self.get_equity(current_price)
        self.equity_history.append(equity)


def risk_check_intrabar(
    broker: Broker,
    df: pd.DataFrame,
    idx: int,
    risk: RiskConfig,
    config: EventConfig,
) -> Optional[str]:
    """
    Check for stop loss, take profit, or trailing stop within a bar.
    
    Returns:
        'sl' if stop loss hit, 'tp' if take profit hit, 'trailing' if trailing stop hit, None otherwise
    """
    if broker.position == 0 or risk is None:
        return None
    
    high = df[config.high_col].iloc[idx]
    low = df[config.low_col].iloc[idx]
    close = df[config.price_col].iloc[idx]
    
    # Get ATR if needed
    atr_value = None
    if risk.sl_atr_k or risk.tp_atr_k or risk.trailing_atr_k:
        if risk.atr_col in df.columns:
            atr_value = df[risk.atr_col].iloc[idx]
        else:
            # Fallback: calculate ATR if not present
            atr_value = _calculate_atr_single(df, idx, period=14)
    
    if broker.position > 0:  # Long position
        # Check stop loss
        if risk.sl_pct:
            sl_price = broker.entry_price * (1 - risk.sl_pct)
        elif risk.sl_atr_k and atr_value is not None:
            sl_price = broker.entry_price - risk.sl_atr_k * atr_value
        else:
            sl_price = None
        
        if sl_price and low <= sl_price:
            exit_price = min(sl_price, close)
            broker.exit_position(exit_price, idx, df.index[idx], reason='sl')
            return 'sl'
        
        # Check take profit
        if risk.tp_pct:
            tp_price = broker.entry_price * (1 + risk.tp_pct)
        elif risk.tp_atr_k and atr_value is not None:
            tp_price = broker.entry_price + risk.tp_atr_k * atr_value
        else:
            tp_price = None
        
        if tp_price and high >= tp_price:
            exit_price = max(tp_price, close)
            broker.exit_position(exit_price, idx, df.index[idx], reason='tp')
            return 'tp'
        
        # Check trailing stop (simplified: check at bar close)
        if risk.trailing_pct or risk.trailing_atr_k:
            # This would need to track trailing high across bars
            # For simplicity, check if current high triggers trailing stop
            if risk.trailing_pct:
                trailing_sl = high * (1 - risk.trailing_pct)
            elif risk.trailing_atr_k and atr_value is not None:
                trailing_sl = high - risk.trailing_atr_k * atr_value
            else:
                trailing_sl = None
            
            if trailing_sl and trailing_sl > broker.entry_price and low <= trailing_sl:
                exit_price = min(trailing_sl, close)
                broker.exit_position(exit_price, idx, df.index[idx], reason='trailing')
                return 'trailing'
    
    elif broker.position < 0:  # Short position
        # Check stop loss
        if risk.sl_pct:
            sl_price = broker.entry_price * (1 + risk.sl_pct)
        elif risk.sl_atr_k and atr_value is not None:
            sl_price = broker.entry_price + risk.sl_atr_k * atr_value
        else:
            sl_price = None
        
        if sl_price and high >= sl_price:
            exit_price = max(sl_price, close)
            broker.exit_position(exit_price, idx, df.index[idx], reason='sl')
            return 'sl'
        
        # Check take profit
        if risk.tp_pct:
            tp_price = broker.entry_price * (1 - risk.tp_pct)
        elif risk.tp_atr_k and atr_value is not None:
            tp_price = broker.entry_price - risk.tp_atr_k * atr_value
        else:
            tp_price = None
        
        if tp_price and low <= tp_price:
            exit_price = min(tp_price, close)
            broker.exit_position(exit_price, idx, df.index[idx], reason='tp')
            return 'tp'
        
        # Check trailing stop
        if risk.trailing_pct or risk.trailing_atr_k:
            if risk.trailing_pct:
                trailing_sl = low * (1 + risk.trailing_pct)
            elif risk.trailing_atr_k and atr_value is not None:
                trailing_sl = low + risk.trailing_atr_k * atr_value
            else:
                trailing_sl = None
            
            if trailing_sl and trailing_sl < broker.entry_price and high >= trailing_sl:
                exit_price = max(trailing_sl, close)
                broker.exit_position(exit_price, idx, df.index[idx], reason='trailing')
                return 'trailing'
    
    return None


def _calculate_atr_single(df: pd.DataFrame, idx: int, period: int = 14) -> float:
    """Calculate ATR for a single index (simplified)."""
    if idx < period:
        return 0.0
    
    if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
        return 0.0
    
    high = df['high'].iloc[idx - period + 1:idx + 1]
    low = df['low'].iloc[idx - period + 1:idx + 1]
    close = df['close'].iloc[idx - period:idx]
    
    tr1 = high - low
    tr2 = (high.iloc[1:] - close).abs()
    tr3 = (low.iloc[1:] - close).abs()
    
    tr = pd.concat([tr1.iloc[1:], tr2, tr3], axis=1).max(axis=1)
    return tr.mean() if not tr.empty else 0.0


def run_event_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    cfg: Optional[EventConfig] = None,
    risk: Optional[RiskConfig] = None,
    risk_exit: Optional[RiskExitEngineConfig] = None,
    regime_series: Optional[pd.Series] = None,
    trend_consensus: Optional[pd.Series] = None,
    max_trades: int = 100,
) -> Dict[str, Any]:
    """
    Run event-driven backtest.
    
    Args:
        df: DataFrame with OHLCV data
        signals: Series with signals (-1, 0, 1)
        cfg: EventConfig (defaults to EventConfig())
        risk: Optional RiskConfig for SL/TP/Trailing
        
    Returns:
        Dictionary with 'summary', 'equity', 'returns', and 'trades'
    """
    if cfg is None:
        cfg = EventConfig()
    
    if df.empty or signals.empty:
        return {
            'summary': {},
            'equity': pd.Series(dtype=float),
            'returns': pd.Series(dtype=float),
            'trades': pd.DataFrame(),
        }
    
    # Align signals
    if isinstance(signals, pd.Series):
        signals = signals.reindex(df.index, method='ffill').fillna(0)
    else:
        signals = pd.Series(signals, index=df.index[:len(signals)]).reindex(df.index, method='ffill').fillna(0)
    
    # Infer frequency if not provided
    freq_label = cfg.freq or infer_freq_label_from_index(df.index)

    # Ensure ATR is available for advanced risk engine (and also useful for legacy ATR stops)
    if risk_exit is not None:
        df = ensure_atr14(df)

    broker = Broker(cfg)
    
    price_col = cfg.price_col if cfg.price_col in df.columns else 'close'
    open_col = cfg.open_col if cfg.open_col in df.columns else 'open'
    
    last_trade_exit_idx = -1
    min_bars_between_trades = 2  # Minimum bars between trades for more natural behavior
    
    # Run backtest
    roundtrip_cost_bps = (cfg.commission + (cfg.slippage_bps / 10000.0)) * 2.0 * 10000.0

    for i in range(len(df)):
        # Giới hạn số lệnh - dừng ngay khi đạt đủ max_trades
        if len(broker.trades) >= max_trades:
            break
            
        timestamp = df.index[i]
        
        # Get prices
        if cfg.use_next_open and i < len(df) - 1:
            entry_price = df[open_col].iloc[i + 1] if open_col in df.columns else df[price_col].iloc[i + 1]
        else:
            entry_price = df[price_col].iloc[i]
        
        current_price = df[price_col].iloc[i]
        
        # Advanced risk/exit engine has priority over legacy risk config.
        if risk_exit is not None:
            risk_exit_check_intrabar(
                broker=broker,
                df=df,
                idx=i,
                cfg=cfg,
                signals=signals,
                regime_series=regime_series,
                trend_consensus=trend_consensus,
                engine=risk_exit,
            )
        elif risk:
            risk_check_intrabar(broker, df, i, risk, cfg)
        
        signal = signals.iloc[i] if i < len(signals) else 0
        # Exit position if signal changes (legacy behavior; advanced engine can still rely on this as safety)
        if broker.position != 0:
            if (broker.position > 0 and signal <= 0) or (broker.position < 0 and signal >= 0):
                broker.exit_position(current_price, i, timestamp, reason='signal')
                last_trade_exit_idx = i
        
        # Enter new position (chỉ nếu chưa đạt giới hạn và đủ thời gian từ lệnh trước)
        can_enter = (i - last_trade_exit_idx >= min_bars_between_trades) if last_trade_exit_idx >= 0 else True
        if signal != 0 and broker.position == 0 and len(broker.trades) < max_trades and can_enter:
            if cfg.enable_cost_budget_check:
                if not cfg.expected_edge_col or cfg.expected_edge_col not in df.columns:
                    # If edge column is not available, skip trade when strict cost-budget mode is enabled.
                    continue
                expected_edge_bps = pd.to_numeric(df[cfg.expected_edge_col], errors='coerce').iloc[i]
                if np.isnan(expected_edge_bps):
                    continue
                min_required_bps = roundtrip_cost_bps + float(cfg.edge_buffer_bps)
                if float(expected_edge_bps) < float(min_required_bps):
                    continue

            # Calculate position size (simplified: use fixed fraction of equity)
            equity = broker.get_equity(current_price)
            position_size = (equity * 0.1) / entry_price  # Use 10% of equity
            if signal < 0:
                position_size = -position_size
            
            broker.enter_position(entry_price, position_size, int(signal), i, timestamp)
            
            # Set stop loss and take profit if risk config provided (legacy RiskConfig)
            if risk and broker.current_trade:
                atr_value = None
                if risk.sl_atr_k or risk.tp_atr_k:
                    if risk.atr_col in df.columns:
                        atr_value = df[risk.atr_col].iloc[i]
                
                if position_size > 0:  # Long
                    if risk.sl_pct:
                        broker.current_trade['stop_loss'] = entry_price * (1 - risk.sl_pct)
                    elif risk.sl_atr_k and atr_value is not None:
                        broker.current_trade['stop_loss'] = entry_price - risk.sl_atr_k * atr_value
                    
                    if risk.tp_pct:
                        broker.current_trade['take_profit'] = entry_price * (1 + risk.tp_pct)
                    elif risk.tp_atr_k and atr_value is not None:
                        broker.current_trade['take_profit'] = entry_price + risk.tp_atr_k * atr_value
                    
                    if risk.sl_pct:
                        broker.current_trade['risk_percent'] = risk.sl_pct
                    elif risk.sl_atr_k and atr_value is not None:
                        broker.current_trade['risk_percent'] = (risk.sl_atr_k * atr_value) / entry_price
                else:  # Short
                    if risk.sl_pct:
                        broker.current_trade['stop_loss'] = entry_price * (1 + risk.sl_pct)
                    elif risk.sl_atr_k and atr_value is not None:
                        broker.current_trade['stop_loss'] = entry_price + risk.sl_atr_k * atr_value
                    
                    if risk.tp_pct:
                        broker.current_trade['take_profit'] = entry_price * (1 - risk.tp_pct)
                    elif risk.tp_atr_k and atr_value is not None:
                        broker.current_trade['take_profit'] = entry_price - risk.tp_atr_k * atr_value
                    
                    if risk.sl_pct:
                        broker.current_trade['risk_percent'] = risk.sl_pct
                    elif risk.sl_atr_k and atr_value is not None:
                        broker.current_trade['risk_percent'] = (risk.sl_atr_k * atr_value) / entry_price

            # For advanced risk/exit engine, we store entry regime for later regime-change exit
            if risk_exit is not None and broker.current_trade is not None and regime_series is not None:
                try:
                    broker.current_trade["entry_regime"] = str(regime_series.iloc[i])
                except Exception:
                    broker.current_trade["entry_regime"] = None
        
        # Update equity
        broker.update_equity(current_price)
    
    # Close any open position at the end (nếu chưa đạt giới hạn)
    if broker.position != 0 and len(broker.trades) < max_trades:
        final_price = df[price_col].iloc[-1]
        broker.exit_position(final_price, len(df) - 1, df.index[-1], reason='end')
    
    # Convert to Series
    # Lưu ý: equity_history thường có độ dài = len(df) hoặc len(df) + 1 (do có điểm equity ban đầu).
    # Để tránh lỗi "Length of values (...) does not match length of index (...)" ta trim hoặc pad cho khớp.
    equity_values = list(broker.equity_history)
    if len(equity_values) > len(df):
        # Trim về cùng độ dài với df (bỏ điểm cuối dư thừa nếu có)
        equity_values = equity_values[:len(df)]
    
    equity_series = pd.Series(equity_values, index=df.index[:len(equity_values)])
    if len(equity_series) < len(df):
        # Pad nếu thiếu (ffill tới hết index của df)
        equity_series = equity_series.reindex(df.index, method='ffill')
    
    returns = equity_series.pct_change().fillna(0)
    
    # Create trades DataFrame
    if broker.trades:
        trades_df = pd.DataFrame(broker.trades)
        # Giới hạn chính xác max_trades lệnh (không thừa)
        if len(trades_df) > max_trades:
            trades_df = trades_df.head(max_trades)
    else:
        trades_df = pd.DataFrame()
    
    # Calculate summary metrics
    summary = performance_summary(equity_series, returns, freq_label)
    summary['freq'] = freq_label
    summary['has_sufficient_data'] = has_min_bars_for_freq(len(df), freq_label)
    
    # Add trade statistics
    if not trades_df.empty:
        pnl_col = 'pnl_net' if 'pnl_net' in trades_df.columns else ('pnl' if 'pnl' in trades_df.columns else None)
        gross_col = 'pnl_gross' if 'pnl_gross' in trades_df.columns else pnl_col

    if not trades_df.empty and pnl_col is not None:
        summary['TotalTrades'] = len(trades_df)
        summary['WinningTrades'] = len(trades_df[trades_df[pnl_col] > 0])
        summary['LosingTrades'] = len(trades_df[trades_df[pnl_col] < 0])
        if summary['TotalTrades'] > 0:
            summary['WinRate'] = summary['WinningTrades'] / summary['TotalTrades']
        else:
            summary['WinRate'] = 0.0
        if summary['LosingTrades'] > 0:
            avg_win = trades_df[trades_df[pnl_col] > 0][pnl_col].mean() if summary['WinningTrades'] > 0 else 0
            avg_loss = abs(trades_df[trades_df[pnl_col] < 0][pnl_col].mean())
            summary['ProfitFactor'] = (avg_win * summary['WinningTrades']) / (avg_loss * summary['LosingTrades']) if avg_loss > 0 else np.inf
        else:
            summary['ProfitFactor'] = np.inf

        summary['AvgNetPnL'] = float(trades_df[pnl_col].mean())
        summary['NetPnLSum'] = float(trades_df[pnl_col].sum())
        if gross_col is not None:
            summary['GrossPnLSum'] = float(trades_df[gross_col].sum())
        if 'total_cost' in trades_df.columns:
            summary['TotalCosts'] = float(trades_df['total_cost'].sum())
            summary['AvgCostPerTrade'] = float(trades_df['total_cost'].mean())
    else:
        summary['TotalTrades'] = 0
        summary['WinningTrades'] = 0
        summary['LosingTrades'] = 0
        summary['WinRate'] = 0.0
        summary['ProfitFactor'] = 0.0
        summary['AvgNetPnL'] = 0.0
        summary['NetPnLSum'] = 0.0
        summary['GrossPnLSum'] = 0.0
        summary['TotalCosts'] = 0.0
        summary['AvgCostPerTrade'] = 0.0
    
    return {
        'summary': summary,
        'equity': equity_series,
        'returns': returns,
        'trades': trades_df,
    }
