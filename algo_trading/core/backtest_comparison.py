"""
COMPARISON: Old vs New Backtest Engine
======================================

File này so sánh kết quả giữa hệ thống cũ và mới để minh họa sự khác biệt.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any

from algo_trading.core.backtest_vectorized import vectorized_pnl, BacktestConfig as OldBacktestConfig
from algo_trading.core.metrics import performance_summary as old_performance_summary

from algo_trading.core.backtest_production import (
    validate_market_data,
    ExecutionConfig,
    ExecutionSimulator,
    RiskConfig,
    RiskEngine,
    EquityCurveEngine,
)
from algo_trading.core.metrics_production import (
    calculate_comprehensive_metrics,
    format_metrics_for_display,
)


def compare_backtest_engines(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 10000.0,
    commission: float = 0.001,
) -> Dict[str, Any]:
    """
    So sánh kết quả giữa hệ thống cũ và mới.
    
    Returns:
        Dict với:
        - old_metrics: Metrics từ hệ thống cũ
        - new_metrics: Metrics từ hệ thống mới
        - differences: Sự khác biệt
        - warnings: Cảnh báo về các vấn đề phát hiện
    """
    warnings = []
    
    # ========================================================================
    # OLD SYSTEM
    # ========================================================================
    old_config = OldBacktestConfig(
        initial_capital=initial_capital,
        commission=commission,
        allow_short=True,
    )
    
    old_equity, old_returns = vectorized_pnl(df, signals, old_config)
    old_metrics_dict = old_performance_summary(old_equity, old_returns, freq='1h')
    
    # ========================================================================
    # NEW SYSTEM
    # ========================================================================
    # 1. Validate data
    validation_result = validate_market_data(df)
    if not validation_result.is_valid:
        warnings.append(f"Data validation issues: {validation_result.issues}")
    
    cleaned_df = validation_result.cleaned_df if validation_result.cleaned_df is not None else df
    
    # 2. Setup execution simulator
    exec_config = ExecutionConfig(
        taker_fee_bps=commission * 10000,  # Convert to bps
        slippage_bps=5.0,
        execution_delay_bars=1,
    )
    
    # 3. Setup risk engine
    risk_config = RiskConfig(
        max_position_size_pct=1.0,  # Allow full position for comparison
        max_leverage=1.0,
    )
    
    # 4. Run simplified backtest (full implementation sẽ phức tạp hơn)
    # Tạm thời dùng cách đơn giản để so sánh
    
    # Calculate returns với execution costs
    prices = cleaned_df['close']
    returns = prices.pct_change().fillna(0)
    positions = signals.shift(1).fillna(0)
    
    # Apply execution costs
    position_changes = positions.diff().abs()
    fees = position_changes * prices * (exec_config.taker_fee_bps / 10000)
    slippage = position_changes * prices * (exec_config.slippage_bps / 10000)
    total_costs = fees + slippage
    
    # Strategy returns
    strategy_returns = positions * returns
    
    # Net returns (after costs)
    net_returns = strategy_returns - (total_costs / initial_capital)
    
    # Equity curve
    new_equity = initial_capital * (1 + net_returns).cumprod()
    new_returns = net_returns
    
    # 5. Calculate metrics
    new_metrics = calculate_comprehensive_metrics(
        equity=new_equity,
        returns=new_returns,
        positions=positions,
        freq="1h",
    )
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    differences = {}
    
    # CAGR
    old_cagr = old_metrics_dict.get('CAGR', 0)
    new_cagr = new_metrics.cagr
    if not np.isnan(old_cagr) and not np.isnan(new_cagr):
        differences['CAGR'] = {
            'old': old_cagr,
            'new': new_cagr,
            'diff_pct': ((new_cagr - old_cagr) / abs(old_cagr) * 100) if old_cagr != 0 else 0,
        }
    
    # Sharpe
    old_sharpe = old_metrics_dict.get('Sharpe', 0)
    new_sharpe = new_metrics.sharpe_ratio
    if not np.isnan(old_sharpe) and not np.isnan(new_sharpe):
        differences['Sharpe'] = {
            'old': old_sharpe,
            'new': new_sharpe,
            'diff_pct': ((new_sharpe - old_sharpe) / abs(old_sharpe) * 100) if old_sharpe != 0 else 0,
        }
    
    # Max Drawdown
    old_mdd = old_metrics_dict.get('MaxDrawdown', 0)
    new_mdd = new_metrics.max_drawdown
    if not np.isnan(old_mdd) and not np.isnan(new_mdd):
        differences['MaxDD'] = {
            'old': old_mdd,
            'new': new_mdd,
            'diff_pct': ((new_mdd - old_mdd) / abs(old_mdd) * 100) if old_mdd != 0 else 0,
        }
    
    # Check for suspicious metrics
    if old_sharpe > 3.0:
        warnings.append(f"⚠️ Old Sharpe ratio quá cao ({old_sharpe:.2f}), có thể do annualization sai")
    
    if old_cagr > 1.0:  # > 100%
        warnings.append(f"⚠️ Old CAGR quá cao ({old_cagr*100:.1f}%), có thể không thực tế")
    
    if abs(differences.get('Sharpe', {}).get('diff_pct', 0)) > 20:
        warnings.append("⚠️ Sharpe ratio khác biệt > 20%, có thể do annualization factor sai")
    
    return {
        'old_metrics': old_metrics_dict,
        'new_metrics': format_metrics_for_display(new_metrics),
        'differences': differences,
        'warnings': warnings,
        'old_equity': old_equity,
        'new_equity': new_equity,
    }


def print_comparison_report(comparison_result: Dict[str, Any]):
    """Print comparison report"""
    print("=" * 80)
    print("BACKTEST ENGINE COMPARISON REPORT")
    print("=" * 80)
    print()
    
    print("OLD SYSTEM METRICS:")
    print("-" * 80)
    old = comparison_result['old_metrics']
    for key, value in old.items():
        if isinstance(value, float):
            if 'Return' in key or 'CAGR' in key or 'Drawdown' in key:
                print(f"  {key:20s}: {value*100:8.2f}%")
            else:
                print(f"  {key:20s}: {value:8.4f}")
        else:
            print(f"  {key:20s}: {value}")
    print()
    
    print("NEW SYSTEM METRICS:")
    print("-" * 80)
    new = comparison_result['new_metrics']
    for key, value in new.items():
        print(f"  {key:20s}: {value}")
    print()
    
    print("DIFFERENCES:")
    print("-" * 80)
    diffs = comparison_result['differences']
    for metric, diff in diffs.items():
        print(f"  {metric}:")
        print(f"    Old:  {diff['old']:.4f}")
        print(f"    New:  {diff['new']:.4f}")
        print(f"    Diff:  {diff['diff_pct']:+.2f}%")
    print()
    
    if comparison_result['warnings']:
        print("WARNINGS:")
        print("-" * 80)
        for warning in comparison_result['warnings']:
            print(f"  {warning}")
        print()







































