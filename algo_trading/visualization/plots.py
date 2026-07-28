"""
Backward compatibility: visualization.plots re-exports from viz module

All plotting functions are now in algo_trading.viz.plots
"""
from algo_trading.viz.plots import (
    plot_candlestick,
    plot_equity_curve,
    plot_drawdown,
    plot_volatility,
    plot_correlation_heatmap,
    alpha_beta_scatter,
    quick_dashboard,
    plot_trade_pnl_distribution,
    plot_trade_timeline,
    plot_cumulative_pnl,
    plot_winrate_metrics,
)

__all__ = [
    'plot_candlestick',
    'plot_equity_curve',
    'plot_drawdown',
    'plot_volatility',
    'plot_correlation_heatmap',
    'alpha_beta_scatter',
    'quick_dashboard',
    'plot_trade_pnl_distribution',
    'plot_trade_timeline',
    'plot_cumulative_pnl',
    'plot_winrate_metrics',
]
