"""Visualization module"""
from .plots import (
    plot_candlestick,
    plot_equity_curve,
    plot_drawdown,
    plot_volatility,
    plot_correlation_heatmap,
    alpha_beta_scatter,
    quick_dashboard,
)

try:
    from .tradingview import (
        prepare_tradingview_data,
        create_tradingview_html,
        create_tradingview_pinescript,
    )
    __all__ = [
        'plot_candlestick',
        'plot_equity_curve',
        'plot_drawdown',
        'plot_volatility',
        'plot_correlation_heatmap',
        'alpha_beta_scatter',
        'quick_dashboard',
        'prepare_tradingview_data',
        'create_tradingview_html',
        'create_tradingview_pinescript',
    ]
except ImportError:
    __all__ = [
        'plot_candlestick',
        'plot_equity_curve',
        'plot_drawdown',
        'plot_volatility',
        'plot_correlation_heatmap',
        'alpha_beta_scatter',
        'quick_dashboard',
    ]

