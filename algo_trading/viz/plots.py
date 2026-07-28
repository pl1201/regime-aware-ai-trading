from __future__ import annotations
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd

# Matplotlib base
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Optional seaborn for heatmap
try:
    import seaborn as sns  # type: ignore
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

# Optional plotly
try:
    import plotly.graph_objects as go  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

from algo_trading.core.metrics import drawdown_series


# -----------------------------
# Candlestick (Matplotlib)
# -----------------------------

def plot_candlestick(
    df: pd.DataFrame,
    price_cols: Dict[str, str] | None = None,
    overlays: Dict[str, pd.Series] | None = None,
    signals: pd.Series | None = None,
    title: str = "Candlestick",
    figsize: tuple[int, int] = (12, 6),
    use_plotly: bool = False,
):
    """
    Vẽ biểu đồ nến kèm overlay indicator và tín hiệu mua/bán.
    - df: DataFrame có index thời gian và các cột open/high/low/close
    - price_cols: map tên cột {'open','high','low','close'} nếu khác mặc định
    - overlays: dict tên -> Series (ví dụ {'SMA20': df['SMA20']})
    - signals: Series {-1,0,1} (tuỳ chọn) để đánh dấu mũi tên buy/sell khi tín hiệu thay đổi
    - use_plotly: nếu True và plotly có sẵn -> dùng plotly
    """
    price_cols = price_cols or {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}
    o, h, l, c = [price_cols.get(k, k) for k in ['open','high','low','close']]
    if use_plotly and _HAS_PLOTLY:
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df[o], high=df[h], low=df[l], close=df[c])])
        if overlays:
            for name, s in overlays.items():
                fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode='lines'))
        if signals is not None:
            sig = signals.reindex(df.index).fillna(0)
            chg = sig.diff().fillna(sig)
            buys = chg > 0
            sells = chg < 0
            fig.add_trace(go.Scatter(
                x=df.index[buys], y=df[c][buys], mode='markers', name='Buy',
                marker_symbol='triangle-up', marker_color='green', marker_size=10))
            fig.add_trace(go.Scatter(
                x=df.index[sells], y=df[c][sells], mode='markers', name='Sell',
                marker_symbol='triangle-down', marker_color='red', marker_size=10))
        fig.update_layout(title=title, xaxis_rangeslider_visible=False, template='plotly_white')
        return fig

    # Matplotlib fallback
    x = mdates.date2num(pd.to_datetime(df.index))
    o_, h_, l_, c_ = df[o].values, df[h].values, df[l].values, df[c].values

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)

    # tính width theo khoảng cách thời gian trung bình
    if len(x) > 1:
        dx = np.median(np.diff(x))
    else:
        dx = 1.0
    w = dx * 0.6

    for i, xi in enumerate(x):
        color = 'green' if c_[i] >= o_[i] else 'red'
        # wick
        ax.plot([xi, xi], [l_[i], h_[i]], color=color, linewidth=1)
        # body
        y0 = min(o_[i], c_[i])
        height = abs(c_[i] - o_[i])
        rect = Rectangle((xi - w/2, y0), w, height if height>0 else dx*0.001, facecolor=color, edgecolor=color, alpha=0.7)
        ax.add_patch(rect)

    # overlays
    if overlays:
        for name, s in overlays.items():
            ax.plot(s.index, s.values, label=name, linewidth=1.2)
        ax.legend(loc='upper left')

    # signals
    if signals is not None:
        sig = signals.reindex(df.index).fillna(0)
        chg = sig.diff().fillna(sig)
        buys = chg > 0
        sells = chg < 0
        ax.scatter(df.index[buys], df[c][buys], marker='^', color='green', s=60, label='Buy')
        ax.scatter(df.index[sells], df[c][sells], marker='v', color='red', s=60, label='Sell')

    ax.xaxis_date()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


# -----------------------------
# Equity curve, Drawdown, Volatility
# -----------------------------

def plot_equity_curve(equity: pd.Series, title: str = 'Equity Curve', figsize=(10,4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(equity.index, equity.values, label='Equity', color='blue')
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_drawdown(equity: pd.Series, title: str = 'Drawdown', figsize=(10,3)):
    dd = drawdown_series(equity).fillna(0.0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
    ax.plot(dd.index, dd.values, color='red', linewidth=1)
    ax.set_title(f"{title} (Min: {dd.min():.2%})")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def plot_volatility(returns: pd.Series, window: int = 30, annualize_factor: Optional[float] = None,
                    title: str = 'Rolling Volatility', figsize=(10,3)):
    r = returns.fillna(0.0)
    rolling_std = r.rolling(window).std(ddof=1)
    if annualize_factor is not None:
        rolling_std = rolling_std * np.sqrt(annualize_factor)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rolling_std.index, rolling_std.values, color='orange')
    ax.set_title(f"{title} (window={window})")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


# -----------------------------
# Correlation Heatmap
# -----------------------------

def plot_correlation_heatmap(data: pd.DataFrame, title: str = 'Correlation Heatmap', figsize=(8,6)):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=figsize)
    if _HAS_SEABORN:
        sns.heatmap(corr, ax=ax, annot=False, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    else:
        im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
    ax.set_title(title)
    plt.tight_layout()
    return fig


# -----------------------------
# Alpha vs Beta Scatter
# -----------------------------

def alpha_beta_scatter(returns: pd.Series, benchmark: pd.Series,
                       title: str = 'Alpha vs Beta', figsize=(6,5)):
    """
    Vẽ scatter giữa benchmark và returns, kèm đường hồi quy:
    r_t = alpha + beta * m_t + eps
    """
    r = returns.align(benchmark, join='inner')[0]
    m = benchmark.align(returns, join='inner')[0]
    # align properly
    r, m = r.align(m, join='inner')
    x = m.values
    y = r.values
    if len(x) < 2:
        raise ValueError('Không đủ điểm để hồi quy alpha/beta')
    beta, alpha = np.polyfit(x, y, deg=1)
    y_hat = alpha + beta * x
    # R^2
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-12
    r2 = 1 - ss_res/ss_tot

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, alpha=0.5, label='Data')
    # regression line
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, alpha + beta*xs, color='red', label=f'fit: y={alpha:.4f}+{beta:.2f}x (R²={r2:.2f})')
    ax.set_xlabel('Benchmark Returns')
    ax.set_ylabel('Strategy Returns')
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    return fig


# -----------------------------
# Dashboard nhanh: Giá + Overlay + Equity + Drawdown
# -----------------------------

def quick_dashboard(df: pd.DataFrame, overlays: Dict[str, pd.Series] | None,
                    equity: pd.Series, signals: pd.Series | None = None,
                    price_cols: Dict[str,str] | None = None,
                    use_plotly: bool = False):
    """Vẽ nhanh 3 biểu đồ: Candlestick, Equity, Drawdown."""
    if use_plotly and _HAS_PLOTLY:
        fig1 = plot_candlestick(df, price_cols=price_cols, overlays=overlays, signals=signals, title='Price', use_plotly=True)
        fig1.show()
        fig2 = plot_equity_curve(equity, 'Equity Curve')
        fig3 = plot_drawdown(equity, 'Drawdown')
        return (fig1, fig2, fig3)
    else:
        fig1 = plot_candlestick(df, price_cols=price_cols, overlays=overlays, signals=signals, title='Price', use_plotly=False)
        fig2 = plot_equity_curve(equity, 'Equity Curve')
        fig3 = plot_drawdown(equity, 'Drawdown')
        return (fig1, fig2, fig3)


# -----------------------------
# Trade Statistics Visualization
# -----------------------------

def plot_trade_pnl_distribution(
    trades: pd.DataFrame,
    title: str = "Phân bố PnL của Trades",
    figsize: tuple[int, int] = (12, 6),
    use_plotly: bool = False,
):
    """
    Vẽ biểu đồ phân bố PnL của các trades, phân biệt win/loss.
    """
    if trades is None or trades.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Không có dữ liệu trades", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # Xác định cột PnL
    pnl_col = None
    for col in ['pnl', 'profit', 'net_pnl', 'return_pct', 'return']:
        if col in trades.columns:
            pnl_col = col
            break
    
    if pnl_col is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Không tìm thấy cột PnL", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    pnl = trades[pnl_col]
    winning = pnl[pnl > 0]
    losing = pnl[pnl < 0]
    
    if use_plotly and _HAS_PLOTLY:
        fig = go.Figure()
        if len(winning) > 0:
            fig.add_trace(go.Histogram(
                x=winning.values,
                name='Winning Trades',
                marker_color='green',
                opacity=0.7,
                nbinsx=30,
            ))
        if len(losing) > 0:
            fig.add_trace(go.Histogram(
                x=losing.values,
                name='Losing Trades',
                marker_color='red',
                opacity=0.7,
                nbinsx=30,
            ))
        fig.update_layout(
            title=title,
            xaxis_title="PnL",
            yaxis_title="Số lượng trades",
            barmode='overlay',
            template='plotly_white',
        )
        return fig
    
    # Matplotlib
    fig, ax = plt.subplots(figsize=figsize)
    if len(winning) > 0:
        ax.hist(winning.values, bins=30, alpha=0.7, color='green', label=f'Winning ({len(winning)})', edgecolor='black')
    if len(losing) > 0:
        ax.hist(losing.values, bins=30, alpha=0.7, color='red', label=f'Losing ({len(losing)})', edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("PnL")
    ax.set_ylabel("Số lượng trades")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def plot_trade_timeline(
    trades: pd.DataFrame,
    title: str = "Timeline của Trades",
    figsize: tuple[int, int] = (14, 6),
    use_plotly: bool = False,
):
    """
    Vẽ timeline các trades với màu sắc phân biệt win/loss.
    """
    if trades is None or trades.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Không có dữ liệu trades", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # Xác định cột PnL và thời gian
    pnl_col = None
    for col in ['pnl', 'profit', 'net_pnl', 'return_pct', 'return']:
        if col in trades.columns:
            pnl_col = col
            break
    
    time_col = 'exit_time' if 'exit_time' in trades.columns else 'entry_time'
    if time_col not in trades.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Không tìm thấy cột thời gian", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    if pnl_col is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Không tìm thấy cột PnL", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    trades_sorted = trades.sort_values(time_col).copy()
    times = pd.to_datetime(trades_sorted[time_col])
    pnl = trades_sorted[pnl_col]
    
    colors = ['green' if p > 0 else 'red' if p < 0 else 'gray' for p in pnl]
    
    if use_plotly and _HAS_PLOTLY:
        fig = go.Figure()
        winning = pnl > 0
        losing = pnl < 0
        breakeven = pnl == 0
        
        if winning.any():
            fig.add_trace(go.Scatter(
                x=times[winning],
                y=pnl[winning],
                mode='markers',
                name='Win',
                marker=dict(color='green', size=8, symbol='triangle-up'),
            ))
        if losing.any():
            fig.add_trace(go.Scatter(
                x=times[losing],
                y=pnl[losing],
                mode='markers',
                name='Loss',
                marker=dict(color='red', size=8, symbol='triangle-down'),
            ))
        if breakeven.any():
            fig.add_trace(go.Scatter(
                x=times[breakeven],
                y=pnl[breakeven],
                mode='markers',
                name='Breakeven',
                marker=dict(color='gray', size=6, symbol='circle'),
            ))
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig.update_layout(
            title=title,
            xaxis_title="Thời gian",
            yaxis_title="PnL",
            template='plotly_white',
        )
        return fig
    
    # Matplotlib
    fig, ax = plt.subplots(figsize=figsize)
    for i, (t, p, c) in enumerate(zip(times, pnl, colors)):
        marker = '^' if p > 0 else 'v' if p < 0 else 'o'
        ax.scatter(t, p, color=c, s=50, marker=marker, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("PnL")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_cumulative_pnl(
    trades: pd.DataFrame,
    title: str = "Cumulative PnL",
    figsize: tuple[int, int] = (12, 6),
    use_plotly: bool = False,
):
    """
    Vẽ biểu đồ cumulative PnL theo thời gian.
    """
    if trades is None or trades.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Không có dữ liệu trades", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # Xác định cột PnL và thời gian
    pnl_col = None
    for col in ['pnl', 'profit', 'net_pnl', 'return_pct', 'return']:
        if col in trades.columns:
            pnl_col = col
            break
    
    time_col = 'exit_time' if 'exit_time' in trades.columns else 'entry_time'
    if time_col not in trades.columns or pnl_col is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Thiếu dữ liệu cần thiết", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    trades_sorted = trades.sort_values(time_col).copy()
    times = pd.to_datetime(trades_sorted[time_col])
    pnl = trades_sorted[pnl_col]
    cum_pnl = pnl.cumsum()
    
    if use_plotly and _HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=cum_pnl.values,
            mode='lines+markers',
            name='Cumulative PnL',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig.update_layout(
            title=title,
            xaxis_title="Thời gian",
            yaxis_title="Cumulative PnL",
            template='plotly_white',
        )
        return fig
    
    # Matplotlib
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, cum_pnl.values, marker='o', markersize=3, linewidth=2, color='blue', label='Cumulative PnL')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.fill_between(times, 0, cum_pnl.values, where=(cum_pnl >= 0), alpha=0.3, color='green', label='Profit')
    ax.fill_between(times, 0, cum_pnl.values, where=(cum_pnl < 0), alpha=0.3, color='red', label='Loss')
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Cumulative PnL")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_winrate_metrics(
    stats: Dict[str, Any],
    title: str = "Thống kê Winrate",
    figsize: tuple[int, int] = (10, 6),
    use_plotly: bool = False,
):
    """
    Vẽ biểu đồ các metrics liên quan đến winrate.
    """
    if not stats or stats.get('total_trades', 0) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    if use_plotly and _HAS_PLOTLY:
        categories = ['Win', 'Loss', 'Breakeven']
        values = [
            stats.get('winning_trades', 0),
            stats.get('losing_trades', 0),
            stats.get('breakeven_trades', 0),
        ]
        colors = ['green', 'red', 'gray']
        
        fig = go.Figure(data=[go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto',
        )])
        fig.update_layout(
            title=title,
            xaxis_title="Loại trade",
            yaxis_title="Số lượng",
            template='plotly_white',
        )
        return fig
    
    # Matplotlib
    fig, ax = plt.subplots(figsize=figsize)
    categories = ['Win', 'Loss', 'Breakeven']
    values = [
        stats.get('winning_trades', 0),
        stats.get('losing_trades', 0),
        stats.get('breakeven_trades', 0),
    ]
    colors = ['green', 'red', 'gray']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({val/stats["total_trades"]*100:.1f}%)' if stats['total_trades'] > 0 else '0',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel("Số lượng trades")
    ax.set_title(f"{title} - Winrate: {stats.get('winrate', 0):.2f}%")
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    return fig




