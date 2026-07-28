"""
PRODUCTION-GRADE METRICS ENGINE
================================

Tất cả metrics được tính theo công thức chuẩn industry.
Mỗi metric có:
- Công thức chính xác
- Khi nào có ý nghĩa
- Khi nào nên bỏ qua
- Anti-patterns cần tránh

Thiết kế bởi: Quant Researcher
Tiêu chuẩn: Hedge Fund / Prop Firm Grade
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Trading days per year (adjust for crypto: 365, for stocks: 252)
TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_YEAR = 252 * 24  # For crypto
TRADING_MINUTES_PER_YEAR = 252 * 24 * 60  # For crypto


def _get_annualization_factor(freq: str, bars_per_year: Optional[int] = None) -> float:
    """
    Calculate annualization factor cho volatility và Sharpe.
    
    CRITICAL: Phải đúng với timeframe của data.
    - Daily data: sqrt(252)
    - Hourly data: sqrt(252 * 24) hoặc sqrt(365 * 24) cho crypto
    - Minute data: sqrt(252 * 24 * 60)
    
    Args:
        freq: Frequency string ("1d", "1h", "1m", etc.)
        bars_per_year: Explicit bars per year (overrides freq)
    
    Returns:
        Annualization factor (sqrt of periods per year)
    """
    if bars_per_year is not None:
        return np.sqrt(bars_per_year)
    
    freq_lower = freq.lower().strip()
    
    # Daily
    if freq_lower in ("d", "1d", "day", "daily"):
        return np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Hourly
    if freq_lower in ("h", "1h", "hour", "hourly"):
        # Default to crypto (365 days)
        return np.sqrt(365 * 24)
    
    # Minute
    if freq_lower in ("m", "1m", "min", "minute"):
        return np.sqrt(365 * 24 * 60)
    
    # Weekly
    if freq_lower in ("w", "1w", "week", "weekly"):
        return np.sqrt(52)
    
    # Monthly
    if freq_lower in ("M", "1M", "month", "monthly"):
        return np.sqrt(12)
    
    # Default: assume daily
    return np.sqrt(TRADING_DAYS_PER_YEAR)


def _get_periods_per_year(freq: str) -> float:
    """Get number of periods per year"""
    ann_factor = _get_annualization_factor(freq)
    return ann_factor ** 2


# ============================================================================
# RETURNS CALCULATION
# ============================================================================

def calculate_returns(equity: pd.Series, method: str = "simple") -> pd.Series:
    """
    Calculate returns từ equity curve.
    
    CRITICAL: Phải tính từ equity curve, không phải từ prices.
    
    Args:
        equity: Equity curve Series
        method: "simple" hoặc "log"
    
    Returns:
        Returns Series
    
    Formula:
        Simple: r_t = (E_t / E_{t-1}) - 1
        Log: r_t = ln(E_t / E_{t-1})
    
    Khi nào dùng:
        - Simple: Cho returns nhỏ (< 10%), dễ hiểu
        - Log: Cho returns lớn, có tính chất time-additive
    """
    if len(equity) < 2:
        return pd.Series(dtype=float, index=equity.index)
    
    if method == "log":
        returns = np.log(equity / equity.shift(1)).dropna()
    else:
        returns = (equity / equity.shift(1) - 1).dropna()
    
    return returns


# ============================================================================
# CAGR (Compound Annual Growth Rate)
# ============================================================================

def calculate_cagr(
    equity: pd.Series,
    freq: str = "1d",
    bars_per_year: Optional[int] = None,
) -> float:
    """
    Calculate CAGR từ equity curve.
    
    Formula:
        CAGR = (End_Value / Start_Value) ^ (1 / Years) - 1
    
    CRITICAL:
        - Phải dùng calendar time hoặc trading days, không phải số bars
        - Nếu không có timestamp, dùng bars_per_year
    
    Args:
        equity: Equity curve Series
        freq: Frequency string
        bars_per_year: Explicit bars per year
    
    Returns:
        CAGR (decimal, e.g., 0.15 = 15%)
    
    Khi nào có ý nghĩa:
        - Có đủ data (ít nhất 1 năm)
        - Equity curve không có gaps lớn
    
    Khi nào nên bỏ qua:
        - Data < 3 tháng
        - Equity curve có nhiều NaN
    """
    if len(equity) < 2:
        return np.nan
    
    start_value = equity.iloc[0]
    end_value = equity.iloc[-1]
    
    if start_value <= 0 or end_value <= 0:
        return np.nan
    
    # Calculate years
    if isinstance(equity.index, pd.DatetimeIndex):
        # Use actual calendar time
        days = (equity.index[-1] - equity.index[0]).days
        if days <= 0:
            return np.nan
        years = days / 365.25
    else:
        # Use bars_per_year
        if bars_per_year is None:
            bars_per_year = int(_get_periods_per_year(freq))
        years = len(equity) / bars_per_year
    
    if years <= 0:
        return np.nan
    
    # CAGR formula
    cagr = (end_value / start_value) ** (1 / years) - 1
    
    return cagr


# ============================================================================
# SHARPE RATIO (Annualized)
# ============================================================================

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    freq: str = "1d",
    bars_per_year: Optional[int] = None,
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Formula:
        Sharpe = (Mean(Excess Returns) / Std(Returns)) * sqrt(Periods_Per_Year)
    
    CRITICAL ANTI-PATTERN:
        SAI: Sharpe = (Mean / Std) * sqrt(252)  # Nhân sau khi chia
        ĐÚNG: Sharpe = (Mean / Std) * sqrt(252)  # Nhưng phải đúng với timeframe
    
    Args:
        returns: Returns Series (từ equity curve)
        risk_free_rate: Annual risk-free rate (default 0)
        freq: Frequency string
        bars_per_year: Explicit bars per year
    
    Returns:
        Annualized Sharpe ratio
    
    Khi nào có ý nghĩa:
        - Returns có phân phối gần normal
        - Có đủ observations (ít nhất 30-60)
        - Returns không có autocorrelation mạnh
    
    Khi nào nên bỏ qua:
        - Returns có skewness cực lớn
        - Có outliers nghiêm trọng
        - Returns có regime changes
    """
    if len(returns) < 2:
        return np.nan
    
    # Remove NaN
    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        return np.nan
    
    # Calculate annualization factor
    ann_factor = _get_annualization_factor(freq, bars_per_year)
    
    # Annualized risk-free rate per period
    periods_per_year = ann_factor ** 2
    rf_per_period = risk_free_rate / periods_per_year
    
    # Excess returns
    excess_returns = returns_clean - rf_per_period
    
    # Mean and std
    mean_excess = excess_returns.mean()
    std_returns = returns_clean.std(ddof=1)  # Sample std
    
    if std_returns == 0:
        return 0.0 if mean_excess == 0 else np.nan
    
    # Sharpe ratio (annualized)
    sharpe = (mean_excess / std_returns) * ann_factor
    
    return sharpe


# ============================================================================
# SORTINO RATIO
# ============================================================================

def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    freq: str = "1d",
    bars_per_year: Optional[int] = None,
) -> float:
    """
    Calculate annualized Sortino ratio.
    
    Formula:
        Sortino = (Mean(Excess Returns) / Downside_Deviation) * sqrt(Periods_Per_Year)
        
        Downside_Deviation = sqrt(Mean(min(0, Returns - Target)^2))
    
    Args:
        returns: Returns Series
        risk_free_rate: Annual risk-free rate
        target_return: Target return (MAR - Minimum Acceptable Return)
        freq: Frequency string
        bars_per_year: Explicit bars per year
    
    Returns:
        Annualized Sortino ratio
    
    Khi nào có ý nghĩa:
        - Quan tâm đến downside risk
        - Returns có negative skew
    
    Khi nào nên bỏ qua:
        - Returns không có downside (all positive)
    """
    if len(returns) < 2:
        return np.nan
    
    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        return np.nan
    
    ann_factor = _get_annualization_factor(freq, bars_per_year)
    periods_per_year = ann_factor ** 2
    rf_per_period = risk_free_rate / periods_per_year
    
    # Excess returns
    excess_returns = returns_clean - rf_per_period
    
    # Downside deviation
    downside_returns = np.minimum(0, returns_clean - target_return)
    downside_dev = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_dev == 0:
        return 0.0 if excess_returns.mean() == 0 else np.nan
    
    # Sortino ratio
    sortino = (excess_returns.mean() / downside_dev) * ann_factor
    
    return sortino


# ============================================================================
# MAX DRAWDOWN
# ============================================================================

def calculate_max_drawdown(equity: pd.Series) -> float:
    """
    Calculate maximum drawdown từ equity curve.
    
    Formula:
        DD_t = (Equity_t / Peak_t) - 1
        MaxDD = min(DD_t)
    
    CRITICAL: Phải tính từ equity curve, không phải từ returns.
    
    Args:
        equity: Equity curve Series
    
    Returns:
        Max drawdown (negative number, e.g., -0.25 = -25%)
    
    Khi nào có ý nghĩa:
        - Luôn có ý nghĩa
        - Quan trọng cho risk management
    
    Anti-patterns:
        - Tính từ returns thay vì equity
        - Reset equity về initial capital
    """
    if len(equity) < 2:
        return 0.0
    
    # Calculate running peak
    peak = equity.cummax()
    
    # Drawdown
    drawdown = (equity / peak) - 1.0
    
    # Max drawdown
    max_dd = drawdown.min()
    
    return max_dd


# ============================================================================
# CALMAR RATIO
# ============================================================================

def calculate_calmar_ratio(
    equity: pd.Series,
    freq: str = "1d",
    bars_per_year: Optional[int] = None,
) -> float:
    """
    Calculate Calmar ratio.
    
    Formula:
        Calmar = CAGR / |MaxDD|
    
    Args:
        equity: Equity curve Series
        freq: Frequency string
        bars_per_year: Explicit bars per year
    
    Returns:
        Calmar ratio
    
    Khi nào có ý nghĩa:
        - Có đủ data (ít nhất 1 năm)
        - MaxDD > 0
    
    Khi nào nên bỏ qua:
        - MaxDD = 0 (no drawdown)
        - Data < 1 năm
    """
    cagr = calculate_cagr(equity, freq, bars_per_year)
    max_dd = calculate_max_drawdown(equity)
    
    if max_dd == 0:
        return np.nan
    
    calmar = cagr / abs(max_dd)
    
    return calmar


# ============================================================================
# PROFIT FACTOR
# ============================================================================

def calculate_profit_factor(trades_pnl: pd.Series) -> float:
    """
    Calculate profit factor từ trades PnL.
    
    Formula:
        Profit_Factor = Sum(Winning_Trades) / |Sum(Losing_Trades)|
    
    Args:
        trades_pnl: Series of trade PnL
    
    Returns:
        Profit factor
    
    Khi nào có ý nghĩa:
        - Có đủ trades (ít nhất 20-30)
        - Có cả winning và losing trades
    
    Khi nào nên bỏ qua:
        - Tất cả trades đều thắng hoặc đều thua
        - Quá ít trades (< 10)
    """
    if len(trades_pnl) == 0:
        return np.nan
    
    winning = trades_pnl[trades_pnl > 0].sum()
    losing = abs(trades_pnl[trades_pnl < 0].sum())
    
    if losing == 0:
        return np.inf if winning > 0 else np.nan
    
    return winning / losing


# ============================================================================
# WIN RATE
# ============================================================================

def calculate_win_rate(trades_pnl: pd.Series) -> float:
    """
    Calculate win rate.
    
    Formula:
        Win_Rate = Count(Winning_Trades) / Total_Trades
    
    Args:
        trades_pnl: Series of trade PnL
    
    Returns:
        Win rate (0 to 1)
    """
    if len(trades_pnl) == 0:
        return np.nan
    
    winning_trades = (trades_pnl > 0).sum()
    total_trades = len(trades_pnl)
    
    return winning_trades / total_trades


# ============================================================================
# EXPOSURE TIME
# ============================================================================

def calculate_exposure_time(positions: pd.Series) -> float:
    """
    Calculate exposure time (% of time in market).
    
    Formula:
        Exposure_Time = Sum(|Position| > 0) / Total_Bars
    
    Args:
        positions: Position Series
    
    Returns:
        Exposure time (0 to 1)
    """
    if len(positions) == 0:
        return np.nan
    
    in_market = (positions.abs() > 1e-8).sum()
    total_bars = len(positions)
    
    return in_market / total_bars


# ============================================================================
# TURNOVER
# ============================================================================

def calculate_turnover(
    position_changes: pd.Series,
    equity: pd.Series,
) -> float:
    """
    Calculate annualized turnover.
    
    Formula:
        Turnover = Sum(|Position_Changes|) / Avg_Equity * Periods_Per_Year
    
    Args:
        position_changes: Position change Series
        equity: Equity curve Series
        freq: Frequency string
    
    Returns:
        Annualized turnover
    """
    if len(position_changes) == 0 or len(equity) == 0:
        return np.nan
    
    total_turnover = position_changes.abs().sum()
    avg_equity = equity.mean()
    
    if avg_equity == 0:
        return np.nan
    
    # Annualized (assuming daily)
    periods_per_year = TRADING_DAYS_PER_YEAR
    turnover = (total_turnover / avg_equity) * periods_per_year
    
    return turnover


# ============================================================================
# COMPREHENSIVE PERFORMANCE SUMMARY
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return: float
    cagr: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    max_drawdown: float
    volatility: float
    
    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    
    # Portfolio metrics
    exposure_time: float
    turnover: float
    
    # Flags
    is_robust: bool = False
    has_sufficient_data: bool = False


def calculate_comprehensive_metrics(
    equity: pd.Series,
    returns: pd.Series,
    positions: pd.Series,
    trades_pnl: Optional[pd.Series] = None,
    freq: str = "1d",
    bars_per_year: Optional[int] = None,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        equity: Equity curve Series
        returns: Returns Series (from equity)
        positions: Position Series
        trades_pnl: Optional Series of trade PnL
        freq: Frequency string
        bars_per_year: Explicit bars per year
        risk_free_rate: Annual risk-free rate
    
    Returns:
        PerformanceMetrics object
    """
    # Basic returns
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else np.nan
    cagr = calculate_cagr(equity, freq, bars_per_year)
    
    # Risk-adjusted
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, freq, bars_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, 0.0, freq, bars_per_year)
    calmar = calculate_calmar_ratio(equity, freq, bars_per_year)
    
    # Risk
    max_dd = calculate_max_drawdown(equity)
    ann_factor = _get_annualization_factor(freq, bars_per_year)
    volatility = returns.std(ddof=1) * ann_factor if len(returns) > 1 else np.nan
    
    # Trade metrics
    if trades_pnl is not None and len(trades_pnl) > 0:
        total_trades = len(trades_pnl)
        win_rate = calculate_win_rate(trades_pnl)
        profit_factor = calculate_profit_factor(trades_pnl)
        avg_win = trades_pnl[trades_pnl > 0].mean() if (trades_pnl > 0).any() else 0.0
        avg_loss = trades_pnl[trades_pnl < 0].mean() if (trades_pnl < 0).any() else 0.0
    else:
        total_trades = 0
        win_rate = np.nan
        profit_factor = np.nan
        avg_win = np.nan
        avg_loss = np.nan
    
    # Portfolio metrics
    exposure_time = calculate_exposure_time(positions)
    position_changes = positions.diff().abs()
    turnover = calculate_turnover(position_changes, equity)
    
    # Robustness flags
    has_sufficient_data = len(equity) >= 252  # At least 1 year
    is_robust = (
        has_sufficient_data and
        not np.isnan(sharpe) and sharpe > 0 and
        not np.isnan(cagr) and cagr > 0 and
        max_dd < -0.1  # Has meaningful drawdown
    )
    
    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        volatility=volatility,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        exposure_time=exposure_time,
        turnover=turnover,
        is_robust=is_robust,
        has_sufficient_data=has_sufficient_data,
    )


def format_metrics_for_display(metrics: PerformanceMetrics) -> Dict[str, str]:
    """Format metrics for display"""
    return {
        "CAGR": f"{metrics.cagr*100:.2f}%" if not np.isnan(metrics.cagr) else "N/A",
        "Sharpe": f"{metrics.sharpe_ratio:.2f}" if not np.isnan(metrics.sharpe_ratio) else "N/A",
        "Sortino": f"{metrics.sortino_ratio:.2f}" if not np.isnan(metrics.sortino_ratio) else "N/A",
        "MaxDD": f"{metrics.max_drawdown*100:.2f}%" if not np.isnan(metrics.max_drawdown) else "N/A",
        "Calmar": f"{metrics.calmar_ratio:.2f}" if not np.isnan(metrics.calmar_ratio) else "N/A",
        "#Trades": f"{metrics.total_trades}",
        "WinRate": f"{metrics.win_rate*100:.1f}%" if not np.isnan(metrics.win_rate) else "N/A",
        "ProfitFactor": f"{metrics.profit_factor:.2f}" if not np.isnan(metrics.profit_factor) else "N/A",
        "Robust": "✅" if metrics.is_robust else "❌",
    }







































