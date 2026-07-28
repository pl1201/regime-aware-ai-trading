from __future__ import annotations

"""
Metrics utilities: returns, drawdowns, Sharpe/Sortino/Calmar, volatility.

Được tách ra từ algo_trading.utils.metrics để dùng chung cho backtest, optimization, analysis.
"""

import numpy as np
import pandas as pd
import warnings

TRADING_DAYS = 252
TRADING_MINUTES = 252 * 6.5 * 60  # rough for equities; for crypto 365*24*60 may be used


def _annualization_factor(freq: str | None, default_days: int = TRADING_DAYS) -> float:
    """
    Calculate annualization factor cho volatility và Sharpe.
    
    CRITICAL FIX: Phải đúng với timeframe thực tế của data.
    - Daily: sqrt(365) cho crypto (không phải 252 cho stocks)
    - Hourly: sqrt(365*24) cho crypto = sqrt(8760) ≈ 93.6
    - Minute: sqrt(365*24*60) cho crypto
    
    Lỗi cũ: Dùng sqrt(252*24) = sqrt(6048) ≈ 77.8 → Sharpe quá cao!
    """
    if not freq:
        return np.sqrt(default_days)
    
    freq_lower = freq.lower().strip()
    
    # Daily - CRITICAL: Dùng 365 cho crypto, 252 cho stocks
    if freq_lower in ("d", "1d", "day", "daily"):
        return np.sqrt(365) 
    
    # Hourly - CRITICAL FIX
    if freq_lower in ("h", "1h", "hour", "hourly"):
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
    
    return np.sqrt(default_days)


def infer_freq_label_from_index(idx: pd.Index) -> str:
    """Infer a coarse frequency label for annualization (1m/1h/1d/1w)."""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return "1d"
    inferred = pd.infer_freq(idx)
    if inferred:
        inf = inferred.lower()
        if "min" in inf or inf.startswith("t"):
            return "1m"
        if "h" in inf:
            return "1h"
        if "d" in inf:
            return "1d"
        if "w" in inf:
            return "1w"
        if "m" == inf:
            return "1m"
    # Fallback: use median delta
    deltas = idx.to_series().diff().dropna()
    if deltas.empty:
        return "1d"
    median_seconds = deltas.median().total_seconds()
    if median_seconds <= 70:
        return "1m"
    if median_seconds <= 5400:
        return "1h"
    return "1d"


def has_min_bars_for_freq(n: int, freq_label: str) -> bool:
    """Basic sufficiency rule of thumb for sample size by freq."""
    freq = (freq_label or "1d").lower()
    if freq.startswith("1m"):
        return n >= 5000
    if freq.startswith("1h"):
        return n >= 1000
    if freq.startswith("1w"):
        return n >= 52
    return n >= 252 


def to_returns(price: pd.Series | pd.DataFrame, method: str = "log") -> pd.Series:
    s = price["close"] if isinstance(price, pd.DataFrame) and "close" in price.columns else price
    if method == "log":
        return np.log(s).diff().dropna()
    return s.pct_change().dropna()


def cum_returns(returns: pd.Series, start_value: float = 1.0, log: bool = True) -> pd.Series:
    if log:
        cum = returns.cumsum().apply(np.exp)
        return start_value * cum
    return start_value * (1 + returns).cumprod()


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd


def max_drawdown(equity: pd.Series) -> float:
    return drawdown_series(equity).min()


def volatility(returns: pd.Series, freq: str | None = None) -> float:
    ann = _annualization_factor(freq)
    return returns.std(ddof=1) * ann


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, freq: str | None = None) -> float:

    if returns.empty:
        return np.nan
    
    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        warnings.warn("⚠️ Sharpe: không đủ số điểm returns (<2); trả về NaN")
        return np.nan
    
    ann_factor = _annualization_factor(freq)
    periods_per_year = ann_factor ** 2
    
    # Annualized risk-free rate per period
    rf_per_period = rf / periods_per_year if periods_per_year > 0 else 0.0
    
    # Excess returns
    excess_returns = returns_clean - rf_per_period
    
    # Mean and std
    mean_excess = excess_returns.mean()
    std_returns = returns_clean.std(ddof=1)  # Sample std
    
    if std_returns == 0:
        # Kiểm tra xem có phải do không có trades không
        non_zero_returns = (returns_clean != 0).sum()
        if non_zero_returns == 0:
            warnings.warn(
                "⚠️ Sharpe: độ lệch chuẩn returns = 0 vì không có returns nào khác 0.\n"
                "   → Có thể do:\n"
                "   1. Không có trades được thực hiện (bật SL/TP trong risk management)\n"
                "   2. Strategy không tạo signals phù hợp\n"
                "   3. Equity curve không thay đổi\n"
                "   → Giải pháp: Bật Stop Loss (SL) và Take Profit (TP) để có trades"
            )
        else:
            warnings.warn(
                f"⚠️ Sharpe: độ lệch chuẩn returns = 0 (có {non_zero_returns} returns khác 0 nhưng tất cả bằng nhau).\n"
                "   → Có thể do:\n"
                "   1. Tất cả trades có cùng PnL\n"
                "   2. Equity curve không biến động\n"
                "   → Trả về 0.0"
            )
        return 0.0 if mean_excess == 0 else np.nan
    
    # Sharpe ratio (annualized) - CÔNG THỨC CHUẨN
    sharpe = (mean_excess / std_returns) * ann_factor
    

    # Nếu > 10, chắc chắn có lỗi trong tính toán
    if sharpe > 10:
        warnings.warn(
            f"⚠️ Sharpe ratio quá cao ({sharpe:.2f}) - có thể do:\n"
            f"  1. Annualization factor sai (đang dùng: {ann_factor:.2f})\n"
            f"  2. Returns được tính sai (mean: {mean_excess:.6f}, std: {std_returns:.6f})\n"
            f"  3. Equity curve có vấn đề (look-ahead bias, không tính costs đúng)\n"
            f"  4. Data quality issues\n"
            f"  → Capping tại 5.0 để tránh metrics ảo"
        )
        sharpe = min(sharpe, 5.0)
    elif sharpe > 5:
        warnings.warn(f"⚠️ Sharpe ratio rất cao ({sharpe:.2f}), kiểm tra lại tính toán")
    
    return sharpe


def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    downside = np.minimum(0.0, returns - mar)
    return np.sqrt((downside ** 2).mean())


def sortino_ratio(returns: pd.Series, rf: float = 0.0, freq: str | None = None) -> float:
    if returns.empty:
        return np.nan
    dr = downside_deviation(returns, 0.0)
    if dr == 0:
        return 0.0
    ann = _annualization_factor(freq)
    return (returns.mean() - rf / _annualization_factor(freq)) / dr * ann


def compound_annual_growth_rate(equity: pd.Series) -> float:
    """
    Calculate CAGR với calendar time đúng cách.
    
    CRITICAL: Phải dùng calendar days, không phải trading days.
    CRITICAL FIX: Cap CAGR để tránh numerical explosion.
    """
    if equity.empty or len(equity) < 2:
        return np.nan
    
    start = equity.iloc[0]
    end = equity.iloc[-1]
    
    if start <= 0 or end <= 0:
        return np.nan
    
    # CRITICAL: Cap end value nếu quá lớn (numerical issue)
    max_reasonable_ratio = 1e6  # 1 million times
    if end / start > max_reasonable_ratio:
        warnings.warn(
            f"⚠️ Equity ratio quá lớn ({end/start:.2e}), có thể do:\n"
            f"  1. Returns quá cao\n"
            f"  2. Equity curve tính sai\n"
            f"  3. Look-ahead bias\n"
            f"  → Capping tại {max_reasonable_ratio}x để tránh numerical issues"
        )
        end = start * max_reasonable_ratio
    
    # Use actual calendar time
    if isinstance(equity.index, pd.DatetimeIndex):
        days = (equity.index[-1] - equity.index[0]).days
        if days <= 0:
            return 0.0
        years = days / 365.25
    else:
        # Fallback: estimate from length (assume daily)
        years = len(equity) / 365.0
    
    if years <= 0:
        return np.nan
    
    # Minimum years để tránh division issues
    years = max(years, 1.0 / 365.25)  # At least 1 day
    
    # CAGR formula
    cagr = (end / start) ** (1 / years) - 1
    
    # CRITICAL: Cap CAGR tại 10 (1000%) - bất kỳ giá trị nào cao hơn đều không thực tế
    if cagr > 10:  # > 1000%
        warnings.warn(
            f"⚠️ CAGR quá cao ({cagr*100:.1f}%), có thể do:\n"
            f"  1. Equity curve tính sai\n"
            f"  2. Returns quá cao\n"
            f"  3. Look-ahead bias\n"
            f"  → Capping tại 1000% để tránh metrics ảo"
        )
        cagr = 10.0  # Cap at 1000%
    
    return cagr


def calmar_ratio(equity: pd.Series, returns: pd.Series, freq: str | None = None) -> float:
    cagr = compound_annual_growth_rate(equity)
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return np.nan
    return cagr / mdd


def safe_total_return(equity: pd.Series) -> float:
    """
    Tính Total Return với validation và cap để tránh giá trị không thực tế.
    
    CRITICAL: Cap Total Return tại 1000x (100,000%) - bất kỳ giá trị nào cao hơn
    đều có thể do lỗi tính toán, position sizing sai, hoặc leverage quá cao.
    """
    if equity.empty or len(equity) < 2:
        return 0.0
    
    start = equity.iloc[0]
    end = equity.iloc[-1]
    
    if start <= 0:
        return 0.0
    
    ratio = end / start
    
    # Cap tại 1000x (100,000%) - bất kỳ giá trị nào cao hơn đều không thực tế
    if ratio > 1000:
        warnings.warn(
            f"⚠️ Total Return quá cao ({ratio*100:.2f}%), có thể do:\n"
            f"  1. Equity curve tính sai\n"
            f"  2. Position sizing không đúng\n"
            f"  3. Leverage quá cao\n"
            f"  4. Look-ahead bias\n"
            f"  → Capping tại 100,000% để tránh metrics ảo"
        )
        return 1000.0 - 1  # 100,000% - 1
    
    return ratio - 1


def performance_summary(equity: pd.Series, returns: pd.Series, freq: str | None = None) -> dict:
    return {
        "CAGR": compound_annual_growth_rate(equity),
        "Sharpe": sharpe_ratio(returns, 0.0, freq),
        "Sortino": sortino_ratio(returns, 0.0, freq),
        "Calmar": calmar_ratio(equity, returns, freq),
        "MaxDrawdown": max_drawdown(equity),
        "Volatility": volatility(returns, freq),
        "TotalReturn": safe_total_return(equity),
    }

































