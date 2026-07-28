"""
Unit tests cho algo_trading.core.metrics.

Test coverage:
- to_returns (log & pct)
- sharpe_ratio (positive, zero std, cap)
- max_drawdown
- compound_annual_growth_rate (CAGR)
- sortino_ratio
- performance_summary keys
- volatility
- safe_total_return
"""

import numpy as np
import pandas as pd
import pytest

from algo_trading.core.metrics import (
    to_returns,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    drawdown_series,
    compound_annual_growth_rate,
    calmar_ratio,
    volatility,
    performance_summary,
    safe_total_return,
    _annualization_factor,
    infer_freq_label_from_index,
)


class TestToReturns:
    """Test to_returns function."""

    def test_log_returns(self, sample_ohlcv_df):
        """Log returns should work correctlyexists."""
        rets = to_returns(sample_ohlcv_df, method="log")
        assert isinstance(rets, pd.Series)
        assert len(rets) == len(sample_ohlcv_df) - 1  # diff drops first row
        assert not rets.isna().all()

    def test_pct_returns(self, sample_ohlcv_df):
        """Percentage returns should work."""
        rets = to_returns(sample_ohlcv_df, method="pct")
        assert isinstance(rets, pd.Series)
        assert len(rets) == len(sample_ohlcv_df) - 1

    def test_returns_from_series(self):
        """Should work with a plain Series."""
        prices = pd.Series([100, 105, 103, 110, 108])
        rets = to_returns(prices, method="pct")
        assert len(rets) == 4
        assert abs(rets.iloc[0] - 0.05) < 1e-10


class TestSharpeRatio:
    """Test sharpe_ratio function."""

    def test_positive_sharpe_for_uptrend(self, uptrend_ohlcv_df):
        """Uptrend data should produce positive Sharpe."""
        rets = to_returns(uptrend_ohlcv_df, method="log")
        sr = sharpe_ratio(rets, rf=0.0, freq="1h")
        assert sr > 0, f"Expected positive Sharpe for uptrend, got {sr}"

    def test_zero_returns_gives_zero(self):
        """Constant price => zero or NaN Sharpe."""
        prices = pd.Series([100.0] * 100)
        rets = to_returns(prices, method="pct")
        sr = sharpe_ratio(rets, rf=0.0)
        assert sr == 0.0 or np.isnan(sr)

    def test_sharpe_cap(self):
        """Extremely high Sharpe should be capped at 5.0."""
        # Create returns that would produce Sharpe > 10
        np.random.seed(10)
        rets = pd.Series(np.random.normal(0.1, 0.001, 500))
        sr = sharpe_ratio(rets, rf=0.0, freq="1d")
        assert sr <= 5.0, f"Sharpe should be capped at 5.0, got {sr}"

    def test_empty_returns_nan(self):
        """Empty returns should give NaN."""
        rets = pd.Series([], dtype=float)
        sr = sharpe_ratio(rets)
        assert np.isnan(sr)


class TestMaxDrawdown:
    """Test max_drawdown and drawdown_series."""

    def test_known_drawdown(self):
        """Test with known equity curve."""
        equity = pd.Series([100, 110, 105, 95, 100, 80, 90])
        mdd = max_drawdown(equity)
        # Peak was 110, lowest after peak is 80 => DD = 80/110 - 1 = -0.2727
        assert mdd < 0, "Max drawdown should be negative"
        assert abs(mdd - (80 / 110 - 1)) < 0.01

    def test_monotonic_up_no_drawdown(self):
        """Monotonically increasing equity should have 0 drawdown."""
        equity = pd.Series(range(1, 101))
        mdd = max_drawdown(equity)
        assert mdd == 0.0

    def test_drawdown_series_shape(self, sample_ohlcv_df):
        """Drawdown series should have same length as input."""
        equity = sample_ohlcv_df["close"]
        dd = drawdown_series(equity)
        assert len(dd) == len(equity)
        assert (dd <= 0).all()


class TestCAGR:
    """Test compound_annual_growth_rate."""

    def test_cagr_positive_for_uptrend(self, uptrend_ohlcv_df):
        """CAGR should be positive for uptrending equity."""
        equity = uptrend_ohlcv_df["close"]
        cagr = compound_annual_growth_rate(equity)
        assert cagr > 0, f"CAGR should be positive, got {cagr}"

    def test_cagr_cap(self):
        """Extremely high CAGR should be capped."""
        idx = pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC")
        equity = pd.Series([1, 10, 100, 1000, 10000, 100000, 1e6, 1e7, 1e8, 1e9], index=idx)
        cagr = compound_annual_growth_rate(equity)
        assert cagr <= 10.0, f"CAGR should be capped at 10 (1000%), got {cagr}"

    def test_cagr_empty(self):
        """Empty equity should return NaN."""
        cagr = compound_annual_growth_rate(pd.Series([], dtype=float))
        assert np.isnan(cagr)


class TestSortinoRatio:
    """Test sortino_ratio."""

    def test_sortino_positive_for_uptrend(self, uptrend_ohlcv_df):
        """Uptrend should give positive Sortino."""
        rets = to_returns(uptrend_ohlcv_df, method="log")
        sr = sortino_ratio(rets, rf=0.0, freq="1h")
        assert sr > 0, f"Expected positive Sortino, got {sr}"


class TestVolatility:
    """Test volatility function."""

    def test_volatility_positive(self, sample_ohlcv_df):
        """Volatility should be positive for real data."""
        rets = to_returns(sample_ohlcv_df, method="log")
        vol = volatility(rets, freq="1h")
        assert vol > 0


class TestSafeTotalReturn:
    """Test safe_total_return."""

    def test_normal_return(self):
        """Test normal case."""
        equity = pd.Series([100, 120])
        ret = safe_total_return(equity)
        assert abs(ret - 0.2) < 1e-10  # 20%

    def test_cap_extreme_return(self):
        """Should cap returns > 1000x."""
        equity = pd.Series([1, 100000])
        ret = safe_total_return(equity)
        assert ret <= 999.0


class TestPerformanceSummary:
    """Test performance_summary."""

    def test_all_keys_present(self, sample_ohlcv_df):
        """All expected keys should be in the summary."""
        equity = sample_ohlcv_df["close"]
        rets = to_returns(sample_ohlcv_df, method="log")
        summary = performance_summary(equity, rets, freq="1h")
        
        expected_keys = {"CAGR", "Sharpe", "Sortino", "Calmar", "MaxDrawdown", "Volatility", "TotalReturn"}
        assert expected_keys == set(summary.keys())


class TestAnnualizationFactor:
    """Test _annualization_factor."""

    def test_daily_crypto(self):
        """Crypto daily should use 365."""
        f = _annualization_factor("1d")
        assert abs(f - np.sqrt(365)) < 0.01

    def test_hourly_crypto(self):
        """Hourly should use 365*24."""
        f = _annualization_factor("1h")
        assert abs(f - np.sqrt(365 * 24)) < 0.1


class TestInferFreqLabel:
    """Test infer_freq_label_from_index."""

    def test_hourly_index(self):
        """Hourly DatetimeIndex should infer '1h'."""
        idx = pd.date_range("2024-01-01", periods=100, freq="1h")
        label = infer_freq_label_from_index(idx)
        assert label == "1h"

    def test_daily_index(self):
        """Daily DatetimeIndex should infer '1d'."""
        idx = pd.date_range("2024-01-01", periods=100, freq="1D")
        label = infer_freq_label_from_index(idx)
        assert label == "1d"
