"""
Shared pytest fixtures cho trading bot tests.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_df():
    """
    Synthetic OHLCV DataFrame with 500 rows, BTC-like prices.
    Giá bắt đầu từ ~50000, random walk với volatility thấp.
    """
    np.random.seed(42)
    n = 500
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h", tz="UTC")
    
    # Random walk price
    returns = np.random.normal(0.0001, 0.005, n)
    close = 50000 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.003, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.003, n)))
    open_price = close * (1 + np.random.normal(0, 0.001, n))
    volume = np.random.uniform(100, 10000, n)
    
    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    
    return df


@pytest.fixture
def sample_signals(sample_ohlcv_df):
    """
    Matching signal Series cho sample_ohlcv_df.
    Signals: -1, 0, 1 (mostly 0 with some buys/sells).
    """
    np.random.seed(123)
    n = len(sample_ohlcv_df)
    signals = np.zeros(n, dtype=int)
    
    # Tạo ~5% buy signals, ~5% sell signals
    buy_idx = np.random.choice(n, size=n // 20, replace=False)
    remaining = np.setdiff1d(np.arange(n), buy_idx)
    sell_idx = np.random.choice(remaining, size=n // 20, replace=False)
    
    signals[buy_idx] = 1
    signals[sell_idx] = -1
    
    return pd.Series(signals, index=sample_ohlcv_df.index, name="signal")


@pytest.fixture
def uptrend_ohlcv_df():
    """
    OHLCV data with clear uptrend for testing positive returns.
    """
    np.random.seed(99)
    n = 300
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h", tz="UTC")
    
    # Strong uptrend
    returns = np.random.normal(0.002, 0.003, n)
    close = 50000 * np.exp(np.cumsum(returns))
    
    high = close * (1 + np.abs(np.random.normal(0, 0.002, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, n)))
    open_price = close * (1 + np.random.normal(0, 0.001, n))
    volume = np.random.uniform(100, 10000, n)
    
    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
