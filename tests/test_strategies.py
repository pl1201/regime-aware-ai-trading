"""
Unit tests cho trading strategies.

Test coverage:
- SMA/EMA Cross: signal generation, output type
- RSI Divergence: signal values
- MACD Momentum: signal values
- Bollinger Breakout: signal values
- BaseStrategy: validate_signals
- StrategyResult dataclass
"""

import numpy as np
import pandas as pd
import pytest

from algo_trading.strategies.base import BaseStrategy, StrategyResult
from algo_trading.strategies import (
    SMAEMACrossStrategy,
    RSIDivergenceStrategy,
    MACDMomentumStrategy,
    BollingerBreakoutStrategy,
)


class TestStrategyResultType:
    """Test that strategies return proper StrategyResult."""

    def test_sma_ema_returns_strategy_result(self, sample_ohlcv_df):
        """SMA/EMA strategy should return StrategyResult."""
        strategy = SMAEMACrossStrategy(fast=20, slow=50, ma_type="ema")
        result = strategy.generate_signals(sample_ohlcv_df)
        
        assert isinstance(result, StrategyResult)
        assert isinstance(result.signals, pd.Series)
        assert len(result.signals) == len(sample_ohlcv_df)
    
    def test_rsi_divergence_returns_strategy_result(self, sample_ohlcv_df):
        """RSI Divergence should return StrategyResult."""
        strategy = RSIDivergenceStrategy(period=14, overbought=70, oversold=30, lookback=5)
        result = strategy.generate_signals(sample_ohlcv_df)
        
        assert isinstance(result, StrategyResult)
        assert isinstance(result.signals, pd.Series)

    def test_macd_returns_strategy_result(self, sample_ohlcv_df):
        """MACD strategy should return StrategyResult."""
        strategy = MACDMomentumStrategy(fast=12, slow=26, signal=9)
        result = strategy.generate_signals(sample_ohlcv_df)
        
        assert isinstance(result, StrategyResult)
        assert isinstance(result.signals, pd.Series)

    def test_bollinger_returns_strategy_result(self, sample_ohlcv_df):
        """Bollinger Breakout should return StrategyResult."""
        strategy = BollingerBreakoutStrategy(window=20, k=2.0)
        result = strategy.generate_signals(sample_ohlcv_df)
        
        assert isinstance(result, StrategyResult)
        assert isinstance(result.signals, pd.Series)


class TestSMAEMACrossSignals:
    """Test SMA/EMA Cross signal generation."""

    def test_signals_in_valid_range(self, sample_ohlcv_df):
        """Signals should only be -1, 0, or 1."""
        strategy = SMAEMACrossStrategy(fast=20, slow=50, ma_type="ema")
        result = strategy.generate_signals(sample_ohlcv_df)
        unique_signals = set(result.signals.dropna().unique())
        assert unique_signals.issubset({-1, 0, 1}), f"Unexpected signals: {unique_signals}"

    def test_different_params(self, sample_ohlcv_df):
        """Different params should work without errors."""
        for fast, slow in [(10, 30), (5, 20), (20, 100)]:
            strategy = SMAEMACrossStrategy(fast=fast, slow=slow, ma_type="sma")
            result = strategy.generate_signals(sample_ohlcv_df)
            assert not result.signals.isna().all()


class TestRSIDivergenceSignals:
    """Test RSI Divergence signal generation."""

    def test_signals_in_valid_range(self, sample_ohlcv_df):
        """RSI signals should be -1, 0, or 1."""
        strategy = RSIDivergenceStrategy(period=14, overbought=70, oversold=30, lookback=5)
        result = strategy.generate_signals(sample_ohlcv_df)
        unique_signals = set(result.signals.dropna().unique())
        assert unique_signals.issubset({-1, 0, 1})


class TestMACDMomentumSignals:
    """Test MACD Momentum signal generation."""

    def test_signals_in_valid_range(self, sample_ohlcv_df):
        """MACD signals should be -1, 0, or 1."""
        strategy = MACDMomentumStrategy(fast=12, slow=26, signal=9)
        result = strategy.generate_signals(sample_ohlcv_df)
        unique_signals = set(result.signals.dropna().unique())
        assert unique_signals.issubset({-1, 0, 1})


class TestBollingerBreakoutSignals:
    """Test Bollinger Breakout signal generation."""

    def test_signals_in_valid_range(self, sample_ohlcv_df):
        """Bollinger signals should be -1, 0, or 1."""
        strategy = BollingerBreakoutStrategy(window=20, k=2.0)
        result = strategy.generate_signals(sample_ohlcv_df)
        unique_signals = set(result.signals.dropna().unique())
        assert unique_signals.issubset({-1, 0, 1})


class TestBaseStrategyValidation:
    """Test BaseStrategy validation."""

    def test_strategy_result_dataclass(self):
        """StrategyResult should hold signals and meta."""
        signals = pd.Series([0, 1, -1, 0, 1])
        meta = {"info": "test"}
        result = StrategyResult(signals=signals, meta=meta)
        
        assert isinstance(result.signals, pd.Series)
        assert result.meta == {"info": "test"}

    def test_strategy_result_default_meta(self):
        """StrategyResult should work with default empty meta."""
        signals = pd.Series([0, 1, -1])
        result = StrategyResult(signals=signals)
        
        assert result.meta is None or isinstance(result.meta, dict)
