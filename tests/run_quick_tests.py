"""Quick verification script - runs tests without pytest dependency."""
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def run_test(name, fn):
    try:
        fn()
        print(f"  ✅ {name}")
        return True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()
        return False

results = []

# ============================================================
print("\n=== 1. Circuit Breaker Tests ===")
from algo_trading.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

def test_cb_no_trigger():
    cb = CircuitBreaker(max_daily_loss_pct=5.0, max_consecutive_losses=5, initial_capital=10000.0)
    cb.record_trade(-50.0)
    cb.record_trade(-50.0)
    assert not cb.is_triggered(), "Should NOT trigger"
    assert cb.check_can_trade(), "Should allow trading"

def test_cb_trigger_daily():
    cb = CircuitBreaker(max_daily_loss_pct=5.0, max_consecutive_losses=100, initial_capital=10000.0)
    cb.record_trade(-500.0)
    assert cb.is_triggered(), "Should trigger on 5% loss"
    assert not cb.check_can_trade(), "Should block trading"

def test_cb_trigger_consecutive():
    cb = CircuitBreaker(max_daily_loss_pct=99.0, max_consecutive_losses=3, initial_capital=10000.0)
    cb.record_trade(-1.0)
    cb.record_trade(-1.0)
    cb.record_trade(-1.0)
    assert cb.is_triggered(), "Should trigger on 3 consecutive losses"

def test_cb_reset():
    cb = CircuitBreaker(max_daily_loss_pct=5.0, max_consecutive_losses=3, initial_capital=10000.0)
    cb.record_trade(-500.0)
    assert cb.is_triggered()
    cb.reset_daily()
    assert not cb.is_triggered(), "Should reset"
    assert cb.check_can_trade(), "Should allow trading after reset"

def test_cb_status():
    cb = CircuitBreaker(max_daily_loss_pct=5.0, max_consecutive_losses=5, initial_capital=10000.0)
    cb.record_trade(100.0)
    cb.record_trade(-50.0)
    s = cb.get_status()
    assert s["daily_pnl"] == 50.0
    assert s["total_trades_today"] == 2
    assert s["winning_trades_today"] == 1
    assert s["losing_trades_today"] == 1

results.append(run_test("CB: no trigger under limit", test_cb_no_trigger))
results.append(run_test("CB: trigger on daily loss", test_cb_trigger_daily))
results.append(run_test("CB: trigger on consecutive", test_cb_trigger_consecutive))
results.append(run_test("CB: reset", test_cb_reset))
results.append(run_test("CB: status dict", test_cb_status))

# ============================================================
print("\n=== 2. Metrics Tests ===")
import numpy as np
import pandas as pd
from algo_trading.core.metrics import (
    to_returns, sharpe_ratio, max_drawdown, compound_annual_growth_rate,
    performance_summary, safe_total_return, sortino_ratio, volatility,
)

def test_to_returns():
    prices = pd.Series([100, 105, 103, 110])
    rets = to_returns(prices, method="pct")
    assert len(rets) == 3
    assert abs(rets.iloc[0] - 0.05) < 1e-10

def test_sharpe_positive():
    np.random.seed(99)
    rets = pd.Series(np.random.normal(0.002, 0.003, 300))
    sr = sharpe_ratio(rets, rf=0.0, freq="1h")
    assert sr > 0, f"Expected positive Sharpe, got {sr}"

def test_sharpe_cap():
    rets = pd.Series(np.random.normal(0.1, 0.001, 500))
    sr = sharpe_ratio(rets, rf=0.0, freq="1d")
    assert sr <= 5.0, f"Sharpe should be capped at 5.0, got {sr}"

def test_max_drawdown():
    equity = pd.Series([100, 110, 105, 95, 100, 80, 90])
    mdd = max_drawdown(equity)
    assert mdd < 0

def test_cagr():
    idx = pd.date_range("2024-01-01", periods=365, freq="1D", tz="UTC")
    equity = pd.Series(np.linspace(100, 120, 365), index=idx)
    cagr = compound_annual_growth_rate(equity)
    assert cagr > 0

def test_performance_summary_keys():
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    equity = pd.Series(np.linspace(100, 120, 100), index=idx)
    rets = equity.pct_change().dropna()
    s = performance_summary(equity, rets, freq="1h")
    expected = {"CAGR", "Sharpe", "Sortino", "Calmar", "MaxDrawdown", "Volatility", "TotalReturn"}
    assert expected == set(s.keys()), f"Missing keys: {expected - set(s.keys())}"

def test_safe_total_return():
    equity = pd.Series([100, 120])
    ret = safe_total_return(equity)
    assert abs(ret - 0.2) < 1e-10

results.append(run_test("Metrics: to_returns", test_to_returns))
results.append(run_test("Metrics: sharpe positive", test_sharpe_positive))
results.append(run_test("Metrics: sharpe cap", test_sharpe_cap))
results.append(run_test("Metrics: max drawdown", test_max_drawdown))
results.append(run_test("Metrics: CAGR", test_cagr))
results.append(run_test("Metrics: performance_summary keys", test_performance_summary_keys))
results.append(run_test("Metrics: safe_total_return", test_safe_total_return))

# ============================================================
print("\n=== 3. Strategy Tests ===")
from algo_trading.strategies import (
    SMAEMACrossStrategy, RSIDivergenceStrategy,
    MACDMomentumStrategy, BollingerBreakoutStrategy,
)
from algo_trading.strategies.base import StrategyResult

np.random.seed(42)
n = 500
dates = pd.date_range(start="2024-01-01", periods=n, freq="1h", tz="UTC")
returns = np.random.normal(0.0001, 0.005, n)
close = 50000 * np.exp(np.cumsum(returns))
test_df = pd.DataFrame({
    "open": close * (1 + np.random.normal(0, 0.001, n)),
    "high": close * (1 + np.abs(np.random.normal(0, 0.003, n))),
    "low": close * (1 - np.abs(np.random.normal(0, 0.003, n))),
    "close": close,
    "volume": np.random.uniform(100, 10000, n),
}, index=dates)

def test_sma_ema():
    s = SMAEMACrossStrategy(fast=20, slow=50, ma_type="ema")
    r = s.generate_signals(test_df)
    assert isinstance(r, StrategyResult)
    assert len(r.signals) == len(test_df)
    uniq = set(r.signals.dropna().unique())
    assert uniq.issubset({-1, 0, 1}), f"Bad signals: {uniq}"

def test_rsi():
    s = RSIDivergenceStrategy(period=14, overbought=70, oversold=30, lookback=5)
    r = s.generate_signals(test_df)
    assert isinstance(r, StrategyResult)

def test_macd():
    s = MACDMomentumStrategy(fast=12, slow=26, signal=9)
    r = s.generate_signals(test_df)
    assert isinstance(r, StrategyResult)
    uniq = set(r.signals.dropna().unique())
    assert uniq.issubset({-1, 0, 1})

def test_bollinger():
    s = BollingerBreakoutStrategy(window=20, k=2.0)
    r = s.generate_signals(test_df)
    assert isinstance(r, StrategyResult)
    uniq = set(r.signals.dropna().unique())
    assert uniq.issubset({-1, 0, 1})

results.append(run_test("Strategy: SMA/EMA Cross", test_sma_ema))
results.append(run_test("Strategy: RSI Divergence", test_rsi))
results.append(run_test("Strategy: MACD Momentum", test_macd))
results.append(run_test("Strategy: Bollinger Breakout", test_bollinger))

# ============================================================
print("\n=== 4. Module Import Tests ===")

def test_import_train_fe():
    from train.ensemble.feature_engineering import calculate_indicators_enhanced
    assert callable(calculate_indicators_enhanced)

def test_import_train_ci():
    from train.ensemble.class_imbalance import handle_class_imbalance
    assert callable(handle_class_imbalance)

def test_import_train_dq():
    from train.ensemble.data_quality import data_quality_report
    assert callable(data_quality_report)

def test_import_train_lb():
    from train.ensemble.labeling import generate_labels_triple_barrier
    assert callable(generate_labels_triple_barrier)

def test_import_train_ts():
    from train.ensemble.threshold_scoring import optimize_threshold_trading_objective
    assert callable(optimize_threshold_trading_objective)

def test_import_evaluator():
    from algo_trading.live.strategy_evaluator import StrategyEvaluator
    assert hasattr(StrategyEvaluator, '_default_error_result')

results.append(run_test("Import: feature_engineering", test_import_train_fe))
results.append(run_test("Import: class_imbalance", test_import_train_ci))
results.append(run_test("Import: data_quality", test_import_train_dq))
results.append(run_test("Import: labeling", test_import_train_lb))
results.append(run_test("Import: threshold_scoring", test_import_train_ts))
results.append(run_test("Import: strategy_evaluator._default_error_result", test_import_evaluator))

# ============================================================
print("\n" + "=" * 60)
passed = sum(results)
total = len(results)
print(f"Results: {passed}/{total} passed")
if passed == total:
    print("🎉 ALL TESTS PASSED!")
else:
    print(f"⚠️ {total - passed} FAILED")
    sys.exit(1)
