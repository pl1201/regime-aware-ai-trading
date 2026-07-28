# Hướng Dẫn Triển Khai Các Nâng Cấp

## Tổng Quan

Tài liệu này hướng dẫn cách tích hợp các tính năng nâng cấp vào bot trading hiện tại.

---

## 1. Model Performance Monitor

### 1.1. Cài Đặt

File đã được tạo tại: `algo_trading/ml/model_monitor.py`

### 1.2. Tích Hợp Vào Live Trading Bot

**Bước 1:** Import vào `universal_bot.py`:

```python
from algo_trading.ml.model_monitor import ModelPerformanceMonitor
```

**Bước 2:** Thêm vào `LiveTradingBot.__init__()`:

```python
def __init__(self, ...):
    # ... existing code ...
    
    # Model performance monitor
    self.model_monitor = ModelPerformanceMonitor(
        model_path=self.config.model_path or "models/regime_ensemble_optimized.pkl",
        min_winrate=0.45,
        min_sharpe=0.5,
        lookback_days=30,
        baseline_winrate=0.52,  # Từ backtest của bạn
        baseline_sharpe=1.0,    # Từ backtest của bạn
    )
```

**Bước 3:** Log trades khi exit position trong `_exit_position()`:

```python
def _exit_position(self, reason: str = "signal"):
    # ... existing exit logic ...
    
    # Log trade vào monitor
    if self.entry_price and self.exit_price and self.entry_time:
        self.model_monitor.log_trade(
            entry_price=self.entry_price,
            exit_price=self.exit_price,
            direction=self.position_direction,
            entry_time=self.entry_time,
            exit_time=datetime.now(),
            symbol=self.config.symbol,
            exit_reason=reason,
            regime=getattr(self, 'current_regime', None),
            model_version=Path(self.config.model_path).stem if self.config.model_path else None
        )
```

**Bước 4:** Check retrain định kỳ trong `run_once()`:

```python
def run_once(self):
    # ... existing trading logic ...
    
    # Check model performance mỗi ngày (hoặc sau N trades)
    if not hasattr(self, '_last_performance_check'):
        self._last_performance_check = datetime.now()
    
    # Check mỗi 24 giờ
    if (datetime.now() - self._last_performance_check).total_seconds() > 86400:
        needs_retrain, info = self.model_monitor.check_retrain_needed()
        if needs_retrain:
            report = self.model_monitor.get_performance_report()
            logger.warning(f"⚠️ MODEL DRIFT DETECTED!\n{report}")
            # Có thể trigger retrain ở đây
        
        self._last_performance_check = datetime.now()
```

### 1.3. Sử Dụng

Sau khi tích hợp, monitor sẽ tự động:
- Log mỗi trade khi exit
- Tính performance metrics (winrate, Sharpe, drawdown)
- So sánh với baseline
- Alert khi cần retrain

Xem performance report:
```python
report = bot.model_monitor.get_performance_report()
print(report)
```

---

## 2. Enhanced Error Handling & Resilience

### 2.1. Tạo File Resilience

Tạo file `algo_trading/live/resilience.py` với code từ `UPGRADE_RECOMMENDATIONS.md`.

### 2.2. Tích Hợp Vào OKX Client

**Trong `okx_client.py`:**

```python
from algo_trading.live.resilience import retry_with_backoff, CircuitBreaker

class OKXClient(ExchangeClient):
    def __init__(self, ...):
        # ... existing code ...
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def _make_request(self, method: str, endpoint: str, ...):
        # Wrap với circuit breaker
        def _call():
            # ... existing request code ...
            return response.json()
        
        return self.circuit_breaker.call(_call)
```

### 2.3. Fallback Strategy

**Trong `universal_bot.py`:**

```python
def _load_strategy(self):
    try:
        # Load strategy như bình thường
        self.strategy = ...
    except Exception as e:
        logger.error(f"❌ Không load được strategy: {e}")
        logger.info("🔄 Fallback về SMA/EMA strategy")
        # Fallback về strategy đơn giản
        from algo_trading.strategies import SMAEMACrossStrategy
        self.strategy = SMAEMACrossStrategy(fast=20, slow=50)
```

---

## 3. Rate Limiting

### 3.1. Tạo File Rate Limiter

Tạo file `algo_trading/live/rate_limiter.py` với code từ `UPGRADE_RECOMMENDATIONS.md`.

### 3.2. Tích Hợp Vào OKX Client

**Trong `okx_client.py`:**

```python
from algo_trading.live.rate_limiter import RateLimiter

# Global rate limiter cho OKX (20 req/s)
okx_rate_limiter = RateLimiter(max_calls=20, time_window=1.0)

class OKXClient(ExchangeClient):
    def _make_request(self, ...):
        # Wait nếu cần
        okx_rate_limiter.wait_if_needed()
        
        # ... existing request code ...
```

---

## 4. Trade Database

### 4.1. Tạo File Database

Tạo file `algo_trading/data/trade_database.py` với code từ `UPGRADE_RECOMMENDATIONS.md`.

### 4.2. Tích Hợp Vào Live Trading Bot

**Trong `universal_bot.py`:**

```python
from algo_trading.data.trade_database import TradeDatabase

class LiveTradingBot:
    def __init__(self, ...):
        # ... existing code ...
        self.trade_db = TradeDatabase(db_path="data/trades.db")
    
    def _exit_position(self, reason: str = "signal"):
        # ... existing exit logic ...
        
        # Save to database
        trade_data = {
            'symbol': self.config.symbol,
            'direction': self.position_direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time,
            'exit_time': datetime.now(),
            'quantity': self.position_size,
            'pnl': (self.exit_price - self.entry_price) * self.position_direction,
            'pnl_pct': ((self.exit_price - self.entry_price) / self.entry_price) * 100 * self.position_direction,
            'exit_reason': reason,
            'regime': getattr(self, 'current_regime', None),
            'model_version': Path(self.config.model_path).stem if self.config.model_path else None,
        }
        self.trade_db.save_trade(trade_data)
```

### 4.3. Query Trades

```python
# Lấy trades gần đây
trades_df = bot.trade_db.get_trades(symbol="BTCUSDT", limit=100)

# Phân tích
print(trades_df.describe())
```

---

## 5. Dynamic Risk Management

### 5.1. Tạo File Dynamic Risk

Tạo file `algo_trading/core/dynamic_risk.py` với code từ `UPGRADE_RECOMMENDATIONS.md`.

### 5.2. Tích Hợp Vào Live Trading Bot

**Trong `universal_bot.py`:**

```python
from algo_trading.core.dynamic_risk import DynamicRiskManager

class LiveTradingBot:
    def __init__(self, ...):
        # ... existing code ...
        self.dynamic_risk = DynamicRiskManager()
    
    def _calculate_position_size(self, last_price: float, df: pd.DataFrame) -> float:
        # Lấy current regime (từ strategy nếu có)
        current_regime = getattr(self.strategy, 'current_regime', 'calm')
        
        # Dùng dynamic risk
        risk_per_trade = self.dynamic_risk.get_risk_per_trade(current_regime)
        
        # ... existing position sizing logic với risk_per_trade mới ...
    
    def _update_stop_loss_take_profit(self, entry_price: float, direction: int, df: pd.DataFrame):
        # Lấy current regime
        current_regime = getattr(self.strategy, 'current_regime', 'calm')
        
        # Apply multipliers
        sl_multiplier = self.dynamic_risk.get_sl_multiplier(current_regime)
        tp_multiplier = self.dynamic_risk.get_tp_multiplier(current_regime)
        
        # Adjust SL/TP với multipliers
        # ... existing SL/TP logic với multipliers ...
```

---

## 6. Testing

### 6.1. Test Model Monitor

```python
# Test monitor
monitor = ModelPerformanceMonitor(...)

# Simulate trades
for i in range(30):
    monitor.log_trade(...)

# Check performance
report = monitor.get_performance_report()
print(report)

# Check retrain
needs_retrain, info = monitor.check_retrain_needed()
assert isinstance(needs_retrain, bool)
```

### 6.2. Test Rate Limiter

```python
from algo_trading.live.rate_limiter import RateLimiter

limiter = RateLimiter(max_calls=5, time_window=1.0)

# Test
import time
start = time.time()
for i in range(10):
    limiter.wait_if_needed()
    print(f"Call {i+1}")
end = time.time()

# Should take ~1 second (5 calls allowed, then wait)
assert end - start >= 1.0
```

### 6.3. Test Trade Database

```python
from algo_trading.data.trade_database import TradeDatabase

db = TradeDatabase("test_trades.db")

# Save trade
db.save_trade({
    'symbol': 'BTCUSDT',
    'direction': 1,
    'entry_price': 50000,
    'exit_price': 51000,
    ...
})

# Query
trades = db.get_trades(symbol='BTCUSDT', limit=10)
assert len(trades) > 0
```

---

## 7. Deployment Checklist

- [ ] Test Model Monitor với paper trading
- [ ] Test Rate Limiter với API calls
- [ ] Test Trade Database với real trades
- [ ] Test Dynamic Risk Management với different regimes
- [ ] Verify error handling với simulated failures
- [ ] Check Telegram notifications
- [ ] Monitor performance metrics sau 1 tuần
- [ ] Review và adjust thresholds nếu cần

---

## 8. Monitoring & Maintenance

### 8.1. Daily Checks

- Xem performance report từ Model Monitor
- Check trade database có lưu đúng không
- Verify rate limiter không block quá nhiều

### 8.2. Weekly Reviews

- Review performance metrics
- Check model drift
- Adjust risk parameters nếu cần

### 8.3. Monthly Tasks

- Retrain models nếu cần
- Review và optimize thresholds
- Update baseline metrics từ backtest mới

---

## 9. Troubleshooting

### Model Monitor không log trades
- Check `_exit_position()` có gọi `model_monitor.log_trade()` không
- Verify file permissions cho history file

### Rate Limiter block quá nhiều
- Adjust `max_calls` và `time_window`
- Check API rate limits của exchange

### Trade Database lỗi
- Check SQLite file permissions
- Verify schema được tạo đúng

---

## 10. Next Steps

Sau khi implement Phase 1, tiếp tục với:
- Phase 2: Advanced Position Sizing, Trailing Stop
- Phase 3: Analytics Dashboard, Alerts

Xem `UPGRADE_RECOMMENDATIONS.md` để có code examples đầy đủ.
