# Đề Xuất Nâng Cấp Bot Trading

## Tổng Quan

Sau khi phân tích toàn bộ codebase, đây là các đề xuất nâng cấp được ưu tiên theo mức độ quan trọng và tác động.

---

## 🔴 MỨC ĐỘ 1: QUAN TRỌNG - CẦN THỰC HIỆN NGAY

### 1.1. Model Monitoring & Auto-Retraining System

**Vấn đề hiện tại:**
- Không có hệ thống theo dõi performance của model trong live trading
- Không tự động phát hiện model drift (khi model không còn phù hợp với thị trường hiện tại)
- Phải retrain thủ công, không biết khi nào cần retrain

**Giải pháp đề xuất:**

```python
# algo_trading/ml/model_monitor.py
class ModelPerformanceMonitor:
    """
    Monitor model performance và tự động trigger retraining khi cần.
    """
    def __init__(
        self,
        model_path: str,
        min_winrate: float = 0.45,
        min_sharpe: float = 0.5,
        lookback_days: int = 30,
        retrain_threshold: float = 0.15  # Performance giảm >15%
    ):
        self.model_path = model_path
        self.min_winrate = min_winrate
        self.min_sharpe = min_sharpe
        self.lookback_days = lookback_days
        self.retrain_threshold = retrain_threshold
        self.trade_history = []  # Lưu trades để tính metrics
    
    def log_trade(self, entry_price, exit_price, direction, entry_time, exit_time):
        """Log mỗi trade để tính performance metrics."""
        pnl = (exit_price - entry_price) * direction
        self.trade_history.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'pnl': pnl,
            'entry_time': entry_time,
            'exit_time': exit_time,
        })
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Tính winrate, Sharpe, drawdown từ trade history."""
        if len(self.trade_history) < 10:
            return None
        
        df = pd.DataFrame(self.trade_history)
        df['pnl_pct'] = df['pnl'] / df['entry_price'] * 100
        
        winrate = (df['pnl'] > 0).sum() / len(df)
        avg_win = df[df['pnl'] > 0]['pnl_pct'].mean() if (df['pnl'] > 0).any() else 0
        avg_loss = abs(df[df['pnl'] < 0]['pnl_pct'].mean()) if (df['pnl'] < 0).any() else 0
        
        returns = df['pnl_pct'].values
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        return {
            'winrate': winrate,
            'sharpe': sharpe,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(df),
        }
    
    def check_retrain_needed(self) -> bool:
        """Kiểm tra xem có cần retrain không."""
        metrics = self.calculate_performance_metrics()
        if metrics is None:
            return False
        
        if metrics['winrate'] < self.min_winrate:
            return True
        if metrics['sharpe'] < self.min_sharpe:
            return True
        
        # So sánh với baseline (backtest metrics)
        baseline_winrate = 0.52  # Từ backtest
        if metrics['winrate'] < baseline_winrate * (1 - self.retrain_threshold):
            return True
        
        return False
    
    def trigger_retrain(self):
        """Trigger retraining process."""
        # Gọi training script
        # Có thể dùng subprocess hoặc queue để retrain async
        pass
```

**Tích hợp vào `universal_bot.py`:**
- Thêm `ModelPerformanceMonitor` vào `LiveTradingBot`
- Log mỗi trade vào monitor
- Check định kỳ (mỗi ngày) xem có cần retrain không
- Gửi alert qua Telegram khi cần retrain

---

### 1.2. Enhanced Error Handling & Resilience

**Vấn đề hiện tại:**
- Error handling cơ bản, nhưng chưa có retry logic cho API calls
- Không có circuit breaker khi API fail liên tục
- Không có fallback khi model load fail

**Giải pháp đề xuất:**

```python
# algo_trading/live/resilience.py
from functools import wraps
import time
from typing import Callable, Any

class CircuitBreaker:
    """Circuit breaker pattern để tránh spam API khi fail."""
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Gọi function với circuit breaker."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
):
    """Decorator để retry với exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
            return None
        return wrapper
    return decorator
```

**Áp dụng vào `okx_client.py` và `universal_bot.py`:**
- Wrap tất cả API calls với `retry_with_backoff`
- Dùng `CircuitBreaker` cho các API calls quan trọng
- Fallback: Nếu model load fail, dùng strategy đơn giản hơn (ví dụ: SMA/EMA)

---

### 1.3. Rate Limiting & Connection Pooling

**Vấn đề hiện tại:**
- Không có rate limiting cho API calls
- Có thể bị ban nếu gọi API quá nhiều
- Không có connection pooling

**Giải pháp đề xuất:**

```python
# algo_trading/live/rate_limiter.py
import time
from collections import deque
from threading import Lock

class RateLimiter:
    """Rate limiter để tránh vượt quá API limits."""
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = Lock()
    
    def wait_if_needed(self):
        """Đợi nếu cần để không vượt quá rate limit."""
        with self.lock:
            now = time.time()
            # Xóa các calls cũ hơn time_window
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # Xóa lại sau khi sleep
                    while self.calls and self.calls[0] < now:
                        self.calls.popleft()
            
            self.calls.append(time.time())

# OKX rate limits: 20 requests/second
okx_rate_limiter = RateLimiter(max_calls=20, time_window=1.0)
```

**Tích hợp:**
- Wrap mỗi API call trong `okx_client.py` với `okx_rate_limiter.wait_if_needed()`
- Tương tự cho Binance API

---

### 1.4. Database để Lưu Trades & Metrics

**Vấn đề hiện tại:**
- Trades chỉ log vào file text
- Khó query và phân tích performance
- Không có lịch sử lâu dài

**Giải pháp đề xuất:**

```python
# algo_trading/data/trade_database.py
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

class TradeDatabase:
    """SQLite database để lưu trades và metrics."""
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Khởi tạo database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                quantity REAL NOT NULL,
                pnl REAL,
                pnl_pct REAL,
                exit_reason TEXT,
                regime TEXT,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                symbol TEXT NOT NULL,
                winrate REAL,
                sharpe REAL,
                total_trades INTEGER,
                total_pnl REAL,
                max_drawdown REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, symbol)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_trade(self, trade: Dict):
        """Lưu một trade vào database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (
                symbol, direction, entry_price, exit_price,
                entry_time, exit_time, quantity, pnl, pnl_pct,
                exit_reason, regime, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade['symbol'],
            trade['direction'],
            trade['entry_price'],
            trade.get('exit_price'),
            trade['entry_time'],
            trade.get('exit_time'),
            trade['quantity'],
            trade.get('pnl'),
            trade.get('pnl_pct'),
            trade.get('exit_reason'),
            trade.get('regime'),
            trade.get('model_version'),
        ))
        
        conn.commit()
        conn.close()
    
    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """Lấy trades từ database."""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM trades"
        params = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        query += " ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
```

**Tích hợp:**
- Thêm `TradeDatabase` vào `LiveTradingBot`
- Lưu mỗi trade khi entry và update khi exit
- Tạo dashboard để xem performance từ database

---

## 🟡 MỨC ĐỘ 2: QUAN TRỌNG - NÊN THỰC HIỆN SỚM

### 2.1. Dynamic Risk Management theo Regime

**Vấn đề hiện tại:**
- Risk management cố định, không thay đổi theo regime
- Không tận dụng được thông tin regime để điều chỉnh risk

**Giải pháp đề xuất:**

```python
# algo_trading/core/dynamic_risk.py
class DynamicRiskManager:
    """Quản lý risk động theo regime và market conditions."""
    def __init__(self):
        # Risk per trade theo regime
        self.regime_risk_map = {
            'trending': 0.15,      # Risk cao hơn khi trending
            'ranging': 0.08,       # Risk thấp khi ranging
            'volatile': 0.05,      # Risk rất thấp khi volatile
            'calm': 0.10,          # Risk trung bình khi calm
        }
        
        # SL/TP multipliers theo regime
        self.regime_sl_multiplier = {
            'trending': 1.2,       # SL rộng hơn khi trending
            'ranging': 0.8,        # SL chặt hơn khi ranging
            'volatile': 1.5,       # SL rất rộng khi volatile
            'calm': 1.0,
        }
        
        self.regime_tp_multiplier = {
            'trending': 2.0,       # TP xa hơn khi trending
            'ranging': 1.2,        # TP gần hơn khi ranging
            'volatile': 1.5,
            'calm': 1.5,
        }
    
    def get_risk_per_trade(self, current_regime: str) -> float:
        """Lấy risk per trade theo regime."""
        return self.regime_risk_map.get(current_regime, 0.10)
    
    def get_sl_multiplier(self, current_regime: str) -> float:
        """Lấy SL multiplier theo regime."""
        return self.regime_sl_multiplier.get(current_regime, 1.0)
    
    def get_tp_multiplier(self, current_regime: str) -> float:
        """Lấy TP multiplier theo regime."""
        return self.regime_tp_multiplier.get(current_regime, 1.5)
    
    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        sl_distance: float,
        current_regime: str
    ) -> float:
        """Tính position size với dynamic risk."""
        risk_per_trade = self.get_risk_per_trade(current_regime)
        risk_amount = balance * risk_per_trade
        position_size = risk_amount / sl_distance
        return position_size
```

**Tích hợp:**
- Thêm vào `LiveTradingBot._calculate_position_size()`
- Điều chỉnh SL/TP trong `_update_stop_loss_take_profit()` theo regime

---

### 2.2. Advanced Position Sizing (Kelly Criterion, Volatility Targeting)

**Vấn đề hiện tại:**
- Position sizing đơn giản, chỉ dựa trên % balance
- Không tối ưu theo winrate và risk/reward

**Giải pháp đề xuất:**

```python
# algo_trading/core/position_sizing.py
class AdvancedPositionSizing:
    """Advanced position sizing methods."""
    
    @staticmethod
    def kelly_fraction(winrate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly Criterion: f = (p * b - q) / b
        f: fraction of capital to bet
        p: win probability
        q: loss probability (1 - p)
        b: win/loss ratio
        """
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss
        f = (winrate * b - (1 - winrate)) / b
        # Giới hạn Kelly fraction để tránh quá aggressive
        return max(0.0, min(f, 0.25))  # Max 25% per trade
    
    @staticmethod
    def volatility_targeting(
        balance: float,
        target_volatility: float = 0.15,  # 15% annual volatility
        current_volatility: float,
        price: float
    ) -> float:
        """
        Position sizing để đạt target volatility.
        """
        if current_volatility == 0:
            return 0.0
        
        # Tính position size để đạt target volatility
        position_value = balance * (target_volatility / current_volatility)
        position_size = position_value / price
        return position_size
    
    @staticmethod
    def fixed_fractional(
        balance: float,
        risk_per_trade: float,
        sl_distance_pct: float
    ) -> float:
        """Fixed fractional (hiện tại đang dùng)."""
        risk_amount = balance * risk_per_trade
        position_size = risk_amount / (balance * sl_distance_pct)
        return position_size
```

**Tích hợp:**
- Thêm vào `LiveTradingBot` với option chọn method
- Có thể combine: dùng Kelly fraction nhưng giới hạn bởi volatility targeting

---

### 2.3. Trailing Stop Loss Động

**Vấn đề hiện tại:**
- Có trailing stop nhưng chưa tối ưu
- Không điều chỉnh theo volatility

**Giải pháp đề xuất:**

```python
# algo_trading/core/trailing_stop.py
class AdaptiveTrailingStop:
    """Trailing stop loss với adaptive distance theo volatility."""
    def __init__(
        self,
        initial_distance_pct: float = 0.02,
        atr_multiplier: float = 2.0,
        min_distance_pct: float = 0.01,
        max_distance_pct: float = 0.05
    ):
        self.initial_distance_pct = initial_distance_pct
        self.atr_multiplier = atr_multiplier
        self.min_distance_pct = min_distance_pct
        self.max_distance_pct = max_distance_pct
    
    def update_stop(
        self,
        current_price: float,
        entry_price: float,
        direction: int,
        atr: float,
        highest_price: float = None,  # Cho LONG
        lowest_price: float = None     # Cho SHORT
    ) -> float:
        """
        Update trailing stop loss.
        Returns new stop loss price.
        """
        if direction == 1:  # LONG
            if highest_price is None:
                highest_price = current_price
            
            # Tính distance dựa trên ATR
            atr_distance = atr * self.atr_multiplier
            pct_distance = atr_distance / current_price
            
            # Giới hạn trong min/max
            pct_distance = max(self.min_distance_pct, min(pct_distance, self.max_distance_pct))
            
            # Trailing stop: chỉ move up, không move down
            new_stop = highest_price * (1 - pct_distance)
            current_stop = entry_price * (1 - self.initial_distance_pct)
            
            return max(new_stop, current_stop)
        
        else:  # SHORT
            if lowest_price is None:
                lowest_price = current_price
            
            atr_distance = atr * self.atr_multiplier
            pct_distance = atr_distance / current_price
            pct_distance = max(self.min_distance_pct, min(pct_distance, self.max_distance_pct))
            
            new_stop = lowest_price * (1 + pct_distance)
            current_stop = entry_price * (1 + self.initial_distance_pct)
            
            return min(new_stop, current_stop)
```

**Tích hợp:**
- Thêm vào `LiveTradingBot._check_stop_loss_take_profit()`
- Update trailing stop mỗi bar mới

---

### 2.4. Portfolio Management (Multi-Asset Trading)

**Vấn đề hiện tại:**
- Chỉ trade 1 symbol tại một thời điểm
- Không có portfolio-level risk management

**Giải pháp đề xuất:**

```python
# algo_trading/live/portfolio_manager.py
class PortfolioManager:
    """Quản lý portfolio với nhiều assets."""
    def __init__(
        self,
        max_positions: int = 5,
        max_correlation: float = 0.7,  # Không trade assets có correlation > 0.7
        total_risk_budget: float = 0.3  # Tổng risk tối đa 30% portfolio
    ):
        self.max_positions = max_positions
        self.max_correlation = max_correlation
        self.total_risk_budget = total_risk_budget
        self.positions = {}  # {symbol: position_info}
        self.correlation_matrix = None
    
    def can_open_position(self, symbol: str, risk_amount: float) -> bool:
        """Kiểm tra xem có thể mở position mới không."""
        # Check số lượng positions
        if len(self.positions) >= self.max_positions:
            return False
        
        # Check total risk
        current_total_risk = sum(p['risk'] for p in self.positions.values())
        if current_total_risk + risk_amount > self.total_risk_budget:
            return False
        
        # Check correlation với các positions hiện tại
        if self.correlation_matrix is not None:
            for existing_symbol in self.positions:
                corr = self.correlation_matrix.get(symbol, {}).get(existing_symbol, 0)
                if abs(corr) > self.max_correlation:
                    return False
        
        return True
    
    def allocate_risk(self, symbol: str, base_risk: float) -> float:
        """Phân bổ risk cho symbol mới."""
        if not self.can_open_position(symbol, base_risk):
            return 0.0
        
        # Có thể điều chỉnh risk theo số lượng positions hiện tại
        # Ví dụ: nếu đã có nhiều positions, giảm risk cho position mới
        if len(self.positions) > 0:
            allocated_risk = base_risk / (len(self.positions) + 1)
        else:
            allocated_risk = base_risk
        
        return allocated_risk
```

**Tích hợp:**
- Tạo `MultiAssetTradingBot` wrapper cho `LiveTradingBot`
- Chạy nhiều bot instances cho nhiều symbols
- Dùng `PortfolioManager` để điều phối

---

## 🟢 MỨC ĐỘ 3: CẢI THIỆN - CÓ THỂ THỰC HIỆN SAU

### 3.1. Model Versioning & A/B Testing

**Đề xuất:**
- Lưu model versions với metadata (ngày train, metrics, hyperparameters)
- A/B testing: chạy 2 models song song, so sánh performance
- Tự động switch sang model tốt hơn

### 3.2. Real-time Alerts & Notifications

**Đề xuất:**
- Webhook integration (Discord, Slack, email)
- Alert khi: trade executed, SL/TP hit, model drift detected, lỗi nghiêm trọng
- Custom alert rules

### 3.3. Advanced Analytics Dashboard

**Đề xuất:**
- Dashboard với Streamlit/Plotly Dash
- Real-time PnL, winrate, Sharpe ratio
- Trade history với filters
- Regime distribution charts
- Model performance over time

### 3.4. Feature Importance Monitoring

**Đề xuất:**
- Track feature importance changes over time
- Alert khi feature importance thay đổi đột ngột (có thể là drift)
- Feature selection tự động dựa trên importance

### 3.5. Data Quality Checks

**Đề xuất:**
- Validate data trước khi train: check missing values, outliers, data freshness
- Alert khi data quality kém
- Auto-fix hoặc skip bad data

### 3.6. Model Explainability (SHAP)

**Đề xuất:**
- Dùng SHAP để explain predictions
- Hiểu tại sao model đưa ra signal
- Debug và cải thiện model

### 3.7. Paper Trading Improvements

**Đề xuất:**
- Simulate slippage, latency
- Realistic order execution
- Compare paper vs live performance

### 3.8. DCA (Dollar Cost Averaging) Enhancement

**Đề xuất:**
- Smart DCA: chỉ DCA khi có confluence (regime + signal)
- Dynamic DCA spacing theo volatility
- Max DCA levels với risk management

---

## 📊 Ưu Tiên Thực Hiện

### Phase 1 (Tuần 1-2):
1. ✅ Model Monitoring & Auto-Retraining
2. ✅ Enhanced Error Handling & Resilience
3. ✅ Rate Limiting
4. ✅ Trade Database

### Phase 2 (Tuần 3-4):
5. ✅ Dynamic Risk Management
6. ✅ Advanced Position Sizing
7. ✅ Adaptive Trailing Stop
8. ✅ Portfolio Management (nếu cần multi-asset)

### Phase 3 (Tuần 5+):
9. Model Versioning & A/B Testing
10. Real-time Alerts
11. Advanced Analytics Dashboard
12. Feature Importance Monitoring

---

## 🔧 Implementation Notes

### Testing Strategy:
- Test từng component riêng biệt
- Integration tests cho toàn bộ flow
- Paper trading trước khi live

### Migration Path:
- Giữ backward compatibility
- Feature flags để bật/tắt tính năng mới
- Gradual rollout

### Documentation:
- Update README với tính năng mới
- Code comments và docstrings
- User guide cho từng tính năng

---

## 📝 Kết Luận

Các đề xuất trên được sắp xếp theo mức độ ưu tiên và tác động. Bắt đầu với Phase 1 để cải thiện stability và monitoring, sau đó mở rộng với Phase 2 và 3.

**Lưu ý:** Mỗi tính năng nên được implement và test kỹ trước khi deploy lên production. Luôn test trên paper trading trước.
