# Regime-Aware Transformer Strategy - Hướng Dẫn Sử Dụng

## Tổng Quan

Đây là implementation của **Phương án 1: Regime-Aware Conditional Distribution Learning với Transformer Architecture**.

Strategy này kết hợp 3 tầng:

1. **Tầng 1 - Market Description**: HMM để detect market regime
2. **Tầng 2 - Inference/Learning**: Transformer để học conditional return distribution
3. **Tầng 3 - Decision**: Expected Value calculation để quyết định trading

## Kiến Trúc

### 1. Regime Detection (HMM)

**File**: `algo_trading/market_models/regime.py`

**Chức năng**:
- Sử dụng Hidden Markov Model (HMM) để phát hiện các regime ẩn của thị trường
- 4 regimes: `trending`, `ranging`, `volatile`, `calm`
- Input: Technical indicators (RSI, MACD, Bollinger Bands width, ATR, Volume)
- Output: Current regime, regime probabilities, transition matrix

**Cách hoạt động**:
```python
from algo_trading.market_models.regime import detect_regime_hmm

# Detect regime từ DataFrame với indicators
regime_info = detect_regime_hmm(
    df,
    indicators={
        'rsi': rsi_series,
        'macd_hist': macd_hist_series,
        'bb_width': bb_width_series,
        'atr': atr_series
    }
)

# regime_info chứa:
# - current_regime: 'trending', 'ranging', 'volatile', 'calm'
# - regime_probabilities: DataFrame với probabilities
# - transition_matrix: Transition probabilities giữa các regimes
```

### 2. Transformer Distribution Model

**File**: `algo_trading/ml/models/transformer_distribution.py`

**Chức năng**:
- Transformer encoder để học temporal dependencies giữa indicators và returns
- Regime embedding để inject regime information vào Transformer
- Output: Full conditional distribution (quantiles, moments, win probability)

**Architecture**:
```
Input: [batch_size, seq_len, n_features]
  ↓
Feature Embedding: Linear projection
  ↓
Regime Embedding: Inject regime ID
  ↓
Positional Encoding: Add positional information
  ↓
Transformer Encoder: Learn temporal dependencies
  ↓
Distribution Head: Output distribution parameters
  ↓
Output: {
    'quantiles': [q10, q25, q50, q75, q90],
    'mean': expected return,
    'std': standard deviation,
    'skew': skewness,
    'kurt': kurtosis,
    'win_prob': P(return > 0)
}
```

**Cách sử dụng**:
```python
from algo_trading.ml.models.transformer_distribution import TransformerDistributionWrapper

# Tạo model
model = TransformerDistributionWrapper(
    input_dim=50,  # Số lượng features
    n_regimes=4,
    d_model=128,
    nhead=8,
    num_layers=3
)

# Predict
features = np.array([...])  # [seq_len, n_features]
regime_id = 0  # Current regime ID
prediction = model.predict(features, regime_id)

# prediction chứa:
# - quantiles: [q10, q25, q50, q75, q90]
# - mean: expected return
# - std: standard deviation
# - win_prob: P(return > 0)
```

### 3. Feature Engineering

**File**: `algo_trading/ml/features.py`

**Chức năng**:
- Kết hợp indicators và market model outputs thành features
- Tạo lagged features và rolling statistics
- Scale features và tạo sequences cho Transformer

**Features được tạo**:
- Basic price features: returns, log returns
- Indicator features: RSI, MACD, Bollinger Bands, ATR, VWAP, etc.
- Market model features: regime probabilities, volatility forecasts
- Lagged features: giá trị trước đó của features
- Rolling statistics: moving averages, standard deviations

**Cách sử dụng**:
```python
from algo_trading.ml.features import FeatureEngineer, create_features

# Tạo features
engineer = FeatureEngineer(sequence_length=20)
features_df = engineer.create_features(df, indicators, market_models)

# Scale và tạo sequences
features_array = engineer.transform_features(features_df, fit_scaler=True)
features_sequences = engineer.create_sequences(features_array)
```

### 4. Training Pipeline

**File**: `algo_trading/ml/training.py`

**Chức năng**:
- Train Transformer model với walk-forward validation
- Optimize Expected Value thay vì accuracy
- Loss function: Quantile Loss + Distribution Consistency + Win Probability Loss

**Training Process**:
1. Prepare features và targets (future returns)
2. Create sequences cho Transformer
3. Split data (train/validation)
4. Train với Expected Value loss
5. Early stopping dựa trên validation loss

**Cách sử dụng**:
```python
from algo_trading.ml.training import train_transformer_model, walk_forward_validation

# Train model
model = train_transformer_model(
    features=features_sequences,
    returns=future_returns,
    regime_ids=regime_ids,
    model_config={
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3
    },
    training_config={
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-4
    }
)

# Save model
model.save('trained_model.pth')
```

### 5. Strategy Wrapper

**File**: `algo_trading/strategies/ml/regime_transformer_strategy.py`

**Chức năng**:
- Tích hợp tất cả components vào BaseStrategy interface
- Generate trading signals dựa trên Expected Value và regime

**Workflow**:
1. Tính indicators (RSI, MACD, BB, ATR, VWAP, etc.)
2. Detect regime (HMM)
3. Tạo features và predict conditional distribution (Transformer)
4. Tính Expected Value từ distribution
5. Generate signals nếu EV > threshold và regime được phép

**Cách sử dụng**:
```python
from algo_trading.strategies.ml.regime_transformer_strategy import RegimeTransformerStrategy

# Tạo strategy
strategy = RegimeTransformerStrategy(
    model_path='trained_model.pth',
    ev_threshold=0.001,  # Minimum EV để trade
    position_sizing='fixed',  # hoặc 'kelly'
    risk_per_trade=0.02,  # 2% risk per trade
    allowed_regimes=['trending', 'ranging'],
    sequence_length=20
)

# Generate signals
result = strategy.generate_signals(df)

# Signals
signals = result.signals  # pd.Series với -1, 0, +1

# Meta information
meta = result.meta
# - regime: current regime
# - ev_net: Expected Value
# - win_probability: Win probability
# - predicted_distribution: Full distribution
```

## Ví Dụ Hoàn Chỉnh

### Bước 1: Chuẩn bị Data

```python
import pandas as pd
from algo_trading.data_loader import load_data

# Load data
df = load_data('BTCUSDT', '1h')
```

### Bước 2: Tính Indicators và Detect Regime

```python
from algo_trading.indicators import rsi, macd, bollinger_bands, atr
from algo_trading.market_models.regime import detect_regime_hmm

# Tính indicators
indicators = {
    'rsi': rsi(df['close'], 14),
    'macd_hist': macd(df['close'])[2],
    'bb_width': (bollinger_bands(df['close'])[0] - bollinger_bands(df['close'])[2]) / df['close'],
    'atr': atr(df, 14) / df['close']
}

# Detect regime
regime_info = detect_regime_hmm(df, indicators=indicators)
print(f"Current regime: {regime_info['current_regime']}")
```

### Bước 3: Train Model

```python
from algo_trading.ml.features import FeatureEngineer
from algo_trading.ml.training import train_transformer_model

# Tạo features
engineer = FeatureEngineer(sequence_length=20)
features_df = engineer.create_features(df, indicators, {'regime': regime_info})
features_array = engineer.transform_features(features_df, fit_scaler=True)

# Prepare targets (future returns)
returns = df['close'].pct_change().shift(-1).fillna(0).values

# Create sequences
features_sequences = engineer.create_sequences(features_array)
returns_sequences = returns[len(returns) - len(features_sequences):]
regime_ids_sequences = regime_info['regime'].values[len(regime_info['regime']) - len(features_sequences):]

# Train
model = train_transformer_model(
    features_sequences,
    returns_sequences,
    regime_ids_sequences,
    model_config={'d_model': 128, 'nhead': 8, 'num_layers': 3},
    training_config={'batch_size': 32, 'epochs': 50, 'learning_rate': 1e-4}
)

# Save
model.save('regime_transformer_model.pth')
```

### Bước 4: Sử Dụng Strategy

```python
from algo_trading.strategies.ml.regime_transformer_strategy import RegimeTransformerStrategy

# Tạo strategy
strategy = RegimeTransformerStrategy(
    model_path='regime_transformer_model.pth',
    ev_threshold=0.001,
    allowed_regimes=['trending', 'ranging']
)

# Generate signals
result = strategy.generate_signals(df)

# Backtest
from algo_trading.core.backtest_vectorized import vectorized_pnl
equity, returns = vectorized_pnl(df, result.signals, backtest_config)
```

## Giải Thích Code Chi Tiết

### 1. Regime Detection với HMM

**Tại sao dùng HMM?**
- Thị trường có các regime ẩn (trending, ranging, volatile, calm) mà chúng ta không quan sát trực tiếp
- HMM cho phép suy luận regime từ observations (indicators)
- Transition matrix cho biết xác suất chuyển đổi giữa các regimes

**Code giải thích**:
```python
# HMM model với Gaussian emissions
self.model = hmm.GaussianHMM(
    n_components=n_regimes,  # 4 regimes
    covariance_type='full',  # Full covariance matrix
    n_iter=n_iter
)

# Train trên observations (indicators)
self.model.fit(observations)

# Predict regime
regimes = self.model.predict(observations)

# Predict probabilities
probs = self.model.predict_proba(observations)
```

### 2. Transformer Architecture

**Tại sao dùng Transformer?**
- Transformer học được temporal dependencies giữa indicators và returns
- Attention mechanism cho phép model tập trung vào các timesteps quan trọng
- Regime embedding inject regime information vào model

**Code giải thích**:
```python
# Feature embedding: project input to d_model
x = self.feature_embedding(features)  # [batch, seq_len, d_model]

# Regime embedding: inject regime information
regime_emb = self.regime_embedding(regime_ids)  # [batch, d_model]
regime_emb = regime_emb.unsqueeze(1).expand(-1, seq_len, -1)
x = x + regime_emb  # Add regime information

# Positional encoding: add positional information
x = self.pos_encoder(x)

# Transformer encoder: learn temporal dependencies
x = self.transformer_encoder(x)  # [batch, seq_len, d_model]

# Use last timestep for prediction
x_last = x[:, -1, :]  # [batch, d_model]

# Distribution head: output distribution parameters
dist_params = self.distribution_head(x_last)  # [batch, 9]
```

### 3. Expected Value Calculation

**Tại sao dùng Expected Value?**
- Trading không phải bài toán "đúng - sai"
- Quan trọng là Expected Value, không phải accuracy
- EV = P(win) * E[win] - P(loss) * E[loss] - commission

**Code giải thích**:
```python
# Expected Value cho long
ev_long = (
    win_prob * take_profit_pct -      # Expected win
    (1 - win_prob) * stop_loss_pct -  # Expected loss
    commission                         # Transaction cost
)

# Expected Value cho short
ev_short = (
    win_prob * stop_loss_pct -         # Short profit khi giá giảm
    (1 - win_prob) * take_profit_pct - # Short loss khi giá tăng
    commission
)

# Trade nếu EV > threshold
if ev_net > ev_threshold:
    signal = +1 if ev_long > ev_short else -1
```

## Dependencies

Cần cài đặt các thư viện sau:

```bash
pip install torch
pip install hmmlearn
pip install scikit-learn
```

## Lưu Ý

1. **Model phải được train trước**: Strategy cần trained model để hoạt động
2. **Sequence length**: Cần đủ data để tạo sequences (mặc định 20 periods)
3. **Regime detection**: Cần đủ data để train HMM (mặc định 500 periods)
4. **Expected Value threshold**: Điều chỉnh `ev_threshold` để filter signals
5. **Allowed regimes**: Chỉ trade trong các regimes được phép

## Metrics Đánh Giá

Khi đánh giá model, nên xem:
- **Calibration**: PIT (Probability Integral Transform) để kiểm tra distribution calibration
- **Sharpness**: Width của prediction intervals
- **Expected Value**: EV từ conditional distribution
- **Risk-adjusted returns**: Sharpe, Sortino, Calmar
- **Win rate**: Tỷ lệ thắng
- **Risk-Reward ratio**: Tỷ lệ risk/reward

