## Tổng quan kiến trúc strategy ML theo regime

Strategy bạn đang dùng là một **pipeline ML hoàn chỉnh**, gồm các khối chính:

- **Data & Feature Engineering**: tạo bộ features nâng cao (indicators, multi‑timeframe, ICT, sequence…)
- **Regime Detection**: dùng HMM để gán mỗi bar vào một trong 4 regime: `trending`, `ranging`, `volatile`, `calm`
- **Label Creation**: tạo nhãn -1/0/1 dựa trên return tương lai và ATR (PHƯƠNG PHÁP LƯỢNG HÓA)
- **Model Training**:
  - **Ensemble chính** (RandomForest + bandit models + stacking)
  - **Regime‑Specific Models**: một model riêng cho từng regime
- **Backtest & Đánh giá**: vectorized backtest, SL/TP, metrics PnL và risk đầy đủ
- **Triển khai Live**: `RegimeEnsembleStrategy` + `LiveTradingBot` + `OKXClient`

Phần dưới mô tả chi tiết từng bước, kèm theo cách entry, risk, SL/TP, ATR, PnL được kiểm soát và hướng phát triển tiếp.

---

## 1. Data & Feature Engineering

### 1.1. Indicators 1 khung thời gian (core features)

Được định nghĩa trong `calculate_indicators_enhanced` và `_calculate_indicators`:

- **Momentum & Oscillators**
  - **RSI** nhiều khung: 9, 14, 21, 50
  - **MACD**: `macd_line`, `macd_signal`, `macd_hist`
- **Bollinger Bands**
  - `bb_upper`, `bb_lower`, `bb_middle`
  - `bb_width` = (upper − lower) / middle → dùng mạnh trong regime/volatility
  - `bb_position` = (close − lower) / (upper − lower)
- **ATR & Volatility**
  - `atr`, `atr_ratio` (chuẩn hóa theo giá), `atr_20`
  - `volatility_5`, `volatility_20`, `volatility_ratio`
- **Moving Averages & Crossovers**
  - SMA: 20/50/100/200
  - EMA: 20/50/200
  - `sma_20_50_cross`, `ema_20_50_cross`
  - `price_sma20_ratio`, `price_sma50_ratio`
- **Volume & VWAP**
  - `volume`, `volume_ma`, `volume_ratio`
  - `vwap`, `vwap_distance`
- **Market Structure & Candle Anatomy**
  - `higher_high`, `lower_low`, `price_position` trong range 20 bars
  - `momentum_5/10/20`, `roc_10/20`
  - `body_size`, `upper_shadow`, `lower_shadow`, `candle_range`

### 1.2. Lagged & Rolling statistics

- **Lagged features**:
  - `ret_lag1/2/3/5/10`
  - `rsi_lag*`, `macd_hist_lag*`
- **Rolling stats trên returns**:
  - Moving average: `ret_ma5/10/20/50`
  - Volatility: `ret_std*`
  - Higher moments: `ret_skew*`, `ret_kurt*`

### 1.3. Regime features & persistence

Sau khi có chuỗi regime, ta encode:

- **One‑hot regime**: `regime_trending`, `regime_ranging`, `regime_volatile`, `regime_calm`
- **Persistence**: `regime_<name>_persist` = số bars liên tiếp đã ở regime đó  
→ Model học được cả **loại thị trường** lẫn **độ “bền” của chế độ**.

### 1.4. ICT features

- Dùng các hàm:
  - `detect_order_blocks`
  - `ob_confluence_signal`
  - `fib_features`
- Sinh thêm các features:
  - `ict_ob_*`, `ict_ob_zone_*`, `ict_fib_*`  
→ Mục tiêu: giúp model biết khi nào giá đang gần vùng OB/Fibo quan trọng để **lọc bớt entry xấu** (sau này dùng thêm ICT filter trong live).

**✅ SEQUENCE FEATURES (LSTM Extractor) - ĐÃ TÍCH HỢP:**
- **Sequence features ĐÃ được tích hợp vào training pipeline** (`build_feature_matrix_optimized` trong `train_regime_ensemble_optimized.py`).
- Khi train với `use_sequence_features=True`, LSTM extractor sẽ:
  - Load model từ `models/seq_lstm_extractor.pt` (nếu có)
  - Extract 3 features: `seq_score`, `seq_vol`, `seq_trend`
  - Thêm vào feature matrix với prefix `seq_*`
- **Khi dùng trong live trading**: Đảm bảo bật `use_sequence_features=True` trong `RegimeEnsembleStrategy` để match với features đã train.
- **Fallback**: Nếu không load được LSTM model, sẽ dùng deterministic features (volatility, trend) hoặc zero features để giữ số chiều nhất quán.

### 1.5. Multi‑timeframe features (4H, 1D)

Hàm `add_multi_timeframe_features`:

- Resample 1h thành 4h và 1d, tính indicators tương tự, rồi align về 1h:
  - `mtf_4h_*` và `mtf_1d_*` cho tất cả indicators được chọn
  - **Trend alignment** 1h–4h–1d
  - **Price position** trong range 4h/1d
  - **Volatility ratio** 4h/1d so với 1h
  - **RSI divergence** giữa 1h và 4h/1d
- Thêm **multi‑timeframe consensus** `mtf_trend_consensus` = trung bình hướng trend across 1h, 4h, 1d.

### 1.6. Sequence features (LSTM extractor – tùy chọn)

- Pipeline `train_full_pipeline`:
  - Train `_TinyLSTM` trên chuỗi log‑returns để học **latent features** về cấu trúc chuỗi giá.
  - Lưu checkpoint `models/seq_lstm_extractor.pt`.
- Khi bật `use_sequence_features=True` trong strategy:
  - `SequenceFeatureExtractor` load model, sinh các features `seq_*` bổ sung.

---

## 2. Regime Detection (HMM + fallback rules)

Định nghĩa trong `algo_trading/market_models/regime.py`.

### 2.1. HMM RegimeDetector

- Input:
  - Indicators chuẩn hóa: RSI, MACD_hist, BB_width, ATR/close, volume ratio.
- Model:
  - `GaussianHMM` với `n_components = 4` tương ứng 4 regime.
  - Chuẩn hóa bằng `StandardScaler` để tránh covariance sick.
  - Tự xử lý NaN/Inf, thêm nhiễu nhỏ cho cột hằng.
- Output:
  - Chuỗi regime `regime` (0–3) cho mỗi bar.
  - Bảng xác suất `prob_trending`, `prob_ranging`, `prob_volatile`, `prob_calm`.
  - `transition_matrix`, `stationary_distribution`.

### 2.2. Smoothing xác suất

Để tránh việc một regime **chiếm 100% xác suất**:

- Trộn với prior đều \((1/n\_regimes)\) bằng hệ số `alpha = 0.6`:
  - `probs_smoothed = alpha * probs + (1-alpha) * uniform`
  - Chuẩn hóa lại mỗi hàng sum = 1  
→ Regime probabilities **mượt hơn**, giúp regime‑specific ensemble hoạt động ổn định.

### 2.3. Fallback simple rules

Khi HMM không khả dụng hoặc dữ liệu quá ít:

- Dùng rule‑based trên:
  - MACD histogram
  - RSI
  - Bollinger width
  - ATR
- Gán trực tiếp 4 regime, ưu tiên: `volatile` > `trending` > `ranging` > `calm`.

---

## 3. Label Creation & PHƯƠNG PHÁP LƯỢNG HÓA

Trong `train_regime_ensemble_optimized.py`:

- **Horizon**: 5 bars → đo return 5 nến tương lai.
- **Volatility scaling**: dùng **ATR** để chuẩn hóa biên độ:
  - `volatility_method='atr'`, `volatility_window=14`
- **Dynamic threshold**:
  - Tham số `k=1.5` (giảm từ 1.75) để có nhiều tín hiệu direction hơn.
  - **Lọc low‑volatility**: `min_volatility_threshold=0.003`, bỏ các đoạn gần như đi ngang.
- Kết quả:
  - **Label gốc**: \(-1, 0, 1\) (SHORT / NEUTRAL / LONG)
  - Nếu class 0 quá ít (< 1%): chỉ giữ ±1 và ánh xạ sang {0,1} cho binary.

Điều này đảm bảo:

- Entry chỉ được tạo khi **biến động đủ lớn** so với ATR.
- Tỷ lệ LONG/SHORT được cân bằng tốt hơn, giảm noise từ nhiễu nhỏ.

---

## 4. Huấn luyện models

### 4.1. Chuẩn hóa, class weights & SMOTE

- **Scaler**: `RobustScaler` trên toàn bộ feature matrix (ít nhạy với outlier).
- **Class weights**:
  - Tính bằng `compute_class_weight('balanced')` trên labels.
  - Tăng trọng số cho class **±1** (LONG/SHORT) để model chú ý hơn vào tín hiệu vào lệnh.
- **SMOTEN / SMOTE**:
  - Khi imbalance mạnh (min_class_ratio < 0.5), dùng **SMOTEN** để oversample các class hiếm.
  - Sau SMOTE/SMOTEN, recalibrate lại class weights.

### 4.2. Feature selection

- Dùng `SelectKBest` với `mutual_info_classif`:
  - Từ ~200+ features → **chọn tốt nhất ~100 features**.
  - Sau khi chọn xong, **refit lại RobustScaler** đúng số chiều mới để tránh mismatch.
  - Lưu lại `feature_names` để align lúc inference (live).

### 4.3. Ensemble chính (Regime Ensemble)

1. **RandomForest chính**:
   - Train với class weights đã boost.
   - Lưu vào `models/regime_ensemble_optimized.pkl` kèm:
     - `model`
     - `scaler`
     - `feature_names`

2. **Bandit models**:
   - Train thêm các model: XGBoost (`xgb`), LightGBM (`lgb`), CatBoost (`cat`), GradientBoosting (`gb`).
   - Lưu từng model `regime_bandit_*_optimized.pkl` (model + scaler + feature_names).

3. **Stacking ensemble**:
   - Base estimators: các `bandit_*`.
   - Meta model: `LogisticRegression(class_weight='balanced')`.
   - Dùng `StackingClassifier` với CV để tránh leak.
   - Lưu `regime_bandit_stacking_optimized.pkl`.

### 4.4. Regime‑Specific Models

- Được implement trong `RegimeSpecificModels`:
  - Mỗi regime (0–3) có:
    - Một **model** riêng (mặc định XGBoost).
    - Một **RobustScaler** riêng.
  - Input cho mỗi regime được filter theo `regime_ids`.
- Đặc điểm:
  - Tự xử lý **encode/decode** label khi dùng XGBoost (0..K‑1).
  - `predict_proba_single_regime` trả về **2 cột:** `p_short`, `p_long`.
  - Hỗ trợ hai kiểu inference:
    - **Weighted ensemble theo regime probabilities**.
    - **Chọn regime hiện tại** (dùng `regime_ids`).
- Lưu dưới dạng `regime_specific_models_optimized.pkl` gồm:
  - `regime_models` (wrapper)
  - `scaler` (toàn cục cho inference cũ, hiện chủ yếu dùng scaler riêng trong wrapper)
  - `feature_names`.

---

## 5. Generate signals & Entry logic

### 5.1. Tính features & regime trong strategy

`RegimeEnsembleStrategy.generate_signals(df)`:

- Tính lại **indicators** giống hệt lúc train.
- Gọi `detect_regime_hmm` → nhận:
  - `current_regime`, `regime` series, `regime_probabilities`.
- Xây `X` bằng `_build_feature_matrix`:
  - Indicators, returns, lags, rolling, regime features, ICT, sequence (nếu bật).
  - Align index, fill/ffill/bfill, đảm bảo numeric.

### 5.2. Bộ lọc allowed_regimes & thresholds

- Nếu `current_regime` **không** thuộc `allowed_regimes`:
  - Toàn bộ `signals = 0` (không trade).
- Khi dùng **regime‑specific models**:
  - Lấy `regime_ids` và/hoặc `regime_probabilities`.
  - Tính `p_short`, `p_long` từ `RegimeSpecificModels.predict_proba`.
  - Áp dụng **threshold theo từng regime**:

    ```python
    regime_thresholds = {
        0: 0.52,  # trending
        1: 0.55,  # ranging
        2: 0.60,  # volatile
        3: 0.55,  # calm
    }
    ```

  - Quy tắc entry:
    - Nếu `p_long >= th` và `p_long > p_short` → signal = **1 (LONG)**.
    - Nếu `p_short >= th` và `p_short > p_long` → signal = **‑1 (SHORT)**.
    - Ngược lại → **0 (no trade)**.

- Khi dùng **ensemble chung**:
  - Dùng `model.predict_proba(X)`:
    - Map ra `p_short`, `p_long`, `p_neutral`.
  - Entry khi directional proba **vượt qua `proba_threshold` và thắng neutral**:
    - `p_long >= thr` & `p_long > p_short` & `p_long >= p_neutral`.

### 5.3. ICT filter (tuỳ chọn)

- Nếu `use_ict_filter=True`:
  - Tính lại ICT OB + Fibo.
  - Chỉ giữ:
    - LONG khi ở **OB bullish + gần vùng Fibo**.
    - SHORT khi ở **OB bearish + gần vùng Fibo**.
→ Biến ML signal thành **confluence** với vùng giá key ICT, giảm entry xấu.

---

## 6. Risk, SL/TP, ATR, PnL trong backtest & live

### 6.1. Backtest: `StrategyEvaluator`

- Sử dụng **vectorized backtest**:
  - Config: `BacktestConfig(initial_capital, commission, allow_short, freq)`.
  - Nếu bật stops:
    - `RiskConfig(sl_pct, tp_pct)` và hàm `barwise_with_stops`.
- Metrics chính:
  - **Hiệu suất**: Total Return, CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Volatility.
  - **Trade‑level**:
    - `win_rate`, `total_trades`
    - `total_pnl`, `avg_win`, `avg_loss`
    - **RR ratio** = `avg_win / avg_loss`
    - **Profit Factor** = tổng lãi / tổng lỗ
    - Số lệnh **long/short**, thời gian giữ lệnh trung bình.
- Các báo cáo:
  - `strategy_comparison_report.txt`: so sánh tất cả indicators/strategy.
  - `strategy_detailed_report.txt`: phân tích sâu từng strategy (PnL, RR, time, loại chiến lược…).

### 6.2. Live trading: `LiveTradingBot`

**Position sizing (`_calculate_position_size`)**

- Lấy **quote balance**:
  - `MODE=paper`: giả lập 1000 USDT.
  - `MODE=live/demo`: hỏi qua `exchange client`.
- `risk_amount = balance * RISK_PER_TRADE`.
- Nếu có SL:
  - Nếu dùng **% SL**: `sl_distance = price * SL_PCT`.
  - Nếu dùng **ATR‑based SL**: `sl_distance = SL_ATR_K * ATR14`.
  - **Qty = risk_amount / sl_distance**.
- Nếu không có SL: fallback qty = `balance * RISK_PER_TRADE / price`.
- Giới hạn bởi `MAX_POSITION_SIZE` nếu được set.

**SL/TP logic (`_update_stop_loss_take_profit`, `_check_stop_loss_take_profit`)**

- LONG:
  - `SL = entry_price * (1 - SL_PCT)` hoặc `entry - SL_ATR_K * ATR`.
  - `TP = entry_price * (1 + TP_PCT)` hoặc `entry + TP_ATR_K * ATR`.
- SHORT:
  - `SL = entry_price * (1 + SL_PCT)` hoặc `entry + SL_ATR_K * ATR`.
  - `TP = entry_price * (1 - TP_PCT)` hoặc `entry - TP_ATR_K * ATR`.
- Mỗi vòng `run_once`, nếu đang **holding**:
  - Check `current_price` vs SL/TP.
  - Nếu hit → gọi `_exit_position(reason="Stop Loss" / "Take Profit")`.

**Entry/Exit dựa trên signals**

- Mỗi vòng:
  - Lấy `latest_signal` từ strategy (‑1/0/1).
  - Nếu đang **không có position**:
    - `signal > 0` → `_enter_position(1, df)` → BUY.
    - `signal < 0` → `_enter_position(-1, df)` → SELL (SHORT).
  - Nếu **đang holding**:
    - Nếu tín hiệu **đảo chiều ngược hướng** → `_exit_position(reason="signal_change")`.
  - Cooldown bằng `COOL_DOWN_SEC` để tránh spam lệnh.

**PnL & log**

- Bot log:
  - Entry: `🚀 Đã vào BUY/SELL ...`, kèm SL/TP.
  - Exit: `🚪 Đã thoát position (...)`.
- PnL chi tiết hiện đang tập trung trong backtest; live PnL bạn xem trên:
  - Equity & order history của OKX.
  - Hoặc có thể mở rộng sau để lưu trade log riêng.

---

## 7. Hướng phát triển tiếp

### 7.1. Nâng cấp training cho Futures/OKX

- **Data source**:
  - Train trực tiếp trên dữ liệu futures (OKX/Binance futures) thay vì spot.
  - Bổ sung các features:
    - Funding rate, open interest, basis, long/short ratio.
- **Cost model**:
  - Thêm funding fee, maker/taker fee thực tế vào backtest.

### 7.2. Regime detection & thresholds động

- Học **mapping riêng** giữa:
  - Symbol, timeframe, và **regime_thresholds** tối ưu.
- Có thể dùng:
  - Grid search / Bayesian optimization trên `proba_threshold` cho từng regime.
  - Hoặc meta‑model dự đoán **khoảng threshold** dựa trên volatility regime hiện tại.

### 7.3. Dynamic risk management

- Cho phép:
  - `RISK_PER_TRADE` thay đổi theo regime:
    - trending → risk cao hơn.
    - volatile → risk giảm mạnh.
  - SL/TP ATR‑based theo regime:
    - `SL_ATR_K`, `TP_ATR_K` khác nhau cho trending vs ranging.
- Xây thêm module **position sizing theo Kelly fraction giới hạn** hoặc **volatility targeting**.

### 7.4. Portfolio & multi‑asset

- Mở rộng từ 1 symbol (BTCUSDT) sang:
  - Nhiều cặp cùng lúc: ETH, SOL, altcoins.
  - Chia vốn theo **risk budget**.
- Cần:
  - Layer quản lý portfolio.
  - Correlation & diversification (tránh full risk trên các coin highly correlated).

### 7.5. Online learning & model monitoring

- Thiết kế pipeline:
  - Định kỳ (ví dụ mỗi tuần) backtest lại → đánh giá **drift**.
  - Nếu hiệu suất giảm dưới ngưỡng → trigger **retrain**:
    - Cập nhật thêm dữ liệu mới vào training.
  - Lưu version model + metadata (ngày train, tham số, metrics) để trace.

---

## 8. CÁCH SỬ DỤNG TRONG LIVE TRADING

### 8.1. Cấu hình .env cho Regime-Specific Strategy với Sequence Features

Khi dùng `regime_specific` strategy trong live trading, cần đảm bảo:

1. **Train models với sequence features BẬT**:
   ```python
   python train_regime_ensemble_optimized.py
   # Hoặc gọi train_optimized_models với use_sequence_features=True
   ```

2. **Cấu hình .env cho live bot**:
   ```env
   STRATEGY=regime_specific
   STRATEGY_PARAMS={"use_regime_specific": true, "use_sequence_features": true, "regime_specific_model_path": "models/regime_specific_models_optimized.pkl", "sequence_model_path": "models/seq_lstm_extractor.pt", "sequence_len": 64}
   ```

3. **Đảm bảo có các file models**:
   - `models/regime_specific_models_optimized.pkl` (regime-specific models)
   - `models/seq_lstm_extractor.pt` (LSTM sequence extractor)

4. **Lưu ý**: Nếu train với `use_sequence_features=True`, **PHẢI** bật `use_sequence_features=True` trong live trading để match số features. Ngược lại sẽ lỗi vì mismatch features.

---

## 9. Cách đọc nhanh để "nắm xương sống"

- **Huấn luyện**:
  - `train_regime_ensemble_optimized.py`:
    - Data → indicators/MTF/ICT → regimes → labels → feature selection → ensemble + regime‑specific.
- **Regime models**:
  - `algo_trading/market_models/regime.py`
  - `algo_trading/ml/regime_specific_models.py`
- **Chiến lược live**:
  - `algo_trading/strategies/ml/regime_ensemble_strategy.py`
- **Backtest & metrics**:
  - `algo_trading/live/strategy_evaluator.py`
- **Bot live + risk/SL/TP + OKX**:
  - `algo_trading/live/universal_bot.py`
  - `algo_trading/live/okx_client.py`

Nắm được 5 file này là bạn hiểu gần như toàn bộ **phương pháp huấn luyện, cách entry, cách quản lý risk và cách triển khai lên OKX** của project. 

