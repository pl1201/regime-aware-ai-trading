# Hệ thống Algo Trading (Không cần API key)

Dự án cung cấp một hệ sinh thái hoàn chỉnh cho nghiên cứu và triển khai thuật toán giao dịch:
- Bộ chiến lược (13+ chiến lược)
- Mô hình toán học (công thức và giải thích直 quan)
- Loader dữ liệu (CSV/Parquet/Yahoo Finance/Binance Public Kline — không cần API key)
- Bộ chỉ báo kỹ thuật tự cài đặt (không phụ thuộc TA-Lib)
- Backtest (vectorized + event-driven) với SL/TP/Trailing, phí giao dịch, slippage, sizing
- Tối ưu hóa (Grid Search, Genetic Algorithm, Bayesian Optimization, Walk-Forward)
- Trực quan hóa (candlestick, overlay indicator, equity/drawdown, volatility, correlation heatmap, alpha-beta)


## 1) Cài đặt

Yêu cầu Python >= 3.9. Khuyến nghị tạo môi trường ảo.

Cài đặt gói bắt buộc và tùy chọn:

- Bắt buộc: numpy, pandas, matplotlib, requests
- Thường dùng: yfinance, seaborn (heatmap), plotly (candlestick tương tác)
- Tùy chọn cho chiến lược/nâng cao:
  - statsmodels (ARIMA/SARIMA, cointegration tools)
  - arch (GARCH)
  - torch (LSTM demo)
  - scikit-optimize (Bayesian Optimization)

Lệnh nhanh (có thể bỏ bớt phần không cần):

- pip install numpy pandas matplotlib requests yfinance seaborn plotly statsmodels arch torch scikit-optimize

Gợi ý Windows/PowerShell: chạy PowerShell với quyền user (không cần WSL).


## 2) Cấu trúc thư mục

```
algo_trading/
│── data/
│── data_loader/
│   └── loader.py
│── indicators/
│   └── core.py
│── strategies/
│   ├── base.py
│   ├── common_strategies.py
│   └── __init__.py
│── backtest/
│   ├── vectorized.py
│   ├── event_driven.py
│   └── __init__.py
│── optimization/
│   ├── grid_search.py
│   ├── genetic.py
│   ├── bayesian.py
│   └── walk_forward.py
│── visualization/
│   └── plots.py
│── utils/
│   ├── metrics.py
│   └── __init__.py
│── main.py
│── README.md
```


## 3) Loader dữ liệu (không cần API key)

Module: algo_trading.data_loader.loader

Hỗ trợ:
- CSV: load_csv(path)
- Parquet: load_parquet(path)
- Yahoo Finance (yfinance): load_yfinance(ticker, interval, start, end)
- Binance Public Kline API (không cần key): load_binance(symbol, interval, start, end, market='spot')

Các bước xử lý:
- Chuẩn hóa cột OHLCV, DatetimeIndex
- Resample theo timeframe (nếu cần)
- Clean missing
- Normalize (zscore|minmax) — tùy chọn
- Tự động tạo features kỹ thuật cơ bản (SMA/EMA/WMA/RSI/MACD/Bollinger/ATR/VWAP/Z-score)
- Train/test split tiện ích (train_test_split_df)

Ví dụ nhanh:

```
from algo_trading.data_loader.loader import load_data
# CSV
df = load_data('csv', path='./data/BTCUSDT.csv')
# Parquet
df = load_data('parquet', path='./data/ETHUSDT.parquet')
# Yahoo
df = load_data('yfinance', ticker='BTC-USD', interval='1h')
# Binance (public, không key)
df = load_data('binance', symbol='BTCUSDT', interval='1h', start='2023-01-01', end='2023-06-01')
```


## 4) Indicators tự triển khai (không dùng TA-Lib)

Module: algo_trading.indicators.core

- SMA / EMA / WMA
- RSI
- MACD
- Bollinger Bands (MA ± kσ)
- ATR, True Range
- VWAP
- Z-score
- add_basic_indicators(df): thêm nhanh bộ chỉ báo cơ bản vào DataFrame


## 5) Các chiến lược (13+)

Module: algo_trading.strategies.common_strategies

Mỗi chiến lược có:
- Mô tả nguyên lý + quy tắc vào/thoát cơ bản
- Tham số có thể tối ưu
- Quản trị rủi ro (sử dụng RiskConfig của backtest) — đặt ngoài chiến lược
- Ưu/nhược điểm (tổng quan)

Danh sách chính:
- SMA/EMA Crossover (SMAEMACrossStrategy)
- RSI + Divergence (RSIDivergenceStrategy)
- MACD Momentum (MACDMomentumStrategy)
- Bollinger Bands Breakout (BollingerBreakoutStrategy)
- VWAP Mean Reversion (VWAPMeanReversionStrategy)
- Renko Trend Following — xấp xỉ bằng ATR brick (RenkoTrendStrategy)
- Volume Profile Imbalance — xấp xỉ bằng histogram (VolumeProfileImbalanceStrategy)
- Ornstein–Uhlenbeck Mean Reversion (OUProcessMeanReversionStrategy)
- Kalman Filter Forecast (KalmanFilterForecastStrategy)
- ARIMA/SARIMA (ARIMAStrategy)
- LSTM/Transformer (LSTMTransformerStrategy)
- Statistical Arbitrage (cointegration) (StatArbCointegrationStrategy)
- GARCH Volatility (GARCHVolatilityStrategy)

Cách dùng chung:
```
from algo_trading.strategies import SMAEMACrossStrategy
strat = SMAEMACrossStrategy(fast=20, slow=50, ma_type='ema')
signals = strat.generate_signals(df).signals  # Series {-1,0,1}
```


## 6) Mô hình toán học (công thức + giải thích)

- MA (SMA): SMA_t = (1/n) Σ_{i=0..n-1} price_{t-i}. Là trung bình động đơn giản, làm mượt nhiễu.
- EMA: EMA_t = α·price_t + (1-α)·EMA_{t-1}, α = 2/(n+1). Nhạy hơn SMA, phản ứng nhanh hơn.
- WMA: WMA_t = (Σ w_i·price_{t-i}) / (Σ w_i), w_i tăng dần cho dữ liệu gần.
- RSI (period n):
  - Δ_t = price_t - price_{t-1}
  - RS = EMA(gain, n) / EMA(loss, n)
  - RSI = 100 - 100/(1+RS). Đo quá mua (>70) / quá bán (<30).
- MACD: MACD = EMA_fast - EMA_slow; Signal = EMA(MACD, s); Histogram = MACD - Signal. Đo động lượng.
- Bollinger Bands: Middle = MA; Upper = MA + k·σ; Lower = MA - k·σ. Đo băng độ biến động.
- True Range: TR_t = max(High-Low, |High-Close_{t-1}|, |Low-Close_{t-1}|). ATR = SMA(TR, n). Đo độ biến động nội tại.
- VWAP: VWAP_t = Σ(P_t·V_t)/ΣV_t (tích lũy hoặc rolling). Mức giá trung bình theo khối lượng.
- OU Process (mean reversion): dx_t = θ(μ - x_t)dt + σ dW_t. Ước lượng gần bằng AR(1): x_t = a + b·x_{t-1} + ε.
- Kalman Filter (local linear trend): trạng thái [level, trend], F=[[1,1],[0,1]], H=[[1,0]], cập nhật dự báo và hiệu chỉnh theo quan sát.
- ARIMA(p,d,q): mô hình chuỗi thời gian (khác biệt d lần), kết hợp tự hồi quy và nhiễu trung bình trượt; SARIMA thêm yếu tố mùa vụ.
- Cointegration test: Với hai chuỗi X, Y: hồi quy Y = α + βX + ε; kiểm định stationarity của ε (ADF); nếu dừng -> đồng liên kết.
- GARCH(1,1): σ_t^2 = ω + α·ε_{t-1}^2 + β·σ_{t-1}^2. Ước lượng volatility có tính tự hồi quy.
- LSTM cell (đơn giản):
  - i = σ(W_i[x,h_{t-1}] + b_i), f = σ(W_f[...] + b_f), o = σ(W_o[...] + b_o)
  - g = tanh(W_g[...] + b_g)
  - c_t = f ⊙ c_{t-1} + i ⊙ g; h_t = o ⊙ tanh(c_t)

Trực quan: MA/EMA làm mượt; RSI đo sức mạnh; MACD động lượng; Bollinger đo biên; OU/cointegration cho mean-reversion; GARCH cho biến động; Kalman/ARIMA/LSTM cho dự báo.


## 7) Backtest

Module: algo_trading.backtest

- vectorized.py
  - vectorized_pnl: nhanh, dùng vị thế lag; phí dựa trên turnover
  - barwise_with_stops: quét bar-by-bar, hỗ trợ SL/TP/Trailing
  - run_backtest(df, signals, cfg, risk)
- event_driven.py
  - Portfolio/Broker đơn giản, khớp lệnh ở open tiếp theo (slippage + commission)
  - risk_check_intrabar để SL/TP/Trailing trong bar
  - run_event_backtest(df, signals, cfg, risk)

Cấu hình:
- BacktestConfig (vectorized), EventConfig (event)
- RiskConfig: sl_pct/tp_pct/trailing_pct hoặc sl_atr_k/tp_atr_k/trailing_atr_k (atr_col mặc định 'ATR14')

Ví dụ vectorized:
```
from algo_trading.backtest.vectorized import run_backtest, BacktestConfig, RiskConfig
cfg = BacktestConfig(freq='1H', commission=0.0005, slippage_bps=1.0, use_next_open=True)
risk = RiskConfig(sl_atr_k=1.5, tp_atr_k=3.0, trailing_atr_k=1.0)
res = run_backtest(df, signals, cfg=cfg, risk=risk)
print(res['summary'])
```

Ví dụ event-driven:
```
from algo_trading.backtest.event_driven import run_event_backtest, EventConfig
from algo_trading.backtest.vectorized import RiskConfig
cfg = EventConfig(initial_cash=10000, freq='1H', use_next_open=True)
risk = RiskConfig(sl_pct=0.01, tp_pct=0.02)
res = run_event_backtest(df, signals, cfg=cfg, risk=risk)
print(res['summary'])
```


## 8) Tối ưu hóa

Module: algo_trading.optimization

- grid_search.grid_search(df, StrategyCls, param_grid, mode, backtest_kwargs, risk, metric)
- genetic.genetic_search(df, StrategyCls, param_space, ...)
- bayesian.bayesian_optimize(df, StrategyCls, param_space, ...) — cần scikit-optimize (fallback random search nếu không có)
- walk_forward.walk_forward_optimize(df, StrategyCls, method, ...)

Ví dụ (grid search SMA/EMA):
```
from algo_trading.optimization.grid_search import grid_search
from algo_trading.strategies import SMAEMACrossStrategy
pg = {'fast':[10,20,30], 'slow':[50,100], 'ma_type':['ema','sma']}
res = grid_search(df, SMAEMACrossStrategy, pg, mode='vectorized', backtest_kwargs={'cfg_kwargs':{'freq':'1H'}})
print(res['best_params'], res['best_score'])
print(res['results'].head())
```

Walk-forward:
```
from algo_trading.optimization.walk_forward import walk_forward_optimize
wf = walk_forward_optimize(df, SMAEMACrossStrategy, method='grid', param_grid=pg, train_size=2000, test_size=500, step=500)
print(wf['splits'].head())
```


## 9) Trực quan hóa

Module: algo_trading.visualization.plots

- plot_candlestick(df, overlays, signals, use_plotly=False)
- plot_equity_curve(equity)
- plot_drawdown(equity)
- plot_volatility(returns, window, annualize_factor)
- plot_correlation_heatmap(data)
- alpha_beta_scatter(returns, benchmark)
- quick_dashboard(df, overlays, equity, signals)


## 10) Chạy demo end-to-end (main.py)

Ví dụ dùng Yahoo Finance (không key) + SMA/EMA + backtest vectorized + vẽ biểu đồ:

PowerShell (Windows):
```
python -m algo_trading.main --source yfinance --ticker BTC-USD --interval 1h \
  --strategy sma_ema --params '{"fast":20,"slow":50,"ma_type":"ema"}' \
  --allow_short --use_next_open --plot
```

CSV local:
```
python -m algo_trading.main --source csv --path ./data/BTCUSDT.csv \
  --strategy macd --params '{"fast":12,"slow":26,"signal":9}' --plot
```

Binance public (không key):
```
python -m algo_trading.main --source binance --symbol BTCUSDT --interval 1h \
  --strategy bb_breakout --params '{"window":20,"k":2}' --plot
```

Event-driven + ATR stops:
```
python -m algo_trading.main --source yfinance --ticker BTC-USD --interval 1h \
  --strategy vwap_mr --params '{"thr":1.5}' --mode event --use_next_open \
  --sl_atr_k 1.5 --tp_atr_k 3.0 --trailing_atr_k 1.0 --plot
```

Lưu hình:
```
python -m algo_trading.main --source yfinance --ticker BTC-USD --interval 1h \
  --strategy sma_ema --params '{"fast":20,"slow":50,"ma_type":"ema"}' --save_dir ./outputs
```


## 11) Lưu ý quan trọng

- Không sử dụng API key: Yahoo Finance/yfinance và Binance Kline public đều không yêu cầu key.
- Chất lượng dữ liệu ảnh hưởng lớn tới kết quả backtest. Hãy xem xét timezone, resample, missing data.
- Ví dụ OU/ARIMA/Kalman/LSTM là mô phỏng đơn giản cho mục đích minh họa; cần tinh chỉnh/thẩm định khi dùng thực tế.
- Phí giao dịch và slippage được mô hình hóa cơ bản (bps). Với thị trường crypto 24/7 có thể điều chỉnh freq/annualization.
- Quản trị rủi ro: dùng RiskConfig thiết lập SL/TP/Trailing theo % hoặc ATR. Kích thước vị thế có thể theo fixed_size hoặc risk_per_trade trong vectorized.


## 12) Giấy phép & Miễn trừ trách nhiệm

- Mã nguồn phục vụ mục đích học tập/nghiên cứu. Không phải khuyến nghị đầu tư.
- Giao dịch có rủi ro cao. Luôn kiểm thử kỹ và chịu trách nhiệm cho quyết định của chính bạn.






