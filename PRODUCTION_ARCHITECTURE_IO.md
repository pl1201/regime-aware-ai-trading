# Production Architecture and File IO Mapping

Pham vi: thu muc production tai d:/Bot_Trading/production (56 file Python).

## 1) Kien truc tong the

- Entry/Process layer:
  - production/start_trading_bot.py: process manager start/stop/status, quan ly PID, redirect log.
  - production/bot.py: wrapper entrypoint toi strategy bot.
  - production/live_trading_moe_v2.py: live trader dung MOE v2 Enhanced + OKX (ccxt).
- Core trading engine:
  - production/algo_trading/live/: exchange adapter, bot loop, telegram, strategy evaluator.
- ML signal layer:
  - production/algo_trading/ml/: expert models, gating, threshold, calibration, monitoring.
- Risk layer:
  - production/algo_trading/risk/dynamic_risk_manager.py.
- Feature/Indicator layer:
  - production/algo_trading/indicators/ + production/algo_trading/features/.
- Utility/config:
  - production/utils/: setup env, update risk params, check model features.
  - production/config/requirements.txt.
  - production/models/: pkl/pt/csv artifacts.

## 2) Execution flow (input -> output)

1. CLI call: python start_trading_bot.py start|stop|status|dry-run.
2. start_trading_bot.py spawn subprocess chay module universal_bot, ghi trading_bot.pid + trading_bot.log.
3. universal_bot.py nap .env (exchange, symbol, interval, strategy, risk params, API keys).
4. Exchange client fetch OHLCV + account/balance.
5. Feature engineer tao feature tu OHLCV + indicator + multi-timeframe.
6. MOE model du doan probability/signal.
7. Signal quality filter loai tin hieu yeu.
8. Dynamic risk manager tinh position size + SL/TP/trailing.
9. Exchange client dat lenh market/limit, quan ly position.
10. Logging + Telegram notify + monitor model performance.

## 3) File-by-file: chuc nang, input, output

Ghi chu:
- Input = tham so ham/chuong trinh, bien moi truong, du lieu file/network.
- Output = return value, file/log, side effects API.
- Muc do tin cay: High (doc ro flow), Medium (suy ra tu ten + usage).

### A. Root files trong production

1. production/start_trading_bot.py
- Chuc nang: Process manager cho bot (start/stop/status).
- Input: argv action (start|stop|status), file trading_bot.pid, moi truong he thong.
- Output: tao/del PID, spawn/kill subprocess, ghi trading_bot.log, exit code 0/1.
- IO ngoai: process control OS signal.
- Confidence: High.

2. production/bot.py
- Chuc nang: wrapper import va goi main trong live bot.
- Input: khong co input rieng (entrypoint script).
- Output: chuyen flow sang algo_trading.live.universal_bot.main().
- Confidence: High.

3. production/live_trading_moe_v2.py
- Chuc nang: bot live theo MOE v2 Enhanced tren OKX qua ccxt.
- Input: env OKX_API_KEY/OKX_SECRET_KEY/OKX_PASSPHRASE, symbol/timeframe, model_path.
- Output: signal, lenh giao dich, log live_trading.log, cap nhat position state trong RAM.
- IO ngoai: fetch_ohlcv, fetch_balance, create_market_order tu OKX API.
- Confidence: High.

### B. Utilities trong production/utils

4. production/utils/setup_okx_env.py
- Chuc nang: setup thong tin OKX vao .env.
- Input: stdin (api key/secret/passphrase/mode).
- Output: ghi/sua file .env.
- Confidence: High.

5. production/utils/setup_telegram_env.py
- Chuc nang: setup TELEGRAM_BOT_TOKEN/CHAT_ID vao .env.
- Input: stdin.
- Output: ghi/sua .env.
- Confidence: High.

6. production/utils/update_env_risk_params.py
- Chuc nang: cap nhat tham so risk trong .env.
- Input: stdin (RISK_PER_TRADE/SL_PCT/TP_PCT...).
- Output: ghi/sua .env.
- Confidence: High.

7. production/utils/check_model_features.py
- Chuc nang: kiem tra input_dim/model config tu model file.
- Input: duong dan model .pt/.pth.
- Output: print thong tin model/features.
- Confidence: High.

### C. Trading runtime trong production/algo_trading/live

8. production/algo_trading/live/__init__.py
- Chuc nang: package init.
- Input/Output: khong dang ke.
- Confidence: High.

9. production/algo_trading/live/universal_bot.py
- Chuc nang: bot runtime tong quat (route strategy + exchange).
- Input: .env (EXCHANGE, MODE, SYMBOL, INTERVAL, STRATEGY, CHECK_INTERVAL_SEC, HISTORY_LIMIT, API keys).
- Output: vong lap signal->order, logs, thong bao Telegram, history in-memory/file tuy implementation.
- IO ngoai: exchange REST API.
- Confidence: High.

10. production/algo_trading/live/okx_client.py
- Chuc nang: adapter OKX theo interface ExchangeClient.
- Input: key/secret/passphrase, symbol, interval, order params.
- Output: ohlcv dataframe/list, order response, account info.
- IO ngoai: HTTP call OKX API.
- Confidence: High.

11. production/algo_trading/live/exchange_base.py
- Chuc nang: abstract interface cho exchange.
- Input: method contract.
- Output: ABC + NotImplemented stubs.
- Confidence: High.

12. production/algo_trading/live/binance_sma_bot.py
- Chuc nang: strategy bot SMA tren Binance.
- Input: env Binance key/secret + symbol + risk.
- Output: order/log theo SMA signal.
- IO ngoai: Binance API.
- Confidence: Medium-High.

13. production/algo_trading/live/telegram_bot.py
- Chuc nang: gui thong bao signal/trade qua Telegram.
- Input: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, message payload.
- Output: tin nhan Telegram, command handlers.
- IO ngoai: Telegram Bot API.
- Confidence: High.

14. production/algo_trading/live/strategy_evaluator.py
- Chuc nang: danh gia strategy tren data lich su (metrics).
- Input: strategy + DataFrame OHLCV + params.
- Output: metrics (Sharpe/Sortino/drawdown/winrate), report structure.
- Confidence: Medium-High.

15. production/algo_trading/live/indicator_combiner.py
- Chuc nang: ket hop nhieu indicator/strategy thanh 1 signal.
- Input: danh sach strategy + weight + du lieu gia.
- Output: combined signal series/value.
- Confidence: Medium-High.

### D. Risk module

16. production/algo_trading/risk/dynamic_risk_manager.py
- Chuc nang: tinh position size, SL/TP, trailing theo confidence/volatility/regime/drawdown.
- Input: account_balance, entry price, ATR/volatility, confidence, drawdown, config risk.
- Output: size, sl, tp, trailing distance.
- Confidence: High.

### E. Signal filter

17. production/algo_trading/filters/signal_quality_filter.py
- Chuc nang: bo loc chat luong tin hieu (multi condition).
- Input: feature row/dataframe (momentum, volume, vol zone, consensus...).
- Output: boolean pass/fail va/hoac score.
- Confidence: High.

### F. Indicator files trong production/algo_trading/indicators

18. production/algo_trading/indicators/__init__.py
- Chuc nang: expose indicator API.
- Input/Output: package exports.
- Confidence: High.

19. production/algo_trading/indicators/core.py
- Chuc nang: central re-export cac indicator.
- Input: import calls.
- Output: unified import surface.
- Confidence: High.

20. production/algo_trading/indicators/moving_averages.py
- Chuc nang: SMA/EMA/WMA.
- Input: series + period.
- Output: indicator series.
- Confidence: High.

21. production/algo_trading/indicators/rsi.py
- Chuc nang: RSI.
- Input: close series + period.
- Output: RSI series.
- Confidence: High.

22. production/algo_trading/indicators/macd.py
- Chuc nang: MACD, signal, histogram.
- Input: close series + fast/slow/signal params.
- Output: MACD dataframe/series.
- Confidence: High.

23. production/algo_trading/indicators/bollinger_bands.py
- Chuc nang: BB upper/middle/lower.
- Input: close series + window/std.
- Output: bands dataframe.
- Confidence: High.

24. production/algo_trading/indicators/volatility.py
- Chuc nang: ATR/true range.
- Input: OHLC dataframe + period.
- Output: vol series.
- Confidence: High.

25. production/algo_trading/indicators/volume.py
- Chuc nang: volume indicators (VD: VWAP).
- Input: HLCV dataframe.
- Output: volume-derived series.
- Confidence: Medium-High.

26. production/algo_trading/indicators/zscore.py
- Chuc nang: z-score indicator.
- Input: series + rolling window.
- Output: zscore series.
- Confidence: High.

27. production/algo_trading/indicators/composite.py
- Chuc nang: tinh batch indicator tren dataframe.
- Input: OHLCV dataframe + config list.
- Output: dataframe bo sung cot indicator.
- Confidence: High.

28. production/algo_trading/indicators/utils.py
- Chuc nang: helper cho index/validation/transform.
- Input: dataframe/series.
- Output: du lieu da normalize/validate.
- Confidence: Medium-High.

29. production/algo_trading/indicators/ict.py
- Chuc nang: chi bao theo ICT concepts.
- Input: OHLCV dataframe.
- Output: cot feature ICT.
- Confidence: Medium.

### G. Feature files trong production/algo_trading/features

30. production/algo_trading/features/multi_timeframe.py
- Chuc nang: tao feature da khung thoi gian.
- Input: base timeframe data + target timeframes.
- Output: feature dataframe MTF.
- Confidence: High.

31. production/algo_trading/features/seasonality.py
- Chuc nang: seasonal/time-based features (hour/day/week).
- Input: datetime index.
- Output: cyclical features (sin/cos or categorical encodings).
- Confidence: Medium-High.

### H. ML package init

32. production/algo_trading/ml/__init__.py
- Chuc nang: package init/export.
- Input/Output: import surface.
- Confidence: High.

### I. Core ML ensemble files trong production/algo_trading/ml

33. production/algo_trading/ml/dynamic_moe_v2_enhanced.py
- Chuc nang: model MOE v2 Enhanced (4 experts + gating + filter hooks).
- Input: X features (train/infer), y labels, optional regime/filter configs.
- Output: fitted model, predict/predict_proba, selected expert/regime metadata.
- Confidence: High.

34. production/algo_trading/ml/dynamic_moe_v2.py
- Chuc nang: baseline MOE v2.
- Input: X, y.
- Output: gating + expert predictions.
- Confidence: High.

35. production/algo_trading/ml/predict_moe_v2.py
- Chuc nang: wrapper inference tu model artifact.
- Input: latest feature vector, model path, threshold params.
- Output: dict signal/proba/confidence/expert/regime.
- Confidence: High.

36. production/algo_trading/ml/expert_trend_detector.py
- Chuc nang: expert cho market trending.
- Input: X_train, y_train; X_infer.
- Output: class probabilities/prediction.
- Confidence: High.

37. production/algo_trading/ml/expert_range_finder.py
- Chuc nang: expert cho ranging market.
- Input: X_train, y_train; X_infer.
- Output: class probabilities/prediction.
- Confidence: High.

38. production/algo_trading/ml/expert_volatility_breakout.py
- Chuc nang: expert cho breakout regime.
- Input: X_train, y_train; X_infer.
- Output: class probabilities/prediction.
- Confidence: High.

39. production/algo_trading/ml/expert_special_regime.py
- Chuc nang: expert cho regime dac biet/extreme.
- Input: X_train, y_train; X_infer.
- Output: class probabilities/prediction.
- Confidence: High.

40. production/algo_trading/ml/regime_specific_thresholds.py
- Chuc nang: map regime -> threshold long/short.
- Input: regime id + direction.
- Output: threshold float.
- Confidence: High.

41. production/algo_trading/ml/regime_specific_models.py
- Chuc nang: train/predict model rieng theo regime.
- Input: X, y, regime assignments.
- Output: per-regime predictions va ensemble combine.
- Confidence: High.

42. production/algo_trading/ml/probability_calibration.py
- Chuc nang: calibrate probability (isotonic/sigmoid).
- Input: raw proba + validation labels.
- Output: calibrated proba.
- Confidence: High.

43. production/algo_trading/ml/regime_confidence_score.py
- Chuc nang: tinh do tu tin cay regime detection.
- Input: regime probability distribution/features.
- Output: confidence score [0..1].
- Confidence: High.

44. production/algo_trading/ml/model_monitor.py
- Chuc nang: monitor perf model va trigger retrain.
- Input: lich su trade, metrics stream.
- Output: canh bao suy giam, flag should_retrain.
- Confidence: High.

45. production/algo_trading/ml/training.py
- Chuc nang: training/walk-forward pipeline (co sequence/transformer).
- Input: dataset folds, labels, hyperparams.
- Output: trained weights/model objects + metrics theo fold.
- Confidence: High.

46. production/algo_trading/ml/features.py
- Chuc nang: feature engineering cho ML layer.
- Input: OHLCV + indicator/market model data.
- Output: X features/scaled features.
- Confidence: High.

47. production/algo_trading/ml/multi_timeframe.py
- Chuc nang: helper MTF cho ML.
- Input: dataframe + timeframe set.
- Output: resampled/merged features.
- Confidence: High.

48. production/algo_trading/ml/enhanced_multi_timeframe.py
- Chuc nang: MTF nang cao.
- Input: raw OHLCV + config.
- Output: bo feature MTF nang cao.
- Confidence: Medium-High.

49. production/algo_trading/ml/sequence_extractor.py
- Chuc nang: sequence feature extractor (LSTM/TCN/Torch).
- Input: chuoi gia/returns + sequence length.
- Output: sequence embeddings/features.
- Confidence: High.

50. production/algo_trading/ml/label_creation.py
- Chuc nang: tao labels cho bai toan 3 class (-1/0/1).
- Input: returns/threshold rules.
- Output: label series.
- Confidence: High.

51. production/algo_trading/ml/focal_loss.py
- Chuc nang: focal loss utility + class weighting.
- Input: class distribution/predictions.
- Output: weighted loss or class weights.
- Confidence: High.

52. production/algo_trading/ml/feature_importance_analysis.py
- Chuc nang: phan tich importance (SHAP/permutation).
- Input: trained model + X_test/y_test.
- Output: ranking/report/plot importance.
- Confidence: Medium-High.

53. production/algo_trading/ml/moe_optimizer.py
- Chuc nang: optimize hyperparameters (Optuna).
- Input: search space + objective + data.
- Output: best params/trials summary.
- Confidence: Medium-High.

54. production/algo_trading/ml/signal_quality_filter.py
- Chuc nang: bo loc signal trong ML package (co the duplicate voi filters/).
- Input: feature row/df.
- Output: pass/fail score.
- Confidence: Medium (can quy uoc file nao la source of truth).

### J. ML models subpackage

55. production/algo_trading/ml/models/__init__.py
- Chuc nang: package init cho ML model classes.
- Input/Output: exports.
- Confidence: High.

56. production/algo_trading/ml/models/transformer_distribution.py
- Chuc nang: transformer model cho distribution/regime forecasting.
- Input: sequence tensor features.
- Output: logits/distribution predictions.
- Confidence: Medium-High.

## 4) Input/Output tai runtime (muc he thong)

- Input chinh:
  - API credentials: OKX/Binance/Telegram tu .env.
  - Market data: OHLCV tu exchange.
  - Model artifacts: pkl/pt tu production/models.
  - Runtime params: strategy, thresholds, risk, intervals.
- Output chinh:
  - Order placement/cancel qua exchange API.
  - Local logs: trading_bot.log, live_trading.log.
  - PID/process status.
  - Telegram notifications.

## 5) File duplicate/ambiguity can chot

- production/algo_trading/ml/signal_quality_filter.py va production/algo_trading/filters/signal_quality_filter.py co ten/chuc nang trung lap.
- bot.py dang truyen vao binance_sma_bot trong khi production co huong MOE/OKX; can chot entrypoint chuan duy nhat.
- live_trading_moe_v2.py va universal_bot.py co the trung vai tro runtime; can chot file chay chuan de tranh config drift.

## 6) De xuat quy uoc van hanh chuan

1. Chot 1 entrypoint duy nhat cho production (khuyen nghi universal_bot.py qua start_trading_bot.py).
2. Chot 1 signal filter source-of-truth (filters/ hoac ml/).
3. Ghi ro map env var -> file consume trong docs production.
4. Them smoke test import + dry-run mode de verify IO truoc khi dat lenh that.

## 7) Ban chot implementation (da ap dung)

1. Entrypoint duy nhat cho production
- Process manager: production/start_trading_bot.py
- Runtime entrypoint: production/algo_trading/live/universal_bot.py
- Wrapper production/bot.py da tro ve universal_bot.main de tranh split flow.

2. Signal filter source-of-truth
- Chot source-of-truth tai production/algo_trading/filters/signal_quality_filter.py.
- production/algo_trading/ml/signal_quality_filter.py da chuyen thanh compatibility adapter, delegate ve filters/.

3. Dry-run mode
- production/start_trading_bot.py ho tro action moi: dry-run.
- production/algo_trading/live/universal_bot.py ho tro --dry-run (ep MODE=paper, chay 1 vong run_once, khong loop vo han).

4. Smoke test
- Them production/smoke_test_production.py de verify import chain + goi dry-run.

## 8) Env var map (consume map)

### Entry + Runtime control
- MODE -> production/algo_trading/live/universal_bot.py (paper/testnet/live; dry-run ep ve paper).
- EXCHANGE -> production/algo_trading/live/universal_bot.py (okx|binance, route qua create_exchange_client).
- SYMBOL -> production/algo_trading/live/universal_bot.py (symbol giao dich).
- INTERVAL -> production/algo_trading/live/universal_bot.py (khung nen fetch).
- STRATEGY -> production/algo_trading/live/universal_bot.py (chon strategy trong STRATEGY_MAP).
- STRATEGY_PARAMS -> production/algo_trading/live/universal_bot.py (json/literal dict).
- HISTORY_LIMIT -> production/algo_trading/live/universal_bot.py (so nen nap moi vong).
- COOL_DOWN_SEC -> production/algo_trading/live/universal_bot.py (cooldown giua signal).
- CHECK_INTERVAL_SEC -> production/algo_trading/live/universal_bot.py (chu ky loop).

### Exchange credentials
- OKX_API_KEY -> production/algo_trading/live/universal_bot.py -> production/algo_trading/live/okx_client.py.
- OKX_API_SECRET hoac OKX_SECRET_KEY -> production/algo_trading/live/universal_bot.py -> production/algo_trading/live/okx_client.py.
- OKX_PASSPHRASE -> production/algo_trading/live/universal_bot.py -> production/algo_trading/live/okx_client.py.
- OKX_USE_SIMULATED_TRADING -> production/algo_trading/live/universal_bot.py -> production/algo_trading/live/okx_client.py.
- BINANCE_API_KEY -> production/algo_trading/live/universal_bot.py -> BinanceClient.
- BINANCE_API_SECRET -> production/algo_trading/live/universal_bot.py -> BinanceClient.

### MOE strategy params (auto-map khi STRATEGY=moe_v2_enhanced)
- MOE_MODEL_PATH -> production/algo_trading/live/universal_bot.py (strategy param model_path).
- MOE_PROBA_THRESHOLD -> production/algo_trading/live/universal_bot.py.
- MOE_USE_REGIME_SPECIFIC -> production/algo_trading/live/universal_bot.py.
- MOE_USE_DYNAMIC_THRESHOLD -> production/algo_trading/live/universal_bot.py.
- MOE_USE_QUANTILE_THRESHOLD -> production/algo_trading/live/universal_bot.py.
- MOE_TARGET_SIGNAL_RATE -> production/algo_trading/live/universal_bot.py.
- MOE_QUANTILE_WINDOW -> production/algo_trading/live/universal_bot.py.
- MOE_QUANTILE_FLOOR -> production/algo_trading/live/universal_bot.py.

### Risk params
- RISK_PER_TRADE -> production/algo_trading/live/universal_bot.py (position sizing).
- SL_PCT -> production/algo_trading/live/universal_bot.py.
- TP_PCT -> production/algo_trading/live/universal_bot.py.
- TRAILING_PCT -> production/algo_trading/live/universal_bot.py.
- SL_ATR_K -> production/algo_trading/live/universal_bot.py.
- TP_ATR_K -> production/algo_trading/live/universal_bot.py.
- TRAILING_ATR_K -> production/algo_trading/live/universal_bot.py.
- MAX_POSITION_SIZE -> production/algo_trading/live/universal_bot.py.
- MAX_DCA_ORDERS -> production/algo_trading/live/universal_bot.py.

### Telegram notify
- TELEGRAM_BOT_TOKEN -> production/algo_trading/live/universal_bot.py + production/algo_trading/live/telegram_bot.py.
- TELEGRAM_CHAT_ID -> production/algo_trading/live/universal_bot.py + production/algo_trading/live/telegram_bot.py.

## 9) Cac cach sua (thuc hanh)

1. Cach nhanh, an toan (khuyen nghi)
- Dung manager script:
  - python production/start_trading_bot.py dry-run
  - python production/start_trading_bot.py start
  - python production/start_trading_bot.py status
  - python production/start_trading_bot.py stop

2. Cach sua truc tiep runtime
- Chay thang module:
  - python -m production.algo_trading.live.universal_bot --dry-run
- Dung cach nay khi ban muon debug runtime argument level.

3. Cach verify truoc deploy
- Smoke test:
  - python production/smoke_test_production.py
- Muc tieu: check import + dry-run I/O chain khong vao lenh that.

4. Cach rollback neu can
- Tra bot.py ve strategy cu (khong khuyen nghi) neu ban can test legacy.
- Giu adapter ml/signal_quality_filter.py de code cu van import duoc.
