# Bot_Trading Project Structure - Đã Tối Ưu

## 📁 Cấu Trúc Mới

```
d:\Bot_Trading\
│
├── 📂 production/              # ✅ FILES CHO LIVE TRADING
│   ├── live_trading_moe_v2.py  # Main bot (entry point)
│   ├── bot.py                  # Telegram bot integration
│   ├── start_trading_bot.py    # Alternative launcher
│   ├── .env                    # API keys & config
│   │
│   ├── 📂 algo_trading/
│   │   ├── 📂 live/            # OKX client, exchange base
│   │   ├── 📂 ml/              # MOE v2 Enhanced model
│   │   ├── 📂 features/        # Multi-timeframe features
│   │   ├── 📂 indicators/      # Technical indicators
│   │   ├── 📂 risk/            # Risk management
│   │   └── 📂 filters/         # Signal quality filters
│   │
│   ├── 📂 config/              # Model configuration
│   ├── 📂 models/              # Trained model files
│   └── 📂 utils/               # Utility functions
│
├── 📂 archive/                 # 📦 FILES ĐƯA VÀO ARCHIVE
│   ├── 📂 backtests/           # Backtest scripts (15+ files)
│   ├── 📂 training/            # Training scripts (10+ files)
│   ├── 📂 tests/               # Test scripts (20+ files)
│   ├── 📂 reports/             # Báo cáo (15+ files)
│   └── 📂 old_models/          # Old model files
│
├── 📂 docs/                    # 📚 Documentation
│   ├── LIVE_TRADING_SETUP.md
│   └── PROJECT_STRUCTURE.md
│
└── 📂 data/                    # Dữ liệu BTC/USDT
    └── okx_1h.csv
```

## 📊 Thống Kê

### Trước khi tối ưu
- **Tổng files**: ~180+ Python files
- **Báo cáo**: 15+ markdown files
- **Test files**: 20+ files
- **Backtest scripts**: 15+ versions
- **Training scripts**: 10+ versions

### Sau khi tối ưu
- **Production**: ~60 files (cần cho live trading)
- **Archive**: ~120 files (đưa vào archive, không xóa)
- **Giảm độ phức tạp**: ~65%

## 🗂️ Files Đã Di Chuyển

### Vào `archive/backtests/`
- `backtest_moe_v2*.py` (5 files)
- `backtest_regime*.py` (3 files)
- `display_*.py` (2 files)
- `run_backtest_with_results.py`

### Vào `archive/training/`
- `train_moe_v2*.py` (6 files)
- `train_regime*.py` (2 files)
- `train_with_new_data.py`

### Vào `archive/tests/`
- `test_*.py` (15 files)
- `quick_test_*.py` (5 files)
- `final_eval*.py` (2 files)

### Vào `archive/reports/`
- `BAO_CAO_*.md` (7 files)
- `CAI_THIEN_*.md` (1 file)
- `FIXES_*.md` (2 files)
- `IMPROVEMENTS_*.md` (1 file)
- `HOW_TO_USE_*.md` (1 file)
- `QUICK_START_*.md` (1 file)
- `RUN_BOT.md`, `START_OKX_BOT.md`, etc.

### Xóa (không cần)
- `*.bat` files (5 files)
- `*.pid` files (3 files)
- `*.zip`, `*.tar.gz` (2 files)
- `*.log` files (3 files)
- Scripts: `analyze_data.py`, `check_dca_env.py`, `optimize_ict_parameters.py`, `repair_and_test.py`, `run_*.py`, `update_env_risk_params_auto.py`, `zip_creator.py`

## ✅ Files Còn Lại trong Production

### Entry Points
- `live_trading_moe_v2.py` - Main bot
- `bot.py` - Telegram integration
- `start_trading_bot.py` - Alternative launcher

### Core Modules
- `algo_trading/live/` - OKX trading client
- `algo_trading/ml/dynamic_moe_v2_enhanced.py` - Core model
- `algo_trading/features/multi_timeframe.py` - Feature generation
- `algo_trading/indicators/` - Technical indicators (20+ files)
- `algo_trading/risk/dynamic_risk_manager.py` - Risk management
- `algo_trading/filters/signal_quality_filter.py` - Signal filtering

### Configuration
- `.env` - API keys
- `config/*.yaml` - Model config
- `models/*.pkl` - Trained models

### Utilities
- `utils/*.py` - Data loading, helpers

## 🚀 Cách Sử Dụng

### Chạy Live Trading
```bash
cd production
python live_trading_moe_v2.py
```

### Access Archive Files (nếu cần)
```bash
# Backtest
cd archive/backtests
python backtest_moe_v2_enhanced_detailed.py

# Training
cd archive/training
python train_moe_v2_enhanced_final.py
```

## 📝 Lưu Ý

1. **Không xóa archive** - Giữ lại để reference khi cần
2. **Production folder** - Chỉ chứa files cần cho live trading
3. **Backup** - Đã có đầy đủ trong archive trước khi xóa
4. **Models** - Đã copy vào production/models/

## 🎯 Next Steps

1. ✅ Cấu trúc đã được tổ chức lại
2. ⏳ Cấu hình `.env` với OKX API keys
3. ⏳ Chạy `production/live_trading_moe_v2.py`
4. ⏳ Monitor performance trong 2-4 tuần
5. ⏳ Switch từ paper trading sang live trading

---
**Date**: 2026-03-25
**Status**: ✅ Production-ready structure
