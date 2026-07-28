# Production Trading Bot - H1 Enhanced

## Overview

Production-ready H1 trading system with HMM regime detection and MTF confirmation.

### Available Models

| Model | Trades | WR | PF | Return | Timeframe |
|-------|--------|-----|-----|--------|-----------|
| **H1 Enhanced** | 140 | 71.4% | **8.24** | +156.5% | H1 |
| **MOE v3 HMM** | 921 | 52.4% | 1.31 | +182.5% | H1 |
| H1 Hybrid | 169 | 55.0% | 1.35 | +30.6% | H1 |

## Quick Start

```bash
# 1. Setup
cd production
pip install -r config/requirements.txt

# 2. Configure
cp .env.h1 .env
# Edit .env with your OKX API keys

# 3. Test
python smoke_test_production.py

# 4. Start
python start_trading_bot.py start
```

## Architecture

```
production/
├── algo_trading/
│   ├── ml/
│   │   ├── h1_enhanced_model.py       # H1 Enhanced (HMM + MTF) - Best
│   │   ├── dynamic_moe_v3_hmm_mtf.py  # MOE v3 with HMM
│   │   ├── h1_hybrid_model.py         # H1 Hybrid (simple)
│   │   └── models/                    # Trained models (.pkl)
│   ├── features/
│   │   └── h1_features.py             # H1 feature engineering
│   ├── live/
│   │   ├── universal_bot.py           # Live trading bot
│   │   └── okx_client.py              # OKX exchange client
│   └── risk/
│       └── dynamic_risk_manager.py    # Risk management
├── train_h1_enhanced.py               # Training script
├── train_moe_v3_hmm_mtf.py            # MOE v3 training
├── evaluate_h1_full.py                # Backtesting
├── start_trading_bot.py               # Start live trading
├── .env.h1                            # Configuration template
└── archive/                           # Deprecated M15 files
```

## Key Features

### HMM Regime Detection
- 4 states: trending, ranging, volatile, calm
- **Only trade in TRENDING regime**
- Skip 79% of sideway time (avoid whipsaw)

### Multi-Timeframe Confirmation (MTF)
- H1 base signals
- H4 momentum alignment
- D1 trend confirmation

### Risk Management
- 2% risk per trade
- SL: 1.5% / TP: 3% (1:2 R:R)
- Max position: 5%

## Configuration (.env)

```env
# Mode
MODE=paper              # paper/live
EXCHANGE=okx
SYMBOL=BTCUSDT
INTERVAL=1h
STRATEGY=h1_hybrid

# OKX Credentials
OKX_API_KEY=your_key
OKX_API_SECRET=your_secret
OKX_PASSPHRASE=your_passphrase

# Risk
RISK_PER_TRADE=0.02
SL_PCT=0.015
TP_PCT=0.03
```

## Operations

```bash
# Dry run (no real orders)
python start_trading_bot.py dry-run

# Start trading
python start_trading_bot.py start

# Check status
python start_trading_bot.py status

# Stop
python start_trading_bot.py stop
```

## Training (Optional - models already trained)

```bash
# Train H1 Enhanced
python train_h1_enhanced.py

# Train MOE v3
python train_moe_v3_hmm_mtf.py --timeframe 1h

# Evaluate
python evaluate_h1_full.py
```

## Why H1 over M15?

| Metric | M15 | H1 |
|--------|-----|-----|
| Cost/move ratio | 40% | **12%** |
| Signal-to-noise | 0.17 | **0.45+** |
| Profit Factor | 0.6-1.0 | **1.35-8.24** |

## Archived Files

All M15-related files moved to `archive/` folder. Use H1 timeframe instead.
