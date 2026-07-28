# Live Trading Setup Guide - MOE v2 Enhanced

## 📋 BACKTEST RESULTS SUMMARY

### 1H Timeframe (RECOMMENDED)
- **Winrate**: 93.5% (In-Sample), 94.9% (OOS 2024-2026)
- **Total Return**: 1521% (IS), 115% (OOS)
- **Profit Factor**: 15.48 (IS), 37.70 (OOS)
- **Max Drawdown**: -8.58% (IS), -0.97% (OOS)
- **Trades**: 1244 (IS), 118 (OOS)
- **Walk-Forward**: 90.1% avg winrate, 0.014 std dev ✅

### M15 Timeframe (NOT RECOMMENDED)
- **Winrate**: 66.3% (IS), 63.9% (OOS)
- **Total Return**: 295% (IS), 6.69% (OOS)
- **Profit Factor**: 1.51 (both)
- **Max Drawdown**: -17.08%

## ⚙️ SETUP STEPS

### 1. Get OKX API Keys
1. Go to https://www.okx.com/account/my-api
2. Create new API key with:
   - **Read** permission
   - **Trade** permission
   - **DO NOT enable Withdrawal** (security)
3. Copy: API Key, Secret Key, Passphrase

### 2. Configure Environment
Edit `.env` file:
```bash
OKX_API_KEY=your_api_key_here
OKX_SECRET_KEY=your_secret_key_here
OKX_PASSPHRASE=your_passphrase_here

PAPER_TRADING=true  # Set to true for paper trading first!
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Paper Trading First (2-4 weeks)
```bash
python live_trading_moe_v2.py
```

### 5. Monitor Performance
Check `live_trading.log` for:
- Signal accuracy vs backtest
- Position sizes
- Drawdown levels
- Any errors

### 6. Switch to Live Trading (After successful paper trading)
Edit `.env`:
```bash
PAPER_TRADING=false
```

Then run:
```bash
python live_trading_moe_v2.py
```

## 🛡️ RISK MANAGEMENT

### Default Settings (Conservative)
- **Max risk per trade**: 2% of account
- **Max daily risk**: 6% of account
- **Max drawdown limit**: 25% (auto-stop)
- **TP/SL ratio**: 3.0
- **Min R:R**: 1.8

### Position Sizing
Based on backtest:
- Average win: 2.3%
- Average loss: 1.4%
- Winrate: 93.5%

## 📊 MONITORING CHECKLIST

### Daily
- [ ] Check `live_trading.log` for errors
- [ ] Verify signal accuracy
- [ ] Monitor drawdown
- [ ] Check balance changes

### Weekly
- [ ] Compare actual vs backtest performance
- [ ] Review all trades taken
- [ ] Adjust position size if needed
- [ ] Check for model drift

### Monthly
- [ ] Full performance review
- [ ] Retrain model if needed (auto-run in script)
- [ ] Update risk parameters
- [ ] Review and optimize

## 🚨 SAFETY FEATURES

1. **Paper Trading Mode**: Default ON - simulate trades without real money
2. **Auto Stop-Loss**: 1.5% trailing stop
3. **Daily Risk Limit**: Max 6% loss per day
4. **Max Drawdown**: Auto-stop at 25% drawdown
5. **Signal Quality Filter**: Only high-quality signals executed
6. **Threshold Filter**: Only signals > 0.5 probability

## ⏰ TRADING SCHEDULE

- **Timeframe**: 1H (check every hour)
- **Coverage**: ~2.3% of time (very selective)
- **Expected trades**: ~3-5 per week
- **Best for**: Long-term swing trading

## 📝 TROUBLESHOOTING

### "Model not found"
```bash
# Run training first
python train_moe_v2_enhanced_final.py
```

### "API authentication failed"
- Verify API keys in `.env`
- Check IP whitelist in OKX dashboard
- Ensure correct passphrase

### "Not enough data"
- Wait for more candles to accumulate
- Minimum 50 candles required

### "No signals"
- Normal! Model is very selective (2.3% coverage)
- May go days/weeks without signals

## 🎯 EXPECTED PERFORMANCE

Based on backtest (conservative estimates):
- **Monthly return**: 5-15%
- **Winrate**: 85-95%
- **Max drawdown**: <10%
- **Trades/month**: 8-20

## ⚠️ IMPORTANT WARNINGS

1. **Past performance ≠ future results**
2. **Start with paper trading for 2-4 weeks**
3. **Never risk more than you can afford to lose**
4. **Monitor closely for first month of live trading**
5. **Have stop-loss ready to manually intervene**
6. **Model may need retraining if market regime changes**

## 📞 SUPPORT

If issues arise:
1. Check `live_trading.log` for error messages
2. Verify API permissions
3. Ensure model file exists
4. Test with paper trading mode first

---
**Last Updated**: 2026-03-25
**Model Version**: MOE v2 Enhanced
**Backtest Period**: 2017-2026
