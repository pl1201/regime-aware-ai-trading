
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load as joblib_load

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from algo_trading.data_loader.loader import load_data
from algo_trading.backtest.vectorized import run_backtest, BacktestConfig, RiskConfig
from algo_trading.strategies.ml.regime_ensemble_strategy import RegimeEnsembleStrategy
from algo_trading.core.metrics import performance_summary, safe_total_return

def run_comparison():
    print("=" * 80)
    print("🚀 DYNAMIC THRESHOLD COMPARISON TEST")
    print("=" * 80)

    # 1. Config
    model_path = "models/regime_ensemble_optimized.pkl" # Prefer optimized model
    if not os.path.exists(model_path):
        # Fallback to regime specific if optimized not found, or handle error
        model_path = "models/regime_specific_models_optimized.pkl"
        use_regime_specific = True
    else:
        use_regime_specific = False # Or based on file content

    # Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    symbol = "BTCUSDT"  
    interval = "1h"
    start_date = "2024-01-01"
    
    # 2. Load Data
    print(f"📥 Loading data for {symbol} ({interval}) from {start_date}...")
    try:
        df = load_data(
            source='binance',
            symbol=symbol,
            interval=interval,
            start=start_date,
            end=None, 
            market='spot',
            add_features=True
        )
        print(f"✅ Loaded {len(df)} bars.")
    except Exception as e:
        print(f"❌ Data load failed: {e}")
        return

    # 3. Setup comparison
    configs = [
        {"name": "Baseline (Static Threshold)", "use_dynamic": False, "threshold": 0.55},
        {"name": "Dynamic Threshold (Volatility)", "use_dynamic": True, "threshold": 0.55},
    ]

    results = []

    for cfg in configs:
        print(f"\n▶️ Testing: {cfg['name']} ...")
        
        try:
            # Init strategy
            # Logic to handle model type loading similar to strategy class
            strategy = RegimeEnsembleStrategy(
                model_path=model_path if not use_regime_specific else None,
                regime_specific_model_path=model_path if use_regime_specific else None,
                use_regime_specific=use_regime_specific,
                use_dynamic_threshold=cfg["use_dynamic"],
                proba_threshold=cfg["threshold"]
            )
            
            # Generate signals
            res = strategy.generate_signals(df)
            signals = res.signals
            meta = res.meta
            
            # Backtest
            risk = RiskConfig(sl_pct=0.02, tp_pct=0.04)
            bt_cfg = BacktestConfig(initial_capital=10000.0, commission=0.0005)
            
            bt_res = run_backtest(df, signals, cfg=bt_cfg, risk=risk)
            
            # Extract metrics
            equity = bt_res.get('equity', pd.Series())
            trades = bt_res.get('trades', pd.DataFrame())
            
            total_ret = 0.0
            sharpe = 0.0
            win_rate = 0.0
            trade_count = len(trades)
            avg_threshold = meta.get("avg_dynamic_threshold", cfg["threshold"])
            
            if not equity.empty:
                 total_ret = safe_total_return(equity)
                 if len(equity) > 1:
                     rets = equity.pct_change().fillna(0)
                     if rets.std() > 0:
                         sharpe = rets.mean() / rets.std() * np.sqrt(24*365) # Annualized rough
            
            if not trades.empty:
                win_rate = (trades['pnl'] > 0).mean()

            results.append({
                "Configuration": cfg['name'],
                "Total Return": f"{total_ret*100:.2f}%",
                "Sharpe": f"{sharpe:.2f}",
                "Trades": trade_count,
                "Win Rate": f"{win_rate*100:.1f}%",
                "Avg Threshold": f"{avg_threshold:.4f}"
            })
            
        except Exception as e:
            print(f"❌ Error in {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Report
    print("\n" + "="*80)
    print("📊 COMPARISON RESULTS")
    print("="*80)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    run_comparison()
