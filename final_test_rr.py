import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.append(ROOT)

from backtest_regime_ensemble_advanced import backtest_regime_ensemble_advanced

def final_test():
    print("--- FINAL EVALUATION START ---")
    model_path = os.path.join("models", "regime_bandit_stacking_optimized.pkl")
    
    # Last 10 days for definitive results
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    print(f"Backtesting from: {start_date}")
    
    try:
        result = backtest_regime_ensemble_advanced(
            model_path=model_path,
            source='binance',
            symbol='BTCUSDT',
            interval='1h',
            start=start_date,
            end=None,
            sl_pct=0.02,
            tp_pct=0.04,
            leverage=1.0,
            commission=0.0005,
            max_trades=20,
            proba_threshold=0.35, # Aggressive for testing
            use_dynamic_threshold=False,
            use_regime_specific_params=True,
            allowed_regimes=["trending", "ranging", "volatile", "calm"],
            validation_split=0.0,
        )
        
        if result:
            train_res = result.get('train', {})
            trades = train_res.get('trades', pd.DataFrame())
            
            # CRITICAL: Write results to a file we CAN read
            output_lines = [
                f"TOTAL_TRADES: {len(trades)}",
            ]
            
            if not trades.empty:
                winning = (trades['pnl'] > 0).sum()
                winrate = (winning / len(trades)) * 100
                avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning > 0 else 0
                avg_loss = abs(trades[trades['pnl'] < 0]['pnl'].mean()) if (trades['pnl'] < 0).any() else 0.0001
                rr = avg_win / avg_loss
                
                output_lines.append(f"WIN_RATE: {winrate:.2f}")
                output_lines.append(f"RR_RATIO: {rr:.2f}")
                output_lines.append(f"PROFIT_FACTOR: {winning/abs(len(trades)-winning) if len(trades)-winning > 0 else 99:.2f}")

            with open("FINAL_RESULTS_RR.txt", "w") as f:
                f.write("\n".join(output_lines))
            print("Successfully wrote FINAL_RESULTS_RR.txt")
        else:
            print("Result is None")
    except Exception as e:
        with open("FINAL_ERROR.txt", "w") as f:
            f.write(str(e))

if __name__ == "__main__":
    final_test()
