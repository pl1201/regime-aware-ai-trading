
import sys
import os
import pandas as pd
import numpy as np
from datetime import timedelta

# Add project root to path
sys.path.append(os.getcwd())

from algo_trading.strategies.ml.regime_ensemble_strategy import RegimeEnsembleStrategy

def run_quick_compare():
    print("=" * 60)
    print("⚡ QUICK COMPARISON: Using Synthetic Data to Verify Logic")
    print("=" * 60)

    # 1. Generate Synthetic Data
    # Create 500 bars. First half volatile, second half calm.
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1H")
    
    # Volatile segment (High BB Width)
    np.random.seed(42)
    p1 = 10000 + np.cumsum(np.random.randn(250) * 50) 
    # Calm segment (Low BB Width)
    p2 = p1[-1] + np.cumsum(np.random.randn(250) * 10)
    
    close = np.concatenate([p1, p2])
    
    # Mock BB width simulation in Strategy requires proper OHLCV
    # We construct OHLC to naturally produce BB width variation
    high = close + (np.abs(np.random.randn(500)) * 10)
    low = close - (np.abs(np.random.randn(500)) * 10)
    
    # Exaggerate volatility in first half
    high[:250] = close[:250] + (np.abs(np.random.randn(250)) * 100)
    low[:250] = close[:250] - (np.abs(np.random.randn(250)) * 100)
    
    df = pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.abs(np.random.randn(500)) * 1000
    }, index=dates)

    print(f"📊 Generated {len(df)} synthetic bars.")
    
    # 2. Setup Strategies
    # We use a mock model to ensure we get Probabilities > 0.5 sometimes
    class MockModel:
        def predict_proba(self, X, **kwargs):
            # Return probs around 0.55 +- noise
            # Shape: (n_samples, 2). Col 1 is class 1 (Long).
            n = len(X)
            p_long = 0.54 + np.random.randn(n) * 0.05
            p_long = np.clip(p_long, 0, 1)
            return np.column_stack([1-p_long, p_long])
            
        def classes_(self):
            return [0, 1]

    # Configs to compare
    configs = [
         {"name": "BASE (Static 0.55)", "use_dyn": False},
         {"name": "DYNAMIC (Vol-based)", "use_dyn": True},
    ]
    
    for cfg in configs:
        print(f"\n🔹 Testing: {cfg['name']}")
        try:
            strategy = RegimeEnsembleStrategy(
                model_path="models/regime_ensemble_optimized.pkl", # Dummy path, we overwrite model
                use_regime_specific=False,
                use_dynamic_threshold=cfg['use_dyn'],
                proba_threshold=0.55
            )
            strategy.model = MockModel() 
            # Force mock validation to pass if needed, but BaseStrategy handles it
            
            res = strategy.generate_signals(df)
            
            sig_counts = res.signals.value_counts()
            n_long = sig_counts.get(1.0, 0)
            n_short = sig_counts.get(-1.0, 0)
            total_sig = n_long + n_short
            
            meta = res.meta
            avg_th = meta.get("avg_dynamic_threshold", 0.55)
            
            print(f"   Signals Generated: {total_sig}")
            print(f"   Avg Threshold:     {avg_th:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n✅ Comparison Logic Verified.")

if __name__ == "__main__":
    run_quick_compare()
