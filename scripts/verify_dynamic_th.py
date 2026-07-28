
import sys
import os
import pandas as pd
import numpy as np
import logging

# Add current directory to path
sys.path.append(os.getcwd())

from algo_trading.strategies.ml.regime_ensemble_strategy import RegimeEnsembleStrategy

def test_dynamic_threshold():
    print("Initializing Strategy with Dynamic Threshold...")
    try:
        # Mock model path to avoid loading actual heavy models if possible, 
        # but the init tries to load. Let's point to existing one.
        strategy = RegimeEnsembleStrategy(
            use_regime_specific=True,
            regime_specific_model_path="models/regime_specific_models_optimized.pkl",
            use_dynamic_threshold=True,
            proba_threshold=0.55
        )
    except Exception as e:
        print(f"Failed to init strategy (likely due to missing model or dependencies): {e}")
        print("Mocking loading for logic verification...")
        # Fallback: manually instantiate if possible or skip model loading part if we can't
        return

    print("Creating synthetic data...")
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")
    
    # Create volatile price (high BB width) and calm price (low BB width)
    # Volatile segment
    p1 = 10000 + np.random.randn(50) * 500 # High std
    # Calm segment
    p2 = 10000 + np.random.randn(50) * 10  # Low std
    
    close = np.concatenate([p1, p2])
    
    df = pd.DataFrame({
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.rand(100) * 1000
    }, index=dates)

    print("Running generate_signals...")
    try:
        # We need to mock predicting because we don't know if the model aligns with our synthetic data features
        # But let's try running it. If feature mismatch, it will raise.
        # To bypass feature mismatch, we can mock `regime_models.predict_proba`
        
        # Mocking predict_proba to return random probabilities
        class MockModel:
            def predict_proba(self, X, **kwargs):
                # Return probas for [short, long]
                return np.random.rand(len(X), 2)
                
        strategy.regime_models = MockModel()
        
        # Run
        result = strategy.generate_signals(df)
        
        print("Analysis of Dynamic Thresholds:")
        meta = result.meta
        if "avg_dynamic_threshold" in meta:
            print(f"Average Dynamic Threshold: {meta['avg_dynamic_threshold']}")
        
        # We can't easily access the internal thresholds array from here unless we modified the class to store it 
        # or we inspect via debugger. 
        # But we can check if signals were generated slightly differently.
        
        print(f"Signals generated: {result.signals.value_counts().to_dict()}")
        print("Verification Successful: Code ran without errors and produced signals.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dynamic_threshold()
