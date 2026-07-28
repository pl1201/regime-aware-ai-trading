
import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algo_trading.live.universal_bot import load_config_from_env, create_exchange_client, STRATEGY_MAP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestOKXIntegration")

def main():
    print("=" * 60)
    print("🚀 TEST OKX INTEGRATION WITH REGIME ENSEMBLE STRATEGY")
    print("=" * 60)

    # 1. Load Config
    try:
        config = load_config_from_env()
        print(f"✅ Config loaded:")
        print(f"   Exchange: {config.exchange}")
        print(f"   Symbol: {config.symbol}")
        print(f"   Strategy: {config.strategy_name}")
        print(f"   Params: {json.dumps(config.strategy_params, indent=2)}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return

    # 2. Init Exchange
    print("\n[Connecting to Exchange...]")
    try:
        client = create_exchange_client(config)
        price = client.get_last_price(config.symbol)
        print(f"✅ Exchange Connected. Current {config.symbol} Price: {price}")
        
        # Test get_current_position
        print("\n[Testing Position Detection...]")
        if hasattr(client, "get_current_position"):
            pos = client.get_current_position(config.symbol)
            print(f"✅ get_current_position() works! Size: {pos}")
        else:
            print(f"❌ Error: get_current_position() NOT FOUND in client!")
            
    except Exception as e:
        print(f"❌ Failed to connect exchange: {e}")
        return

    # 3. Fetch Data
    print("\n[Fetching Live Data...]")
    try:
        df = client.get_klines_df(config.symbol, config.interval, limit=300)
        if df.empty:
            print("❌ No data received.")
            return
        print(f"✅ Data received: {len(df)} bars. Last close: {df['close'].iloc[-1]}")
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return

    # 4. Init Strategy
    print("\n[Initializing Strategy...]")
    try:
        strat_class = STRATEGY_MAP.get(config.strategy_name)
        if not strat_class:
            print(f"❌ Strategy {config.strategy_name} not found in map.")
            return
        
        strategy = strat_class(**config.strategy_params)
        print(f"✅ Strategy {strategy.name} initialized.")
        
        # Check active features
        if hasattr(strategy, "use_dynamic_threshold"):
            print(f"   - Dynamic Threshold: {strategy.use_dynamic_threshold}")
        if hasattr(strategy, "use_sequence_features"):
            print(f"   - Sequence Features: {strategy.use_sequence_features}")
        
    except Exception as e:
        print(f"❌ Failed to init strategy: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Generate Signals
    print("\n[Generating Signals...]")
    try:
        result = strategy.generate_signals(df)
        signals = result.signals
        meta = result.meta
        
        last_signal = signals.iloc[-1]
        print(f"✅ Signal Generated!")
        print(f"   Last Signal: {last_signal}")
        print(f"   Current Regime: {meta.get('current_regime')}")
        
        print("\n[Debug Info]")
        # Print some meta keys relevant to new features
        keys = ["proba_threshold", "avg_dynamic_threshold", "signal_count", "using_regime_specific"]
        for k in keys:
             if k in meta:
                 print(f"   {k}: {meta[k]}")
                 
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
