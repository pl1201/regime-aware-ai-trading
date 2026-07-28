"""
Train and Backtest H1 Enhanced Model

Features:
- HMM Regime Detection (trending/ranging/volatile/calm)
- Multi-Timeframe Confirmation (H1 + H4 + D1)
- Only trade in TRENDING regime
- Skip sideway completely
"""
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Import model
from algo_trading.ml.h1_enhanced_model import H1EnhancedModel, backtest_enhanced


def load_data():
    """Load H1 data."""
    data_path = Path(__file__).parent.parent / "data" / "okx_1h.csv"
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize(None)  # Remove timezone
    
    return df


def main():
    print("=" * 70)
    print("H1 ENHANCED MODEL - HMM REGIME + MTF CONFIRMATION")
    print("=" * 70)
    
    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} bars: {df.index[0]} to {df.index[-1]}")
    
    # Split train/test
    train_end = pd.Timestamp('2024-12-31')
    test_start = pd.Timestamp('2025-01-01')
    
    train_df = df[df.index < train_end]
    test_df = df[df.index >= test_start]
    
    print(f"Train: {len(train_df)} bars ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"Test: {len(test_df)} bars ({test_df.index[0]} to {test_df.index[-1]})")
    
    # Initialize model
    model = H1EnhancedModel()
    
    # Configure for strict trend-only trading
    model.config.update({
        'min_trend_prob': 0.55,      # Need 55% confidence in trending regime
        'skip_ranging': True,         # Skip ranging completely
        'skip_volatile': False,       # Trade in volatile but reduce size
        'volatile_size_mult': 0.5,    # Half position in volatile
        'require_mtf_confirm': True,  # Require H4+D1 alignment
        'min_mtf_score': 0.6,         # 60% MTF alignment needed
        'min_confidence': 0.45,       # Min model confidence
        'adx_min': 22,                # Min ADX
        'momentum_confirm': True,     # Require momentum alignment
        'holding_period': 8,          # 8 hour holding
    })
    
    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)
    
    # Train
    metrics = model.train(train_df)
    print(f"\nTraining completed:")
    print(f"  Samples: {metrics['train_samples']}")
    print(f"  Accuracy: {metrics['train_accuracy']:.1%}")
    print(f"  Features: {metrics['features']}")
    print(f"  Label distribution: {metrics['label_dist']}")
    
    # Backtest on test set
    print("\n" + "-" * 70)
    print("BACKTEST (2025)")
    print("-" * 70)
    
    results = backtest_enhanced(test_df, model, holding=8)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"\nResults:")
    print(f"  Trades: {results['trades']}")
    print(f"  Win Rate: {results['win_rate']:.1%}")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")
    print(f"  Total Return: {results['total_return']*100:.1f}%")
    print(f"  Avg Return/Trade: {results['avg_return']*100:.2f}%")
    print(f"  Max Drawdown: {results['max_drawdown']*100:.1f}%")
    
    # Show regime breakdown
    print("\n" + "-" * 70)
    print("REGIME BREAKDOWN")
    print("-" * 70)
    
    if results.get('trades_detail'):
        trades_df = pd.DataFrame(results['trades_detail'])
        regime_stats = trades_df.groupby('regime').agg({
            'return': ['count', 'mean', lambda x: (x > 0).mean()]
        })
        regime_stats.columns = ['count', 'avg_ret', 'win_rate']
        print(regime_stats)
    
    # Try different configurations
    print("\n" + "-" * 70)
    print("SENSITIVITY ANALYSIS")
    print("-" * 70)
    
    configs = [
        {'min_confidence': 0.40, 'min_mtf_score': 0.5, 'adx_min': 20},
        {'min_confidence': 0.45, 'min_mtf_score': 0.6, 'adx_min': 22},
        {'min_confidence': 0.50, 'min_mtf_score': 0.65, 'adx_min': 25},
        {'min_confidence': 0.55, 'min_mtf_score': 0.7, 'adx_min': 28},
    ]
    
    for cfg in configs:
        model.config.update(cfg)
        res = backtest_enhanced(test_df, model, holding=8)
        
        if 'error' not in res:
            print(f"Conf={cfg['min_confidence']:.2f}, MTF={cfg['min_mtf_score']:.2f}, ADX={cfg['adx_min']}: "
                  f"{res['trades']:3d} trades, WR={res['win_rate']:.1%}, PF={res['profit_factor']:.2f}, "
                  f"Ret={res['total_return']*100:+.1f}%")
        else:
            print(f"Conf={cfg['min_confidence']:.2f}, MTF={cfg['min_mtf_score']:.2f}, ADX={cfg['adx_min']}: No trades")
    
    # Save best model
    print("\n" + "-" * 70)
    print("SAVING MODEL")
    print("-" * 70)
    
    # Use best config
    model.config.update({
        'min_confidence': 0.45,
        'min_mtf_score': 0.6,
        'adx_min': 22,
    })
    
    model.save('h1_enhanced')
    print(f"\nModel saved!")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
