"""
Training script for Dynamic MOE v3 with HMM + MTF

This combines:
- 4 Expert Models (specialized for each regime)
- HMM Regime Detection
- Multi-Timeframe Confirmation (H4 + D1)
- Walk-forward validation

Usage:
    python train_moe_v3_hmm_mtf.py --timeframe 1h --test-year 2025
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from datetime import datetime
import json
import sys
import io

# Fix Windows encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import joblib

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from algo_trading.ml.dynamic_moe_v3_hmm_mtf import DynamicMOE_v3_HMM_MTF
try:
    from algo_trading.features.h1_features import H1Features
    HAS_H1_FEATURES = True
except ImportError:
    HAS_H1_FEATURES = False

warnings.filterwarnings('ignore')


def build_inline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features inline when H1Features not available."""
    data = df.copy()
    feats = pd.DataFrame(index=data.index)
    
    # EMAs
    feats['ema_9'] = data['close'].ewm(span=9).mean()
    feats['ema_21'] = data['close'].ewm(span=21).mean()
    feats['ema_50'] = data['close'].ewm(span=50).mean()
    feats['ema_200'] = data['close'].ewm(span=200).mean()
    
    # EMA crosses
    feats['ema_9_21'] = (feats['ema_9'] - feats['ema_21']) / feats['ema_21']
    feats['ema_21_50'] = (feats['ema_21'] - feats['ema_50']) / feats['ema_50']
    feats['ema_50_200'] = (feats['ema_50'] - feats['ema_200']) / feats['ema_200']
    
    # Price position
    feats['price_ema50'] = (data['close'] - feats['ema_50']) / feats['ema_50']
    feats['price_ema200'] = (data['close'] - feats['ema_200']) / feats['ema_200']
    
    # Momentum
    feats['mom_6'] = data['close'].pct_change(6)
    feats['mom_12'] = data['close'].pct_change(12)
    feats['mom_24'] = data['close'].pct_change(24)
    feats['mom_48'] = data['close'].pct_change(48)
    
    # RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    feats['rsi'] = 100 - 100 / (1 + rs)
    feats['rsi_14'] = feats['rsi']
    feats['rsi_9'] = 100 - 100 / (1 + delta.where(delta > 0, 0).rolling(9).mean() / 
                                      (-delta.where(delta < 0, 0)).rolling(9).mean() + 1e-10)
    
    # MACD
    ema12 = data['close'].ewm(span=12).mean()
    ema26 = data['close'].ewm(span=26).mean()
    feats['macd'] = ema12 - ema26
    feats['macd_signal'] = feats['macd'].ewm(span=9).mean()
    feats['macd_histogram'] = feats['macd'] - feats['macd_signal']
    feats['macd_hist'] = feats['macd_histogram']
    
    # Bollinger Bands
    bb_mid = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feats['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    feats['bb_width'] = (bb_upper - bb_lower) / bb_mid
    
    # ATR
    high = data['high']
    low = data['low']
    close = data['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    feats['atr'] = tr.rolling(14).mean()
    feats['atr_pct'] = feats['atr'] / close
    
    # ADX
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / (feats['atr'] + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (feats['atr'] + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    feats['adx'] = dx.rolling(14).mean()
    
    # Volume
    feats['vol_ma_20'] = data['volume'].rolling(20).mean()
    feats['vol_ratio'] = data['volume'] / (feats['vol_ma_20'] + 1e-10)
    
    # Range position (20 period)
    h20 = high.rolling(20).max()
    l20 = low.rolling(20).min()
    feats['range_pos_20'] = (close - l20) / (h20 - l20 + 1e-10)
    
    return feats


def load_data(timeframe: str) -> pd.DataFrame:
    """Load OHLCV data."""
    base = Path(__file__).parent.parent / 'data'
    
    if timeframe in ['1h', 'h1', 'H1']:
        path = base / 'okx_1h.csv'
    else:
        path = base / 'okx_15m.csv'
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
    print(f"📊 Loaded {len(df)} bars from {path.name}")
    return df


def build_labels(df: pd.DataFrame, lookahead: int = 8, threshold: float = 0.01) -> pd.Series:
    """Build trading labels based on forward returns."""
    future_ret = df['close'].shift(-lookahead) / df['close'] - 1
    
    labels = pd.Series(0, index=df.index)  # Hold/neutral
    labels[future_ret > threshold] = 1     # Long
    labels[future_ret < -threshold] = -1   # Short
    
    return labels


def walk_forward_backtest(
    model: DynamicMOE_v3_HMM_MTF,
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    labels: pd.Series,
    test_start: str,
    trade_cost: float = 0.0006,
) -> dict:
    """Run walk-forward backtest on test period."""
    test_mask = df.index >= test_start
    
    df_test = df[test_mask].copy()
    features_test = features_df[test_mask].copy()
    X_test = features_test.values
    y_test = labels[test_mask].values
    
    print(f"\n📊 Walk-Forward Backtest: {test_start} onward")
    print(f"   Test samples: {len(X_test)}")
    
    # Get predictions with regime filtering
    signals, confidences, regimes, size_mult = model.predict_with_regime(
        X_test, features_test, df_test
    )
    
    # Calculate actual returns
    actual_returns = df_test['close'].pct_change(8).shift(-8).values  # 8-bar lookahead
    
    results = {
        'total_bars': len(X_test),
        'regimes': {},
    }
    
    # Regime distribution
    for i, name in enumerate(model.REGIME_NAMES):
        count = (regimes == i).sum()
        results['regimes'][name] = {'count': count, 'pct': count / len(regimes) * 100}
    
    # Calculate trading metrics
    trade_signals = (signals != 0)
    n_trades = trade_signals.sum()
    
    if n_trades == 0:
        print("   ⚠️ No trades generated!")
        results['trades'] = 0
        return results
    
    # Trade returns
    trade_returns = np.where(
        signals == 1,
        actual_returns - trade_cost,  # Long
        np.where(signals == -1, -actual_returns - trade_cost, 0)  # Short
    )
    
    valid_mask = ~np.isnan(trade_returns) & trade_signals
    trade_rets = trade_returns[valid_mask]
    
    wins = (trade_rets > 0).sum()
    losses = (trade_rets <= 0).sum()
    
    results.update({
        'trades': n_trades,
        'win_rate': wins / n_trades * 100 if n_trades > 0 else 0,
        'total_return': trade_rets.sum() * 100,
        'profit_factor': (
            trade_rets[trade_rets > 0].sum() / abs(trade_rets[trade_rets < 0].sum())
            if (trade_rets < 0).any() else float('inf')
        ),
        'avg_return_per_trade': trade_rets.mean() * 100 if len(trade_rets) > 0 else 0,
    })
    
    # Per-regime breakdown
    for i, name in enumerate(model.REGIME_NAMES):
        regime_mask = (regimes == i) & valid_mask
        regime_trades = trade_signals & (regimes == i)
        
        if regime_trades.sum() > 0:
            regime_rets = trade_returns[regime_mask]
            regime_wins = (regime_rets > 0).sum()
            
            results['regimes'][name].update({
                'trades': regime_trades.sum(),
                'win_rate': regime_wins / regime_trades.sum() * 100 if regime_trades.sum() > 0 else 0,
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train MOE v3 with HMM + MTF')
    parser.add_argument('--timeframe', '-t', default='1h', help='Timeframe: 1h or 15m')
    parser.add_argument('--test-year', type=int, default=2025, help='Test year for walk-forward')
    parser.add_argument('--lookahead', type=int, default=8, help='Bars lookahead for labels')
    parser.add_argument('--threshold', type=float, default=0.008, help='Return threshold for labels')
    parser.add_argument('--skip-ranging', action='store_true', default=True, help='Skip ranging markets')
    parser.add_argument('--skip-calm', action='store_true', default=False, help='Skip calm markets')
    parser.add_argument('--no-mtf', action='store_true', help='Disable MTF confirmation')
    parser.add_argument('--no-hmm', action='store_true', help='Disable HMM regime')
    parser.add_argument('--save-model', action='store_true', help='Save trained model')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DYNAMIC MOE v3 - HMM + MTF TRAINING")
    print("=" * 70)
    print(f"Timeframe: {args.timeframe}")
    print(f"Test Year: {args.test_year}")
    print(f"HMM Regime: {'ON' if not args.no_hmm else 'OFF'}")
    print(f"MTF Confirm: {'ON' if not args.no_mtf else 'OFF'}")
    print(f"Skip Ranging: {'ON' if args.skip_ranging else 'OFF'}")
    print()
    
    # Load data
    df = load_data(args.timeframe)
    
    # Build features
    print("\n" + "-" * 40)
    print("STEP 1: BUILD FEATURES")
    print("-" * 40)
    
    if HAS_H1_FEATURES:
        feature_builder = H1Features()
        features_df = feature_builder.build_features(df)
    else:
        # Inline features
        features_df = build_inline_features(df)
    
    # Add additional features for HMM
    features_df['rsi'] = features_df.get('rsi_14', features_df.get('rsi_9', 50))
    features_df['macd_hist'] = features_df.get('macd_histogram', 0)
    features_df['bb_width'] = features_df.get('bb_width', 0.02)
    features_df['atr_pct'] = features_df.get('atr_pct', 0.01)
    
    # Fill NaN
    features_df = features_df.ffill().fillna(0)
    df = df.iloc[-len(features_df):]
    
    print(f"Features: {features_df.shape[1]}")
    print(f"Samples: {len(features_df)}")
    
    # Build labels
    labels = build_labels(df, lookahead=args.lookahead, threshold=args.threshold)
    
    # Train/Test split
    test_start = f"{args.test_year}-01-01"
    train_mask = df.index < test_start
    
    X_train = features_df[train_mask].values
    y_train = labels[train_mask].values
    df_train = df[train_mask]
    features_train = features_df[train_mask]
    
    print(f"\nTrain: {train_mask.sum()} samples (until {args.test_year})")
    print(f"Test: {(~train_mask).sum()} samples ({args.test_year} onward)")
    
    # Label distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nLabel distribution (train):")
    for u, c in zip(unique, counts):
        lbl = {-1: 'Short', 0: 'Hold', 1: 'Long'}.get(u, str(u))
        print(f"  {lbl}: {c} ({c/len(y_train)*100:.1f}%)")
    
    # Create model
    print("\n" + "-" * 40)
    print("STEP 2: TRAIN MODEL")
    print("-" * 40)
    
    model = DynamicMOE_v3_HMM_MTF(
        n_experts=4,
        use_hmm=not args.no_hmm,
        use_mtf=not args.no_mtf,
        skip_ranging=args.skip_ranging,
    )
    
    if args.skip_calm:
        model.config['skip_calm'] = True
    
    # Train
    metrics = model.fit(
        X_train, y_train, features_train, df_train
    )
    
    print("\nTraining complete!")
    
    # Walk-forward backtest
    print("\n" + "-" * 40)
    print("STEP 3: WALK-FORWARD BACKTEST")
    print("-" * 40)
    
    results = walk_forward_backtest(
        model, df, features_df, labels, test_start
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Regime Distribution (Test):")
    for name, data in results.get('regimes', {}).items():
        print(f"   {name.capitalize()}: {data['count']} bars ({data['pct']:.1f}%)")
        if 'trades' in data:
            print(f"      Trades: {data['trades']}, WR: {data['win_rate']:.1f}%")
    
    print(f"\n📈 Trading Performance:")
    print(f"   Total Trades: {results.get('trades', 0)}")
    print(f"   Win Rate: {results.get('win_rate', 0):.1f}%")
    print(f"   Profit Factor: {results.get('profit_factor', 0):.2f}")
    print(f"   Total Return: {results.get('total_return', 0):+.1f}%")
    print(f"   Avg Return/Trade: {results.get('avg_return_per_trade', 0):+.3f}%")
    
    # Deployment recommendation
    print("\n" + "-" * 40)
    print("DEPLOYMENT ASSESSMENT")
    print("-" * 40)
    
    pf = results.get('profit_factor', 0)
    wr = results.get('win_rate', 0)
    trades = results.get('trades', 0)
    
    ready = pf > 1.5 and wr > 55 and trades > 20
    
    if ready:
        print("✅ MODEL READY FOR DEPLOYMENT")
        print(f"   - PF {pf:.2f} > 1.5 ✓")
        print(f"   - WR {wr:.1f}% > 55% ✓")
        print(f"   - Trades {trades} > 20 ✓")
    else:
        print("⚠️ MODEL NEEDS IMPROVEMENT")
        if pf <= 1.5:
            print(f"   - PF {pf:.2f} < 1.5 ✗")
        if wr <= 55:
            print(f"   - WR {wr:.1f}% < 55% ✗")
        if trades <= 20:
            print(f"   - Trades {trades} < 20 ✗")
    
    # Save model if requested
    if args.save_model:
        save_path = Path(__file__).parent / 'algo_trading' / 'ml' / 'models'
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_path = save_path / f'moe_v3_{args.timeframe}_{args.test_year}.pkl'
        model.save(str(model_path))
        
        # Save config
        config = {
            'timeframe': args.timeframe,
            'test_year': args.test_year,
            'use_hmm': not args.no_hmm,
            'use_mtf': not args.no_mtf,
            'skip_ranging': args.skip_ranging,
            'results': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in results.items() if k != 'regimes'},
            'trained_at': datetime.now().isoformat(),
        }
        config_path = save_path / f'moe_v3_{args.timeframe}_{args.test_year}_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Model saved to {model_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
