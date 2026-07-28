"""
H1 Enhanced Model - PROPER Walk-Forward Backtest
No data leakage - train only on past data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load H1 data."""
    data_path = Path(__file__).parent.parent / "data" / "okx_1h.csv"
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize(None)
    return df


def build_features(df):
    """Build comprehensive features."""
    data = df.copy()
    
    # Momentum
    for p in [1, 2, 4, 6, 8, 12, 24, 48]:
        data[f'mom_{p}'] = data['close'].pct_change(p)
    
    # EMAs
    for p in [9, 21, 50, 100, 200]:
        data[f'ema_{p}'] = data['close'].ewm(span=p).mean()
    
    # EMA crosses
    data['ema_9_21'] = (data['ema_9'] - data['ema_21']) / data['close']
    data['ema_21_50'] = (data['ema_21'] - data['ema_50']) / data['close']
    data['ema_50_200'] = (data['ema_50'] - data['ema_200']) / data['close']
    data['price_ema50'] = (data['close'] - data['ema_50']) / data['close']
    data['price_ema200'] = (data['close'] - data['ema_200']) / data['close']
    
    # RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    data['rsi_norm'] = (data['rsi'] - 50) / 50
    
    # ATR
    tr = pd.concat([
        data['high'] - data['low'],
        abs(data['high'] - data['close'].shift()),
        abs(data['low'] - data['close'].shift())
    ], axis=1).max(axis=1)
    data['atr'] = tr.rolling(14).mean()
    data['atr_pct'] = data['atr'] / data['close']
    
    # Bollinger
    data['bb_mid'] = data['close'].rolling(20).mean()
    data['bb_std'] = data['close'].rolling(20).std()
    data['bb_pos'] = (data['close'] - data['bb_mid']) / (2 * data['bb_std'] + 1e-10)
    
    # Volume
    data['vol_ratio'] = data['volume'] / (data['volume'].rolling(20).mean() + 1)
    
    # ADX
    up = data['high'] - data['high'].shift()
    down = data['low'].shift() - data['low']
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_di = pd.Series(plus_dm, index=data.index).rolling(14).mean() / data['atr'] * 100
    minus_di = pd.Series(minus_dm, index=data.index).rolling(14).mean() / data['atr'] * 100
    data['adx'] = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100).rolling(14).mean()
    data['di_diff'] = (plus_di - minus_di) / 100
    
    # Structure
    data['range_pos'] = (data['close'] - data['low'].rolling(24).min()) / \
                        (data['high'].rolling(24).max() - data['low'].rolling(24).min() + 1e-10)
    
    # MTF alignment (within H1 only)
    mom_cols = ['mom_4', 'mom_8', 'mom_12', 'mom_24', 'mom_48']
    data['mtf_bull'] = sum([(data[c] > 0).astype(float) for c in mom_cols]) / len(mom_cols)
    
    # Trend alignment
    data['trend_align'] = (
        (data['ema_9'] > data['ema_21']).astype(float) +
        (data['ema_21'] > data['ema_50']).astype(float) +
        (data['ema_50'] > data['ema_100']).astype(float) +
        (data['close'] > data['ema_200']).astype(float)
    ) / 4
    
    # MTF context from H1 (simulate H4/D1)
    data['h4_mom'] = data['close'].pct_change(4*4)  # 4 H1 bars = 1 H4 bar, 4 H4 bars
    data['d1_mom'] = data['close'].pct_change(24*5)  # 5 days
    data['h4_trend'] = (data['close'].ewm(span=9*4).mean() > data['close'].ewm(span=21*4).mean()).astype(float)
    data['d1_trend'] = (data['close'].ewm(span=9*24).mean() > data['close'].ewm(span=21*24).mean()).astype(float)
    data['mtf_score'] = (data['trend_align'] + data['h4_trend'] + data['d1_trend']) / 3
    
    return data


def detect_regime_simple(data):
    """Simple regime detection (no HMM to avoid leakage)."""
    regime = pd.Series(1, index=data.index)  # Default: ranging
    
    # Trending: ADX > 25 AND clear direction
    trending = (data['adx'] > 25) & (abs(data['di_diff']) > 0.1)
    regime[trending] = 0
    
    # Volatile: high ATR
    atr_q80 = data['atr_pct'].rolling(200, min_periods=50).quantile(0.8)
    volatile = data['atr_pct'] > atr_q80
    regime[volatile & ~trending] = 2
    
    # Calm: low ATR + low ADX
    atr_q20 = data['atr_pct'].rolling(200, min_periods=50).quantile(0.2)
    calm = (data['atr_pct'] < atr_q20) & (data['adx'] < 20)
    regime[calm] = 3
    
    return regime


def main():
    print("=" * 70)
    print("H1 ENHANCED - PROPER WALK-FORWARD BACKTEST")
    print("=" * 70)
    
    # Load and prepare data
    df = load_data()
    print(f"\nLoaded {len(df)} bars: {df.index[0]} to {df.index[-1]}")
    
    data = build_features(df)
    data['regime'] = detect_regime_simple(data)
    
    # Create labels (8-hour holding, 0.8% threshold)
    holding = 8
    threshold = 0.008
    future_ret = data['close'].shift(-holding) / data['close'] - 1
    data['label'] = np.where(future_ret > threshold, 1, np.where(future_ret < -threshold, 2, 0))
    
    data = data.dropna()
    
    # Feature list
    feats = [
        'mom_1', 'mom_4', 'mom_8', 'mom_12', 'mom_24', 'mom_48',
        'ema_9_21', 'ema_21_50', 'ema_50_200',
        'price_ema50', 'price_ema200',
        'rsi_norm', 'atr_pct', 'bb_pos', 'vol_ratio',
        'adx', 'di_diff', 'range_pos',
        'mtf_bull', 'trend_align', 'mtf_score',
        'h4_mom', 'd1_mom', 'h4_trend', 'd1_trend'
    ]
    
    # STRICT train/test split
    train_end = pd.Timestamp('2024-12-31')
    
    train_data = data[data.index < train_end].copy()
    test_data = data[data.index >= pd.Timestamp('2025-01-01')].copy()
    
    print(f"\nTrain: {len(train_data)} bars (ending {train_data.index[-1]})")
    print(f"Test: {len(test_data)} bars (starting {test_data.index[0]})")
    
    # Only train on TRENDING regime
    train_trending = train_data[train_data['regime'] == 0]
    print(f"Training on trending regime only: {len(train_trending)} samples")
    print(f"Train label dist: {train_trending['label'].value_counts().sort_index().to_dict()}")
    
    X_train = train_trending[feats]
    y_train = train_trending['label']
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    print("\nTraining model...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    print(f"Train accuracy: {model.score(X_train_scaled, y_train):.1%}")
    
    # Predict on test
    X_test = test_data[feats]
    X_test_scaled = scaler.transform(X_test)
    
    pred = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)
    
    # Apply filters and backtest
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS (2025)")
    print("=" * 70)
    
    configs = [
        {'min_conf': 0.35, 'adx_min': 20, 'mtf_min': 0.5, 'trend_only': False},
        {'min_conf': 0.40, 'adx_min': 22, 'mtf_min': 0.6, 'trend_only': True},
        {'min_conf': 0.45, 'adx_min': 25, 'mtf_min': 0.65, 'trend_only': True},
        {'min_conf': 0.50, 'adx_min': 25, 'mtf_min': 0.7, 'trend_only': True},
        {'min_conf': 0.55, 'adx_min': 28, 'mtf_min': 0.75, 'trend_only': True},
    ]
    
    for cfg in configs:
        signals = np.where(pred == 1, 1, np.where(pred == 2, -1, 0))
        max_proba = proba.max(axis=1)
        
        # Apply filters
        mask = np.ones(len(signals), dtype=bool)
        
        # Confidence filter
        mask &= max_proba >= cfg['min_conf']
        
        # ADX filter
        mask &= test_data['adx'].values >= cfg['adx_min']
        
        # MTF filter (directional)
        mtf = test_data['mtf_score'].values
        mask &= np.where(pred == 1, mtf >= cfg['mtf_min'], mtf <= (1 - cfg['mtf_min']))
        
        # Regime filter
        if cfg['trend_only']:
            mask &= test_data['regime'].values == 0  # Trending only
        
        # Momentum confirmation
        mtf_bull = test_data['mtf_bull'].values
        mask &= np.where(pred == 1, mtf_bull >= 0.6, mtf_bull <= 0.4)
        
        signals[~mask] = 0
        
        # Calculate returns
        rets = []
        i = 0
        while i < len(signals) - holding:
            if signals[i] == 0:
                i += 1
                continue
            
            entry = test_data['close'].iloc[i]
            exit_p = test_data['close'].iloc[i + holding]
            ret = (exit_p / entry - 1) * signals[i] - 0.001  # 0.1% costs
            rets.append(ret)
            i += holding  # Skip holding period
        
        if len(rets) == 0:
            print(f"Conf≥{cfg['min_conf']:.2f}, ADX≥{cfg['adx_min']}, MTF≥{cfg['mtf_min']:.2f}, Trend={cfg['trend_only']}: No trades")
            continue
        
        rets = np.array(rets)
        wins = rets > 0
        win_sum = rets[wins].sum() if wins.any() else 0
        loss_sum = abs(rets[~wins].sum()) if (~wins).any() else 1e-10
        pf = win_sum / loss_sum
        
        longs = sum(signals == 1)
        shorts = sum(signals == -1)
        
        print(f"Conf≥{cfg['min_conf']:.2f}, ADX≥{cfg['adx_min']}, MTF≥{cfg['mtf_min']:.2f}: "
              f"{len(rets):3d} trades (L:{longs}, S:{shorts}), "
              f"WR={wins.mean():.1%}, PF={pf:.2f}, Ret={rets.sum()*100:+.1f}%")
    
    # Feature importance
    print("\n" + "-" * 70)
    print("TOP 10 FEATURES")
    print("-" * 70)
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    for f, v in imp.head(10).items():
        print(f"  {f}: {v:.3f}")
    
    print("\n" + "=" * 70)
    print("DONE - No data leakage in this test")
    print("=" * 70)


if __name__ == '__main__':
    main()
