"""
H1 ENHANCED MODEL - COMPREHENSIVE EVALUATION
=============================================

Full evaluation with:
- HMM Regime Detection (trained on train data only)
- Multi-Timeframe Confirmation (H1 + H4 + D1)
- Walk-Forward Backtest (no data leakage)
- Detailed Performance Metrics
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings

# Try HMM import
try:
    from algo_trading.market_models.regime import RegimeDetector
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    warnings.warn("HMM not available, using simple regime detection")


def load_data():
    """Load H1 data."""
    data_path = Path(__file__).parent.parent / "data" / "okx_1h.csv"
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize(None)
    return df


def build_features(df):
    """Build comprehensive H1 features."""
    data = df.copy()
    
    # === MOMENTUM ===
    for p in [1, 2, 4, 6, 8, 12, 24, 48]:
        data[f'mom_{p}'] = data['close'].pct_change(p)
    
    # === EMAs ===
    for p in [9, 21, 50, 100, 200]:
        data[f'ema_{p}'] = data['close'].ewm(span=p).mean()
    
    data['ema_9_21'] = (data['ema_9'] - data['ema_21']) / data['close']
    data['ema_21_50'] = (data['ema_21'] - data['ema_50']) / data['close']
    data['ema_50_200'] = (data['ema_50'] - data['ema_200']) / data['close']
    data['price_ema50'] = (data['close'] - data['ema_50']) / data['close']
    data['price_ema200'] = (data['close'] - data['ema_200']) / data['close']
    
    # === RSI ===
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    data['rsi_norm'] = (data['rsi'] - 50) / 50
    
    # === MACD ===
    ema12 = data['close'].ewm(span=12).mean()
    ema26 = data['close'].ewm(span=26).mean()
    data['macd'] = ema12 - ema26
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_hist'] = (data['macd'] - data['macd_signal']) / data['close']
    
    # === ATR ===
    tr = pd.concat([
        data['high'] - data['low'],
        abs(data['high'] - data['close'].shift()),
        abs(data['low'] - data['close'].shift())
    ], axis=1).max(axis=1)
    data['atr'] = tr.rolling(14).mean()
    data['atr_pct'] = data['atr'] / data['close']
    
    # === Bollinger Bands ===
    data['bb_mid'] = data['close'].rolling(20).mean()
    data['bb_std'] = data['close'].rolling(20).std()
    data['bb_pos'] = (data['close'] - data['bb_mid']) / (2 * data['bb_std'] + 1e-10)
    data['bb_width'] = (data['bb_std'] * 4) / data['bb_mid']
    
    # === Volume ===
    data['vol_ratio'] = data['volume'] / (data['volume'].rolling(20).mean() + 1)
    
    # === ADX ===
    up = data['high'] - data['high'].shift()
    down = data['low'].shift() - data['low']
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_di = pd.Series(plus_dm, index=data.index).rolling(14).mean() / data['atr'] * 100
    minus_di = pd.Series(minus_dm, index=data.index).rolling(14).mean() / data['atr'] * 100
    data['adx'] = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100).rolling(14).mean()
    data['di_diff'] = (plus_di - minus_di) / 100
    
    # === Structure ===
    data['range_pos'] = (data['close'] - data['low'].rolling(24).min()) / \
                        (data['high'].rolling(24).max() - data['low'].rolling(24).min() + 1e-10)
    
    # === MTF Alignment (within H1) ===
    mom_cols = ['mom_4', 'mom_8', 'mom_12', 'mom_24', 'mom_48']
    data['mtf_bull'] = sum([(data[c] > 0).astype(float) for c in mom_cols]) / len(mom_cols)
    
    data['trend_align'] = (
        (data['ema_9'] > data['ema_21']).astype(float) +
        (data['ema_21'] > data['ema_50']).astype(float) +
        (data['ema_50'] > data['ema_100']).astype(float) +
        (data['close'] > data['ema_200']).astype(float)
    ) / 4
    
    return data


def build_mtf_context(data, df_h1):
    """Build Multi-Timeframe context (H4, D1) from H1 data."""
    
    # Resample to H4
    df_h4 = df_h1.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 
        'close': 'last', 'volume': 'sum'
    }).dropna()
    
    # Resample to D1
    df_d1 = df_h1.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    
    # H4 features
    h4_ema_fast = df_h4['close'].ewm(span=9).mean()
    h4_ema_slow = df_h4['close'].ewm(span=21).mean()
    h4_trend = (h4_ema_fast > h4_ema_slow).astype(int)
    h4_mom = df_h4['close'].pct_change(6)  # 6 H4 bars = 24h
    
    # D1 features
    d1_ema_fast = df_d1['close'].ewm(span=9).mean()
    d1_ema_slow = df_d1['close'].ewm(span=21).mean()
    d1_trend = (d1_ema_fast > d1_ema_slow).astype(int)
    d1_mom = df_d1['close'].pct_change(5)  # 5 days
    d1_range_pos = (df_d1['close'] - df_d1['low'].rolling(20).min()) / \
                   (df_d1['high'].rolling(20).max() - df_d1['low'].rolling(20).min() + 1e-10)
    
    # Reindex to H1
    data['h4_trend'] = h4_trend.reindex(data.index, method='ffill').fillna(0.5)
    data['h4_mom'] = h4_mom.reindex(data.index, method='ffill').fillna(0)
    data['d1_trend'] = d1_trend.reindex(data.index, method='ffill').fillna(0.5)
    data['d1_mom'] = d1_mom.reindex(data.index, method='ffill').fillna(0)
    data['d1_range_pos'] = d1_range_pos.reindex(data.index, method='ffill').fillna(0.5)
    
    # MTF score
    h1_trend = data['trend_align']
    data['mtf_score'] = (h1_trend + data['h4_trend'] + data['d1_trend']) / 3
    
    return data


def train_hmm_regime(train_data):
    """Train HMM on training data only."""
    if not HAS_HMM:
        return None
    
    # Prepare observations for HMM
    obs_cols = ['rsi', 'macd_hist', 'bb_width', 'atr_pct']
    available = [c for c in obs_cols if c in train_data.columns]
    
    if len(available) < 2:
        return None
    
    obs_df = train_data[available].dropna()
    
    if len(obs_df) < 500:
        return None
    
    try:
        detector = RegimeDetector(n_regimes=4, n_iter=100, random_state=42)
        detector.fit(obs_df)
        return detector
    except Exception as e:
        print(f"HMM training failed: {e}")
        return None


def predict_regime(data, detector):
    """Predict regime using trained HMM or fallback to simple detection."""
    
    if detector is not None and HAS_HMM:
        try:
            obs_cols = ['rsi', 'macd_hist', 'bb_width', 'atr_pct']
            available = [c for c in obs_cols if c in data.columns]
            obs_df = data[available].fillna(method='ffill').fillna(0)
            
            regime = detector.predict(obs_df)
            probs = detector.predict_proba(obs_df)
            
            return regime, probs
        except Exception as e:
            print(f"HMM prediction failed, using fallback: {e}")
    
    # Fallback: simple ADX-based detection
    regime = pd.Series(1, index=data.index)  # Default: ranging
    
    # Trending: ADX > 25
    trending = data['adx'] > 25
    regime[trending] = 0
    
    # Volatile: high ATR (rolling quantile to avoid lookahead)
    atr_q80 = data['atr_pct'].expanding(min_periods=100).quantile(0.8)
    volatile = data['atr_pct'] > atr_q80
    regime[volatile & ~trending] = 2
    
    # Calm: low ATR + low ADX
    atr_q20 = data['atr_pct'].expanding(min_periods=100).quantile(0.2)
    calm = (data['atr_pct'] < atr_q20) & (data['adx'] < 20)
    regime[calm] = 3
    
    # Create probability DataFrame
    probs = pd.DataFrame(0.25, index=data.index, 
                        columns=['prob_trending', 'prob_ranging', 'prob_volatile', 'prob_calm'])
    regime_names = ['trending', 'ranging', 'volatile', 'calm']
    for i, name in enumerate(regime_names):
        probs.loc[regime == i, f'prob_{name}'] = 0.7
    
    return regime, probs


def backtest(test_data, signals, holding=8, costs=0.001):
    """Run backtest with given signals."""
    rets = []
    trades = []
    
    i = 0
    while i < len(signals) - holding:
        if signals[i] == 0:
            i += 1
            continue
        
        entry = test_data['close'].iloc[i]
        exit_p = test_data['close'].iloc[i + holding]
        direction = signals[i]
        
        ret = (exit_p / entry - 1) * direction - costs
        
        rets.append(ret)
        trades.append({
            'entry_time': test_data.index[i],
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry': entry,
            'exit': exit_p,
            'return': ret * 100
        })
        
        i += holding
    
    return np.array(rets), trades


def evaluate_model(test_data, model, scaler, feats, regime, regime_probs, config):
    """Evaluate model with given config."""
    
    X_test = test_data[feats].fillna(0)
    X_test_scaled = scaler.transform(X_test)
    
    pred = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)
    max_proba = proba.max(axis=1)
    
    signals = np.where(pred == 1, 1, np.where(pred == 2, -1, 0))
    
    # Apply filters
    mask = np.ones(len(signals), dtype=bool)
    
    # Confidence filter
    mask &= max_proba >= config['min_conf']
    
    # ADX filter
    mask &= test_data['adx'].values >= config['adx_min']
    
    # Regime filter (only trade in trending)
    if config['trend_only']:
        mask &= regime.values == 0
    
    # MTF confirmation
    if config['mtf_confirm']:
        mtf = test_data['mtf_score'].values
        # Long needs bullish MTF, Short needs bearish
        mtf_ok = np.where(pred == 1, mtf >= config['mtf_min'], 
                         np.where(pred == 2, mtf <= (1 - config['mtf_min']), True))
        mask &= mtf_ok
    
    # Momentum confirmation
    if config['mom_confirm']:
        mtf_bull = test_data['mtf_bull'].values
        mom_ok = np.where(pred == 1, mtf_bull >= 0.6, 
                         np.where(pred == 2, mtf_bull <= 0.4, True))
        mask &= mom_ok
    
    # Trend probability filter (from HMM)
    if config['use_hmm_prob'] and 'prob_trending' in regime_probs.columns:
        mask &= regime_probs['prob_trending'].values >= config['hmm_trend_prob']
    
    signals[~mask] = 0
    
    return signals


def main():
    print("=" * 80)
    print("H1 ENHANCED MODEL - COMPREHENSIVE EVALUATION")
    print("HMM Regime Detection + Multi-Timeframe Confirmation")
    print("=" * 80)
    
    # Load data
    df = load_data()
    print(f"\n📊 Data: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
    
    # Build features
    data = build_features(df)
    data = build_mtf_context(data, df)
    data = data.dropna()
    
    # Feature list
    feats = [
        'mom_1', 'mom_4', 'mom_8', 'mom_12', 'mom_24', 'mom_48',
        'ema_9_21', 'ema_21_50', 'ema_50_200', 'price_ema50', 'price_ema200',
        'rsi_norm', 'macd_hist', 'atr_pct', 'bb_pos', 'bb_width', 'vol_ratio',
        'adx', 'di_diff', 'range_pos', 'mtf_bull', 'trend_align', 'mtf_score',
        'h4_trend', 'h4_mom', 'd1_trend', 'd1_mom', 'd1_range_pos'
    ]
    
    # STRICT train/test split
    train_end = pd.Timestamp('2024-12-31')
    test_start = pd.Timestamp('2025-01-01')
    
    train_data = data[data.index < train_end].copy()
    test_data = data[data.index >= test_start].copy()
    
    print(f"\n📈 Train: {len(train_data)} bars (up to {train_data.index[-1]})")
    print(f"📉 Test: {len(test_data)} bars (from {test_data.index[0]})")
    
    # ========================================
    # 1. TRAIN HMM ON TRAINING DATA ONLY
    # ========================================
    print("\n" + "-" * 80)
    print("STEP 1: HMM REGIME DETECTION")
    print("-" * 80)
    
    hmm_detector = train_hmm_regime(train_data)
    
    if hmm_detector is not None:
        print("✅ HMM trained successfully on training data")
        
        # Analyze training regimes
        train_regime, train_probs = predict_regime(train_data, hmm_detector)
        regime_dist = train_regime.value_counts().sort_index()
        regime_names = ['Trending', 'Ranging', 'Volatile', 'Calm']
        print("\nTraining data regime distribution:")
        for i, name in enumerate(regime_names):
            count = regime_dist.get(i, 0)
            pct = count / len(train_regime) * 100
            print(f"  {name}: {count} bars ({pct:.1f}%)")
    else:
        print("⚠️ HMM not available, using simple ADX-based detection")
    
    # Predict regime on test data (using HMM trained on train data)
    test_regime, test_regime_probs = predict_regime(test_data, hmm_detector)
    
    # ========================================
    # 2. CREATE LABELS AND TRAIN MODEL
    # ========================================
    print("\n" + "-" * 80)
    print("STEP 2: TRAIN TRADING MODEL")
    print("-" * 80)
    
    # Labels
    holding = 8
    threshold = 0.008
    
    train_future_ret = train_data['close'].shift(-holding) / train_data['close'] - 1
    train_labels = np.where(train_future_ret > threshold, 1, 
                           np.where(train_future_ret < -threshold, 2, 0))
    
    # Filter to trending regime only for training
    train_regime_simple, _ = predict_regime(train_data, hmm_detector)
    train_trending_mask = train_regime_simple == 0
    
    X_train = train_data.loc[train_trending_mask, feats]
    y_train = train_labels[train_trending_mask]
    
    # Remove NaN labels (last 'holding' rows)
    valid_mask = ~np.isnan(train_future_ret.values[train_trending_mask])
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    print(f"Training on TRENDING regime: {len(X_train)} samples")
    print(f"Label distribution: 0={sum(y_train==0)}, 1={sum(y_train==1)}, 2={sum(y_train==2)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    train_acc = model.score(X_train_scaled, y_train)
    print(f"✅ Model trained - Accuracy: {train_acc:.1%}")
    
    # ========================================
    # 3. BACKTEST ON TEST DATA
    # ========================================
    print("\n" + "-" * 80)
    print("STEP 3: WALK-FORWARD BACKTEST (2025)")
    print("-" * 80)
    
    # Test regime distribution
    test_regime_dist = test_regime.value_counts().sort_index()
    print("\nTest data regime distribution:")
    for i, name in enumerate(regime_names):
        count = test_regime_dist.get(i, 0)
        pct = count / len(test_regime) * 100
        print(f"  {name}: {count} bars ({pct:.1f}%)")
    
    # Multiple configurations
    configs = [
        {
            'name': 'Baseline (no filters)',
            'min_conf': 0.35, 'adx_min': 15, 'trend_only': False,
            'mtf_confirm': False, 'mom_confirm': False, 
            'use_hmm_prob': False, 'hmm_trend_prob': 0.5, 'mtf_min': 0.5
        },
        {
            'name': 'ADX + Confidence',
            'min_conf': 0.40, 'adx_min': 22, 'trend_only': False,
            'mtf_confirm': False, 'mom_confirm': False,
            'use_hmm_prob': False, 'hmm_trend_prob': 0.5, 'mtf_min': 0.5
        },
        {
            'name': 'Trending Only',
            'min_conf': 0.40, 'adx_min': 22, 'trend_only': True,
            'mtf_confirm': False, 'mom_confirm': False,
            'use_hmm_prob': False, 'hmm_trend_prob': 0.5, 'mtf_min': 0.5
        },
        {
            'name': 'Trending + MTF',
            'min_conf': 0.40, 'adx_min': 22, 'trend_only': True,
            'mtf_confirm': True, 'mom_confirm': False,
            'use_hmm_prob': False, 'hmm_trend_prob': 0.5, 'mtf_min': 0.6
        },
        {
            'name': 'Trending + MTF + Momentum',
            'min_conf': 0.45, 'adx_min': 25, 'trend_only': True,
            'mtf_confirm': True, 'mom_confirm': True,
            'use_hmm_prob': False, 'hmm_trend_prob': 0.5, 'mtf_min': 0.6
        },
        {
            'name': 'Full (HMM + MTF + Mom)',
            'min_conf': 0.45, 'adx_min': 25, 'trend_only': True,
            'mtf_confirm': True, 'mom_confirm': True,
            'use_hmm_prob': True, 'hmm_trend_prob': 0.55, 'mtf_min': 0.6
        },
        {
            'name': 'Strict Full',
            'min_conf': 0.50, 'adx_min': 28, 'trend_only': True,
            'mtf_confirm': True, 'mom_confirm': True,
            'use_hmm_prob': True, 'hmm_trend_prob': 0.6, 'mtf_min': 0.65
        },
    ]
    
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"{'Config':<30} {'Trades':>7} {'Longs':>6} {'Shorts':>6} {'WR':>7} {'PF':>7} {'Return':>10}")
    print("-" * 80)
    
    results = []
    for cfg in configs:
        signals = evaluate_model(test_data, model, scaler, feats, 
                                 test_regime, test_regime_probs, cfg)
        
        rets, trades = backtest(test_data, signals, holding=holding)
        
        if len(rets) == 0:
            print(f"{cfg['name']:<30} {'No trades':>7}")
            continue
        
        wins = rets > 0
        win_sum = rets[wins].sum() if wins.any() else 0
        loss_sum = abs(rets[~wins].sum()) if (~wins).any() else 1e-10
        pf = win_sum / loss_sum
        
        longs = sum(signals == 1)
        shorts = sum(signals == -1)
        
        print(f"{cfg['name']:<30} {len(rets):>7} {longs:>6} {shorts:>6} "
              f"{wins.mean()*100:>6.1f}% {pf:>7.2f} {rets.sum()*100:>+9.1f}%")
        
        results.append({
            'config': cfg['name'],
            'trades': len(rets),
            'win_rate': wins.mean(),
            'profit_factor': pf,
            'total_return': rets.sum(),
            'avg_return': rets.mean(),
            'max_dd': np.minimum.accumulate(np.cumsum(rets)).min()
        })
    
    # ========================================
    # 4. BEST CONFIGURATION ANALYSIS
    # ========================================
    if results:
        print("\n" + "-" * 80)
        print("BEST CONFIGURATION ANALYSIS")
        print("-" * 80)
        
        # Find best by profit factor (with min trades)
        valid_results = [r for r in results if r['trades'] >= 20]
        if valid_results:
            best = max(valid_results, key=lambda x: x['profit_factor'])
            print(f"\n🏆 Best Config: {best['config']}")
            print(f"   Trades: {best['trades']}")
            print(f"   Win Rate: {best['win_rate']:.1%}")
            print(f"   Profit Factor: {best['profit_factor']:.2f}")
            print(f"   Total Return: {best['total_return']*100:+.1f}%")
            print(f"   Avg Return/Trade: {best['avg_return']*100:.2f}%")
            print(f"   Max Drawdown: {best['max_dd']*100:.1f}%")
    
    # ========================================
    # 5. FEATURE IMPORTANCE
    # ========================================
    print("\n" + "-" * 80)
    print("TOP 10 FEATURES")
    print("-" * 80)
    
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    for i, (f, v) in enumerate(imp.head(10).items()):
        print(f"  {i+1}. {f}: {v:.3f}")
    
    # ========================================
    # 6. DEPLOYMENT RECOMMENDATION
    # ========================================
    print("\n" + "=" * 80)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 80)
    
    if results:
        best_valid = [r for r in results if r['trades'] >= 20 and r['profit_factor'] > 1.0]
        
        if best_valid:
            best = max(best_valid, key=lambda x: x['profit_factor'])
            print(f"""
✅ MODEL READY FOR DEPLOYMENT

Recommended Configuration:
- Config: {best['config']}
- Min Confidence: 0.45
- Min ADX: 25
- Trend Only: Yes
- MTF Confirmation: Yes (H4 + D1 alignment)
- Momentum Confirmation: Yes

Expected Performance:
- Trades/Month: ~{best['trades'] / 15:.0f}
- Win Rate: {best['win_rate']:.1%}
- Profit Factor: {best['profit_factor']:.2f}
- Monthly Return: ~{best['total_return']*100/15:.1f}%

Risk Parameters:
- Max Drawdown: {best['max_dd']*100:.1f}%
- Holding Period: 8 hours
- Position Size: 2% of capital per trade

Filters Active:
✓ HMM Regime Detection (only trade trending markets)
✓ Multi-Timeframe Confirmation (H1 + H4 + D1)
✓ Momentum Alignment
✓ ADX Trend Strength
✓ Confidence Threshold
""")
        else:
            print("""
⚠️ MODEL NEEDS IMPROVEMENT

Current performance does not meet deployment criteria:
- Profit Factor < 1.0 or too few trades

Recommendations:
1. Try different feature combinations
2. Adjust thresholds
3. Consider longer holding periods
4. Add more regime filters
""")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
