"""
H1 Enhanced Trading Model with HMM Regime Detection + MTF Confirmation

Key Features:
1. HMM Regime Detection (trending/ranging/volatile/calm)
2. Multi-Timeframe Confirmation (H1 + H4 + D1 alignment)
3. Only trade in TRENDING regime with high confidence
4. Skip sideway/ranging completely
5. Reduce position size in volatile regime

Performance Target: PF > 1.5, WR > 55%
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import joblib
import warnings

# Try imports
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from algo_trading.market_models.regime import RegimeDetector, detect_regime_hmm
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    warnings.warn("HMM regime detection not available")


class H1EnhancedModel:
    """
    H1 Trading Model with:
    - HMM Regime Detection
    - Multi-Timeframe Confirmation
    - Trend-Only Trading (skip sideway)
    """
    
    # Regime constants
    REGIME_TRENDING = 0
    REGIME_RANGING = 1
    REGIME_VOLATILE = 2
    REGIME_CALM = 3
    
    REGIME_NAMES = ['trending', 'ranging', 'volatile', 'calm']
    
    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            model_dir = Path(__file__).parent / 'models'
        self.model_dir = Path(model_dir)
        
        # Models
        self.trend_model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.regime_detector = None
        
        # Configuration
        self.config = {
            # Regime thresholds
            'min_trend_prob': 0.55,      # Min probability to consider "trending"
            'skip_ranging': True,         # Skip trades in ranging regime
            'skip_volatile': False,       # Trade in volatile but reduce size
            'volatile_size_mult': 0.5,    # Position size multiplier in volatile
            
            # MTF confirmation
            'require_mtf_confirm': True,  # Require H4+D1 alignment
            'min_mtf_score': 0.6,         # Min MTF alignment score (0-1)
            
            # Signal thresholds
            'min_confidence': 0.50,       # Min model confidence
            'holding_period': 8,          # 8 hours
            
            # Entry conditions
            'adx_min': 20,                # Min ADX for trend strength
            'momentum_confirm': True,     # Require momentum alignment
        }
        
        self.feature_names = None
        self.is_fitted = False
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build comprehensive H1 features."""
        data = df.copy()
        
        # === MOMENTUM (multi-scale) ===
        for p in [1, 2, 4, 6, 8, 12, 24, 48]:
            data[f'mom_{p}'] = data['close'].pct_change(p)
        
        # === TREND EMAs ===
        for p in [9, 21, 50, 100, 200]:
            data[f'ema_{p}'] = data['close'].ewm(span=p, adjust=False).mean()
        
        # EMA crosses (normalized)
        data['ema_9_21'] = (data['ema_9'] - data['ema_21']) / data['close']
        data['ema_21_50'] = (data['ema_21'] - data['ema_50']) / data['close']
        data['ema_50_100'] = (data['ema_50'] - data['ema_100']) / data['close']
        data['ema_50_200'] = (data['ema_50'] - data['ema_200']) / data['close']
        
        # Price vs EMAs
        data['price_ema50'] = (data['close'] - data['ema_50']) / data['close']
        data['price_ema100'] = (data['close'] - data['ema_100']) / data['close']
        data['price_ema200'] = (data['close'] - data['ema_200']) / data['close']
        
        # === RSI ===
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        data['rsi_norm'] = (data['rsi'] - 50) / 50
        
        # === VOLATILITY ===
        tr = pd.concat([
            data['high'] - data['low'],
            abs(data['high'] - data['close'].shift()),
            abs(data['low'] - data['close'].shift())
        ], axis=1).max(axis=1)
        data['atr'] = tr.rolling(14).mean()
        data['atr_pct'] = data['atr'] / data['close']
        
        # Bollinger Bands
        data['bb_mid'] = data['close'].rolling(20).mean()
        data['bb_std'] = data['close'].rolling(20).std()
        data['bb_pos'] = (data['close'] - data['bb_mid']) / (2 * data['bb_std'] + 1e-10)
        data['bb_width'] = (data['bb_std'] * 4) / data['bb_mid']
        
        # === VOLUME ===
        data['vol_ratio'] = data['volume'] / (data['volume'].rolling(20).mean() + 1)
        
        # === ADX (trend strength) ===
        up = data['high'] - data['high'].shift()
        down = data['low'].shift() - data['low']
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        plus_di = pd.Series(plus_dm, index=data.index).rolling(14).mean() / data['atr'] * 100
        minus_di = pd.Series(minus_dm, index=data.index).rolling(14).mean() / data['atr'] * 100
        data['adx'] = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100).rolling(14).mean()
        data['di_diff'] = (plus_di - minus_di) / 100
        
        # === STRUCTURE ===
        data['high_24'] = data['high'].rolling(24).max()
        data['low_24'] = data['low'].rolling(24).min()
        data['range_pos'] = (data['close'] - data['low_24']) / (data['high_24'] - data['low_24'] + 1e-10)
        
        # === MTF ALIGNMENT ===
        # Count bullish momentum alignments
        mom_cols = ['mom_4', 'mom_8', 'mom_12', 'mom_24', 'mom_48']
        data['mtf_bull'] = sum([(data[c] > 0).astype(float) for c in mom_cols]) / len(mom_cols)
        
        # EMA trend alignment (0 = bearish, 1 = bullish)
        data['trend_align'] = (
            (data['ema_9'] > data['ema_21']).astype(float) +
            (data['ema_21'] > data['ema_50']).astype(float) +
            (data['ema_50'] > data['ema_100']).astype(float) +
            (data['close'] > data['ema_200']).astype(float)
        ) / 4
        
        return data
    
    def build_mtf_context(self, df_h1: pd.DataFrame, 
                          df_h4: Optional[pd.DataFrame] = None,
                          df_d1: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build Multi-Timeframe context features.
        
        Args:
            df_h1: H1 OHLCV data
            df_h4: H4 OHLCV data (optional, will resample from H1 if None)
            df_d1: D1 OHLCV data (optional, will resample from H1 if None)
        
        Returns:
            DataFrame with MTF context columns added to H1 data
        """
        data = df_h1.copy()
        
        # Resample to H4 if not provided
        if df_h4 is None:
            df_h4 = df_h1.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        # Resample to D1 if not provided
        if df_d1 is None:
            df_d1 = df_h1.resample('1D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        # H4 trend
        h4_ema_fast = df_h4['close'].ewm(span=9).mean()
        h4_ema_slow = df_h4['close'].ewm(span=21).mean()
        h4_trend = (h4_ema_fast > h4_ema_slow).astype(int)
        h4_trend = h4_trend.reindex(data.index, method='ffill')
        data['h4_trend'] = h4_trend
        
        # H4 momentum
        h4_mom = df_h4['close'].pct_change(4)
        h4_mom = h4_mom.reindex(data.index, method='ffill')
        data['h4_mom'] = h4_mom
        
        # D1 trend
        d1_ema_fast = df_d1['close'].ewm(span=9).mean()
        d1_ema_slow = df_d1['close'].ewm(span=21).mean()
        d1_trend = (d1_ema_fast > d1_ema_slow).astype(int)
        d1_trend = d1_trend.reindex(data.index, method='ffill')
        data['d1_trend'] = d1_trend
        
        # D1 momentum
        d1_mom = df_d1['close'].pct_change(5)
        d1_mom = d1_mom.reindex(data.index, method='ffill')
        data['d1_mom'] = d1_mom
        
        # D1 position in range
        d1_high_20 = df_d1['high'].rolling(20).max()
        d1_low_20 = df_d1['low'].rolling(20).min()
        d1_range_pos = (df_d1['close'] - d1_low_20) / (d1_high_20 - d1_low_20 + 1e-10)
        d1_range_pos = d1_range_pos.reindex(data.index, method='ffill')
        data['d1_range_pos'] = d1_range_pos
        
        # MTF alignment score
        # 1.0 = all timeframes bullish aligned
        # 0.0 = all timeframes bearish aligned
        # 0.5 = mixed
        h1_trend = (data['ema_9'] > data['ema_21']).astype(float) if 'ema_9' in data.columns else 0.5
        data['mtf_score'] = (h1_trend + data['h4_trend'] + data['d1_trend']) / 3
        
        return data
    
    def detect_regime(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Detect market regime using HMM.
        
        Returns:
            regime_series: Series with regime IDs (0=trending, 1=ranging, 2=volatile, 3=calm)
            regime_probs: DataFrame with probabilities for each regime
        """
        if HAS_HMM:
            try:
                result = detect_regime_hmm(df, n_regimes=4)
                return result['regime'], result['regime_probabilities']
            except Exception as e:
                warnings.warn(f"HMM failed, using simple detection: {e}")
        
        # Fallback: simple ADX-based detection
        data = self.build_features(df) if 'adx' not in df.columns else df
        
        regime = pd.Series(1, index=df.index)  # Default: ranging
        
        # Trending: ADX > 25
        trending_mask = data['adx'] > 25
        regime[trending_mask] = 0
        
        # Volatile: high ATR
        if 'atr_pct' in data.columns:
            atr_high = data['atr_pct'] > data['atr_pct'].rolling(100).quantile(0.8)
            regime[atr_high & ~trending_mask] = 2
        
        # Calm: low ATR + low ADX
        if 'atr_pct' in data.columns:
            atr_low = data['atr_pct'] < data['atr_pct'].rolling(100).quantile(0.3)
            adx_low = data['adx'] < 20
            regime[atr_low & adx_low] = 3
        
        # Create probability DataFrame (simple)
        probs = pd.DataFrame(0.0, index=df.index, 
                            columns=['prob_trending', 'prob_ranging', 'prob_volatile', 'prob_calm'])
        for i, r in enumerate(self.REGIME_NAMES):
            probs.loc[regime == i, f'prob_{r}'] = 0.8
            probs.loc[regime != i, f'prob_{r}'] = 0.2 / 3
        
        return regime, probs
    
    def create_labels(self, df: pd.DataFrame, threshold: float = 0.008) -> np.ndarray:
        """Create trading labels."""
        holding = self.config['holding_period']
        future_ret = df['close'].shift(-holding) / df['close'] - 1
        
        labels = np.zeros(len(df))
        labels[future_ret > threshold] = 1   # Long
        labels[future_ret < -threshold] = 2  # Short
        
        return labels
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            df: H1 OHLCV data with DatetimeIndex
        
        Returns:
            Training metrics
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for training")
        
        # Build features
        data = self.build_features(df)
        data = self.build_mtf_context(data)
        
        # Detect regime
        regime, regime_probs = self.detect_regime(df)
        data['regime'] = regime
        for col in regime_probs.columns:
            data[col] = regime_probs[col]
        
        # Create labels
        labels = self.create_labels(df)
        data['label'] = labels
        
        # Drop NaN
        data = data.dropna()
        
        # Feature selection
        self.feature_names = [
            'mom_1', 'mom_4', 'mom_8', 'mom_12', 'mom_24', 'mom_48',
            'ema_9_21', 'ema_21_50', 'ema_50_100', 'ema_50_200',
            'price_ema50', 'price_ema100', 'price_ema200',
            'rsi_norm', 'atr_pct', 'bb_pos', 'bb_width',
            'vol_ratio', 'adx', 'di_diff',
            'range_pos', 'mtf_bull', 'trend_align',
            'h4_trend', 'h4_mom', 'd1_trend', 'd1_mom', 'd1_range_pos', 'mtf_score'
        ]
        
        # Filter to only trending regime for training
        # This makes the model specialized for trending markets
        if self.config['skip_ranging']:
            train_mask = data['regime'] == self.REGIME_TRENDING
            print(f"Training on trending regime only: {train_mask.sum()} samples")
        else:
            train_mask = pd.Series(True, index=data.index)
        
        X = data.loc[train_mask, self.feature_names]
        y = data.loc[train_mask, 'label']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.trend_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            min_samples_leaf=50,
            subsample=0.8,
            random_state=42
        )
        self.trend_model.fit(X_scaled, y)
        
        self.is_fitted = True
        
        # Metrics
        train_acc = self.trend_model.score(X_scaled, y)
        
        return {
            'train_samples': len(X),
            'train_accuracy': train_acc,
            'features': len(self.feature_names),
            'label_dist': dict(pd.Series(y).value_counts().sort_index())
        }
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trading signals.
        
        Returns:
            signals: 1=long, -1=short, 0=no trade
            confidences: model confidence (0-1)
            regimes: regime IDs
            size_mult: position size multiplier (0-1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        # Build features
        data = self.build_features(df)
        data = self.build_mtf_context(data)
        
        # Detect regime
        regime, regime_probs = self.detect_regime(df)
        data['regime'] = regime
        
        # Initialize outputs
        n = len(data)
        signals = np.zeros(n)
        confidences = np.zeros(n)
        size_mult = np.ones(n)
        
        # Get available features
        available_feats = [f for f in self.feature_names if f in data.columns]
        if len(available_feats) < len(self.feature_names):
            missing = set(self.feature_names) - set(available_feats)
            for m in missing:
                data[m] = 0.0
        
        X = data[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        proba = self.trend_model.predict_proba(X_scaled)
        pred = self.trend_model.predict(X_scaled)
        
        for i in range(n):
            current_regime = regime.iloc[i]
            conf = np.max(proba[i])
            
            # === REGIME FILTER ===
            # Skip ranging regime completely
            if self.config['skip_ranging'] and current_regime == self.REGIME_RANGING:
                continue
            
            # Skip volatile if configured (or reduce size)
            if current_regime == self.REGIME_VOLATILE:
                if self.config['skip_volatile']:
                    continue
                else:
                    size_mult[i] = self.config['volatile_size_mult']
            
            # === TREND PROBABILITY CHECK ===
            trend_prob = regime_probs['prob_trending'].iloc[i] if 'prob_trending' in regime_probs.columns else 0.5
            if trend_prob < self.config['min_trend_prob']:
                continue
            
            # === MTF CONFIRMATION ===
            if self.config['require_mtf_confirm']:
                mtf_score = data['mtf_score'].iloc[i] if 'mtf_score' in data.columns else 0.5
                
                # For long: need bullish MTF alignment
                if pred[i] == 1 and mtf_score < self.config['min_mtf_score']:
                    continue
                # For short: need bearish MTF alignment
                if pred[i] == 2 and mtf_score > (1 - self.config['min_mtf_score']):
                    continue
            
            # === ADX CHECK ===
            adx = data['adx'].iloc[i] if 'adx' in data.columns else 25
            if adx < self.config['adx_min']:
                continue
            
            # === MOMENTUM CONFIRMATION ===
            if self.config['momentum_confirm']:
                mtf_bull = data['mtf_bull'].iloc[i] if 'mtf_bull' in data.columns else 0.5
                if pred[i] == 1 and mtf_bull < 0.6:  # Long needs bullish momentum
                    continue
                if pred[i] == 2 and mtf_bull > 0.4:  # Short needs bearish momentum
                    continue
            
            # === CONFIDENCE CHECK ===
            if conf < self.config['min_confidence']:
                continue
            
            # === GENERATE SIGNAL ===
            if pred[i] == 1:
                signals[i] = 1
            elif pred[i] == 2:
                signals[i] = -1
            confidences[i] = conf
        
        return signals, confidences, regime.values, size_mult
    
    def save(self, name: str = 'h1_enhanced'):
        """Save model to disk."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.trend_model, self.model_dir / f'{name}_model.pkl')
        joblib.dump(self.scaler, self.model_dir / f'{name}_scaler.pkl')
        joblib.dump({
            'feature_names': self.feature_names,
            'config': self.config
        }, self.model_dir / f'{name}_config.pkl')
        
        print(f"Model saved to {self.model_dir}")
    
    def load(self, name: str = 'h1_enhanced') -> bool:
        """Load model from disk."""
        try:
            model_path = self.model_dir / f'{name}_model.pkl'
            scaler_path = self.model_dir / f'{name}_scaler.pkl'
            config_path = self.model_dir / f'{name}_config.pkl'
            
            if model_path.exists():
                self.trend_model = joblib.load(model_path)
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            if config_path.exists():
                loaded = joblib.load(config_path)
                self.feature_names = loaded.get('feature_names')
                self.config.update(loaded.get('config', {}))
            
            self.is_fitted = self.trend_model is not None
            return self.is_fitted
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def backtest_enhanced(df: pd.DataFrame, model: H1EnhancedModel, 
                      holding: int = 8, costs: float = 0.001) -> Dict[str, Any]:
    """
    Backtest the enhanced model.
    
    Returns:
        Performance metrics
    """
    signals, confs, regimes, size_mult = model.predict(df)
    
    rets = []
    trades = []
    
    i = 0
    while i < len(signals) - holding:
        if signals[i] == 0:
            i += 1
            continue
        
        entry = df['close'].iloc[i]
        exit_p = df['close'].iloc[i + holding]
        direction = signals[i]
        size = size_mult[i]
        
        ret = (exit_p / entry - 1) * direction - costs
        ret *= size  # Apply position sizing
        
        rets.append(ret)
        trades.append({
            'entry_time': df.index[i],
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'regime': model.REGIME_NAMES[int(regimes[i])],
            'confidence': confs[i],
            'size': size,
            'return': ret * 100
        })
        
        i += holding  # Skip holding period
    
    if len(rets) == 0:
        return {'error': 'No trades'}
    
    rets = np.array(rets)
    wins = rets > 0
    
    win_sum = rets[wins].sum() if wins.any() else 0
    loss_sum = abs(rets[~wins].sum()) if (~wins).any() else 1e-10
    
    return {
        'trades': len(rets),
        'win_rate': wins.mean(),
        'profit_factor': win_sum / loss_sum,
        'total_return': rets.sum(),
        'avg_return': rets.mean(),
        'max_drawdown': np.minimum.accumulate(np.cumsum(rets)).min(),
        'trades_detail': trades[:20]  # First 20 trades for inspection
    }
