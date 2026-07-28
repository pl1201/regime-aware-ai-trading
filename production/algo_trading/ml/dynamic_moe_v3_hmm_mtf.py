
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Tuple, List, Any
import warnings
import joblib
from pathlib import Path

# Try HMM import from different locations
HAS_HMM = False
RegimeDetector = None

try:
    # Try from algo_trading (main package)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from algo_trading.market_models.regime import RegimeDetector
    HAS_HMM = True
except ImportError:
    try:
        # Try relative import
        from hmmlearn import hmm
        HAS_HMM = True
        
        # Create simple RegimeDetector inline
        class RegimeDetector:
            """Simple HMM-based regime detector."""
            REGIME_NAMES = ['trending', 'ranging', 'volatile', 'calm']
            
            def __init__(self, n_regimes=4, n_iter=100, random_state=42):
                self.n_regimes = n_regimes
                self.model = hmm.GaussianHMM(
                    n_components=n_regimes,
                    covariance_type='diag',
                    n_iter=n_iter,
                    random_state=random_state
                )
                self.scaler = StandardScaler()
                self.is_fitted = False
            
            def fit(self, X):
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled)
                self.is_fitted = True
                return self
            
            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict(X_scaled)
            
            def predict_proba(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict_proba(X_scaled)
    except ImportError:
        warnings.warn("HMM not available - using ADX-based regime detection fallback")

# Check if expert classes exist
HAS_EXPERTS = False
try:
    from .expert_trend_detector import TrendDetectorExpert
    from .expert_range_finder import RangeFinderExpert  
    from .expert_volatility_breakout import VolatilityBreakoutExpert
    from .expert_special_regime import SpecialRegimeExpert
    HAS_EXPERTS = True
except ImportError:
    pass


class DynamicMOE_v3_HMM_MTF:
    """
    Dynamic Mixture of Experts v3 with:
    - HMM Regime Detection (trending/ranging/volatile/calm)
    - Multi-Timeframe Confirmation (H4 + D1)
    - Expert routing based on regime
    - Skip sideway markets
    """
    
    # Regime constants (matching HMM output)
    REGIME_TRENDING = 0
    REGIME_RANGING = 1
    REGIME_VOLATILE = 2
    REGIME_CALM = 3
    
    REGIME_NAMES = ['trending', 'ranging', 'volatile', 'calm']
    
    def __init__(
        self,
        n_experts: int = 4,
        random_state: int = 42,
        use_hmm: bool = True,
        use_mtf: bool = True,
        skip_ranging: bool = True,
    ):
        self.n_experts = n_experts
        self.random_state = random_state
        self.use_hmm = use_hmm and HAS_HMM
        self.use_mtf = use_mtf
        self.skip_ranging = skip_ranging
        
        # Models
        self.experts = []
        self.scaler = StandardScaler()
        self.hmm_detector = None
        
        # Configuration
        self.config = {
            # HMM thresholds
            'min_trend_prob': 0.50,      # Min HMM trending probability
            'skip_ranging': skip_ranging,
            'skip_calm': False,          # Trade calm with reduced size
            
            # MTF confirmation
            'require_mtf': use_mtf,
            'min_mtf_score': 0.6,        # 60% timeframe alignment
            
            # Signal thresholds
            'min_confidence': 0.45,
            'conf_gap': 0.03,            # Min gap between long/short
            
            # Expert routing
            'trending_experts': [0],      # TrendDetector for trending
            'volatile_experts': [2],      # VolatilityBreakout for volatile
            'calm_experts': [0, 3],       # Trend + Special for calm
            
            # Position sizing
            'volatile_size_mult': 0.5,
            'calm_size_mult': 0.8,
        }
        
        self.feature_names = None
        self.mtf_feature_names = None
        self.is_fitted = False
        self.classes_ = np.array([-1, 0, 1])
    
    def _create_experts(self):
        """Create 4 expert models."""
        if HAS_EXPERTS:
            self.experts = [
                TrendDetectorExpert(random_state=self.random_state),
                RangeFinderExpert(random_state=self.random_state + 1),
                VolatilityBreakoutExpert(random_state=self.random_state + 2),
                SpecialRegimeExpert(random_state=self.random_state + 3),
            ]
        else:
            # Fallback: simple GradientBoosting experts
            self.experts = [
                GradientBoostingClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.08,
                    min_samples_leaf=50, random_state=self.random_state + i
                )
                for i in range(self.n_experts)
            ]
    
    def build_mtf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build Multi-Timeframe features from base timeframe."""
        data = df.copy()
        
        # Resample to H4 (for M15: 16 bars, for H1: 4 bars)
        interval_minutes = self._detect_interval(df)
        h4_bars = max(1, 240 // interval_minutes)
        d1_bars = max(1, 1440 // interval_minutes)
        
        # H4 features
        data['h4_close'] = data['close'].rolling(h4_bars).mean()
        data['h4_mom'] = data['close'].pct_change(h4_bars * 6)  # 6 H4 periods
        data['h4_ema_fast'] = data['close'].ewm(span=9 * h4_bars // 4).mean()
        data['h4_ema_slow'] = data['close'].ewm(span=21 * h4_bars // 4).mean()
        data['h4_trend'] = (data['h4_ema_fast'] > data['h4_ema_slow']).astype(float)
        
        # D1 features
        data['d1_close'] = data['close'].rolling(d1_bars).mean()
        data['d1_mom'] = data['close'].pct_change(d1_bars * 5)  # 5 days
        data['d1_ema_fast'] = data['close'].ewm(span=9 * d1_bars // 4).mean()
        data['d1_ema_slow'] = data['close'].ewm(span=21 * d1_bars // 4).mean()
        data['d1_trend'] = (data['d1_ema_fast'] > data['d1_ema_slow']).astype(float)
        
        # D1 range position
        d1_high = data['high'].rolling(d1_bars * 20).max()
        d1_low = data['low'].rolling(d1_bars * 20).min()
        data['d1_range_pos'] = (data['close'] - d1_low) / (d1_high - d1_low + 1e-10)
        
        # Base timeframe trend alignment
        data['ema_9'] = data['close'].ewm(span=9).mean()
        data['ema_21'] = data['close'].ewm(span=21).mean()
        data['ema_50'] = data['close'].ewm(span=50).mean()
        data['base_trend'] = (
            (data['ema_9'] > data['ema_21']).astype(float) +
            (data['ema_21'] > data['ema_50']).astype(float)
        ) / 2
        
        # MTF alignment score (0 = all bearish, 1 = all bullish)
        data['mtf_score'] = (data['base_trend'] + data['h4_trend'] + data['d1_trend']) / 3
        
        self.mtf_feature_names = [
            'h4_mom', 'h4_trend', 'd1_mom', 'd1_trend', 'd1_range_pos', 'mtf_score'
        ]
        
        return data
    
    def _detect_interval(self, df: pd.DataFrame) -> int:
        """Detect data interval in minutes."""
        if len(df) < 2:
            return 60  # Default H1
        
        diff = (df.index[1] - df.index[0]).total_seconds() / 60
        return int(diff)
    
    def detect_regime(self, df: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Detect market regime using HMM or fallback.
        
        Returns:
            regime: Series with regime IDs
            regime_probs: DataFrame with probabilities
        """
        if self.use_hmm and self.hmm_detector is not None:
            try:
                # Use pre-trained HMM
                obs_cols = ['rsi', 'macd_hist', 'bb_width', 'atr_pct']
                available = [c for c in obs_cols if c in features.columns]
                
                if len(available) >= 2:
                    obs_df = features[available].ffill().fillna(0)
                    regime = self.hmm_detector.predict(obs_df)
                    probs = self.hmm_detector.predict_proba(obs_df)
                    return regime, probs
            except Exception as e:
                warnings.warn(f"HMM prediction failed: {e}")
        
        # Fallback: ADX-based detection
        regime = pd.Series(self.REGIME_RANGING, index=df.index)
        
        if 'adx' in features.columns:
            adx = features['adx']
            trending = adx > 25
            regime[trending] = self.REGIME_TRENDING
            
            if 'atr_pct' in features.columns:
                atr = features['atr_pct']
                atr_high = atr > atr.expanding().quantile(0.8)
                regime[atr_high & ~trending] = self.REGIME_VOLATILE
                
                atr_low = atr < atr.expanding().quantile(0.2)
                regime[atr_low & (adx < 20)] = self.REGIME_CALM
        
        # Create probability DataFrame
        probs = pd.DataFrame(0.25, index=df.index,
                            columns=[f'prob_{n}' for n in self.REGIME_NAMES])
        for i, name in enumerate(self.REGIME_NAMES):
            probs.loc[regime == i, f'prob_{name}'] = 0.7
        
        return regime, probs
    
    def train_hmm(self, train_df: pd.DataFrame, train_features: pd.DataFrame):
        """Train HMM on training data only."""
        if not self.use_hmm or not HAS_HMM:
            return
        
        obs_cols = ['rsi', 'macd_hist', 'bb_width', 'atr_pct']
        available = [c for c in obs_cols if c in train_features.columns]
        
        if len(available) < 2:
            warnings.warn("Not enough features for HMM")
            return
        
        obs_df = train_features[available].dropna()
        
        if len(obs_df) < 500:
            warnings.warn("Not enough data for HMM training")
            return
        
        try:
            self.hmm_detector = RegimeDetector(n_regimes=4, n_iter=100, random_state=42)
            self.hmm_detector.fit(obs_df)
            print(f"✅ HMM trained on {len(obs_df)} samples")
        except Exception as e:
            warnings.warn(f"HMM training failed: {e}")
            self.hmm_detector = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features_df: pd.DataFrame,
        df: pd.DataFrame,
        regime_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the MOE model.
        
        Args:
            X: Feature matrix
            y: Target labels (-1, 0, 1)
            features_df: DataFrame with named features
            df: Original OHLCV data
            regime_ids: Pre-computed regime IDs (optional)
        """
        self.feature_names = list(features_df.columns)
        self.classes_ = np.unique(y)
        
        # Build MTF features
        mtf_df = self.build_mtf_features(df)
        for col in self.mtf_feature_names:
            if col in mtf_df.columns and col not in features_df.columns:
                features_df[col] = mtf_df[col]
        
        # Train HMM on training data
        self.train_hmm(df, features_df)
        
        # Detect regime (using trained HMM)
        regime, regime_probs = self.detect_regime(df, features_df)
        
        if regime_ids is None:
            regime_ids = regime.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create experts
        self._create_experts()
        
        # Train each expert on its specialized regime
        metrics = {}
        
        for expert_idx in range(self.n_experts):
            # Get data for this expert's regime
            regime_mask = regime_ids == expert_idx
            
            if regime_mask.sum() < 50:
                # Use all data if insufficient regime samples
                regime_mask = np.ones(len(X), dtype=bool)
            
            X_expert = X_scaled[regime_mask]
            y_expert = y[regime_mask]
            
            if len(X_expert) > 0:
                try:
                    if hasattr(self.experts[expert_idx], 'fit'):
                        self.experts[expert_idx].fit(
                            X_expert, y_expert,
                            features_df.iloc[regime_mask] if hasattr(self.experts[expert_idx], 'fit') else None
                        )
                    else:
                        self.experts[expert_idx].fit(X_expert, y_expert)
                    
                    # Calculate accuracy
                    if hasattr(self.experts[expert_idx], 'score'):
                        acc = self.experts[expert_idx].score(X_expert, y_expert)
                    else:
                        pred = self.experts[expert_idx].predict(X_expert)
                        acc = (pred == y_expert).mean()
                    
                    metrics[f'expert_{expert_idx}_samples'] = len(X_expert)
                    metrics[f'expert_{expert_idx}_accuracy'] = acc
                    print(f"  Expert {expert_idx} ({self.REGIME_NAMES[expert_idx]}): "
                          f"{len(X_expert)} samples, acc={acc:.1%}")
                except Exception as e:
                    warnings.warn(f"Expert {expert_idx} training failed: {e}")
        
        self.is_fitted = True
        
        # Regime distribution
        for i, name in enumerate(self.REGIME_NAMES):
            count = (regime_ids == i).sum()
            metrics[f'regime_{name}_count'] = count
        
        return metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        
        # Average predictions from all experts
        all_probs = []
        for expert in self.experts:
            if hasattr(expert, 'predict_proba'):
                try:
                    probs = expert.predict_proba(X_scaled)
                    all_probs.append(probs)
                except:
                    pass
        
        if len(all_probs) == 0:
            return np.ones((len(X), 3)) / 3
        
        return np.mean(all_probs, axis=0)
    
    def predict_with_regime(
        self,
        X: np.ndarray,
        features_df: pd.DataFrame,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with regime-aware expert routing and MTF confirmation.
        
        Returns:
            signals: 1=long, -1=short, 0=no trade
            confidences: model confidence
            regimes: detected regimes
            size_mult: position size multipliers
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        n = len(X)
        signals = np.zeros(n)
        confidences = np.zeros(n)
        size_mult = np.ones(n)
        
        # Build MTF features
        mtf_df = self.build_mtf_features(df)
        for col in self.mtf_feature_names:
            if col in mtf_df.columns and col not in features_df.columns:
                features_df[col] = mtf_df[col].values
        
        # Detect regime
        regime, regime_probs = self.detect_regime(df, features_df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        for i in range(n):
            current_regime = regime.iloc[i] if hasattr(regime, 'iloc') else regime[i]
            
            # === REGIME FILTER ===
            # Skip ranging if configured
            if self.config['skip_ranging'] and current_regime == self.REGIME_RANGING:
                continue
            
            # Skip calm if configured
            if self.config.get('skip_calm', False) and current_regime == self.REGIME_CALM:
                continue
            
            # === HMM PROBABILITY CHECK ===
            if self.use_hmm and 'prob_trending' in regime_probs.columns:
                trend_prob = regime_probs['prob_trending'].iloc[i]
                if current_regime == self.REGIME_TRENDING and trend_prob < self.config['min_trend_prob']:
                    continue
            
            # === SELECT EXPERT BASED ON REGIME ===
            if current_regime == self.REGIME_TRENDING:
                expert_indices = self.config['trending_experts']
            elif current_regime == self.REGIME_VOLATILE:
                expert_indices = self.config['volatile_experts']
                size_mult[i] = self.config['volatile_size_mult']
            elif current_regime == self.REGIME_CALM:
                expert_indices = self.config['calm_experts']
                size_mult[i] = self.config['calm_size_mult']
            else:
                continue  # Skip unknown regimes
            
            # Get predictions from selected experts
            row = X_scaled[[i]]
            expert_preds = []
            expert_probs = []
            
            for idx in expert_indices:
                if idx < len(self.experts):
                    expert = self.experts[idx]
                    try:
                        pred = expert.predict(row)[0]
                        if hasattr(expert, 'predict_proba'):
                            prob = expert.predict_proba(row)[0]
                        else:
                            prob = np.array([0.33, 0.34, 0.33])
                        expert_preds.append(pred)
                        expert_probs.append(prob)
                    except:
                        pass
            
            if len(expert_preds) == 0:
                continue
            
            # Ensemble expert predictions
            avg_prob = np.mean(expert_probs, axis=0)
            pred = np.argmax(avg_prob)
            conf = np.max(avg_prob)
            
            # Map prediction to signal
            if len(self.classes_) == 3:
                signal = self.classes_[pred]
            else:
                signal = pred - 1  # Assume 0,1,2 -> -1,0,1
            
            # === CONFIDENCE CHECK ===
            if conf < self.config['min_confidence']:
                continue
            
            # === CONFIDENCE GAP CHECK ===
            sorted_probs = np.sort(avg_prob)[::-1]
            if len(sorted_probs) >= 2:
                gap = sorted_probs[0] - sorted_probs[1]
                if gap < self.config['conf_gap']:
                    continue
            
            # === MTF CONFIRMATION ===
            if self.config['require_mtf']:
                mtf_score = features_df['mtf_score'].iloc[i] if 'mtf_score' in features_df.columns else 0.5
                
                # Long needs bullish MTF
                if signal == 1 and mtf_score < self.config['min_mtf_score']:
                    continue
                # Short needs bearish MTF
                if signal == -1 and mtf_score > (1 - self.config['min_mtf_score']):
                    continue
            
            # === GENERATE SIGNAL ===
            signals[i] = signal
            confidences[i] = conf
        
        return signals, confidences, regime.values if hasattr(regime, 'values') else regime, size_mult
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Simple predict (without regime filtering)."""
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        return self.classes_[pred_idx]
    
    def save(self, path: str):
        """Save model to disk."""
        save_dict = {
            'experts': self.experts,
            'scaler': self.scaler,
            'hmm_detector': self.hmm_detector,
            'config': self.config,
            'feature_names': self.feature_names,
            'mtf_feature_names': self.mtf_feature_names,
            'classes_': self.classes_,
            'is_fitted': self.is_fitted,
        }
        joblib.dump(save_dict, path)
        print(f"✅ Model saved to {path}")
    
    def load(self, path: str) -> bool:
        """Load model from disk."""
        try:
            loaded = joblib.load(path)
            self.experts = loaded['experts']
            self.scaler = loaded['scaler']
            self.hmm_detector = loaded.get('hmm_detector')
            self.config = loaded['config']
            self.feature_names = loaded['feature_names']
            self.mtf_feature_names = loaded.get('mtf_feature_names', [])
            self.classes_ = loaded['classes_']
            self.is_fitted = loaded['is_fitted']
            return True
        except Exception as e:
            warnings.warn(f"Failed to load model: {e}")
            return False
