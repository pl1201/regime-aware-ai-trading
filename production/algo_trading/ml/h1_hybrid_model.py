"""
H1 Hybrid Trading Model

Combines:
- Trend Following (when ADX > 25)
- Mean Reversion (when ADX <= 25 + RSI extremes)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict


class H1HybridModel:
    """
    Hybrid H1 trading model.
    
    Performance (2025 backtest):
    - Conf >= 45%: 169 trades, WR=55%, PF=1.35, Return=+30.6%
    - Conf >= 50%: 52 trades, WR=50%, PF=1.58, Return=+16.9%
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            model_dir = Path(__file__).parent / 'models'
        self.model_dir = Path(model_dir)
        
        self.trend_model = None
        self.reversion_model = None
        self.config = {
            'adx_threshold': 25,
            'holding_period': 6,
            'min_confidence': 0.45,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
        }
        self.feature_names = None
        self.is_fitted = False
    
    def load(self, prefix: str = '') -> bool:
        """Load trained models."""
        try:
            trend_path = self.model_dir / f'{prefix}trend_model.pkl'
            reversion_path = self.model_dir / f'{prefix}reversion_model.pkl'
            config_path = self.model_dir / f'{prefix}config.pkl'
            
            if trend_path.exists():
                self.trend_model = joblib.load(trend_path)
            if reversion_path.exists():
                self.reversion_model = joblib.load(reversion_path)
            if config_path.exists():
                loaded_config = joblib.load(config_path)
                self.config.update(loaded_config)
                self.feature_names = loaded_config.get('feature_names')
            
            self.is_fitted = self.trend_model is not None
            return self.is_fitted
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trading signals.
        
        Args:
            features: DataFrame with H1 features
        
        Returns:
            signals: 1=long, -1=short, 0=no trade
            confidences: prediction confidence (0-1)
            regimes: 1=trending, 0=ranging
        """
        if not self.is_fitted:
            raise ValueError("Model not loaded. Call load() first.")
        
        n = len(features)
        signals = np.zeros(n)
        confidences = np.zeros(n)
        regimes = (features['adx'] > self.config['adx_threshold']).astype(int).values
        
        # Ensure correct feature order
        if self.feature_names:
            features = features[self.feature_names]
        
        for i in range(n):
            row = features.iloc[[i]]
            regime = regimes[i]
            rsi = features['rsi'].iloc[i] if 'rsi' in features.columns else 50
            
            if regime == 1 and self.trend_model is not None:
                # Trending: use trend model
                pred = self.trend_model.predict(row)[0]
                proba = self.trend_model.predict_proba(row)[0]
                conf = np.max(proba)
                
                if conf >= self.config['min_confidence']:
                    if pred == 1:
                        signals[i] = 1
                    elif pred == 2:
                        signals[i] = -1
                    confidences[i] = conf
                    
            elif regime == 0 and self.reversion_model is not None:
                # Ranging: use reversion model with RSI filter
                pred = self.reversion_model.predict(row)[0]
                proba = self.reversion_model.predict_proba(row)[0]
                conf = np.max(proba)
                
                if conf >= self.config['min_confidence']:
                    # Mean reversion: only trade RSI extremes
                    if pred == 1 and rsi < self.config['rsi_oversold']:
                        signals[i] = 1
                        confidences[i] = conf
                    elif pred == 2 and rsi > self.config['rsi_overbought']:
                        signals[i] = -1
                        confidences[i] = conf
        
        return signals, confidences, regimes
    
    def predict_single(self, features: pd.Series) -> Tuple[int, float, int]:
        """
        Predict for a single bar.
        
        Returns:
            signal: 1=long, -1=short, 0=no trade
            confidence: 0-1
            regime: 1=trending, 0=ranging
        """
        df = pd.DataFrame([features])
        signals, confs, regimes = self.predict(df)
        return int(signals[0]), float(confs[0]), int(regimes[0])
    
    def get_position_size(self, confidence: float, regime: int,
                          base_size: float = 0.02) -> float:
        """
        Calculate position size based on confidence and regime.
        
        Args:
            confidence: Signal confidence (0-1)
            regime: 1=trending, 0=ranging
            base_size: Base position size (2% of capital)
        
        Returns:
            Position size multiplier (0.5x to 1.5x of base)
        """
        # Scale by confidence
        conf_mult = 0.5 + confidence  # 0.5x to 1.5x
        
        # Reduce in ranging regime (more uncertain)
        regime_mult = 1.0 if regime == 1 else 0.8
        
        return base_size * conf_mult * regime_mult


# Singleton instance
_model = None

def get_h1_model(model_dir: Optional[Path] = None) -> H1HybridModel:
    """Get or create H1 model instance."""
    global _model
    if _model is None:
        _model = H1HybridModel(model_dir)
        # Try to load from algo_trading_H1/models
        h1_models = Path(__file__).parent.parent.parent / 'algo_trading_H1' / 'models'
        if h1_models.exists():
            _model.model_dir = h1_models
            _model.load()
    return _model
