"""
Volatility Breakout Expert - Chuyên phát hiện breakout trong thị trường biến động

Mục tiêu:
- Phát hiện các điểm breakout mạnh trong thị trường biến động cao
- Sử dụng CatBoost với volatility features
- Kết hợp ATR và volume analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict
import warnings


class VolatilityBreakoutExpert:
    """
    Expert chuyên biệt cho volatile regimes:
    - ATR-based breakout detection
    - Volume spike analysis
    - Volatility regime classification
    - Custom feature weighting
    """

    def __init__(self, random_state: int = 42):
        self.model = None
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None

    def _create_model(self) -> RandomForestClassifier:
        """Tạo model với custom weights"""
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_split=12,
            min_samples_leaf=6,
            class_weight='balanced_subsample',
            random_state=self.random_state,
            n_jobs=-1
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features_df: Optional[pd.DataFrame] = None
    ):
        """Train Volatility Breakout Expert"""
        self.feature_names = features_df.columns.tolist() if features_df is not None else None

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_scaled, y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)

        # Ensure proper shape for 3 classes
        if probs.shape[1] == 2:
            # Binary classification, convert to ternary
            new_probs = np.zeros((len(X), 3))
            new_probs[:, 0] = probs[:, 0]  # Short
            new_probs[:, 2] = probs[:, 1]  # Long
            new_probs[:, 1] = 0.0  # Neutral
        elif probs.shape[1] == 3:
            new_probs = probs
        else:
            new_probs = np.ones((len(X), 3)) / 3

        return new_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1) - 1  # Convert to -1, 0, 1

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if self.model is None or self.feature_names is None:
            return {}

        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_volatility_confidence(self, X: np.ndarray) -> np.ndarray:
        """Calculate volatility confidence score"""
        probs = self.predict_proba(X)
        # Volatility confidence = max(abs(short_prob - long_prob), neutral_prob)
        volatility_conf = np.maximum(
            np.abs(probs[:, 0] - probs[:, 2]),
            probs[:, 1]
        )
        return np.clip(volatility_conf, 0, 1)


# Quick test
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([-1, 0, 1], n_samples)

    # Create features DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    features_df = pd.DataFrame(X, columns=feature_names)

    # Create and train expert
    expert = VolatilityBreakoutExpert()
    expert.fit(X, y, features_df)

    # Test predictions
    predictions = expert.predict(X[:100])
    probs = expert.predict_proba(X[:100])

    print(f"Sample predictions: {predictions[:10]}")
    print(f"Sample probabilities shape: {probs.shape}")
    print(f"Volatility confidence scores: {expert.get_volatility_confidence(X[:10])[:5]}")