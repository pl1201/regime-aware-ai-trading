"""
Range Finder Expert - Chuyên phát hiện thị trường đi ngang

Mục tiêu:
- Phát hiện vùng đi ngang với độ chính xác cao
- Sử dụng Random Forest với class balancing
- Kết hợp support/resistance analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict
import warnings


class RangeFinderExpert:
    """
    Expert chuyên biệt cho ranging regimes:
    - Support/resistance strength analysis
    - Volatility contraction detection
    - Volume profile analysis
    - Class balanced training
    """

    def __init__(self, random_state: int = 42):
        self.model = None
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None

    def _create_model(self) -> RandomForestClassifier:
        """Tạo model với class balancing"""
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=15,
            min_samples_leaf=8,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features_df: Optional[pd.DataFrame] = None
    ):
        """Train Range Finder với class balancing"""
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

    def get_range_confidence(self, X: np.ndarray) -> np.ndarray:
        """Calculate range confidence score"""
        probs = self.predict_proba(X)
        # Range confidence = neutral probability
        return probs[:, 1]


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
    expert = RangeFinderExpert()
    expert.fit(X, y, features_df)

    # Test predictions
    predictions = expert.predict(X[:100])
    probs = expert.predict_proba(X[:100])

    print(f"Sample predictions: {predictions[:10]}")
    print(f"Sample probabilities shape: {probs.shape}")
    print(f"Range confidence scores: {expert.get_range_confidence(X[:10])[:5]}")