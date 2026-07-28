"""
Trend Detector Expert - Chuyên phát hiện xu hướng mạnh

Mục tiêu:
- Phát hiện xu hướng mạnh với độ chính xác cao
- Sử dụng Focal Loss để tập trung vào các mẫu khó
- Kết hợp multi-timeframe confirmation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict
import warnings

from .focal_loss import FocalLossOptimizer


class TrendDetectorExpert:
    """
    Expert chuyên biệt cho trending regimes:
    - Multi-timeframe trend confirmation
    - Momentum divergence detection
    - Volume trend confirmation
    - Focal Loss training
    """

    def __init__(self, random_state: int = 42):
        self.model = None
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.class_weights = None

    def _create_model(self, use_focal_loss: bool = True) -> GradientBoostingClassifier:
        """Tạo model với Focal Loss weighting"""
        return GradientBoostingClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=10,
            validation_fraction=0.1,
            n_iter_no_change=15,
            tol=1e-4,
            random_state=self.random_state
        )

    def _calculate_focal_weights(self, y: np.ndarray) -> np.ndarray:
        """Tính sample weights dựa trên Focal Loss"""
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))

        optimizer = FocalLossOptimizer(class_dist)
        class_weights = optimizer.get_class_weights()

        # Map weights to samples
        weights = np.ones(len(y))
        for i, label in enumerate(y):
            if label in class_dist:
                class_idx = list(class_dist.keys()).index(label)
                if class_idx < len(class_weights):
                    weights[i] = class_weights[class_idx]

        return weights

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features_df: Optional[pd.DataFrame] = None,
        use_focal_loss: bool = True
    ):
        """Train Trend Detector với Focal Loss"""
        self.feature_names = features_df.columns.tolist() if features_df is not None else None

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Calculate focal loss weights
        if use_focal_loss:
            sample_weights = self._calculate_focal_weights(y)
        else:
            sample_weights = None

        # Create and train model
        self.model = self._create_model(use_focal_loss)

        if sample_weights is not None:
            self.model.fit(X_scaled, y, sample_weight=sample_weights)
        else:
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

    def get_trend_strength(self, X: np.ndarray) -> np.ndarray:
        """Calculate trend strength score"""
        probs = self.predict_proba(X)
        # Trend strength = max probability - neutral probability
        trend_strength = np.max(probs, axis=1) - probs[:, 1]
        return np.clip(trend_strength, 0, 1)


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
    expert = TrendDetectorExpert()
    expert.fit(X, y, features_df, use_focal_loss=True)

    # Test predictions
    predictions = expert.predict(X[:100])
    probs = expert.predict_proba(X[:100])

    print(f"Sample predictions: {predictions[:10]}")
    print(f"Sample probabilities shape: {probs.shape}")
    print(f"Feature importance: {expert.get_feature_importance()[:5]}")