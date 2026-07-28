"""
Specialized Expert for Extreme Market Regimes

Handles special market conditions like:
- High volatility breakouts
- Strong trending periods
- Consolidation phases
- News-driven movements
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union
import warnings


class SpecialRegimeExpert:
    """
    Specialized expert for extreme market regimes
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features=0.5,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features_df: Optional[pd.DataFrame] = None
    ):
        """
        Train the special regime expert

        Args:
            X: Feature matrix
            y: Target labels
            features_df: DataFrame with additional features
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Add special regime features if available
        if features_df is not None:
            enhanced_X = self._add_special_features(X_scaled, features_df)
        else:
            enhanced_X = X_scaled

        # Train model
        self.model.fit(enhanced_X, y)
        self.is_fitted = True

        # Store feature importance
        self.feature_importance_ = self.model.feature_importances_

        return self

    def _add_special_features(
        self,
        X: np.ndarray,
        features_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Add special regime features to enhance predictions

        Args:
            X: Scaled feature matrix
            features_df: DataFrame with additional features

        Returns:
            Enhanced feature matrix
        """
        n_samples = len(X)
        enhanced_features = []

        # Volatility regime indicator
        if 'volatility' in features_df.columns:
            vol = features_df['volatility'].values
            vol_mean = np.mean(vol)
            vol_std = np.std(vol)

            # High volatility indicator
            high_volatility = (vol > (vol_mean + 1.5 * vol_std)).astype(float)
            enhanced_features.append(high_volatility.reshape(-1, 1))

            # Low volatility indicator
            low_volatility = (vol < (vol_mean - 0.5 * vol_std)).astype(float)
            enhanced_features.append(low_volatility.reshape(-1, 1))

        # Trend strength indicator
        if 'trend_1h' in features_df.columns and 'trend_4h' in features_df.columns:
            trend_1h = features_df['trend_1h'].values
            trend_4h = features_df['trend_4h'].values

            # Strong trend indicator (both timeframes agree)
            strong_trend = ((np.sign(trend_1h) == np.sign(trend_4h)) &
                           (np.abs(trend_1h) > 0.3) &
                           (np.abs(trend_4h) > 0.3)).astype(float)
            enhanced_features.append(strong_trend.reshape(-1, 1))

        # Volume confirmation
        if 'volume_ratio_5' in features_df.columns:
            vol_ratio = features_df['volume_ratio_5'].values
            high_volume = (vol_ratio > 2.0).astype(float)
            enhanced_features.append(high_volume.reshape(-1, 1))

        # Combine enhanced features with original
        if enhanced_features:
            enhanced_X = np.hstack([X] + enhanced_features)
            return enhanced_X
        else:
            return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities

        Args:
            X: Feature matrix

        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        try:
            return self.model.predict_proba(X_scaled)
        except Exception as e:
            warnings.warn(f"Special regime expert prediction failed: {e}")
            n_samples = len(X_scaled)
            n_classes = len(self.model.classes_)
            return np.full((n_samples, n_classes), 1.0 / n_classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Feature matrix

        Returns:
            Class predictions
        """
        probs = self.predict_proba(X)
        return self.model.classes_[np.argmax(probs, axis=1)]

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance"""
        return self.feature_importance_ if self.feature_importance_ is not None else np.array([])


# Quick test
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([-1, 0, 1], n_samples)

    # Create features DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    features_df = pd.DataFrame(X, columns=feature_names)
    features_df['volatility'] = np.random.uniform(0.005, 0.04, n_samples)
    features_df['trend_1h'] = np.random.uniform(-1, 1, n_samples)
    features_df['trend_4h'] = np.random.uniform(-1, 1, n_samples)
    features_df['volume_ratio_5'] = np.random.uniform(0.5, 2.5, n_samples)

    # Create and train expert
    expert = SpecialRegimeExpert(random_state=42)
    print("Training Special Regime Expert...")
    expert.fit(X, y, features_df)

    # Test predictions
    predictions = expert.predict(X[:100])
    probabilities = expert.predict_proba(X[:100])

    print(f"Sample predictions: {predictions[:10]}")
    print(f"Sample probabilities shape: {probabilities.shape}")
    print(f"Feature importance shape: {expert.get_feature_importance().shape}")