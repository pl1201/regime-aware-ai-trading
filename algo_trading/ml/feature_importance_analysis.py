"""
Feature Importance Analysis Module

Phân tích và loại bỏ các features kém quan trọng để:
- Giảm overfitting
- Tăng tốc độ inference
- Cải thiện khả năng giải thích

Uses permutation importance to measure feature impact.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
from pathlib import Path

try:
    from sklearn.inspection import permutation_importance
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available for feature importance")


class FeatureImportanceAnalyzer:
    """
    Phân tích tầm quan trọng của features sử dụng permutation importance
    """

    def __init__(
        self,
        n_repeats: int = 10,
        random_state: int = 42,
        threshold: float = 0.001  # Loại bỏ features có importance < threshold
    ):
        """
        Args:
            n_repeats: Số lần hoán vị để tính importance
            random_state: Random seed
            threshold: Ngưỡng loại bỏ features (nếu importance < threshold)
        """
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.threshold = threshold
        self.feature_importances_ = None
        self.selected_features_ = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, model: any = None) -> 'FeatureImportanceAnalyzer':
        """
        Fit analyzer trên data và model

        Args:
            X: Features DataFrame
            y: Target labels
            model: Trained model (nếu không có, dùng RandomForest làm baseline)

        Returns:
            Self
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for feature importance")

        # Use RandomForest as baseline if no model provided
        if model is None:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X, y)

        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Store feature importances
        self.feature_importances_ = pd.Series(
            perm_importance.importances_mean,
            index=X.columns
        ).sort_values(ascending=False)

        # Select features above threshold
        self.selected_features_ = self.feature_importances_[
            self.feature_importances_ >= self.threshold
        ].index.tolist()

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filter features to keep only important ones

        Args:
            X: Original features DataFrame

        Returns:
            Filtered DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Call fit() before transform()")

        return X[self.selected_features_]

    def get_importance_summary(self) -> pd.DataFrame:
        """
        Get summary of feature importances

        Returns:
            DataFrame with feature importance stats
        """
        if not self.is_fitted:
            raise ValueError("Call fit() before getting summary")

        summary = pd.DataFrame({
            'feature': self.feature_importances_.index,
            'importance': self.feature_importances_.values,
            'is_selected': self.feature_importances_.values >= self.threshold
        }).sort_values('importance', ascending=False)

        return summary

    def plot_importance(self, top_n: int = 20):
        """
        Plot feature importances (requires matplotlib)

        Args:
            top_n: Number of top features to plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available for plotting")
            return

        if not self.is_fitted:
            raise ValueError("Call fit() before plotting")

        top_features = self.feature_importances_.head(top_n)
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Permutation Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def get_redundant_features(self, correlation_threshold: float = 0.8) -> List[str]:
        """
        Find redundant features based on correlation

        Args:
            correlation_threshold: Correlation threshold to consider features redundant

        Returns:
            List of redundant feature names
        """
        if not self.is_fitted:
            raise ValueError("Call fit() before finding redundant features")

        # Only consider selected features
        X_selected = self.transform(self.feature_importances_.index.to_frame().T)
        if len(X_selected.columns) < 2:
            return []

        # Calculate correlation matrix
        corr_matrix = X_selected.corr().abs()

        # Find highly correlated pairs
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        redundant = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

        return redundant


def analyze_and_filter_features(
    X: pd.DataFrame,
    y: pd.Series,
    model: any = None,
    threshold: float = 0.001,
    correlation_threshold: float = 0.8,
    n_repeats: int = 10
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Convenience function to analyze and filter features

    Args:
        X: Features DataFrame
        y: Target labels
        model: Trained model (optional)
        threshold: Importance threshold
        correlation_threshold: Correlation threshold for redundancy
        n_repeats: Number of repeats for permutation importance

    Returns:
        Tuple of (filtered_features, metadata)
    """
    analyzer = FeatureImportanceAnalyzer(
        threshold=threshold,
        n_repeats=n_repeats
    )
    analyzer.fit(X, y, model)

    # Get filtered features
    X_filtered = analyzer.transform(X)

    # Get redundant features
    redundant_features = analyzer.get_redundant_features(correlation_threshold)

    # Summary
    summary = analyzer.get_importance_summary()
    n_features_before = len(X.columns)
    n_features_after = len(X_filtered.columns)
    n_redundant = len(redundant_features)

    metadata = {
        'n_features_before': n_features_before,
        'n_features_after': n_features_after,
        'n_redundant': n_redundant,
        'redundant_features': redundant_features,
        'importance_summary': summary,
        'selected_features': analyzer.selected_features_
    }

    print(f"Feature importance analysis completed:")
    print(f"  - Features before: {n_features_before}")
    print(f"  - Features after: {n_features_after} ({n_features_before - n_features_after} removed)")
    print(f"  - Redundant features: {n_redundant}")

    return X_filtered, metadata
