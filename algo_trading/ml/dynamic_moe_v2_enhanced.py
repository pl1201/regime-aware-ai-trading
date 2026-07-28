"""
Enhanced Dynamic Mixture of Experts v2.1 with all improvements integrated

Mục tiêu:
- Tích hợp Signal Quality Filter
- Tích hợp Dynamic Risk Management
- Cải thiện winrate 58-62%
- Giảm drawdown < 25%
- Tăng R:R ratio > 2.0
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Optional, Union, List, Tuple, Dict
import joblib
import warnings

# Local imports
from .signal_quality_filter import SignalQualityFilter, FilterConfig
from .focal_loss import FocalLossOptimizer, create_focal_loss
from ..risk.dynamic_risk_manager import DynamicRiskManager, RiskConfig
from .expert_trend_detector import TrendDetectorExpert
from .expert_range_finder import RangeFinderExpert
from .expert_volatility_breakout import VolatilityBreakoutExpert
from .expert_special_regime import SpecialRegimeExpert  # Thêm expert mới


class DynamicMOE_v2_Enhanced:
    """
    Enhanced Dynamic Mixture of Experts v2.1 with:
    1. Signal quality filtering
    2. Dynamic risk management
    3. Focal Loss training
    4. Regime-aware expert selection
    5. Specialized expert components
    """

    def __init__(
        self,
        n_experts: int = 4,  # Tăng từ 3 lên 4 experts
        random_state: int = 42,
        use_focal_loss: bool = True,
        enable_signal_filter: bool = True,
        enable_risk_management: bool = True
    ):
        self.n_experts = n_experts
        self.random_state = random_state
        self.use_focal_loss = use_focal_loss
        self.enable_signal_filter = enable_signal_filter
        self.enable_risk_management = enable_risk_management

        self.experts = []
        self.gating_network = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

        # Enhanced components
        self.signal_filter = SignalQualityFilter(FilterConfig(
            enable_filter=enable_signal_filter,
            debug=False
        )) if enable_signal_filter else None

        self.risk_manager = DynamicRiskManager(RiskConfig(
            enable_dynamic_sizing=enable_risk_management,
            enable_adaptive_sltp=enable_risk_management
        )) if enable_risk_management else None

        self.confidence_threshold = 0.6
        self.classes_ = np.array([-1, 0, 1], dtype=int)

        # Performance tracking
        self.training_history = []
        self.validation_scores = []

    def _create_experts(self):
        """Create enhanced expert models"""
        self.experts = []

        # Expert 1: Trend Detector (XGBoost with Focal Loss weighting)
        expert1 = TrendDetectorExpert(random_state=self.random_state)

        # Expert 2: Range Finder (Random Forest with class balancing)
        expert2 = RangeFinderExpert(random_state=self.random_state + 1)

        # Expert 3: Volatility Expert (Random Forest with custom weights)
        expert3 = VolatilityBreakoutExpert(random_state=self.random_state + 2)

        # Expert 4: Special Regime Expert (for extreme market conditions)
        expert4 = SpecialRegimeExpert(random_state=self.random_state + 3)

        self.experts = [expert1, expert2, expert3, expert4]

    def _train_expert(
        self,
        expert_idx: int,
        X: np.ndarray,
        y: np.ndarray,
        regime_ids: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        features_df: Optional[pd.DataFrame] = None
    ):
        """Train individual expert with enhanced features"""
        # Filter data for this expert's regime
        if regime_ids is not None and len(regime_ids) == len(X):
            regime_mask = regime_ids == expert_idx
        else:
            # If no regime info or mismatched lengths, use all data
            regime_mask = np.ones(len(X), dtype=bool)

        if np.sum(regime_mask) < 32:  # Need minimum samples
            regime_mask = np.ones(len(X), dtype=bool)  # Use all data if insufficient

        X_expert = X[regime_mask]
        y_expert = y[regime_mask]

        if sample_weights is not None and len(sample_weights) == len(X):
            weights_expert = sample_weights[regime_mask]
        else:
            weights_expert = None

        if len(X_expert) > 0:
            # Train expert model
            if hasattr(self.experts[expert_idx], 'fit'):
                try:
                    if weights_expert is not None:
                        self.experts[expert_idx].fit(X_expert, y_expert, features_df.iloc[regime_mask] if features_df is not None else None)
                    else:
                        self.experts[expert_idx].fit(X_expert, y_expert, features_df.iloc[regime_mask] if features_df is not None else None)
                except Exception as e:
                    warnings.warn(f"Expert {expert_idx} training failed: {e}")
                    # Fallback to simple fit
                    self.experts[expert_idx].fit(X_expert, y_expert)
        else:
            # Fallback training with all data
            if hasattr(self.experts[expert_idx], 'fit'):
                try:
                    if sample_weights is not None:
                        self.experts[expert_idx].fit(X, y, features_df)
                    else:
                        self.experts[expert_idx].fit(X, y, features_df)
                except Exception as e:
                    warnings.warn(f"Expert {expert_idx} fallback training failed: {e}")
                    # Last resort simple fit
                    self.experts[expert_idx].fit(X, y)

    def _calculate_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Calculate sample weights using Focal Loss approach"""
        if not self.use_focal_loss:
            return np.ones(len(y))

        # Calculate class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))

        # Get Focal Loss weights
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
        regime_ids: Optional[np.ndarray] = None,
        features_df: Optional[pd.DataFrame] = None
    ):
        """
        Train the enhanced MOE model

        Args:
            X: Feature matrix
            y: Target labels
            regime_ids: Regime assignments for each sample
            features_df: DataFrame with features for signal filtering
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize regime IDs if not provided
        if regime_ids is None:
            # Deterministic fallback để tái lập kết quả.
            regime_ids = (np.arange(len(X)) % self.n_experts).astype(int)
        else:
            regime_ids = np.asarray(regime_ids, dtype=int)
            regime_ids = np.clip(regime_ids, 0, self.n_experts - 1)
            # Ensure regime_ids matches X length
            if len(regime_ids) != len(X):
                print(f"Warning: regime_ids length ({len(regime_ids)}) doesn't match X length ({len(X)}). Using fallback.")
                regime_ids = (np.arange(len(X)) % self.n_experts).astype(int)

        # Calculate sample weights
        sample_weights = self._calculate_sample_weights(y)

        # Create and train experts
        self._create_experts()
        for i in range(self.n_experts):
            self._train_expert(i, X_scaled, y, regime_ids, sample_weights, features_df)

        # Train gating network từ regimes thật.
        self.gating_network = LogisticRegression(
            max_iter=1500,
            class_weight='balanced',
            random_state=self.random_state,
            C=0.5  # Regularization
        )
        self.gating_network.fit(X_scaled, regime_ids, sample_weight=sample_weights)

        # Validate on holdout set
        if len(X_scaled) > 1000:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=self.random_state
            )

            # Re-train on subset for validation
            val_regime_ids = regime_ids[:len(X_val)] if len(regime_ids) >= len(X_val) else None
            self._fit_validation(X_train, y_train, val_regime_ids)

            # Validate
            val_score = self.score(X_val, y_val)
            self.validation_scores.append(val_score)

            print(f"Validation Score: {val_score:.4f}")

        self.is_fitted = True
        return self

    def _fit_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_ids: Optional[np.ndarray] = None
    ):
        """Fit on validation subset"""
        # Simple re-fit for validation scoring
        # Re-train experts on validation subset
        for i, expert in enumerate(self.experts):
            try:
                # Get regime mask for this expert
                if regime_ids is not None and len(regime_ids) == len(X):
                    regime_mask = (regime_ids == i)
                    if np.sum(regime_mask) > 10:  # Need at least 10 samples
                        X_expert = X[regime_mask]
                        y_expert = y[regime_mask]
                        if len(X_expert) > 0:
                            expert.fit(X_expert, y_expert)
                else:
                    # If no regime info, train on all data
                    expert.fit(X, y)
            except Exception as e:
                print(f"Warning: Failed to re-fit expert {i} on validation set: {e}")
                continue

        # Re-train gating network
        try:
            if regime_ids is not None and len(regime_ids) == len(X):
                self.gating_network.fit(X, regime_ids)
        except Exception as e:
            print(f"Warning: Failed to re-fit gating network on validation set: {e}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities with enhanced filtering

        Args:
            X: Feature matrix

        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Scale features
        X_scaled = self.scaler.transform(X)

        n_samples = len(X_scaled)
        n_classes = len(self.classes_)

        # Expert probs shape: [N, E, C]
        expert_probs = np.zeros((n_samples, self.n_experts, n_classes), dtype=float)
        for i, expert in enumerate(self.experts):
            try:
                probs = expert.predict_proba(X_scaled)
                expert_probs[:, i, :] = probs

                # Normalize if needed
                row_sum = expert_probs[:, i, :].sum(axis=1, keepdims=True)
                zero_mask = row_sum.squeeze() <= 1e-8
                if np.any(zero_mask):
                    expert_probs[zero_mask, i, :] = 1.0 / n_classes
                    row_sum = expert_probs[:, i, :].sum(axis=1, keepdims=True)
                expert_probs[:, i, :] = expert_probs[:, i, :] / np.clip(row_sum, 1e-8, None)
            except Exception as e:
                warnings.warn(f"Expert {i} prediction failed: {e}")
                expert_probs[:, i, :] = 1.0 / n_classes

        # Expert weights từ gating network với improved routing
        try:
            gate_probs = self.gating_network.predict_proba(X_scaled)
            gate_classes = getattr(self.gating_network, 'classes_', np.arange(gate_probs.shape[1]))
            gate_weights = np.zeros((len(X_scaled), self.n_experts), dtype=float)
            for j, cls in enumerate(gate_classes):
                cls_idx = int(np.clip(int(cls), 0, self.n_experts - 1))
                gate_weights[:, cls_idx] += gate_probs[:, j]
            gate_weights = gate_weights / np.clip(gate_weights.sum(axis=1, keepdims=True), 1e-8, None)

            # Apply confidence-based weighting adjustment
            max_gate_probs = np.max(gate_probs, axis=1)
            confidence_adjustment = np.clip(max_gate_probs, 0.5, 1.0)  # Minimum 50% confidence
            gate_weights = gate_weights * confidence_adjustment[:, None]
            gate_weights = gate_weights / np.clip(gate_weights.sum(axis=1, keepdims=True), 1e-8, None)
        except Exception as e:
            warnings.warn(f"Gating network failed: {e}")
            gate_weights = np.full((len(X_scaled), self.n_experts), 1.0 / self.n_experts)

        # Weighted blend theo experts -> [N, C]
        avg_probs = np.sum(expert_probs * gate_weights[:, :, None], axis=1)
        avg_probs = np.clip(avg_probs, 0.0, 1.0)
        avg_probs = avg_probs / np.clip(avg_probs.sum(axis=1, keepdims=True), 1e-8, None)

        return avg_probs

    def predict(
        self,
        X: np.ndarray,
        features_df: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Predict class labels with signal filtering

        Args:
            X: Feature matrix
            features_df: DataFrame with features for filtering

        Returns:
            Class predictions
        """
        probs = self.predict_proba(X)

        # Apply signal quality filter if enabled
        if self.signal_filter is not None and features_df is not None:
            filtered_preds, filtered_probs, filter_stats = self.signal_filter.filter(
                self.classes_[np.argmax(probs, axis=1)],
                probs,
                features_df
            )
            # Return filtered predictions
            return filtered_preds
        else:
            return self.classes_[np.argmax(probs, axis=1)]

    def predict_with_confidence(
        self,
        X: np.ndarray,
        features_df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores

        Args:
            X: Feature matrix
            features_df: DataFrame with features for filtering

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        probs = self.predict_proba(X)
        predictions = self.classes_[np.argmax(probs, axis=1)]
        confidence = np.max(probs, axis=1)

        # Apply signal quality filter if enabled
        if self.signal_filter is not None and features_df is not None:
            filtered_preds, filtered_probs, filter_stats = self.signal_filter.filter(
                predictions,
                probs,
                features_df
            )
            # Return filtered predictions and their confidence
            filtered_confidence = np.max(filtered_probs, axis=1) if len(filtered_probs) > 0 else np.array([])
            return filtered_preds, filtered_confidence

        return predictions, confidence

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score"""
        # Skip scoring during fit process
        if not hasattr(self, 'is_fitted') or not self.is_fitted:
            return 0.0

        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from all experts"""
        importance_dict = {}
        for i, expert in enumerate(self.experts):
            if hasattr(expert, 'get_feature_importance'):
                try:
                    importance_dict[f'expert_{i}'] = expert.get_feature_importance()
                except Exception as e:
                    warnings.warn(f"Could not get feature importance for expert {i}: {e}")
        return importance_dict

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'n_experts': self.n_experts,
            'is_fitted': self.is_fitted,
            'classes': self.classes_.tolist(),
            'validation_scores': self.validation_scores,
            'signal_filter_enabled': self.signal_filter is not None,
            'risk_management_enabled': self.risk_manager is not None,
            'focal_loss_enabled': self.use_focal_loss
        }


def save_moe_v2_enhanced(model: DynamicMOE_v2_Enhanced, filepath: str):
    """Save enhanced MOE v2 model"""
    joblib.dump(model, filepath)


def load_moe_v2_enhanced(filepath: str) -> DynamicMOE_v2_Enhanced:
    """Load enhanced MOE v2 model"""
    loaded = joblib.load(filepath)
    if isinstance(loaded, dict) and 'model' in loaded:
        return loaded['model']
    return loaded


# Quick test
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([-1, 0, 1], n_samples)

    # Create features DataFrame for filtering
    feature_names = [f'feature_{i}' for i in range(n_features)]
    features_df = pd.DataFrame(X, columns=feature_names)
    features_df['trend_1h'] = np.random.uniform(-1, 1, n_samples)
    features_df['trend_4h'] = np.random.uniform(-1, 1, n_samples)
    features_df['volatility'] = np.random.uniform(0.005, 0.04, n_samples)
    features_df['volume_ratio_5'] = np.random.uniform(0.5, 2.5, n_samples)

    # Create and train model
    model = DynamicMOE_v2_Enhanced(
        n_experts=3,
        use_focal_loss=True,
        enable_signal_filter=True,
        enable_risk_management=True
    )

    print("Training enhanced MOE v2...")
    model.fit(X, y, features_df=features_df)

    # Test predictions
    predictions, confidence = model.predict_with_confidence(X[:100], features_df.iloc[:100])

    print(f"Sample predictions: {predictions[:10]}")
    print(f"Sample confidence: {confidence[:10]}")
    print(f"Model info: {model.get_model_info()}")