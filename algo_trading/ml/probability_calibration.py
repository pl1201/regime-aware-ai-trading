"""
Probability Calibration Module

Cải thiện probability estimates của models để:
1. Threshold optimization chính xác hơn
2. Uncertainty-aware decision making
3. Better risk management

Methods:
- Isotonic Regression: Non-parametric, more flexible
- Sigmoid (Platt Scaling): Parametric, works well with small data
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
import warnings
from pathlib import Path

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available")


class ProbabilityCalibrator:
    """
    Calibrate probability outputs của classification models

    Sử dụng isotonic regression hoặc sigmoid calibration để:
    - Cải thiện calibration của probabilities
    - Giảm overconfident predictions
    - Better uncertainty estimation
    """

    def __init__(
        self,
        method: str = 'isotonic',  # 'isotonic' or 'sigmoid'
        cv_folds: int = 5,
        min_samples_per_class: int = 50
    ):
        """
        Args:
            method: Calibration method ('isotonic' or 'sigmoid')
            cv_folds: Số folds cho cross-validation
            min_samples_per_class: Minimum samples cần cho mỗi class
        """
        self.method = method
        self.cv_folds = cv_folds
        self.min_samples_per_class = min_samples_per_class
        self.calibrated_models: Dict[str, any] = {}
        self.is_calibrated = False

    def calibrate(
        self,
        model: any,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
        model_name: str = "model"
    ) -> any:
        """
        Calibrate model với calibration data

        Args:
            model: Trained sklearn-compatible classifier
            X_calib: Calibration features
            y_calib: Calibration labels
            model_name: Name của model

        Returns:
            Calibrated model
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for calibration")

        # Check class balance
        class_counts = y_calib.value_counts()
        min_count = class_counts.min()

        if min_count < self.min_samples_per_class:
            warnings.warn(
                f"Class imbalance detected (min={min_count}). "
                f"Consider using stratified calibration."
            )

        # Choose calibration method
        if self.method == 'isotonic':
            # Isotonic regression: non-parametric, more flexible
            base_estimator = None  # Use model directly
            method = 'isotonic'
        else:  # sigmoid
            # Platt scaling: parametric, better for small data
            base_estimator = LogisticRegression()
            method = 'sigmoid'

        # Create calibrated classifier
        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv=self.cv_folds,
            n_jobs=-1
        )

        # Fit calibration
        calibrated_model.fit(X_calib, y_calib)
        self.calibrated_models[model_name] = calibrated_model
        self.is_calibrated = True

        print(f"OK: Calibrated {model_name} with {self.method} method")
        return calibrated_model

    def predict_calibrated_proba(
        self,
        X: pd.DataFrame,
        model_name: str = "model"
    ) -> np.ndarray:
        """
        Get calibrated probabilities

        Args:
            X: Features
            model_name: Name of calibrated model

        Returns:
            Calibrated probabilities
        """
        if model_name not in self.calibrated_models:
            raise ValueError(f"Model '{model_name}' not calibrated")

        return self.calibrated_models[model_name].predict_proba(X)

    def get_calibration_quality(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate calibration quality

        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Name of calibrated model

        Returns:
            Dict với calibration metrics
        """
        from sklearn.metrics import brier_score_loss, log_loss

        # Get calibrated probabilities
        proba = self.predict_calibrated_proba(X_test, model_name)

        # Calculate metrics
        metrics = {}

        # Log loss (lower is better)
        metrics['log_loss'] = log_loss(y_test, proba)

        # Brier score (for binary, lower is better)
        if proba.shape[1] == 2:
            metrics['brier_score'] = brier_score_loss(y_test, proba[:, 1])

        # Expected Calibration Error (ECE)
        metrics['ece'] = self._calculate_ece(proba, y_test)

        # Maximum Calibration Error (MCE)
        metrics['mce'] = self._calculate_mce(proba, y_test)

        return metrics

    def _calculate_ece(
        self,
        proba: np.ndarray,
        y_true: pd.Series,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error

        Args:
            proba: Predicted probabilities
            y_true: True labels
            n_bins: Number of bins

        Returns:
            ECE value
        """
        # Get predicted class and confidence
        predicted_classes = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)
        accuracies = (predicted_classes == y_true.values).astype(float)

        # Bin by confidence
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & \
                     (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

        return ece

    def _calculate_mce(
        self,
        proba: np.ndarray,
        y_true: pd.Series,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Maximum Calibration Error

        Args:
            proba: Predicted probabilities
            y_true: True labels
            n_bins: Number of bins

        Returns:
            MCE value
        """
        predicted_classes = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)
        accuracies = (predicted_classes == y_true.values).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & \
                     (confidences <= bin_boundaries[i + 1])

            if in_bin.sum() > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                mce = max(mce, np.abs(avg_accuracy - avg_confidence))

        return mce


class RegimeAwareCalibrator:
    """
    Regime-aware probability calibration

    Calibrate riêng cho từng regime để:
    - Better calibration trong từng market condition
    - Adapt với regime-specific biases
    """

    def __init__(
        self,
        method: str = 'isotonic',
        cv_folds: int = 3,  # Giảm folds cho mỗi regime
        min_samples_per_regime: int = 100
    ):
        self.method = method
        self.cv_folds = cv_folds
        self.min_samples_per_regime = min_samples_per_regime
        self.calibrators: Dict[int, ProbabilityCalibrator] = {}
        self.regime_mapping: Dict[int, str] = {}

    def calibrate_by_regime(
        self,
        model: any,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: pd.Series,
        model_name: str = "model"
    ) -> Dict[int, any]:
        """
        Calibrate model separately for each regime

        Args:
            model: Trained model
            X: Features
            y: Labels
            regimes: Regime labels
            model_name: Model name

        Returns:
            Dict với calibrated models per regime
        """
        calibrated_by_regime = {}

        for regime_id in regimes.unique():
            # Filter data for this regime
            mask = regimes == regime_id
            X_regime = X[mask]
            y_regime = y[mask]

            # Check minimum samples
            if len(X_regime) < self.min_samples_per_regime:
                warnings.warn(
                    f"Not enough samples for regime {regime_id} "
                    f"(n={len(X_regime)}). Skipping calibration."
                )
                continue

            # Create calibrator for this regime
            calibrator = ProbabilityCalibrator(
                method=self.method,
                cv_folds=self.cv_folds
            )

            # Calibrate
            calibrated_model = calibrator.calibrate(
                model, X_regime, y_regime,
                model_name=f"{model_name}_regime_{regime_id}"
            )

            self.calibrators[regime_id] = calibrator
            calibrated_by_regime[regime_id] = calibrated_model

        return calibrated_by_regime

    def predict_calibrated_proba_regime_aware(
        self,
        X: pd.DataFrame,
        regime_id: int,
        model_name: str = "model"
    ) -> np.ndarray:
        """
        Get calibrated probabilities for specific regime

        Args:
            X: Features
            regime_id: Current regime ID
            model_name: Model name

        Returns:
            Calibrated probabilities
        """
        if regime_id not in self.calibrators:
            raise ValueError(f"No calibrator for regime {regime_id}")

        return self.calibrators[regime_id].predict_calibrated_proba(
            X, f"{model_name}_regime_{regime_id}"
        )


def apply_calibration_to_ensemble(
    models_dict: Dict[str, any],
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    method: str = 'isotonic',
    cv_folds: int = 5
) -> Dict[str, ProbabilityCalibrator]:
    """
    Apply calibration to all models in ensemble

    Args:
        models_dict: Dict với model_name -> model
        X_calib: Calibration features
        y_calib: Calibration labels
        method: Calibration method
        cv_folds: Number of CV folds

    Returns:
        Dict với model_name -> ProbabilityCalibrator
    """
    calibrators = {}

    for model_name, model in models_dict.items():
        calibrator = ProbabilityCalibrator(
            method=method,
            cv_folds=cv_folds
        )

        calibrator.calibrate(model, X_calib, y_calib, model_name)
        calibrators[model_name] = calibrator

    return calibrators
