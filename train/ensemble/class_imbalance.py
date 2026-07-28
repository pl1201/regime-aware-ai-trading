"""
Class Imbalance Handling cho ML models.

Tách từ train_regime_ensemble_models_advanced.py

Includes:
- calculate_class_weights: Tính class weights cho imbalanced datasets
- handle_class_imbalance: SMOTE/SMOTEN oversampling
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np

try:
    from imblearn.over_sampling import SMOTE, SMOTEN
    HAS_SMOTE = True
    HAS_SMOTEN = True
except ImportError:
    HAS_SMOTE = False
    HAS_SMOTEN = False


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Tính class weights để dùng trong models."""
    unique_classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(unique_classes)
    
    weights = {}
    for cls, count in zip(unique_classes, counts):
        weights[int(cls)] = total / (n_classes * count)
    
    return weights


def handle_class_imbalance(
    X: np.ndarray, y: np.ndarray, method: str = "smoten"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Xử lý class imbalance với SMOTEN (SMOTE cho multi-class).
    
    Args:
        X: Feature matrix
        y: Labels array
        method: 'smoten' hoặc 'smote'
    
    Returns:
        Tuple (X_resampled, y_resampled)
    """
    if method == "smoten" and HAS_SMOTEN:
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            warnings.warn(
                f"⚠️ SMOTEN bị bỏ qua: chỉ có 1 class (classes={unique_classes})."
            )
            return X, y
        
        class_counts = [int((y == cls).sum()) for cls in unique_classes]
        min_class_count = min(class_counts)
        
        if min_class_count < 2:
            warnings.warn(
                f"⚠️ SMOTEN bị bỏ qua: class nhỏ nhất có <2 samples. "
                f"Class counts: {dict(zip(unique_classes, class_counts))}"
            )
            return X, y
        
        k_neighbors = min(5, min_class_count - 1)
        if k_neighbors <= 0:
            return X, y
        
        try:
            smoten = SMOTEN(random_state=42, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smoten.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            warnings.warn(f"⚠️ Lỗi SMOTEN, bỏ qua: {e}")
            return X, y
    
    elif method == "smote" and HAS_SMOTE:
        y_binary = (y != 0).astype(int)
        unique_classes = np.unique(y_binary)
        if len(unique_classes) < 2:
            return X, y
        
        minority_count = int((y_binary == 1).sum())
        if minority_count < 2:
            return X, y
        
        k_neighbors = min(5, minority_count - 1)
        if k_neighbors <= 0:
            return X, y
        
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_binary_resampled = smote.fit_resample(X, y_binary)
        y_resampled = np.zeros(len(X_resampled))
        signal_indices = np.where(y_binary_resampled == 1)[0]
        if len(signal_indices) > 0:
            original_signals = y[y != 0]
            if len(original_signals) > 0:
                y_resampled[signal_indices[: len(original_signals)]] = original_signals[
                    : len(signal_indices)
                ]
        return X_resampled, y_resampled

    return X, y
