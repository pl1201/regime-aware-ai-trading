"""
Regime Confidence Score Module

Tính toán độ tin cậy của regime prediction để:
- Tránh giao dịch khi uncertainty cao
- Cải thiện risk management
- Tăng winrate bằng cách filter signals

Uses entropy của probability distribution để tính confidence.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
import warnings

try:
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Install with: pip install scipy")


class RegimeConfidenceScorer:
 

    def __init__(
        self,
        min_confidence_threshold: float = 0.3,
        use_entropy: bool = True,
        use_stability: bool = True
    ):
        """
        Args:
            min_confidence_threshold: Ngưỡng confidence tối thiểu để giao dịch
            use_entropy: Có sử dụng entropy không
            use_stability: Có sử dụng regime stability không
        """
        if not SCIPY_AVAILABLE and use_entropy:
            warnings.warn("scipy not available, entropy-based confidence disabled")
            use_entropy = False

        self.min_confidence_threshold = min_confidence_threshold
        self.use_entropy = use_entropy
        self.use_stability = use_stability

    def calculate_confidence(
        self,
        regime_probabilities: Union[np.ndarray, pd.DataFrame],
        regime_history: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> float:
        """
        Tính confidence score từ regime probabilities

        Args:
            regime_probabilities: [n_regimes] hoặc [n_samples, n_regimes]
            regime_history: Lịch sử regime IDs (optional)

        Returns:
            Confidence score (0-1, cao = confident)
        """
        # Handle single sample
        if isinstance(regime_probabilities, np.ndarray) and regime_probabilities.ndim == 1:
            probs = regime_probabilities
        elif isinstance(regime_probabilities, pd.DataFrame):
            if len(regime_probabilities) == 1:
                probs = regime_probabilities.iloc[0].values
            else:
                # Use last row for multiple samples
                probs = regime_probabilities.iloc[-1].values
        else:
            probs = regime_probabilities

        # Normalize probabilities
        probs = np.array(probs)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            probs = np.ones_like(probs) / len(probs)

        confidence = 1.0

        # 1. Entropy-based confidence
        if self.use_entropy and SCIPY_AVAILABLE:
            # Entropy cao = uncertainty cao
            entropy_val = entropy(probs)

            # Normalize entropy to 0-1 range
            n_regimes = len(probs)
            if n_regimes > 1:
                max_entropy = np.log(n_regimes)
                normalized_entropy = entropy_val / max_entropy if max_entropy > 0 else 0
                # Confidence = 1 - normalized_entropy
                entropy_confidence = 1.0 - normalized_entropy
                confidence *= entropy_confidence

        # 2. Stability-based confidence
        if self.use_stability and regime_history is not None:
            stability_confidence = self._calculate_stability_confidence(regime_history)
            confidence *= stability_confidence

        # Clamp to [0, 1]
        confidence = np.clip(confidence, 0.0, 1.0)
        return float(confidence)

    def _calculate_stability_confidence(
        self,
        regime_history: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Tính confidence dựa trên regime stability

        Args:
            regime_history: Lịch sử regime IDs

        Returns:
            Stability confidence (0-1)
        """
        if len(regime_history) < 2:
            return 1.0

        # Convert to numpy array
        if isinstance(regime_history, pd.Series):
            history = regime_history.values
        else:
            history = np.array(regime_history)

        # Tính số lần regime thay đổi
        changes = np.diff(history) != 0
        change_rate = np.mean(changes)

        # Stability cao = ít thay đổi = confidence cao
        # Stability thấp = nhiều thay đổi = confidence thấp
        stability_confidence = 1.0 - change_rate
        return float(stability_confidence)

    def should_trade(
        self,
        regime_probabilities: Union[np.ndarray, pd.DataFrame],
        regime_history: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> bool:
        """
        Kiểm tra có nên giao dịch không dựa trên confidence

        Args:
            regime_probabilities: Regime probabilities
            regime_history: Lịch sử regime (optional)

        Returns:
            True nếu confidence đủ cao để giao dịch
        """
        confidence = self.calculate_confidence(regime_probabilities, regime_history)
        return confidence >= self.min_confidence_threshold

    def get_confidence_metrics(
        self,
        regime_probabilities: Union[np.ndarray, pd.DataFrame],
        regime_history: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, float]:
        """
        Get detailed confidence metrics

        Args:
            regime_probabilities: Regime probabilities
            regime_history: Lịch sử regime (optional)

        Returns:
            Dict với confidence metrics
        """
        confidence = self.calculate_confidence(regime_probabilities, regime_history)

        metrics = {
            'confidence_score': confidence,
            'should_trade': confidence >= self.min_confidence_threshold,
            'min_threshold': self.min_confidence_threshold
        }

        # Add entropy if available
        if self.use_entropy and SCIPY_AVAILABLE:
            if isinstance(regime_probabilities, pd.DataFrame):
                if len(regime_probabilities) == 1:
                    probs = regime_probabilities.iloc[0].values
                else:
                    probs = regime_probabilities.iloc[-1].values
            else:
                probs = regime_probabilities

            probs = np.array(probs)
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)

            entropy_val = entropy(probs)
            n_regimes = len(probs)
            if n_regimes > 1:
                max_entropy = np.log(n_regimes)
                normalized_entropy = entropy_val / max_entropy if max_entropy > 0 else 0
            else:
                normalized_entropy = 0.0

            metrics['entropy'] = float(entropy_val)
            metrics['normalized_entropy'] = float(normalized_entropy)

        return metrics


def calculate_regime_confidence(
    regime_probabilities: Union[np.ndarray, pd.DataFrame],
    regime_history: Optional[Union[np.ndarray, pd.Series]] = None,
    min_confidence_threshold: float = 0.3
) -> float:
    """
    Convenience function để tính regime confidence

    Args:
        regime_probabilities: Regime probabilities
        regime_history: Lịch sử regime (optional)
        min_confidence_threshold: Ngưỡng confidence tối thiểu

    Returns:
        Confidence score (0-1)
    """
    scorer = RegimeConfidenceScorer(min_confidence_threshold=min_confidence_threshold)
    return scorer.calculate_confidence(regime_probabilities, regime_history)


def should_trade_based_on_confidence(
    regime_probabilities: Union[np.ndarray, pd.DataFrame],
    regime_history: Optional[Union[np.ndarray, pd.Series]] = None,
    min_confidence_threshold: float = 0.3
) -> bool:
    """
    Convenience function để kiểm tra có nên giao dịch không

    Args:
        regime_probabilities: Regime probabilities
        regime_history: Lịch sử regime (optional)
        min_confidence_threshold: Ngưỡng confidence tối thiểu

    Returns:
        True nếu nên giao dịch
    """
    scorer = RegimeConfidenceScorer(min_confidence_threshold=min_confidence_threshold)
    return scorer.should_trade(regime_probabilities, regime_history)