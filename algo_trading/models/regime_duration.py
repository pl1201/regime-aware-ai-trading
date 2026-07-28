"""
Regime Duration Modeling Module

Dự đoán khi nào regime sẽ kết thúc để:
- Timing exit tốt hơn
- Tránh entry cuối regime
- Dự đoán regime change
- Tối ưu SL/TP
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings


class RegimeDurationModel:
    """
    Model dự đoán regime duration và timing
    """

    def __init__(
        self,
        regime_names: List[str] = None,
        min_duration_threshold: float = 1.5,
        max_duration_threshold: float = 3.0
    ):
        """
        Args:
            regime_names: Tên các regime (default: ['trending', 'ranging', 'volatile', 'calm'])
            min_duration_threshold: Ngưỡng tối thiểu để cảnh báo regime sắp kết thúc
            max_duration_threshold: Ngưỡng tối đa để cảnh báo regime có thể kéo dài
        """
        self.regime_names = regime_names or ['trending', 'ranging', 'volatile', 'calm']
        self.min_duration_threshold = min_duration_threshold
        self.max_duration_threshold = max_duration_threshold
        self.duration_stats = {}  # Stats cho từng regime
        self.is_fitted = False

    def fit(self, regime_series: pd.Series) -> 'RegimeDurationModel':
        """
        Fit model với regime history

        Args:
            regime_series: Series với regime IDs (0, 1, 2, 3)

        Returns:
            Self
        """
        if len(regime_series) < 10:
            warnings.warn("Not enough data to fit regime duration model")
            return self

        # Calculate duration stats for each regime
        durations = self._calculate_regime_durations(regime_series)

        for regime_id in range(len(self.regime_names)):
            regime_durations = durations.get(regime_id, [])
            if len(regime_durations) > 0:
                self.duration_stats[regime_id] = {
                    'mean': np.mean(regime_durations),
                    'std': np.std(regime_durations),
                    'median': np.median(regime_durations),
                    'min': np.min(regime_durations),
                    'max': np.max(regime_durations),
                    'count': len(regime_durations)
                }
            else:
                # Default values nếu không có data
                self.duration_stats[regime_id] = {
                    'mean': 50.0,
                    'std': 20.0,
                    'median': 45.0,
                    'min': 10.0,
                    'max': 100.0,
                    'count': 0
                }

        self.is_fitted = True
        return self

    def _calculate_regime_durations(self, regime_series: pd.Series) -> Dict[int, List[int]]:
        """
        Calculate duration cho từng regime

        Args:
            regime_series: Series với regime IDs

        Returns:
            Dict với regime_id -> list of durations
        """
        durations = {i: [] for i in range(len(self.regime_names))}
        current_regime = None
        current_duration = 0

        for regime in regime_series:
            if regime == current_regime:
                current_duration += 1
            else:
                if current_regime is not None and current_duration > 0:
                    durations[current_regime].append(current_duration)
                current_regime = regime
                current_duration = 1

        # Add last duration
        if current_regime is not None and current_duration > 0:
            durations[current_regime].append(current_duration)

        return durations

    def get_current_duration(self, regime_history: List[int], current_regime: int) -> int:
        """
        Get current duration của regime hiện tại

        Args:
            regime_history: Lịch sử regime IDs
            current_regime: Regime hiện tại

        Returns:
            Current duration
        """
        if not regime_history:
            return 0

        duration = 0
        # Đếm ngược từ cuối
        for regime in reversed(regime_history):
            if regime == current_regime:
                duration += 1
            else:
                break
        return duration

    def get_expected_duration(self, regime_id: int) -> float:
        """
        Get expected duration cho regime

        Args:
            regime_id: Regime ID

        Returns:
            Expected duration (mean)
        """
        if not self.is_fitted or regime_id not in self.duration_stats:
            return 50.0  # Default

        return self.duration_stats[regime_id]['mean']

    def get_duration_confidence(self, regime_id: int, current_duration: int) -> float:
        """
        Get confidence score dựa trên duration

        Args:
            regime_id: Regime ID
            current_duration: Current duration

        Returns:
            Confidence score (0-1)
        """
        if not self.is_fitted or regime_id not in self.duration_stats:
            return 0.5

        stats = self.duration_stats[regime_id]
        mean_duration = stats['mean']

        if mean_duration <= 0:
            return 0.5

        # Normalize current duration
        normalized_duration = current_duration / mean_duration

        # Confidence thấp khi duration gần hết hoặc quá dài
        if normalized_duration < 0.5:
            # Regime mới bắt đầu - confidence cao
            confidence = 0.8
        elif normalized_duration < 0.8:
            # Regime ổn định - confidence cao
            confidence = 0.9
        elif normalized_duration < 1.2:
            # Regime gần hết - confidence giảm
            confidence = 1.0 - (normalized_duration - 0.8) / 0.4
        else:
            # Regime kéo dài quá lâu - confidence thấp
            confidence = max(0.2, 1.0 - (normalized_duration - 1.2) * 0.3)

        return np.clip(confidence, 0.1, 1.0)

    def get_regime_change_probability(self, regime_id: int, current_duration: int) -> float:
        """
        Get probability regime sẽ thay đổi

        Args:
            regime_id: Regime ID
            current_duration: Current duration

        Returns:
            Probability (0-1)
        """
        if not self.is_fitted or regime_id not in self.duration_stats:
            return 0.1

        stats = self.duration_stats[regime_id]
        mean_duration = stats['mean']

        if mean_duration <= 0:
            return 0.1

        normalized_duration = current_duration / mean_duration

        # Probability tăng khi duration tăng
        if normalized_duration < 0.5:
            prob = 0.05  # Rất thấp
        elif normalized_duration < 0.8:
            prob = 0.1   # Thấp
        elif normalized_duration < 1.0:
            prob = 0.3   # Trung bình
        elif normalized_duration < 1.5:
            prob = 0.6   # Cao
        else:
            prob = min(0.9, 0.5 + (normalized_duration - 1.5) * 0.2)  # Rất cao

        return np.clip(prob, 0.05, 0.95)

    def get_duration_features(
        self,
        regime_history: List[int],
        current_regime: int
    ) -> Dict[str, float]:
        """
        Get tất cả duration features

        Args:
            regime_history: Lịch sử regime IDs
            current_regime: Regime hiện tại

        Returns:
            Dict với duration features
        """
        current_duration = self.get_current_duration(regime_history, current_regime)
        expected_duration = self.get_expected_duration(current_regime)
        confidence = self.get_duration_confidence(current_regime, current_duration)
        change_prob = self.get_regime_change_probability(current_regime, current_duration)

        # Duration ratio
        duration_ratio = current_duration / expected_duration if expected_duration > 0 else 1.0

        # Regime status
        is_early = duration_ratio < 0.5
        is_mature = 0.5 <= duration_ratio < 1.0
        is_late = duration_ratio >= 1.0
        is_very_late = duration_ratio >= 1.5

        features = {
            'current_duration': current_duration,
            'expected_duration': expected_duration,
            'duration_ratio': duration_ratio,
            'duration_confidence': confidence,
            'regime_change_probability': change_prob,
            'is_early_regime': is_early,
            'is_mature_regime': is_mature,
            'is_late_regime': is_late,
            'is_very_late_regime': is_very_late,
            'should_reduce_risk': is_late or is_very_late,
            'should_avoid_entry': is_very_late,
            'should_exit_early': is_very_late,
            'should_increase_tp': is_early,
            'should_reduce_sl': is_late
        }

        return features

    def get_trading_recommendations(
        self,
        regime_history: List[int],
        current_regime: int
    ) -> Dict[str, any]:
        """
        Get trading recommendations dựa trên regime duration

        Args:
            regime_history: Lịch sử regime IDs
            current_regime: Regime hiện tại

        Returns:
            Dict với recommendations
        """
        features = self.get_duration_features(regime_history, current_regime)
        regime_name = self.regime_names[current_regime] if current_regime < len(self.regime_names) else f"regime_{current_regime}"

        recommendations = {
            'regime_name': regime_name,
            'current_duration': features['current_duration'],
            'expected_duration': features['expected_duration'],
            'duration_ratio': features['duration_ratio'],
            'confidence': features['duration_confidence'],
            'change_probability': features['regime_change_probability']
        }

        # Entry recommendations
        if features['is_very_late_regime']:
            recommendations['entry'] = 'avoid'  # Tránh vào lệnh
            recommendations['entry_reason'] = f'Regime {regime_name} da keo dai qua lau ({features["duration_ratio"]:.1f}x expected)'
        elif features['is_early_regime']:
            recommendations['entry'] = 'favorable'  # Thuận lợi để vào lệnh
            recommendations['entry_reason'] = f'Regime {regime_name} moi bat dau ({features["current_duration"]} bars)'
        else:
            recommendations['entry'] = 'neutral'
            recommendations['entry_reason'] = f'Regime {regime_name} ổn định ({features["current_duration"]} bars)'

        # Exit recommendations
        if features['is_very_late_regime']:
            recommendations['exit'] = 'early'  # Thoát sớm
            recommendations['exit_reason'] = f'Regime {regime_name} sap ket thuc (change probability: {features["change_probability"]:.1%})'
        elif features['is_late_regime']:
            recommendations['exit'] = 'cautious'  # Thoát thận trọng
            recommendations['exit_reason'] = f'Regime {regime_name} gan het (duration ratio: {features["duration_ratio"]:.1f})'
        else:
            recommendations['exit'] = 'normal'
            recommendations['exit_reason'] = f'Regime {regime_name} con on dinh'

        # Risk management
        if features['is_very_late_regime']:
            recommendations['risk'] = 'reduce'  # Giảm rủi ro
            recommendations['risk_reason'] = 'Regime sap ket thuc'
        elif features['is_late_regime']:
            recommendations['risk'] = 'cautious'
            recommendations['risk_reason'] = 'Regime gan het'
        else:
            recommendations['risk'] = 'normal'
            recommendations['risk_reason'] = 'Regime on dinh'

        return recommendations


def create_regime_duration_model(
    regime_names: List[str] = None,
    min_duration_threshold: float = 1.5,
    max_duration_threshold: float = 3.0
) -> RegimeDurationModel:
    """
    Convenience function để tạo regime duration model

    Args:
        regime_names: Tên các regime
        min_duration_threshold: Ngưỡng tối thiểu
        max_duration_threshold: Ngưỡng tối đa

    Returns:
        RegimeDurationModel instance
    """
    return RegimeDurationModel(
        regime_names=regime_names,
        min_duration_threshold=min_duration_threshold,
        max_duration_threshold=max_duration_threshold
    )