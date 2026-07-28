"""
Regime-Specific Thresholds Module

Điều chỉnh probability thresholds theo từng regime để:
- Tăng winrate 5-10%
- Adapt với market conditions
- Giảm false signals trong volatile regimes
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import warnings


class RegimeSpecificThresholds:
    """
    Quản lý thresholds cho từng regime

    Regime-specific thresholds giúp:
    - Trending: Thấp hơn (dễ vào lệnh theo trend)
    - Ranging: Cao hơn (cần tín hiệu mạnh)
    - Volatile: Rất cao (tránh false signals)
    - Calm: Trung bình
    """

    # Default thresholds cho từng regime
    DEFAULT_THRESHOLDS = {
        0: {'long': 0.50, 'short': 0.50, 'description': 'trending'},      # Trending: dễ vào lệnh
        1: {'long': 0.60, 'short': 0.60, 'description': 'ranging'},       # Ranging: cần tín hiệu mạnh
        2: {'long': 0.65, 'short': 0.65, 'description': 'volatile'},      # Volatile: rất cao
        3: {'long': 0.55, 'short': 0.55, 'description': 'calm'},          # Calm: trung bình
    }

    def __init__(
        self,
        custom_thresholds: Optional[Dict[int, Dict[str, float]]] = None,
        use_regime_aware: bool = True
    ):
        """
        Args:
            custom_thresholds: Custom thresholds cho từng regime
            use_regime_aware: Có sử dụng regime-aware thresholds không
        """
        self.use_regime_aware = use_regime_aware

        if custom_thresholds:
            self.thresholds = custom_thresholds
        else:
            self.thresholds = self.DEFAULT_THRESHOLDS.copy()

        # Validate thresholds
        self._validate_thresholds()

    def _validate_thresholds(self):
        """Validate rằng thresholds hợp lệ"""
        for regime_id, regime_thresholds in self.thresholds.items():
            for direction, threshold in regime_thresholds.items():
                if direction in ['long', 'short']:
                    if not 0.0 <= threshold <= 1.0:
                        raise ValueError(
                            f"Threshold {direction} cho regime {regime_id} "
                            f"phải nằm trong khoảng [0, 1], got {threshold}"
                        )

    def get_threshold(
        self,
        regime_id: int,
        direction: str = 'long'
    ) -> float:
        """
        Get threshold cho regime và direction cụ thể

        Args:
            regime_id: Regime ID (0-3)
            direction: 'long' hoặc 'short'

        Returns:
            Threshold value
        """
        if not self.use_regime_aware:
            # Return default 0.5 nếu không dùng regime-aware
            return 0.5

        if regime_id not in self.thresholds:
            warnings.warn(
                f"Regime {regime_id} không có threshold, dùng default 0.55"
            )
            return 0.55

        threshold = self.thresholds[regime_id].get(direction, 0.55)
        return threshold

    def get_all_thresholds(self, regime_id: int) -> Dict[str, float]:
        """
        Get tất cả thresholds cho một regime

        Args:
            regime_id: Regime ID

        Returns:
            Dict với thresholds
        """
        if regime_id not in self.thresholds:
            return {'long': 0.55, 'short': 0.55}
        return self.thresholds[regime_id].copy()

    def adjust_signal(
        self,
        p_long: float,
        p_short: float,
        p_neutral: float,
        regime_id: int
    ) -> float:
        """
        Adjust signal dựa trên regime-specific thresholds

        Args:
            p_long: Xác suất LONG
            p_short: Xác suất SHORT
            p_neutral: Xác suất NEUTRAL
            regime_id: Current regime ID

        Returns:
            Signal: 1.0 (LONG), -1.0 (SHORT), 0.0 (NEUTRAL)
        """
        if not self.use_regime_aware:
            # Use default threshold 0.5
            threshold = 0.5
        else:
            threshold_long = self.get_threshold(regime_id, 'long')
            threshold_short = self.get_threshold(regime_id, 'short')

            # Check LONG signal
            if p_long >= threshold_long and p_long > p_short and p_long >= p_neutral:
                return 1.0

            # Check SHORT signal
            if p_short >= threshold_short and p_short > p_long and p_short >= p_neutral:
                return -1.0

            return 0.0

        # Fallback to default logic
        if p_long >= threshold and p_long > p_short and p_long >= p_neutral:
            return 1.0
        elif p_short >= threshold and p_short > p_long and p_short >= p_neutral:
            return -1.0
        else:
            return 0.0

    def set_threshold(
        self,
        regime_id: int,
        direction: str,
        value: float
    ):
        """
        Set threshold cho regime cụ thể

        Args:
            regime_id: Regime ID
            direction: 'long' hoặc 'short'
            value: Threshold value
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold phải nằm trong khoảng [0, 1], got {value}")

        if regime_id not in self.thresholds:
            self.thresholds[regime_id] = {
                'long': 0.55,
                'short': 0.55,
                'description': f'regime_{regime_id}'
            }

        self.thresholds[regime_id][direction] = value

    def get_threshold_summary(self) -> pd.DataFrame:
        """
        Get summary của tất cả thresholds

        Returns:
            DataFrame với thresholds
        """
        rows = []
        for regime_id, regime_thresholds in self.thresholds.items():
            rows.append({
                'regime_id': regime_id,
                'description': regime_thresholds.get('description', f'regime_{regime_id}'),
                'long_threshold': regime_thresholds.get('long', 0.55),
                'short_threshold': regime_thresholds.get('short', 0.55)
            })

        return pd.DataFrame(rows)


def get_regime_specific_thresholds(
    regime_id: int,
    custom_thresholds: Optional[Dict[int, Dict[str, float]]] = None
) -> Dict[str, float]:
    """
    Convenience function để get thresholds cho regime

    Args:
        regime_id: Regime ID
        custom_thresholds: Custom thresholds (optional)

    Returns:
        Dict với long/short thresholds
    """
    manager = RegimeSpecificThresholds(custom_thresholds)
    return manager.get_all_thresholds(regime_id)


def apply_regime_threshold_to_signal(
    p_long: float,
    p_short: float,
    p_neutral: float,
    regime_id: int,
    custom_thresholds: Optional[Dict[int, Dict[str, float]]] = None
) -> float:
    """
    Convenience function để apply regime thresholds

    Args:
        p_long: Probability LONG
        p_short: Probability SHORT
        p_neutral: Probability NEUTRAL
        regime_id: Current regime ID
        custom_thresholds: Custom thresholds (optional)

    Returns:
        Signal: 1.0, -1.0, or 0.0
    """
    manager = RegimeSpecificThresholds(custom_thresholds)
    return manager.adjust_signal(p_long, p_short, p_neutral, regime_id)
