"""
Compatibility adapter cho signal filter trong nhom ML.

Source-of-truth da duoc chot tai:
- algo_trading.filters.signal_quality_filter

File nay giu API toi thieu de tranh vo import cu.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

from algo_trading.filters.signal_quality_filter import signal_quality_filter


@dataclass
class FilterConfig:
    """Config toi thieu cho adapter."""
    enable_filter: bool = True
    debug: bool = False


class SignalQualityFilter:
    """Adapter class de tuong thich voi code huan luyen cu."""

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self.filter_stats_history: List[Dict] = []

    def filter(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        features: Optional[pd.DataFrame] = None,
        additional_mask: Optional[np.ndarray] = None,
        timestamps: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        if not self.config.enable_filter:
            return predictions, probabilities, {"enabled": False}

        n_samples = len(predictions)
        quality_mask = np.ones(n_samples, dtype=bool)

        if features is not None and len(features) == n_samples:
            quality_mask &= signal_quality_filter(features)

        if additional_mask is not None and len(additional_mask) == n_samples:
            quality_mask &= additional_mask.astype(bool)

        filtered_predictions = predictions[quality_mask]
        filtered_probabilities = probabilities[quality_mask]

        stats = {
            "enabled": True,
            "total_signals": int(n_samples),
            "filtered_signals": int(quality_mask.sum()),
            "filter_rate": float(1.0 - quality_mask.mean()) if n_samples else 0.0,
            "retention_rate": float(quality_mask.mean()) if n_samples else 0.0,
        }
        self.filter_stats_history.append(stats)

        if self.config.debug:
            print(
                f"[SignalQualityFilter] kept={stats['filtered_signals']}/"
                f"{stats['total_signals']} ({stats['retention_rate']:.2%})"
            )

        return filtered_predictions, filtered_probabilities, stats

    def get_filter_report(self) -> str:
        if not self.filter_stats_history:
            return "No filter history available."

        total = sum(s["total_signals"] for s in self.filter_stats_history)
        kept = sum(s["filtered_signals"] for s in self.filter_stats_history)
        retention = (kept / total) if total else 0.0
        return (
            "Signal Quality Filter Report\n"
            f"Total signals: {total}\n"
            f"Kept signals: {kept}\n"
            f"Retention: {retention:.2%}"
        )
