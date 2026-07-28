"""
Signal Quality Filter - Lọc tín hiệu chất lượng cao

Mục tiêu:
- Giảm số lượng tín hiệu từ 436 → 200-250
- Tăng winrate từ 49% → 55-60%
- Chỉ giữ lại tín hiệu có chất lượng cao
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class FilterConfig:
    """Cấu hình bộ lọc tín hiệu"""
    # Multi-timeframe consensus
    min_tf_consensus: float = 0.7  # 70% timeframe đồng thuận

    # Volatility filter
    volatility_floor: float = 0.008  # Volatility tối thiểu (tránh flat)
    volatility_ceiling: float = 0.03  # Volatility tối đa (tránh quá biến động)

    # Volume confirmation
    min_volume_ratio: float = 1.5  # Volume > 1.5x trung bình (tăng từ 1.2)
    volume_ma_period: int = 20  # MA period cho volume

    # Regime confidence
    min_regime_confidence: float = 0.65  # Tăng từ 0.6 lên 0.65

    # ICT confluence
    min_ict_confluence: int = 2  # Tối thiểu 2 ICT signals

    # Probability threshold
    min_probability: float = 0.65  # Tăng từ 0.55 lên 0.65

    # Market regime filters
    allow_trending: bool = True
    allow_ranging: bool = True
    allow_volatile: bool = True

    # Time-based filters
    avoid_weekend_trades: bool = True  # Tránh trade cuối tuần
    avoid_low_liquidity_hours: bool = True  # Tránh giờ thanh khoản thấp

    # Divergence filters
    enable_divergence_filter: bool = True  # Bật bộ lọc divergence

    # Enable/disable filter
    enable_filter: bool = True

    # Debug mode
    debug: bool = False


class SignalQualityFilter:
    """
    Bộ lọc chất lượng tín hiệu với các điều kiện:
    1. Multi-timeframe trend consensus
    2. Volatility filter (tránh flat markets và quá biến động)
    3. Volume confirmation
    4. Regime confidence threshold
    5. ICT confluence (Order Blocks + Fibonacci)
    6. Probability threshold
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self.filter_stats_history: List[Dict] = []

    def _calculate_quality_scores(
        self,
        features: pd.DataFrame,
        probabilities: np.ndarray
    ) -> np.ndarray:
        """
        Tính điểm chất lượng cho từng tín hiệu

        Args:
            features: DataFrame chứa các features
            probabilities: Predicted probabilities [N, 3]

        Returns:
            Quality scores [N]
        """
        n_samples = len(features)
        quality_scores = np.ones(n_samples)

        # 1. Multi-timeframe trend consensus score
        if 'trend_1h' in features.columns and 'trend_4h' in features.columns:
            tf_consensus = np.abs(
                features['trend_1h'].values +
                features['trend_4h'].values +
                features.get('trend_1d', features['trend_1h']).values
            ) / 3
            consensus_score = np.clip(tf_consensus / 0.7, 0, 1)
            quality_scores *= (0.5 + 0.5 * consensus_score)

        # 2. Volatility score
        if 'volatility' in features.columns or 'atr_14' in features.columns:
            vol = features.get('volatility', features.get('atr_14', pd.Series([0.015] * n_samples))).values
            vol_floor = self.config.volatility_floor
            vol_ceiling = self.config.volatility_ceiling

            # Score cao nhất khi volatility ở mức trung bình
            vol_scores = np.ones(n_samples)
            low_vol_mask = vol < vol_floor
            high_vol_mask = vol > vol_ceiling

            # Penalize low volatility
            vol_scores[low_vol_mask] = 0.3 + 0.7 * (vol[low_vol_mask] / vol_floor)

            # Penalize high volatility
            vol_scores[high_vol_mask] = 1.0 - 0.7 * ((vol[high_vol_mask] - vol_ceiling) / vol_ceiling)
            vol_scores = np.clip(vol_scores, 0.3, 1.0)

            quality_scores *= vol_scores

        # 3. Volume confirmation score
        if 'volume_ratio_5' in features.columns:
            volume_ratio = features['volume_ratio_5'].values
            volume_scores = np.clip((volume_ratio - 0.8) / 0.4, 0.5, 1.0)
            quality_scores *= volume_scores

        # 4. Regime confidence score
        regime_confidence = np.max(probabilities, axis=1)
        confidence_scores = np.clip((regime_confidence - 0.4) / 0.4, 0, 1)
        quality_scores *= (0.5 + 0.5 * confidence_scores)

        # 5. ICT confluence score (nếu có)
        if any(c.startswith('ob_confluence') for c in features.columns) or any(c.startswith('fib_confluence') for c in features.columns):
            ob_cols = [c for c in features.columns if c.startswith('ob_confluence')]
            fib_cols = [c for c in features.columns if c.startswith('fib_confluence')]

            ob_score = features[ob_cols].mean(axis=1).fillna(0) if ob_cols else pd.Series(0, index=features.index)
            fib_score = features[fib_cols].mean(axis=1).fillna(0) if fib_cols else pd.Series(0, index=features.index)
            ict_score = (ob_score.values + fib_score.values) / 2
            quality_scores *= (0.5 + 0.5 * ict_score)

        # 6. Divergence filter score (nếu có)
        if self.config.enable_divergence_filter:
            if 'regular_bullish_div' in features.columns or 'regular_bearish_div' in features.columns:
                # Tín hiệu divergence mạnh sẽ được ưu tiên
                div_bullish = features.get('regular_bullish_div', pd.Series([0] * n_samples)).values
                div_bearish = features.get('regular_bearish_div', pd.Series([0] * n_samples)).values
                hidden_bullish = features.get('hidden_bullish_div', pd.Series([0] * n_samples)).values
                hidden_bearish = features.get('hidden_bearish_div', pd.Series([0] * n_samples)).values

                # Tổng điểm divergence (0-4)
                div_score = div_bullish + div_bearish + hidden_bullish + hidden_bearish
                div_score = np.clip(div_score / 2.0, 0, 1)  # Normalize về 0-1
                quality_scores *= (0.7 + 0.3 * div_score)  # Tối thiểu 70% score

        return np.clip(quality_scores, 0, 1)

    def filter(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        features: Optional[pd.DataFrame] = None,
        additional_mask: Optional[np.ndarray] = None,
        timestamps: Optional[pd.Series] = None  # Thêm timestamps để filter theo thời gian
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Lọc tín hiệu chất lượng cao

        Args:
            predictions: Array predictions [N]
            probabilities: Array probabilities [N, 3]
            features: DataFrame chứa features (optional)
            additional_mask: Mask bổ sung từ bên ngoài (optional)
            timestamps: Thời gian của các tín hiệu (optional)

        Returns:
            Tuple of (filtered_predictions, filtered_probabilities, filter_stats)
        """
        if not self.config.enable_filter:
            return predictions, probabilities, {'enabled': False}

        n_samples = len(predictions)

        # Initialize mask với tất cả True
        quality_mask = np.ones(n_samples, dtype=bool)

        # 1. Regime confidence filter
        regime_confidence = np.max(probabilities, axis=1)
        confidence_mask = regime_confidence >= self.config.min_regime_confidence

        # 2. Probability threshold filter
        prob_mask = regime_confidence >= self.config.min_probability

        # 3. Features-based filter (nếu có features)
        features_mask = np.ones(n_samples, dtype=bool)
        if features is not None:
            quality_scores = self._calculate_quality_scores(features, probabilities)
            features_mask = quality_scores >= 0.7  # Tăng threshold từ 0.6 lên 0.7

            if self.config.debug:
                print(f"Quality scores: min={quality_scores.min():.3f}, max={quality_scores.max():.3f}, mean={quality_scores.mean():.3f}")

        # 4. Time-based filters (nếu có timestamps)
        time_mask = np.ones(n_samples, dtype=bool)
        if timestamps is not None and (self.config.avoid_weekend_trades or self.config.avoid_low_liquidity_hours):
            time_mask = self._apply_time_filters(timestamps)

        # 5. Additional mask (nếu có)
        if additional_mask is not None:
            features_mask &= additional_mask

        # Kết hợp tất cả điều kiện
        quality_mask = confidence_mask & prob_mask & features_mask & time_mask

        # Apply filter
        filtered_predictions = predictions[quality_mask]
        filtered_probabilities = probabilities[quality_mask]

        # Tính thống kê
        filter_rate = quality_mask.mean()
        stats = {
            'enabled': True,
            'total_signals': int(n_samples),
            'filtered_signals': int(quality_mask.sum()),
            'filter_rate': float(filter_rate),
            'retention_rate': float(1 - filter_rate),
            'confidence_mean': float(regime_confidence.mean()),
            'confidence_filtered_mean': float(regime_confidence[quality_mask].mean()) if quality_mask.any() else 0,
            'quality_scores_mean': float(quality_scores.mean()) if features is not None else 0,
        }

        # Lưu history
        self.filter_stats_history.append(stats)

        if self.config.debug:
            print(f"\n{'='*60}")
            print(f"🔍 SIGNAL QUALITY FILTER STATS")
            print(f"{'='*60}")
            print(f"Total signals:      {stats['total_signals']}")
            print(f"Filtered signals:   {stats['filtered_signals']}")
            print(f"Filter rate:        {stats['filter_rate']:.2%}")
            print(f"Retention rate:     {stats['retention_rate']:.2%}")
            print(f"Confidence (all):   {stats['confidence_mean']:.3f}")
            print(f"Confidence (kept):  {stats['confidence_filtered_mean']:.3f}")
            print(f"{'='*60}\n")

        return filtered_predictions, filtered_probabilities, stats

    def get_filter_report(self) -> str:
        """Tạo báo cáo filter statistics"""
        if not self.filter_stats_history:
            return "No filter history available."

        total_filtered = sum(s['filtered_signals'] for s in self.filter_stats_history)
        total_signals = sum(s['total_signals'] for s in self.filter_stats_history)
        avg_filter_rate = np.mean([s['filter_rate'] for s in self.filter_stats_history])

        report = f"""
📊 SIGNAL QUALITY FILTER REPORT
{'='*50}
Total signals processed: {total_signals}
Total signals filtered:  {total_filtered}
Average filter rate:     {avg_filter_rate:.2%}

Recent filters:
"""
        for i, stats in enumerate(self.filter_stats_history[-5:]):
            report += f"  Filter {len(self.filter_stats_history)-4+i}: {stats['filtered_signals']}/{stats['total_signals']} ({stats['filter_rate']:.2%})\n"

        return report

    def _apply_time_filters(self, timestamps: pd.Series) -> np.ndarray:
        """
        Áp dụng time-based filters

        Args:
            timestamps: Thời gian của các tín hiệu

        Returns:
            Boolean mask cho các tín hiệu được phép trade
        """
        n_samples = len(timestamps)
        time_mask = np.ones(n_samples, dtype=bool)

        # Chuyển timestamps về datetime nếu cần
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.to_datetime(timestamps)

        # Tránh trade cuối tuần (Saturday, Sunday)
        if self.config.avoid_weekend_trades:
            weekend_mask = timestamps.weekday < 5  # Chỉ trade Mon-Fri
            time_mask &= weekend_mask

        # Tránh giờ thanh khoản thấp (22:00-02:00 UTC)
        if self.config.avoid_low_liquidity_hours:
            hour_mask = ~((timestamps.hour >= 22) | (timestamps.hour < 2))
            time_mask &= hour_mask

        return time_mask

    def reset_stats(self):
        """Reset filter statistics"""
        self.filter_stats_history = []


# Quick test
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    predictions = np.random.choice([-1, 0, 1], n_samples)
    probabilities = np.random.dirichlet([1, 1, 1], n_samples)

    features = pd.DataFrame({
        'trend_1h': np.random.uniform(-1, 1, n_samples),
        'trend_4h': np.random.uniform(-1, 1, n_samples),
        'volatility': np.random.uniform(0.005, 0.04, n_samples),
        'volume_ratio_5': np.random.uniform(0.5, 2.5, n_samples),
        'ob_confluence': np.random.randint(0, 2, n_samples),
        'fib_confluence': np.random.randint(0, 2, n_samples),
    })

    # Create filter
    config = FilterConfig(
        min_regime_confidence=0.6,
        min_probability=0.55,
        enable_filter=True,
        debug=True
    )
    filter_obj = SignalQualityFilter(config)

    # Apply filter
    filtered_preds, filtered_probs, stats = filter_obj.filter(
        predictions, probabilities, features
    )

    print(f"\nOriginal: {len(predictions)} signals")
    print(f"Filtered: {len(filtered_preds)} signals")
    print(f"Retention: {len(filtered_preds)/len(predictions):.2%}")
