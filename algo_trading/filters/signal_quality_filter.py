"""
Signal quality filter for MOE v2 model
Filters out low-quality trading signals to improve winrate
"""
import numpy as np
import pandas as pd
from typing import Union, Dict, Any

def signal_quality_filter(
    features: Union[pd.DataFrame, Dict[str, np.ndarray]],
    price_col: str = 'close'
) -> np.ndarray:
    """
    Advanced signal quality filter to reduce false signals.

    Args:
        features: Input features (DataFrame or dict)
        price_col: Column name for price data

    Returns:
        Boolean array indicating which signals pass quality filter
    """
    # Convert to DataFrame if dict
    if isinstance(features, dict):
        df = pd.DataFrame(features)
    else:
        df = features.copy()

    # Initialize filter conditions and weights
    conditions = []
    weights = []

    # 1. Multi-timeframe trend confirmation (stricter)
    if 'multi_tf_trend_consensus' in df.columns:
        trend_filter = df['multi_tf_trend_consensus'].abs() >= 1
        conditions.append(trend_filter)
        weights.append(0.3)  # Weight 30%

    # 2. Near strong supply/demand zones
    if 'near_supply_zone' in df.columns and 'near_demand_zone' in df.columns:
        zone_filter = (df['near_supply_zone'] > 0.35) | (df['near_demand_zone'] > 0.35)
        conditions.append(zone_filter)
        weights.append(0.25)  # Weight 25%

    # 3. Volatility filter (avoid extremes + too-calm zones)
    if 'volatility_normalized' in df.columns:
        v = pd.to_numeric(df['volatility_normalized'], errors='coerce').fillna(0.0)
        low, high = v.quantile(0.15), v.quantile(0.85)
        vol_filter = (v >= low) & (v <= high)
        conditions.append(vol_filter)
        weights.append(0.2)  # Weight 20%
    elif 'ATR14' in df.columns and price_col in df.columns:
        vol_normalized = df['ATR14'] / df[price_col]
        low, high = vol_normalized.quantile(0.15), vol_normalized.quantile(0.85)
        vol_filter = (vol_normalized >= low) & (vol_normalized <= high)
        conditions.append(vol_filter)
        weights.append(0.2)  # Weight 20%

    # 4. Volume confirmation
    if 'volume' in df.columns:
        vol_ma = df['volume'].rolling(20, min_periods=1).mean()
        volume_filter = df['volume'] > (vol_ma * 1.05)
        conditions.append(volume_filter)
        weights.append(0.15)  # Weight 15%

    # 5. Momentum confirmation
    if 'RSI14' in df.columns:
        momentum_filter = (df['RSI14'] > 35) & (df['RSI14'] < 65)
        conditions.append(momentum_filter)
        weights.append(0.1)  # Weight 10%
    if 'momentum_confirmation' in df.columns:
        mc_filter = pd.to_numeric(df['momentum_confirmation'], errors='coerce').fillna(0.0) >= 0.55
        conditions.append(mc_filter)
        weights.append(0.1)  # Weight 10%

    # 6. Price action confirmation
    if 'close' in df.columns and 'open' in df.columns:
        # Avoid trades against strong momentum
        price_change = df['close'] / df['open'] - 1
        price_action_filter = abs(price_change) < 0.03
        conditions.append(price_action_filter)
        weights.append(0.1)  # Weight 10%

    # Combine all conditions with weighted OR logic
    if conditions:
        # Calculate weighted score
        weighted_score = np.zeros(len(df), dtype=float)
        for condition, weight in zip(conditions, weights):
            weighted_score += condition.astype(float) * weight

        # Pass if weighted score > 0.3 (giảm từ 0.5 xuống 0.3 để tăng tín hiệu)
        quality_filter = weighted_score > 0.3
        return quality_filter.values
    else:
        # If no conditions available, pass all signals
        return np.ones(len(df), dtype=bool)

def enhanced_signal_scoring(
    predictions: np.ndarray,
    features: Union[pd.DataFrame, Dict[str, np.ndarray]]
) -> np.ndarray:
    """
    Enhanced signal scoring based on multiple factors.

    Args:
        predictions: Model predictions (probabilities)
        features: Input features

    Returns:
        Enhanced signal scores (0-1 range)
    """
    # Convert to DataFrame if dict
    if isinstance(features, dict):
        df = pd.DataFrame(features)
    else:
        df = features.copy()

    # Handle predictions - ensure it's 2D
    if predictions.ndim == 1:
        scores = np.asarray(predictions, dtype=float).copy()
    else:
        scores = np.asarray(predictions).max(axis=1).copy()

    # 1. Trend strength bonus
    if 'multi_tf_trend_consensus' in df.columns:
        trend_strength = abs(df['multi_tf_trend_consensus'])
        scores = scores * (1 + trend_strength * 0.5)  # Up to 50% bonus

    # 2. Zone proximity bonus
    zone_bonus = 0
    if 'near_demand_zone' in df.columns:
        zone_bonus += df['near_demand_zone'] * 0.3
    if 'near_supply_zone' in df.columns:
        zone_bonus += df['near_supply_zone'] * 0.3
    scores = scores * (1 + zone_bonus)

    # 3. Volume confirmation bonus
    if 'volume' in df.columns:
        vol_ma = df['volume'].rolling(20, min_periods=1).mean()
        volume_conf = (df['volume'] > vol_ma).astype(int)
        scores = scores * (1 + volume_conf * 0.2)  # 20% bonus

    # 4. Volatility normalization penalty
    if 'ATR14' in df.columns and 'close' in df.columns:
        vol_normalized = df['ATR14'] / df['close']
        vol_penalty = np.clip(vol_normalized / vol_normalized.quantile(0.5), 0.5, 2.0)
        scores = scores / vol_penalty  # Reduce score in high vol

    # Normalize scores to 0-1 range
    scores = np.clip(scores, 0, 1)
    return scores

def apply_signal_filter(
    predictions: np.ndarray,
    features: Union[pd.DataFrame, Dict[str, np.ndarray]],
    threshold: float = 0.6
) -> np.ndarray:
    """
    Apply quality filter to model predictions.

    Args:
        predictions: Model predictions
        features: Input features
        threshold: Minimum quality score threshold

    Returns:
        Filtered predictions
    """
    # Apply quality filter
    quality_mask = signal_quality_filter(features)

    # Apply enhanced scoring
    enhanced_scores = enhanced_signal_scoring(predictions, features)

    # Apply final threshold
    final_signals = (enhanced_scores > threshold) & quality_mask

    return final_signals.astype(int)

# Example usage
if __name__ == "__main__":
    # Example data
    np.random.seed(42)
    n_samples = 1000

    # Simulate features
    features = {
        'multi_tf_trend_consensus': np.random.choice([-1, 0, 1], n_samples),
        'near_supply_zone': np.random.random(n_samples),
        'near_demand_zone': np.random.random(n_samples),
        'ATR14': np.random.random(n_samples) * 100 + 50,
        'close': np.random.random(n_samples) * 50000 + 30000,
        'volume': np.random.random(n_samples) * 1000000,
        'RSI14': np.random.random(n_samples) * 100,
        'open': np.random.random(n_samples) * 50000 + 30000
    }

    # Simulate predictions
    predictions = np.random.random(n_samples)

    # Apply filter
    filtered_signals = apply_signal_filter(predictions, features, threshold=0.6)

    print(f"Original signals: {len(predictions)}")
    print(f"Filtered signals: {filtered_signals.sum()}")
    print(f"Reduction: {(1 - filtered_signals.sum() / len(predictions)) * 100:.1f}%")