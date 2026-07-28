
import numpy as np
import pandas as pd
from typing import Union, Dict, Any


def _to_num(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(default)

def signal_quality_filter(
    features: Union[pd.DataFrame, Dict[str, np.ndarray]],
    price_col: str = 'close'
) -> np.ndarray:

    # Convert to DataFrame if dict
    if isinstance(features, dict):
        df = pd.DataFrame(features)
    else:
        df = features.copy()

    # Initialize filter conditions and weights
    conditions = []
    weights = []

    # 1. Multi-timeframe trend confirmation with persistence check
    if 'multi_tf_trend_consensus' in df.columns:
        consensus = _to_num(df['multi_tf_trend_consensus'])
        trend_filter = consensus.abs() >= 0.8
        trend_persistence = consensus.rolling(4, min_periods=1).mean().abs() >= 0.6
        trend_filter = trend_filter & trend_persistence
        conditions.append(trend_filter)
        weights.append(0.28)  # Weight 28%

    # 2. Near strong supply/demand zones
    if 'near_supply_zone' in df.columns and 'near_demand_zone' in df.columns:
        supply = _to_num(df['near_supply_zone'])
        demand = _to_num(df['near_demand_zone'])
        zone_filter = (supply > 0.42) | (demand > 0.42)
        conditions.append(zone_filter)
        weights.append(0.22)  # Weight 22%

    # 3. Volatility filter (avoid extremes + too-calm zones)
    if 'volatility_normalized' in df.columns:
        v = _to_num(df['volatility_normalized'])
        low, high = v.quantile(0.15), v.quantile(0.88)
        vol_filter = (v >= low) & (v <= high)
        conditions.append(vol_filter)
        weights.append(0.18)  # Weight 18%
    elif 'ATR14' in df.columns and price_col in df.columns:
        atr = _to_num(df['ATR14'])
        px = _to_num(df[price_col], default=np.nan).replace(0, np.nan)
        vol_normalized = (atr / px).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        low, high = vol_normalized.quantile(0.15), vol_normalized.quantile(0.88)
        vol_filter = (vol_normalized >= low) & (vol_normalized <= high)
        conditions.append(vol_filter)
        weights.append(0.18)  # Weight 18%

    # 4. Volume confirmation
    if 'volume' in df.columns:
        vol = _to_num(df['volume'])
        vol_ma = vol.rolling(20, min_periods=1).mean()
        volume_filter = vol > (vol_ma * 1.03)
        conditions.append(volume_filter)
        weights.append(0.12)  # Weight 12%

    # 5. Momentum confirmation
    if 'RSI14' in df.columns:
        rsi = _to_num(df['RSI14'])
        momentum_filter = (rsi > 35) & (rsi < 65)
        conditions.append(momentum_filter)
        weights.append(0.08)  # Weight 8%
    if 'momentum_confirmation' in df.columns:
        mc_filter = _to_num(df['momentum_confirmation']) >= 0.55
        conditions.append(mc_filter)
        weights.append(0.08)  # Weight 8%

    # 6. Price action confirmation
    if 'close' in df.columns and 'open' in df.columns:
        # Avoid trades against strong momentum
        c = _to_num(df['close'], default=np.nan)
        o = _to_num(df['open'], default=np.nan).replace(0, np.nan)
        price_change = (c / o - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        candle_body_filter = price_change.abs() < 0.02

        # Avoid frequent local direction flips.
        recent_dir = np.sign(price_change).rolling(3, min_periods=1).sum().abs()
        persistence_filter = recent_dir >= 1
        price_action_filter = candle_body_filter & persistence_filter
        conditions.append(price_action_filter)
        weights.append(0.09)  # Weight 9%

    # 7. ICT/Fibonacci confluence across timeframes
    ob_cols = [c for c in df.columns if c.startswith('ob_confluence')]
    fib_cols = [c for c in df.columns if c.startswith('fib_confluence')]
    if ob_cols or fib_cols:
        ob_score = df[ob_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).fillna(0.0) if ob_cols else pd.Series(0.0, index=df.index)
        fib_score = df[fib_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).fillna(0.0) if fib_cols else pd.Series(0.0, index=df.index)
        ict_score = (ob_score + fib_score) / 2.0
        ict_filter = ict_score >= 0.42
        conditions.append(ict_filter)
        weights.append(0.18)  # Weight 18%

    # Combine all conditions with weighted OR logic
    if conditions:
        # Calculate weighted score.
        total_weight = max(float(sum(weights)), 1e-9)
        weighted_score = np.zeros(len(df), dtype=float)
        for condition, weight in zip(conditions, weights):
            weighted_score += condition.astype(float) * weight
        weighted_score = weighted_score / total_weight

        # Adaptive quality gate: tighter in high churn periods, slightly looser in calm periods.
        if 'close' in df.columns and 'open' in df.columns:
            c = _to_num(df['close'], default=np.nan)
            o = _to_num(df['open'], default=np.nan).replace(0, np.nan)
            body = (c / o - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0).abs()
            churn = body.rolling(16, min_periods=1).mean()
            churn_norm = (churn - churn.quantile(0.2)) / max((churn.quantile(0.8) - churn.quantile(0.2)), 1e-9)
            churn_norm = np.clip(churn_norm, 0.0, 1.0)
            dynamic_gate = 0.50 + 0.08 * churn_norm
        else:
            dynamic_gate = 0.52

        quality_filter = weighted_score >= dynamic_gate
        return np.asarray(quality_filter, dtype=bool)
    else:
        # If no conditions available, pass all signals
        return np.ones(len(df), dtype=bool)

def enhanced_signal_scoring(
    predictions: np.ndarray,
    features: Union[pd.DataFrame, Dict[str, np.ndarray]]
) -> np.ndarray:
    # Convert to DataFrame if dict
    if isinstance(features, dict):
        df = pd.DataFrame(features)
    else:
        df = features.copy()

    scores = np.asarray(predictions, dtype=float).copy()

    # 1. Trend strength bonus with cap
    if 'multi_tf_trend_consensus' in df.columns:
        trend_strength = abs(_to_num(df['multi_tf_trend_consensus']))
        scores = scores * (1 + np.clip(trend_strength, 0.0, 1.5) * 0.35)

    # 2. Zone proximity bonus
    zone_bonus = 0.0
    if 'near_demand_zone' in df.columns:
        zone_bonus += _to_num(df['near_demand_zone']) * 0.25
    if 'near_supply_zone' in df.columns:
        zone_bonus += _to_num(df['near_supply_zone']) * 0.25
    scores = scores * (1 + zone_bonus)

    # 3. Volume confirmation bonus
    if 'volume' in df.columns:
        vol = _to_num(df['volume'])
        vol_ma = vol.rolling(20, min_periods=1).mean()
        volume_conf = (vol > vol_ma * 1.01).astype(int)
        scores = scores * (1 + volume_conf * 0.12)

    # 4. Volatility normalization penalty
    if 'ATR14' in df.columns and 'close' in df.columns:
        atr = _to_num(df['ATR14'])
        close = _to_num(df['close'], default=np.nan).replace(0, np.nan)
        vol_normalized = (atr / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mid = max(float(vol_normalized.quantile(0.5)), 1e-9)
        vol_penalty = np.clip(vol_normalized / mid, 0.75, 1.60)
        scores = scores / vol_penalty

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