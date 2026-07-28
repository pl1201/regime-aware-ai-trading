"""
Multi-timeframe feature engineering for enhanced signal filtering
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


TF_TO_BARS = {
    '1h': 1,
    '4h': 4,
    '1d': 24,
    '1w': 24 * 7,
}

def create_enhanced_multi_timeframe_features(
    df: pd.DataFrame,
    timeframes: List[str] = ['1h', '4h', '1d', '1w']
) -> pd.DataFrame:
    """
    Create enhanced multi-timeframe features for signal quality filtering.

    Args:
        df: Base timeframe data (1h)
        timeframes: List of timeframes to include

    Returns:
        DataFrame with enhanced multi-timeframe features
    """
    # Start with base features
    enhanced_df = df.copy()

    # Add multi-timeframe trend consensus
    enhanced_df['multi_tf_trend_consensus'] = calculate_multi_tf_trend_consensus(df, timeframes)

    # Add supply/demand zone proximity
    enhanced_df['near_supply_zone'] = calculate_supply_zone_proximity(df)
    enhanced_df['near_demand_zone'] = calculate_demand_zone_proximity(df)

    # Add volatility features
    enhanced_df['volatility_normalized'] = calculate_normalized_volatility(df)

    # Add momentum confirmation
    enhanced_df['momentum_confirmation'] = calculate_momentum_confirmation(df)

    # Add volume confirmation
    enhanced_df['volume_trend_confirmation'] = calculate_volume_trend_confirmation(df)

    return enhanced_df

def calculate_multi_tf_trend_consensus(df: pd.DataFrame, timeframes: List[str]) -> pd.Series:
    """
    Calculate trend consensus across multiple timeframes.

    Returns:
        -1 for bearish consensus, 0 for neutral, 1 for bullish consensus
    """
    if 'close' not in df.columns or len(df) == 0:
        return pd.Series(0, index=df.index)

    close = pd.to_numeric(df['close'], errors='coerce').ffill().bfill()
    votes = []
    weights = []

    for tf in timeframes:
        tf = str(tf).lower()
        bars = TF_TO_BARS.get(tf)
        if bars is None:
            continue

        if bars <= 1:
            lookback = 6
            ret = close.pct_change(lookback)
        else:
            # Resample sang timeframe lớn hơn, sau đó align ngược về index gốc.
            close_tf = close.resample(tf).last().dropna()
            if len(close_tf) < 4:
                continue
            ret_tf = close_tf.pct_change(3)
            ret = ret_tf.reindex(close.index, method='ffill')

        vote = np.sign(ret.fillna(0.0))
        vote[np.abs(ret.fillna(0.0)) < 1e-4] = 0.0
        votes.append(vote)
        weights.append(float(max(1, bars)))

    if not votes:
        return pd.Series(0, index=df.index)

    vote_matrix = np.vstack([v.values for v in votes])
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    weighted_score = (vote_matrix * w).sum(axis=0) / (w.sum() + 1e-8)

    consensus = pd.Series(0, index=df.index, dtype=int)
    consensus[weighted_score >= 0.35] = 1
    consensus[weighted_score <= -0.35] = -1
    return consensus

def calculate_supply_zone_proximity(df: pd.DataFrame) -> pd.Series:
    """
    Calculate proximity to supply zones (resistance levels).

    Returns:
        Normalized proximity score (0-1)
    """
    # Simplified implementation - in practice, you'd identify actual supply zones
    # For now, we'll use a proxy based on recent highs

    # Calculate recent highs
    recent_high = df['high'].rolling(20).max()
    distance_to_high = (recent_high - df['close']) / df['close']

    # Normalize to 0-1 range (closer to high = higher score)
    proximity_score = np.clip(1 - distance_to_high, 0, 1)

    return proximity_score

def calculate_demand_zone_proximity(df: pd.DataFrame) -> pd.Series:
    """
    Calculate proximity to demand zones (support levels).

    Returns:
        Normalized proximity score (0-1)
    """
    # Simplified implementation - in practice, you'd identify actual demand zones
    # For now, we'll use a proxy based on recent lows

    # Calculate recent lows
    recent_low = df['low'].rolling(20).min()
    distance_to_low = (df['close'] - recent_low) / df['close']

    # Normalize to 0-1 range (closer to low = higher score)
    proximity_score = np.clip(1 - distance_to_low, 0, 1)

    return proximity_score

def calculate_normalized_volatility(df: pd.DataFrame) -> pd.Series:
    """
    Calculate normalized volatility for filtering.

    Returns:
        Normalized volatility (0-1 range)
    """
    # Calculate ATR-based volatility
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    atr14 = pd.Series(tr).rolling(14).mean()

    # Normalize volatility
    vol_normalized = atr14 / df['close']

    # Scale to 0-1 range
    vol_min = vol_normalized.quantile(0.1)
    vol_max = vol_normalized.quantile(0.9)
    vol_score = (vol_normalized - vol_min) / (vol_max - vol_min)
    vol_score = np.clip(vol_score, 0, 1)

    return vol_score

def calculate_momentum_confirmation(df: pd.DataFrame) -> pd.Series:
    """
    Calculate momentum confirmation for signal quality.

    Returns:
        Momentum confirmation score (0-1)
    """
    # RSI-based momentum
    rsi = calculate_rsi(df['close'], 14)

    # MACD-based momentum
    macd_line, signal_line = calculate_macd(df['close'])
    macd_histogram = macd_line - signal_line

    # Combine momentum indicators
    # RSI in middle range (30-70) is preferred
    rsi_score = 1 - np.abs(rsi - 50) / 50
    rsi_score = np.clip(rsi_score, 0, 1)

    # MACD histogram direction
    macd_score = (macd_histogram > 0).astype(int)

    # Combined score
    momentum_score = (rsi_score + macd_score) / 2

    return pd.Series(momentum_score, index=df.index)

def calculate_volume_trend_confirmation(df: pd.DataFrame) -> pd.Series:
    """
    Calculate volume trend confirmation.

    Returns:
        Volume confirmation score (0-1)
    """
    if 'volume' not in df.columns:
        return pd.Series(1, index=df.index)  # Default to confirmed if no volume data

    # Volume moving averages
    vol_ma_short = df['volume'].rolling(5).mean()
    vol_ma_long = df['volume'].rolling(20).mean()

    # Volume trend confirmation (above average volume)
    vol_confirmation = (df['volume'] > vol_ma_long).astype(int)

    # Normalize to 0-1 range
    vol_score = vol_confirmation.astype(float)

    return vol_score

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line

def prepare_enhanced_multi_timeframe_data_for_training(
    df: pd.DataFrame,
    target_lookahead: int = 1,
    return_dataframe: bool = False,
    label_mode: str = 'ternary',
    move_threshold: float = 0.001,
    neutral_quantile: float = 0.35,
) -> Union[
    Tuple[np.ndarray, np.ndarray, List[str]],
    Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame],
]:
    """
    Prepare enhanced multi-timeframe data for training with quality features.

    Args:
        df: Input DataFrame with enhanced features
        target_lookahead: Number of periods to look ahead for target

    Returns:
        Tuple of (features, targets, feature_names)
    """
    # Create enhanced multi-timeframe features
    enhanced_df = create_enhanced_multi_timeframe_features(df)

    # Select feature columns
    feature_columns = [
        'multi_tf_trend_consensus',
        'near_supply_zone',
        'near_demand_zone',
        'volatility_normalized',
        'momentum_confirmation',
        'volume_trend_confirmation',
        'RSI14',
        'ATR14',
        'volume'
    ]

    # Add any additional numeric columns that might be useful
    additional_features = [
        col for col in enhanced_df.columns
        if col not in feature_columns
        and col not in ['open', 'high', 'low', 'close', 'timestamp']
        and np.issubdtype(enhanced_df[col].dtype, np.number)
    ]

    feature_columns.extend(additional_features)

    # Ensure all feature columns exist
    feature_columns = [col for col in feature_columns if col in enhanced_df.columns]

    # Extract features
    feature_df = enhanced_df[feature_columns].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    X = feature_df.values

    # Create target (future price movement)
    future_returns = enhanced_df['close'].shift(-target_lookahead) / enhanced_df['close'] - 1
    mode = str(label_mode).lower().strip()
    if mode == 'binary':
        y = (future_returns > move_threshold).astype(int).fillna(0).values
    else:
        # Ternary label: -1 (short), 0 (neutral), 1 (long)
        valid_abs = future_returns.abs().dropna().values
        if len(valid_abs) > 0:
            q_thr = float(np.quantile(valid_abs, np.clip(neutral_quantile, 0.05, 0.80)))
        else:
            q_thr = float(move_threshold)
        thr = float(max(move_threshold, q_thr))

        y_series = pd.Series(0, index=enhanced_df.index, dtype=int)
        y_series[future_returns > thr] = 1
        y_series[future_returns < -thr] = -1
        y = y_series.fillna(0).astype(int).values

    if not return_dataframe:
        return X, y, feature_columns

    aligned_df = enhanced_df.loc[feature_df.index].copy()
    return X, y, feature_columns, aligned_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.random(1000) * 50000 + 30000,
        'high': np.random.random(1000) * 50000 + 30000,
        'low': np.random.random(1000) * 50000 + 30000,
        'close': np.random.random(1000) * 50000 + 30000,
        'volume': np.random.random(1000) * 1000000
    })

    # Add technical indicators
    sample_data['RSI14'] = calculate_rsi(sample_data['close'], 14)
    tr = np.maximum(
        sample_data['high'] - sample_data['low'],
        np.maximum(
            abs(sample_data['high'] - sample_data['close'].shift(1)),
            abs(sample_data['low'] - sample_data['close'].shift(1))
        )
    )
    sample_data['ATR14'] = pd.Series(tr).rolling(14).mean()

    # Test enhanced features
    enhanced_features = create_enhanced_multi_timeframe_features(sample_data)

    print("Enhanced Multi-Timeframe Features:")
    print(enhanced_features[['multi_tf_trend_consensus', 'near_supply_zone',
                           'near_demand_zone', 'volatility_normalized']].head())

    # Test data preparation
    X, y, feature_names = prepare_enhanced_multi_timeframe_data_for_training(sample_data)
    print(f"\nPrepared data shape: {X.shape}")
    print(f"Feature names: {feature_names}")
    print(f"Target distribution: {np.mean(y):.3f}")


# ============================================
# NEW ENHANCEMENTS - Divergence Detection
# ============================================

def detect_divergence_signals(
    df: pd.DataFrame,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Phát hiện divergence signals (bullish/bearish)

    Args:
        df: DataFrame với price và indicator data
        lookback: Số nến để tìm local highs/lows

    Returns:
        DataFrame với divergence signals
    """
    df = df.copy()

    # RSI
    rsi = calculate_rsi(df['close'], 14)

    # Tìm local lows và highs
    price_lows = find_local_lows(df['close'], lookback)
    price_highs = find_local_highs(df['close'], lookback)
    rsi_lows = find_local_lows(rsi, lookback)
    rsi_highs = find_local_highs(rsi, lookback)

    # Regular bullish divergence: price lower low, RSI higher low
    df['regular_bullish_div'] = detect_regular_bullish_divergence(
        df['close'], rsi, lookback
    )

    # Regular bearish divergence: price higher high, RSI lower high
    df['regular_bearish_div'] = detect_regular_bearish_divergence(
        df['close'], rsi, lookback
    )

    # Hidden bullish divergence: price higher low, RSI lower low
    df['hidden_bullish_div'] = detect_hidden_bullish_divergence(
        df['close'], rsi, lookback
    )

    # Hidden bearish divergence: price lower high, RSI higher high
    df['hidden_bearish_div'] = detect_hidden_bearish_divergence(
        df['close'], rsi, lookback
    )

    # Divergence score (tổng số divergence signals)
    div_cols = ['regular_bullish_div', 'regular_bearish_div',
                'hidden_bullish_div', 'hidden_bearish_div']
    df['divergence_score'] = df[div_cols].sum(axis=1)

    return df


def find_local_lows(series: pd.Series, period: int = 20) -> pd.Series:
    """Tìm local lows"""
    lows = pd.Series(np.zeros(len(series)), index=series.index)
    for i in range(period, len(series) - period):
        if series.iloc[i] == series.iloc[i-period:i+period+1].min():
            lows.iloc[i] = 1
    return lows


def find_local_highs(series: pd.Series, period: int = 20) -> pd.Series:
    """Tìm local highs"""
    highs = pd.Series(np.zeros(len(series)), index=series.index)
    for i in range(period, len(series) - period):
        if series.iloc[i] == series.iloc[i-period:i+period+1].max():
            highs.iloc[i] = 1
    return highs


def detect_regular_bullish_divergence(
    price: pd.Series,
    indicator: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """Phát hiện regular bullish divergence"""
    signals = pd.Series(np.zeros(len(price)), index=price.index)

    price_lows = find_local_lows(price, lookback)
    indicator_lows = find_local_lows(indicator, lookback)

    for i in range(lookback * 2, len(price)):
        if price_lows.iloc[i] and indicator_lows.iloc[i]:
            prev_idx = max(0, i - 50)
            if (indicator.iloc[i] > indicator.iloc[prev_idx] and
                price.iloc[i] < price.iloc[prev_idx]):
                signals.iloc[i] = 1

    return signals


def detect_regular_bearish_divergence(
    price: pd.Series,
    indicator: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """Phát hiện regular bearish divergence"""
    signals = pd.Series(np.zeros(len(price)), index=price.index)

    price_highs = find_local_highs(price, lookback)
    indicator_highs = find_local_highs(indicator, lookback)

    for i in range(lookback * 2, len(price)):
        if price_highs.iloc[i] and indicator_highs.iloc[i]:
            prev_idx = max(0, i - 50)
            if (indicator.iloc[i] < indicator.iloc[prev_idx] and
                price.iloc[i] > price.iloc[prev_idx]):
                signals.iloc[i] = 1

    return signals


def detect_hidden_bullish_divergence(
    price: pd.Series,
    indicator: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """Phát hiện hidden bullish divergence"""
    signals = pd.Series(np.zeros(len(price)), index=price.index)

    price_lows = find_local_lows(price, lookback)
    indicator_lows = find_local_lows(indicator, lookback)

    for i in range(lookback * 2, len(price)):
        if price_lows.iloc[i] and indicator_lows.iloc[i]:
            prev_idx = max(0, i - 50)
            if (indicator.iloc[i] < indicator.iloc[prev_idx] and
                price.iloc[i] > price.iloc[prev_idx]):
                signals.iloc[i] = 1

    return signals


def detect_hidden_bearish_divergence(
    price: pd.Series,
    indicator: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """Phát hiện hidden bearish divergence"""
    signals = pd.Series(np.zeros(len(price)), index=price.index)

    price_highs = find_local_highs(price, lookback)
    indicator_highs = find_local_highs(indicator, lookback)

    for i in range(lookback * 2, len(price)):
        if price_highs.iloc[i] and indicator_highs.iloc[i]:
            prev_idx = max(0, i - 50)
            if (indicator.iloc[i] > indicator.iloc[prev_idx] and
                price.iloc[i] < price.iloc[prev_idx]):
                signals.iloc[i] = 1

    return signals