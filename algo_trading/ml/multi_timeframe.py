"""
Multi-timeframe data processing for advanced ML models.
Handles loading, resampling, and feature engineering across multiple timeframes.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

from algo_trading.data_loader.loader import load_data
from algo_trading.indicators.ict import detect_order_blocks, detect_swing_high_low, fib_levels_from_swing, fib_features

# Timeframe configurations for multi-timeframe analysis
MULTI_TIMEFRAME_CONFIG = {
    'primary': '1h',           # Primary timeframe for signals
    'confirmation': '4h',      # Confirmation timeframe
    'trend': '1d',            # Trend confirmation timeframe
    'context': '1w',          # Market context timeframe
}

def load_multi_timeframe_data(
    source: str,
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeframes: Optional[Dict[str, str]] = None,
    market: str = 'spot'
) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple timeframes.

    Args:
        source: Data source ('binance', 'yfinance', etc.)
        symbol: Trading pair (e.g., 'BTCUSDT')
        start: Start date
        end: End date
        timeframes: Dict mapping timeframe names to intervals
        market: Market type ('spot', 'futures')

    Returns:
        Dict mapping timeframe names to DataFrames
    """
    if timeframes is None:
        timeframes = MULTI_TIMEFRAME_CONFIG

    data_dict = {}

    for tf_name, interval in timeframes.items():
        try:
            df = load_data(
                source=source,
                symbol=symbol,
                interval=interval,
                start=start,
                end=end,
                market=market,
                add_features=True  # Add basic indicators
            )
            data_dict[tf_name] = df
            print(f"Loaded {tf_name} timeframe: {interval} - {len(df)} candles")
        except Exception as e:
            print(f"Failed to load {tf_name} timeframe ({interval}): {e}")
            data_dict[tf_name] = pd.DataFrame()

    return data_dict

def align_timeframes(
    data_dict: Dict[str, pd.DataFrame],
    primary_timeframe: str = 'primary'
) -> pd.DataFrame:
    """
    Align multiple timeframes to primary timeframe using forward fill.

    Args:
        data_dict: Dict of DataFrames for each timeframe
        primary_timeframe: Name of primary timeframe to align to

    Returns:
        DataFrame with aligned multi-timeframe features
    """
    if primary_timeframe not in data_dict or data_dict[primary_timeframe].empty:
        raise ValueError(f"Primary timeframe '{primary_timeframe}' not found or empty")

    primary_df = data_dict[primary_timeframe].copy()

    # Add multi-timeframe features
    for tf_name, df in data_dict.items():
        if tf_name == primary_timeframe or df.empty:
            continue

        # Resample higher timeframe to primary timeframe using forward fill
        # This creates a "higher timeframe context" at each primary bar
        for col in df.columns:
            if col in primary_df.columns:
                # Avoid duplicate column names
                aligned_col = f"{col}_{tf_name}"
            else:
                aligned_col = col

            # Forward fill higher timeframe data
            resampled = df[col].reindex(
                primary_df.index,
                method='ffill'  # Use most recent higher timeframe value
            )
            primary_df[aligned_col] = resampled

    return primary_df

def calculate_multi_timeframe_indicators(
    df: pd.DataFrame,
    timeframes: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Calculate indicators that use information from multiple timeframes.

    Args:
        df: Primary timeframe DataFrame with aligned multi-timeframe data
        timeframes: Timeframe configuration

    Returns:
        DataFrame with multi-timeframe indicators
    """
    if timeframes is None:
        timeframes = MULTI_TIMEFRAME_CONFIG

    features = df.copy()

    # Multi-timeframe trend confirmation
    # Check if trend is consistent across timeframes
    trend_confirmations = []
    for tf_name in ['confirmation', 'trend']:
        close_col = f'close_{tf_name}'
        if close_col in features.columns:
            # Calculate trend direction for each timeframe
            ma50 = features[close_col].rolling(50).mean()
            trend_direction = (features[close_col] > ma50).astype(int) - (features[close_col] < ma50).astype(int)
            trend_confirmations.append(trend_direction)

    if len(trend_confirmations) >= 2:
        # Consensus trend: both timeframes agree
        consensus_trend = (trend_confirmations[0] == trend_confirmations[1]).astype(int) * trend_confirmations[0]
        features['multi_tf_trend_consensus'] = consensus_trend.fillna(0)

    # Multi-timeframe volatility
    volatility_measures = []
    for tf_name in timeframes.keys():
        close_col = f'close_{tf_name}' if tf_name != 'primary' else 'close'
        if close_col in features.columns:
            returns = features[close_col].pct_change()
            vol = returns.rolling(20).std()
            volatility_measures.append(vol)

    if len(volatility_measures) >= 2:
        # Volatility divergence (higher timeframe volatility vs primary)
        primary_vol = volatility_measures[0] if len(volatility_measures) > 0 else pd.Series(0, index=features.index)
        higher_vol = volatility_measures[1] if len(volatility_measures) > 1 else pd.Series(0, index=features.index)
        vol_divergence = (higher_vol / (primary_vol + 1e-8)) - 1
        features['volatility_divergence'] = vol_divergence.fillna(0)

    return features

def detect_supply_demand_zones(
    df: pd.DataFrame,
    lookback: int = 50,
    min_body_ratio: float = 0.6
) -> pd.DataFrame:
    """
    Detect supply and demand zones using ICT methodology.

    Args:
        df: DataFrame with OHLC data
        lookback: Lookback period for zone detection
        min_body_ratio: Minimum body to wick ratio for valid OB candles

    Returns:
        DataFrame with supply/demand zone indicators
    """
    # Detect order blocks
    ob_features = detect_order_blocks(df, lookback=lookback, min_body_pct=min_body_ratio)

    # Detect swing points for zone validation
    swings = detect_swing_high_low(df, lookback=3)

    # Calculate zone strength and validity
    close = df['close']
    ob_bull = ob_features['ob_bull_level']
    ob_bear = ob_features['ob_bear_level']

    # Distance to zones (normalized)
    dist_to_demand = (close - ob_bull).abs() / close
    dist_to_supply = (close - ob_bear).abs() / close

    # Zone confluence signals
    ob_confluence = {
        'near_demand_zone': (dist_to_demand < 0.005).astype(float),  # 0.5% tolerance
        'near_supply_zone': (dist_to_supply < 0.005).astype(float),
        'in_demand_zone': (close > ob_bull) & (close < ob_bull * 1.01),  # 1% above demand
        'in_supply_zone': (close < ob_bear) & (close > ob_bear * 0.99),   # 1% below supply
    }

    # Add Fibonacci levels for confluence
    try:
        fib_feats = fib_features(df, lookback=100)
        ob_confluence['fib_dist_nearest'] = fib_feats['fib_dist_nearest']
    except Exception as e:
        print(f"Could not calculate Fibonacci features: {e}")
        ob_confluence['fib_dist_nearest'] = 0.0

    # Combine all zone features
    zone_df = pd.DataFrame(ob_confluence, index=df.index)

    # Add swing information
    zone_df['swing_high'] = swings['swing_high']
    zone_df['swing_low'] = swings['swing_low']

    return zone_df

def create_multi_timeframe_features(
    source: str = 'binance',
    symbol: str = 'BTCUSDT',
    start: Optional[str] = '2023-01-01',
    end: Optional[str] = None,
    timeframes: Optional[Dict[str, str]] = None,
    market: str = 'spot'
) -> pd.DataFrame:
    """
    Create comprehensive multi-timeframe features including supply/demand zones.

    Args:
        source: Data source
        symbol: Trading pair
        start: Start date
        end: End date
        timeframes: Timeframe configuration
        market: Market type

    Returns:
        DataFrame with all multi-timeframe features
    """
    print("Loading multi-timeframe data...")

    # Load data for all timeframes
    data_dict = load_multi_timeframe_data(
        source=source,
        symbol=symbol,
        start=start,
        end=end,
        timeframes=timeframes,
        market=market
    )

    if not data_dict.get('primary', pd.DataFrame()).empty:
        print("Aligning timeframes...")
        # Align timeframes to primary
        aligned_df = align_timeframes(data_dict, 'primary')

        print("Calculating multi-timeframe indicators...")
        # Add multi-timeframe indicators
        mt_features = calculate_multi_timeframe_indicators(aligned_df, timeframes)

        print("Detecting supply/demand zones...")
        # Add supply/demand zone features
        zone_features = detect_supply_demand_zones(mt_features, lookback=50)

        # Combine all features
        final_features = pd.concat([mt_features, zone_features], axis=1)

        print(f"Multi-timeframe feature creation complete: {len(final_features.columns)} features")
        return final_features
    else:
        print("Failed to load primary timeframe data")
        return pd.DataFrame()

# Utility functions for model integration
def get_multi_timeframe_feature_names() -> List[str]:
    """Get list of multi-timeframe feature names for model training."""
    base_features = [
        'multi_tf_trend_consensus',
        'volatility_divergence',
        'near_demand_zone',
        'near_supply_zone',
        'in_demand_zone',
        'in_supply_zone',
        'fib_dist_nearest',
        'swing_high',
        'swing_low'
    ]

    # Add timeframe-specific features
    timeframes = ['confirmation', 'trend', 'context']
    tf_features = []
    for tf in timeframes:
        tf_features.extend([
            f'close_{tf}',
            f'high_{tf}',
            f'low_{tf}',
            f'volume_{tf}',
            f'rsi_14_{tf}',
            f'macd_hist_{tf}'
        ])

    return base_features + tf_features

def prepare_multi_timeframe_data_for_training(
    df: pd.DataFrame,
    target_lookahead: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare multi-timeframe data for ML training.

    Args:
        df: DataFrame with multi-timeframe features
        target_lookahead: Number of periods to look ahead for target

    Returns:
        Tuple of (features, targets, feature_names)
    """
    # Select feature columns (only existing columns)
    feature_columns = [col for col in df.columns if not col.startswith(('open_', 'high_', 'low_', 'close_')) or col == 'close']

    # Add multi-timeframe specific features that exist in the DataFrame
    mt_feature_names = get_multi_timeframe_feature_names()
    available_mt_features = [col for col in mt_feature_names if col in df.columns]
    feature_columns = list(set(feature_columns + available_mt_features))

    # Remove target leakage columns
    exclude_cols = ['future_return', 'target']
    feature_columns = [col for col in feature_columns if col not in exclude_cols]

    # Only select columns that actually exist in the DataFrame
    feature_columns = [col for col in feature_columns if col in df.columns]

    print(f"Selected {len(feature_columns)} features for training")
    print(f"Feature columns: {feature_columns}")

    # Prepare features
    X = df[feature_columns].fillna(0).values

    # Prepare targets (future returns)
    if 'close' in df.columns:
        future_returns = df['close'].shift(-target_lookahead) / df['close'] - 1
        y = (future_returns > 0.001).astype(int)  # 0.1% threshold
        y = y.fillna(0).values
    else:
        y = np.zeros(len(X))

    return X, y, feature_columns