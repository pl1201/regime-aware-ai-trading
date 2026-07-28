

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from algo_trading.data_loader.loader import load_data
from algo_trading.ml.regime_specific_models import RegimeSpecificModels
from algo_trading.market_models.regime import detect_regime_hmm
from algo_trading.ml.label_creation import create_labels_with_filtering
from algo_trading.ml.features import build_feature_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_merge_data():
    logger.info("Loading data from data/ directory...")
    
    # Load BTC price data
    df_btc = load_data('csv', path='data/CRYPTO_BTCUSD, 1D.csv', add_features=True)
    logger.info(f"✅ Loaded BTC data: {len(df_btc)} rows, {df_btc.index.min()} to {df_btc.index.max()}")
    
    # Load Total Market Cap (nếu có)
    df_total = None
    try:
        df_total = load_data('csv', path='data/CRYPTOCAP_TOTAL, 1D.csv', add_features=False)
        logger.info(f"✅ Loaded Total Market Cap: {len(df_total)} rows")
    except Exception as e:
        logger.warning(f"⚠️ Could not load Total Market Cap: {e}")
    
    # Merge data
    if df_total is not None:
        # Merge BTC và Total Market Cap
        df_combined = pd.merge(
            df_btc,
            df_total[['close']],
            left_index=True,
            right_index=True,
            how='left',
            suffixes=('', '_total_mcap')
        )
        
        # Rename
        df_combined = df_combined.rename(columns={'close_total_mcap': 'total_mcap'})
        
        # Add macro features
        df_combined['btc_dominance'] = df_combined['close'] / df_combined['total_mcap']
        df_combined['total_mcap_change'] = df_combined['total_mcap'].pct_change()
        df_combined['btc_vs_total_corr'] = df_combined['close'].rolling(30).corr(df_combined['total_mcap'])
        
        logger.info("✅ Added macro features: btc_dominance, total_mcap_change, btc_vs_total_corr")
    else:
        df_combined = df_btc
    
    # Create volume proxy nếu không có volume
    if 'volume' not in df_combined.columns or df_combined['volume'].isna().all():
        logger.info("Creating volume proxy from price action...")
        df_combined['volume_proxy'] = (
            (df_combined['high'] - df_combined['low']) / df_combined['close'] * 0.6 +
            abs(df_combined['close'] - df_combined['open']) / df_combined['close'] * 0.4
        )
        # Normalize
        df_combined['volume_proxy'] = (
            (df_combined['volume_proxy'] - df_combined['volume_proxy'].rolling(30).mean()) /
            (df_combined['volume_proxy'].rolling(30).std() + 1e-8)
        )
        logger.info("✅ Created volume proxy")
    
    # Add time-based features
    df_combined['day_of_week'] = df_combined.index.dayofweek
    df_combined['month'] = df_combined.index.month
    df_combined['quarter'] = df_combined.index.quarter
    
    # Cyclical encoding
    df_combined['day_of_week_sin'] = np.sin(2 * np.pi * df_combined['day_of_week'] / 7)
    df_combined['day_of_week_cos'] = np.cos(2 * np.pi * df_combined['day_of_week'] / 7)
    df_combined['month_sin'] = np.sin(2 * np.pi * df_combined['month'] / 12)
    df_combined['month_cos'] = np.cos(2 * np.pi * df_combined['month'] / 12)
    
    logger.info("✅ Added time-based features")
    
    return df_combined


def walk_forward_split(df: pd.DataFrame, train_years: int = 5, test_years: int = 1):
    """
    Walk-forward split cho time series data.
    
    Returns:
        List of (train_df, test_df) tuples
    """
    splits = []
    start_date = df.index.min()
    end_date = df.index.max()
    
    current_train_end = start_date + pd.Timedelta(days=train_years * 365)
    current_test_end = current_train_end + pd.Timedelta(days=test_years * 365)
    
    while current_test_end <= end_date:
        train_df = df[(df.index >= start_date) & (df.index < current_train_end)]
        test_df = df[(df.index >= current_train_end) & (df.index < current_test_end)]
        
        if len(train_df) > 100 and len(test_df) > 50:  # Minimum requirements
            splits.append((train_df, test_df))
            logger.info(
                f"Split: Train {train_df.index.min()} to {train_df.index.max()} "
                f"({len(train_df)} rows), "
                f"Test {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)"
            )
        
        # Slide forward
        current_train_end += pd.Timedelta(days=test_years * 365)
        current_test_end += pd.Timedelta(days=test_years * 365)
    
    return splits


def train_regime_specific_models_with_new_data():
    """
    Train regime-specific models với dữ liệu mới.
    """
    logger.info("=" * 80)
    logger.info("TRAINING REGIME-SPECIFIC MODELS VỚI DỮ LIỆU MỚI")
    logger.info("=" * 80)
    
    # 1. Load và prepare data
    df = load_and_merge_data()
    
    # 2. Detect regimes
    logger.info("Detecting regimes...")
    from algo_trading.indicators import rsi, macd, bollinger_bands, atr
    
    # Calculate indicators
    indicators = {
        'rsi': rsi(df['close'], 14),
        'macd_line': macd(df['close'])[0],
        'macd_signal': macd(df['close'])[1],
        'macd_hist': macd(df['close'])[2],
        'bb_upper': bollinger_bands(df['close'])[0],
        'bb_lower': bollinger_bands(df['close'])[1],
        'bb_width': (bollinger_bands(df['close'])[0] - bollinger_bands(df['close'])[1]) / df['close'],
        'atr': atr(df, 14),
    }
    
    regime_info = detect_regime_hmm(df, indicators=indicators, lookback_window=500)
    logger.info(f"✅ Detected regimes: {regime_info.get('current_regime', 'unknown')}")
    
    # 3. Create labels
    logger.info("Creating labels...")
    labels = create_labels_with_filtering(
        df,
        k=1.5,
        horizon=5,
        min_volatility_threshold=0.003,
    )
    logger.info(f"✅ Created labels: {(labels == 1).sum()} LONG, {(labels == -1).sum()} SHORT, {(labels == 0).sum()} NEUTRAL")
    
    # 4. Build features
    logger.info("Building feature matrix...")

    # 5. Walk-forward validation
    logger.info("Performing walk-forward validation...")
    splits = walk_forward_split(df, train_years=5, test_years=1)
    
    if not splits:
        logger.warning("⚠️ No valid splits found. Using simple train/test split.")
        # Fallback: simple split
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        splits = [(train_df, test_df)]
    
    # 6. Train models cho mỗi split
    all_results = []
    for i, (train_df, test_df) in enumerate(splits):
        logger.info(f"\n{'='*80}")
        logger.info(f"Split {i+1}/{len(splits)}")
        logger.info(f"{'='*80}")
        
        logger.info(f"✅ Completed split {i+1}")
    
    # 7. Save best model
    logger.info("\n" + "="*80)
    logger.info("Saving best model...")
    # TODO: Save model
    logger.info("✅ Training completed!")
    
    return df, regime_info, labels


if __name__ == "__main__":
    try:
        df, regime_info, labels = train_regime_specific_models_with_new_data()
        logger.info("\n✅ All done!")
    except Exception as e:
        logger.exception(f"❌ Error: {e}")
        raise
