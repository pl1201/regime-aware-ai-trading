import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    '''Add basic technical indicators'''
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['WMA20'] = df['close'].rolling(window=20).apply(lambda x: np.dot(x, np.arange(1, 21)) / np.sum(np.arange(1, 21)), raw=True)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))

    # MACD
    df['EMA12'] = df['close'].ewm(span=12).mean()
    df['EMA26'] = df['close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9).mean()
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']

    # Bollinger Bands
    df['BB_MID'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_UPPER'] = df['BB_MID'] + (bb_std * 2)
    df['BB_LOWER'] = df['BB_MID'] - (bb_std * 2)

    # ATR
    df['TR'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['ATR14'] = df['TR'].rolling(window=14).mean()

    # VWAP
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # Z-Score
    df['Z20'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()

    return df

def detect_order_blocks(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    '''Detect ICT Order Blocks using swing highs/lows'''
    df = df.copy()

    # Find swing highs and lows
    df['swing_high'] = df['high'].rolling(window=lookback, center=True).apply(lambda x: x[lookback//2] if x[lookback//2] == max(x) else np.nan, raw=False)
    df['swing_low'] = df['low'].rolling(window=lookback, center=True).apply(lambda x: x[lookback//2] if x[lookback//2] == min(x) else np.nan, raw=False)

    # Forward fill to get last known OB level
    df['ob_bull_level'] = df['swing_low'].ffill().bfill()
    df['ob_bear_level'] = df['swing_high'].ffill().bfill()

    # Distance from current price to OB levels
    df['price_to_ob_bull'] = (df['close'] - df['ob_bull_level']) / df['close']
    df['price_to_ob_bear'] = (df['close'] - df['ob_bear_level']) / df['close']

    return df[['ob_bull_level', 'ob_bear_level', 'price_to_ob_bull', 'price_to_ob_bear']]

def fib_features(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    '''Calculate Fibonacci confluence levels from recent swing'''
    df = df.copy()

    # Find recent swing high and low
    swing_high = df['high'].rolling(window=lookback).max().iloc[-1]
    swing_low = df['low'].rolling(window=lookback).min().iloc[-1]

    # Fibonacci levels
    diff = swing_high - swing_low
    fib_levels = {
        'fib_236': swing_high - 0.236 * diff,
        'fib_382': swing_high - 0.382 * diff,
        'fib_500': swing_high - 0.500 * diff,
        'fib_618': swing_high - 0.618 * diff,
        'fib_786': swing_high - 0.786 * diff
    }

    # Find closest level for each row
    results = []
    for idx in df.index:
        current_price = df.loc[idx, 'close']
        distances = {level: abs(current_price - value) for level, value in fib_levels.items()}
        closest_level = min(distances, key=distances.get)
        fib_dist_nearest = distances[closest_level] / diff  # normalized distance
        results.append({
            'fib_dist_nearest': fib_dist_nearest,
            'fib_zone': closest_level
        })

    return pd.DataFrame(results, index=df.index)

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    '''Add all indicators including ICT and Fib'''
    df = add_basic_indicators(df)

    # Add ICT Order Blocks
    ob_df = detect_order_blocks(df)
    df = pd.concat([df, ob_df], axis=1)

    # Add Fibonacci confluence
    fib_df = fib_features(df)
    df = pd.concat([df, fib_df], axis=1)

    return df

# --- Main script to generate dataset ---
if __name__ == "__main__":
    data_dir = "data"
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)

    # Load all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    all_features = []
    all_labels = []

    for file in csv_files:
        print(f"Processing {file}...")
        df = pd.read_csv(os.path.join(data_dir, file), index_col=0, parse_dates=True)

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {file}: missing required columns")
            continue

        # Add all indicators
        df = add_all_indicators(df)

        # Create target: 2% return in next 5 candles
        df['target'] = df['close'].shift(-5) / df['close'] - 1
        df['label'] = (df['target'] > 0.02).astype(int)

        # Keep only rows with complete features and labels
        feature_cols = [
            'SMA20', 'EMA20', 'WMA20', 'RSI14', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
            'BB_UPPER', 'BB_LOWER', 'ATR14', 'VWAP', 'Z20',
            'ob_bull_level', 'ob_bear_level', 'price_to_ob_bull', 'price_to_ob_bear',
            'fib_dist_nearest'
        ]

        df_clean = df[feature_cols + ['label']].dropna()

        if len(df_clean) > 0:
            all_features.append(df_clean[feature_cols])
            all_labels.append(df_clean['label'])

    # Combine all data
    if len(all_features) > 0:
        X = pd.concat(all_features, axis=0)
        y = pd.concat(all_labels, axis=0)

        # Save features and labels
        X.to_csv(os.path.join(output_dir, "X_features.csv"), index=False)
        y.to_csv(os.path.join(output_dir, "y_labels.csv"), index=False)

        print(f"Dataset created: {len(X)} samples")
        print(f"Feature shape: {X.shape}")
        print(f"Label distribution: {y.value_counts().to_dict()}")
    else:
        print("❌ No data processed. Check input files.")