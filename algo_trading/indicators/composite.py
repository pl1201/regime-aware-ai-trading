import pandas as pd
from .moving_averages import sma, ema, wma
from .rsi import rsi
from .bollinger_bands import bollinger_bands
from .volatility import atr
from .volume import vwap
from .zscore import zscore
from .ict import detect_order_blocks

def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    '''Add basic technical indicators'''
    df['SMA20'] = sma(df['close'], 20)
    df['EMA20'] = ema(df['close'], 20)
    df['WMA20'] = wma(df['close'], 20)

    # RSI
    df['RSI14'] = rsi(df['close'], 14)

    # MACD
    df['EMA12'] = ema(df['close'], 12)
    df['EMA26'] = ema(df['close'], 26)
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_SIGNAL'] = ema(df['MACD'], 9)
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']

    # Bollinger Bands
    mid, upper, lower = bollinger_bands(df['close'], 20, 2)
    df['BB_MID'] = mid
    df['BB_UPPER'] = upper
    df['BB_LOWER'] = lower

    # ATR
    df['ATR14'] = atr(df, 14)

    # VWAP
    df['VWAP'] = vwap(df)

    # Z-Score
    df['Z20'] = zscore(df['close'], 20)

    return df

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    '''Add all indicators including ICT and Fib'''
    df = add_basic_indicators(df)

    # Add ICT Order Blocks
    ob_features = detect_order_blocks(df)
    df['ob_bull_level'] = ob_features['ob_bull_level']
    df['ob_bear_level'] = ob_features['ob_bear_level']
    df['price_to_ob_bull'] = (df['close'] - df['ob_bull_level']) / df['close']
    df['price_to_ob_bear'] = (df['close'] - df['ob_bear_level']) / df['close']

    # Add Fibonacci confluence
    from .fibonacci import fib_features

    fib_df = fib_features(df)
    df['fib_dist_nearest'] = fib_df['fib_dist_nearest']
    df['fib_zone'] = fib_df['fib_zone']

    return df
