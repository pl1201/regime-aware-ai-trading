"""
Composite indicators and utilities: add_basic_indicators
"""
import pandas as pd
from .moving_averages import sma, ema, wma
from .rsi import rsi
from .macd import macd
from .bollinger_bands import bollinger_bands
from .zscore import zscore
from .volatility import atr
from .volume import vwap
from .utils import ensure_datetime_index


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:

    df = ensure_datetime_index(df.copy())
    close = df['close']
    df['SMA20'] = sma(close, 20)
    df['EMA20'] = ema(close, 20)
    df['WMA20'] = wma(close, 20)
    df['RSI14'] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    df['MACD'] = macd_line
    df['MACD_SIGNAL'] = signal_line
    df['MACD_HIST'] = hist
    m, u, l = bollinger_bands(close)
    df['BB_MID'] = m
    df['BB_UPPER'] = u
    df['BB_LOWER'] = l
    df['ATR14'] = atr(df, 14)
    df['VWAP'] = vwap(df)
    df['Z20'] = zscore(close, 20)
    return df

