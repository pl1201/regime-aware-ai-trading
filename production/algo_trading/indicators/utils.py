"""
Utility functions for indicators (no dependencies on other indicator modules)
"""
import pandas as pd


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has DatetimeIndex"""
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df = df.set_index(pd.to_datetime(df['time']))
            df = df.drop(columns=['time'])
        else:
            df.index = pd.to_datetime(df.index)
    return df


































































