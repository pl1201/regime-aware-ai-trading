"""Utility functions for Streamlit UI"""
from typing import Dict, Any
import pandas as pd
from algo_trading.data_loader.loader import load_data


def load_dataframe(source: str, **kwargs) -> pd.DataFrame:
    return load_data(source, **kwargs)


def get_load_kwargs(source: str, ticker: str = None, symbol: str = None, 
                    interval: str = None, start: str = None, end: str = None,
                    market: str = 'spot', path: str = None) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if source == 'yfinance':
        kwargs = {'ticker': ticker, 'interval': interval, 'start': start or None, 'end': end or None}
    elif source == 'binance':
        kwargs = {'symbol': symbol, 'interval': interval, 'start': start or None, 'end': end or None, 'market': market}
    elif source in ('csv','parquet'):
        kwargs = {'path': path}
    return kwargs


def load_df_from_sidebar_config(source: str, ticker: str = None, symbol: str = None,
                                interval: str = None, start: str = None, end: str = None,
                                market: str = 'spot', path: str = None) -> pd.DataFrame:
    kwargs = get_load_kwargs(source, ticker, symbol, interval, start, end, market, path)
    return load_dataframe(source, **kwargs)

