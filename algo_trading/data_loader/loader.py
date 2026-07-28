import os
import io
import json
import time
import math
import gzip
import requests
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict

from algo_trading.indicators.core import add_basic_indicators, ensure_datetime_index

def _flatten_yf_columns(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if ticker is not None and ticker in df.columns.get_level_values(-1):
            try:
                df = df.xs(ticker, axis=1, level=-1)
            except Exception:
                pass
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in df.columns]
            except Exception:
                df.columns = ['_'.join([str(x) for x in c if x is not None]) if isinstance(c, tuple) else str(c) for c in df.columns]
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for key in ['open','high','low','close','volume','time','timestamp','date']:
        for c in list(df.columns):
            if c.lower() == key:
                mapping[c] = key
    df = df.rename(columns=mapping)
    # unify timestamp to DatetimeIndex
    if 'time' in df.columns:
        idx = pd.to_datetime(df['time'], utc=True, errors='coerce')
        df = df.drop(columns=['time'])
        df.index = idx
    elif 'timestamp' in df.columns:
        idx = pd.to_datetime(df['timestamp'], utc=True, errors='coerce', unit='ms')
        df = df.drop(columns=['timestamp'])
        df.index = idx
    elif 'date' in df.columns:
        idx = pd.to_datetime(df['date'], utc=True, errors='coerce')
        df = df.drop(columns=['date'])
        df.index = idx
    else:
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
    keep = [c for c in ['open','high','low','close','volume'] if c in df.columns]
    df = df[keep]
    df = df.sort_index()
    return df


def _resample_ohlcv(df: pd.DataFrame, timeframe: Optional[str]) -> pd.DataFrame:
    if not timeframe:
        return df
    tf_map = { '1m':'1T','3m':'3T','5m':'5T','15m':'15T','30m':'30T','45m':'45T',
               '1h':'1H','2h':'2H','4h':'4H','6h':'6H','12h':'12H',
               '1d':'1D','1w':'1W','1mo':'1M' }
    rule = tf_map.get(timeframe.lower(), timeframe)
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    v = df['volume'].resample(rule).sum() if 'volume' in df.columns else None
    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = ['open','high','low','close','volume'] if v is not None else ['open','high','low','close']
    out = out.dropna(how='any')
    return out


def _normalize(df: pd.DataFrame, method: Optional[str] = None) -> pd.DataFrame:
    if not method:
        return df
    out = df.copy()
    if method == 'zscore':
        for c in ['open','high','low','close','volume']:
            if c in out.columns:
                m = out[c].mean(); s = out[c].std(ddof=0) + 1e-12
                out[c] = (out[c] - m)/s
    elif method == 'minmax':
        for c in ['open','high','low','close','volume']:
            if c in out.columns:
                mn = out[c].min(); mx = out[c].max(); rng = (mx - mn) if (mx - mn)!=0 else 1.0
                out[c] = (out[c] - mn)/rng
    return out


def _add_features(df: pd.DataFrame, add_indicators: bool = True) -> pd.DataFrame:
    if add_indicators:
        df = add_basic_indicators(df)
    return df


def train_test_split_df(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_test = int(n * test_size)
    return df.iloc[:-n_test], df.iloc[-n_test:]

# -----------------------------
# Loaders
# -----------------------------

def load_csv(path: str, timeframe: Optional[str] = None, normalize: Optional[str] = None,
             add_features_flag: bool = True, dropna: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardize_columns(df)
    if dropna:
        df = df.dropna()
    df = _resample_ohlcv(df, timeframe)
    df = _add_features(df, add_indicators=add_features_flag)
    df = _normalize(df, normalize)
    return df


def load_parquet(path: str, timeframe: Optional[str] = None, normalize: Optional[str] = None,
                 add_features_flag: bool = True, dropna: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = _standardize_columns(df)
    if dropna:
        df = df.dropna()
    df = _resample_ohlcv(df, timeframe)
    df = _add_features(df, add_indicators=add_features_flag)
    df = _normalize(df, normalize)
    return df


def load_yfinance(ticker: str, start: Optional[str] = None, end: Optional[str] = None,
                  interval: str = '1d', add_features_flag: bool = True,
                  normalize: Optional[str] = None) -> pd.DataFrame:
    """
    Load data từ Yahoo Finance.
    
    CRITICAL: Yahoo Finance có giới hạn:
    - Hourly data: Chỉ trong 730 ngày gần nhất
    - Daily data: Không giới hạn (có thể lấy nhiều năm)
    - Minute data: Chỉ trong 7 ngày gần nhất
    
    Nếu request vượt quá giới hạn, sẽ tự động chia nhỏ thành chunks.
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise ImportError("yfinance chưa được cài đặt. Vui lòng pip install yfinance")
    
    # Parse dates
    if start:
        start_date = pd.to_datetime(start)
    else:
        start_date = pd.Timestamp.now() - pd.Timedelta(days=365)
    
    if end:
        end_date = pd.to_datetime(end)
    else:
        end_date = pd.Timestamp.now()
    
    # Check interval và date range limits
    interval_lower = interval.lower().strip()
    days_diff = (end_date - start_date).days
    today = pd.Timestamp.now()
    days_from_today = (today - end_date).days  # Số ngày từ end_date đến hôm nay
    
    # Yahoo Finance limits
    if interval_lower in ('1h', '1H', 'hour', 'hourly', '60m', '60'):
        max_days = 730  # Hourly: 730 days max từ HÔM NAY
        
        # CRITICAL: Kiểm tra xem date range có nằm trong 730 ngày gần nhất không
        if days_from_today > max_days:
            import warnings
            # Calculate suggested start date
            suggested_start = (today - pd.Timedelta(days=max_days)).date()
            raise ValueError(
                f"❌ KHÔNG THỂ tải hourly data từ {start_date.date()} đến {end_date.date()}\n"
                f"   Yahoo Finance chỉ cho phép hourly data trong {max_days} ngày GẦN NHẤT tính từ hôm nay.\n"
                f"   End date của bạn ({end_date.date()}) cách hôm nay {days_from_today} ngày (vượt quá {max_days} ngày).\n\n"
                f"💡 GIẢI PHÁP:\n"
                f"   1. Dùng daily data (interval='1d') - không giới hạn số năm\n"
                f"   2. Hoặc điều chỉnh date range về {max_days} ngày gần nhất:\n"
                f"      - Start: {suggested_start}\n"
                f"      - End: {today.date()}\n"
                f"   3. Hoặc dùng data source khác (Binance API, etc.) cho historical hourly data"
            )
        
        if days_diff > max_days:
            import warnings
            warnings.warn(
                f"⚠️ Yahoo Finance chỉ cho phép download hourly data trong {max_days} ngày gần nhất.\n"
                f"   Request: {days_diff} ngày ({start_date.date()} -> {end_date.date()})\n"
                f"   → Tự động điều chỉnh về {max_days} ngày gần nhất (từ {end_date - pd.Timedelta(days=max_days)} đến {end_date})"
            )
            # Adjust start date to max_days before end
            start_date = end_date - pd.Timedelta(days=max_days)
    
    elif interval_lower in ('1m', '1M', 'min', 'minute', '5m', '15m', '30m'):
        max_days = 7  # Minute: 7 days max
        if days_diff > max_days:
            import warnings
            warnings.warn(
                f"⚠️ Yahoo Finance chỉ cho phép download minute data trong {max_days} ngày gần nhất.\n"
                f"   Request: {days_diff} ngày ({start_date.date()} -> {end_date.date()})\n"
                f"   → Tự động điều chỉnh về {max_days} ngày gần nhất"
            )
            start_date = end_date - pd.Timedelta(days=max_days)
    
    # Download với dates đã điều chỉnh
    try:
        df = yf.download(
            ticker, 
            start=start_date.strftime('%Y-%m-%d'), 
            end=end_date.strftime('%Y-%m-%d'), 
            interval=interval, 
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        # Nếu vẫn lỗi, thử download với period thay vì start/end
        if '730 days' in str(e) or 'YFPricesMissingError' in str(type(e).__name__):
            import warnings
            warnings.warn(
                f"⚠️ Không thể download {interval} data cho range {start_date.date()} -> {end_date.date()}\n"
                f"   Yahoo Finance giới hạn: {max_days if interval_lower in ('1h', 'hour') else 7} ngày cho {interval} data\n"
                f"   → Thử download với period='max' hoặc dùng daily data thay thế"
            )
            # Thử với period='max' (lấy tối đa có thể)
            try:
                df = yf.download(ticker, period='max', interval=interval, progress=False, auto_adjust=True)
            except:
                # Nếu vẫn fail, suggest daily
                raise ValueError(
                    f"Không thể download {interval} data. "
                    f"Yahoo Finance chỉ cho phép {interval} data trong {max_days} ngày gần nhất.\n"
                    f"Khuyến nghị: Dùng interval='1d' (daily) để lấy data lịch sử dài hơn."
                )
        else:
            raise
    
    if df.empty:
        raise ValueError("Không có dữ liệu tải về từ yfinance")
    if isinstance(df.index, pd.MultiIndex):
        # lấy level cuối (Datetime)
        try:
            df.index = pd.to_datetime(df.index.get_level_values(-1), utc=True)
        except Exception:
            df = df.reset_index()
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                df = df.set_index('Date')
    df = _flatten_yf_columns(df, ticker)
    # chuẩn hóa tên cột
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'})
    # nếu còn thiếu cột volume (một số crypto), tạo volume=0
    if 'volume' not in df.columns and 'Volume' in df.columns:
        df['volume'] = df['Volume']
    # chọn các cột cần thiết
    cols = [c for c in ['open','high','low','close','volume'] if c in df.columns]
    df = df[cols]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.dropna()
    df = _add_features(df, add_indicators=add_features_flag)
    df = _normalize(df, normalize)
    return df


BINANCE_REST = {
    'spot': 'https://api.binance.com/api/v3/klines',
    'futures': 'https://fapi.binance.com/fapi/v1/klines',
}


def _binance_klines(symbol: str, interval: str = '1h', start_ms: Optional[int] = None,
                    end_ms: Optional[int] = None, market: str = 'spot', limit: int = 1000) -> pd.DataFrame:
    url = BINANCE_REST.get(market, BINANCE_REST['spot'])
    params = {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}
    if start_ms is not None:
        params['startTime'] = int(start_ms)
    if end_ms is not None:
        params['endTime'] = int(end_ms)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # each entry: [ openTime, open, high, low, close, volume, closeTime, qav, trades, takerBase, takerQuote, ignore ]
    cols = ['open_time','open','high','low','close','volume','close_time','qav','trades','taker_base','taker_quote','ignore']
    df = pd.DataFrame(data, columns=cols)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.set_index('open_time')
    df = df[['open','high','low','close','volume']]
    return df


def load_binance(symbol: str, interval: str = '1h', start: Optional[str] = None, end: Optional[str] = None,
                 market: str = 'spot', add_features_flag: bool = True, normalize: Optional[str] = None) -> pd.DataFrame:
    # paginate
    if start:
        start_ts = int(pd.Timestamp(start, tz='UTC').timestamp() * 1000)
    else:
        start_ts = None
    if end:
        end_ts = int(pd.Timestamp(end, tz='UTC').timestamp() * 1000)
    else:
        end_ts = None

    frames = []
    current = start_ts
    while True:
        df = _binance_klines(symbol, interval=interval, start_ms=current, end_ms=end_ts, market=market)
        if df.empty:
            break
        frames.append(df)
        last_end = int(df.index[-1].timestamp()*1000)
        # advance by 1ms to avoid overlap
        current = last_end + 1
        if end_ts is not None and current >= end_ts:
            break
        # stop if we got less than limit results (end reached)
        if len(df) < 1000:
            break
        time.sleep(0.2)  # be gentle
    if not frames:
        raise ValueError('Không lấy được dữ liệu từ Binance')
    df_all = pd.concat(frames).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep='first')]
    df_all = df_all.dropna()
    df_all = _add_features(df_all, add_indicators=add_features_flag)
    df_all = _normalize(df_all, normalize)
    return df_all


# public API

def load_data(source: str, **kwargs) -> pd.DataFrame:
    source = source.lower()
    if source == 'csv':
        return load_csv(kwargs['path'], timeframe=kwargs.get('timeframe'), normalize=kwargs.get('normalize'),
                        add_features_flag=kwargs.get('add_features', True), dropna=kwargs.get('dropna', True))
    if source == 'parquet':
        return load_parquet(kwargs['path'], timeframe=kwargs.get('timeframe'), normalize=kwargs.get('normalize'),
                            add_features_flag=kwargs.get('add_features', True), dropna=kwargs.get('dropna', True))
    if source == 'yfinance':
        return load_yfinance(kwargs['ticker'], start=kwargs.get('start'), end=kwargs.get('end'),
                             interval=kwargs.get('interval','1d'), add_features_flag=kwargs.get('add_features', True),
                             normalize=kwargs.get('normalize'))
    if source == 'binance':
        return load_binance(kwargs['symbol'], interval=kwargs.get('interval','1h'), start=kwargs.get('start'), end=kwargs.get('end'),
                            market=kwargs.get('market','spot'), add_features_flag=kwargs.get('add_features', True),
                            normalize=kwargs.get('normalize'))
    raise ValueError('Nguồn dữ liệu không hỗ trợ')

