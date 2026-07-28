"""
Improved Label Creation Module

Triển khai phương pháp tạo labels dựa trên forward-looking returns với threshold động
theo mô tả trong PHUONG_PHAP_LUONG_HOA.md section 9.2.

Công thức:
y_t = {
    +1 nếu (P_{t+h} - P_t) / P_t > θ_+
    -1 nếu (P_{t+h} - P_t) / P_t < -θ_-
    0  ngược lại
}

Trong đó:
- h: horizon (3-5 bars)
- θ_+ = k × σ_t, θ_- = k × σ_t (threshold động dựa trên volatility)
- σ_t: volatility (ATR hoặc std của returns) tại thời điểm t
- k: hệ số (thường 1.5-2.0)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def calculate_volatility(
    df: pd.DataFrame,
    method: str = 'atr',
    window: int = 14
) -> pd.Series:
    """
    Tính volatility (σ_t) để làm threshold động
    
    Args:
        df: DataFrame với price data (cần có 'close', 'high', 'low')
        method: 'atr' hoặc 'std' (standard deviation của returns)
        window: Window size cho rolling calculation
    
    Returns:
        Series với volatility values
    """
    if method == 'atr':
        # Sử dụng ATR (Average True Range)
        from algo_trading.indicators.volatility import atr
        atr_values = atr(df, window=window)
        # Normalize ATR bằng giá (ATR/Price)
        volatility = atr_values / df['close']
    elif method == 'std':
        # Sử dụng standard deviation của returns
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window).std()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'atr' or 'std'")
    
    return volatility.fillna(method='bfill').fillna(method='ffill')


def create_labels_improved(
    df: pd.DataFrame,
    horizon: int = 5,
    k: float = 1.75,
    volatility_method: str = 'atr',
    volatility_window: int = 14,
    min_volatility_threshold: float = 0.005,
    use_relative_atr: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """
    Tạo labels cải thiện dựa trên forward-looking returns với threshold động
    
    Args:
        df: DataFrame với price data (cần có 'close', 'high', 'low')
        horizon: Số bars nhìn về phía trước (h) - thường 3-5
        k: Hệ số cho threshold (thường 1.5-2.0)
        volatility_method: 'atr' hoặc 'std'
        volatility_window: Window size cho volatility calculation
        min_volatility_threshold: Ngưỡng tối thiểu ATR/Price (ví dụ 0.5% = 0.005)
                                  Chỉ giữ samples có volatility >= threshold này
        use_relative_atr: Nếu True, sử dụng ATR/Price làm volatility metric
    
    Returns:
        Tuple (labels, forward_returns)
        - labels: Series với values {-1, 0, 1}
        - forward_returns: Series với forward returns (P_{t+h} - P_t) / P_t
    """
    # Tính forward returns
    prices = df['close']
    forward_prices = prices.shift(-horizon)
    forward_returns = (forward_prices - prices) / prices
    
    # Tính volatility
    volatility = calculate_volatility(df, method=volatility_method, window=volatility_window)
    
    # Tính dynamic thresholds
    threshold_positive = k * volatility
    threshold_negative = -k * volatility
    
    labels = pd.Series(0, index=df.index, dtype=int)
    
    labels.loc[forward_returns > threshold_positive] = 1
    
    labels.loc[forward_returns < threshold_negative] = -1
    
    
    # Lọc bỏ samples có volatility quá thấp (sideways market)
    if use_relative_atr:
        # Tính ATR/Price
        from algo_trading.indicators.volatility import atr
        atr_values = atr(df, window=volatility_window)
        relative_atr = atr_values / prices
        
        # Chỉ giữ samples có relative ATR >= min_volatility_threshold
        low_volatility_mask = relative_atr < min_volatility_threshold
        labels.loc[low_volatility_mask] = np.nan
    
    # Fill NaN với 0 (hoặc có thể drop)
    labels = labels.fillna(0).astype(int)
    
    return labels, forward_returns


def filter_low_volatility_samples(
    df: pd.DataFrame,
    labels: pd.Series,
    features: Optional[pd.DataFrame] = None,
    min_volatility_threshold: float = 0.005,
    volatility_window: int = 14
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
    """
    Lọc bỏ các samples trong giai đoạn thị trường đi ngang (sideways) 
    hoặc có volatility quá thấp
    
    Args:
        df: DataFrame với price data
        labels: Series với labels
        features: Optional DataFrame với features (sẽ được filter cùng)
        min_volatility_threshold: Ngưỡng tối thiểu ATR/Price (ví dụ 0.5% = 0.005)
        volatility_window: Window size cho ATR calculation
    
    Returns:
        Tuple (df_filtered, labels_filtered, features_filtered)
    """
    from algo_trading.indicators.volatility import atr
    
    # Tính ATR/Price
    atr_values = atr(df, window=volatility_window)
    relative_atr = atr_values / df['close']
    
    # Tạo mask: chỉ giữ samples có relative ATR >= threshold
    mask = relative_atr >= min_volatility_threshold
    
    # Filter
    df_filtered = df.loc[mask].copy()
    labels_filtered = labels.loc[mask].copy()
    
    if features is not None:
        features_filtered = features.loc[mask].copy()
    else:
        features_filtered = None
    
    return df_filtered, labels_filtered, features_filtered


def create_labels_with_filtering(
    df: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
    horizon: int = 5,
    k: float = 1.75,
    volatility_method: str = 'atr',
    volatility_window: int = 14,
    min_volatility_threshold: float = 0.005,
    apply_volatility_filter: bool = True
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Tạo labels cải thiện và lọc bỏ low volatility samples
    
    Args:
        df: DataFrame với price data
        features: Optional DataFrame với features
        horizon: Số bars nhìn về phía trước
        k: Hệ số cho threshold
        volatility_method: 'atr' hoặc 'std'
        volatility_window: Window size cho volatility
        min_volatility_threshold: Ngưỡng tối thiểu ATR/Price
        apply_volatility_filter: Có áp dụng volatility filter không
    
    Returns:
        Tuple (labels, forward_returns, df_filtered, features_filtered)
    """
    # Tạo labels
    labels, forward_returns = create_labels_improved(
        df=df,
        horizon=horizon,
        k=k,
        volatility_method=volatility_method,
        volatility_window=volatility_window,
        min_volatility_threshold=min_volatility_threshold,
        use_relative_atr=True
    )
    
    if apply_volatility_filter:
        df_filtered, labels_filtered, features_filtered = filter_low_volatility_samples(
            df=df,
            labels=labels,
            features=features,
            min_volatility_threshold=min_volatility_threshold,
            volatility_window=volatility_window
        )
    else:
        df_filtered = df.copy()
        labels_filtered = labels.copy()
        features_filtered = features.copy() if features is not None else None
    
    return labels_filtered, forward_returns.loc[labels_filtered.index], df_filtered, features_filtered

