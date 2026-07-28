"""
H1 Feature Engineering for algo_trading

Optimized for hourly timeframe - replaces M15 features.
"""
import numpy as np
import pandas as pd
from typing import List


class H1Features:
    """Build features optimized for H1 timeframe."""
    
    FEATURE_NAMES = [
        'ema_cross_9_21', 'ema_cross_21_50', 'price_vs_ema50', 'price_vs_ema200',
        'mom_4', 'mom_8', 'mom_12', 'mom_24', 'mom_48',
        'rsi', 'atr_pct', 'adx', 'di_diff',
        'bb_pos', 'bb_width', 'vol_ratio', 'range_pos_24', 'mtf_align'
    ]
    
    def __init__(self):
        self.feature_names = self.FEATURE_NAMES
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build H1 features from OHLCV data.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
        
        Returns:
            DataFrame with 18 features
        """
        data = df.copy()
        
        # === EMAs ===
        for p in [9, 21, 50, 100, 200]:
            data[f'ema_{p}'] = data['close'].ewm(span=p, adjust=False).mean()
        
        # EMA crosses (normalized)
        data['ema_cross_9_21'] = (data['ema_9'] - data['ema_21']) / data['close']
        data['ema_cross_21_50'] = (data['ema_21'] - data['ema_50']) / data['close']
        data['price_vs_ema50'] = (data['close'] - data['ema_50']) / data['close']
        data['price_vs_ema200'] = (data['close'] - data['ema_200']) / data['close']
        
        # === MOMENTUM ===
        for p in [4, 8, 12, 24, 48]:  # 4h to 2 days
            data[f'mom_{p}'] = data['close'].pct_change(p)
        
        # === RSI ===
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        # === ATR ===
        tr = pd.concat([
            data['high'] - data['low'],
            abs(data['high'] - data['close'].shift()),
            abs(data['low'] - data['close'].shift())
        ], axis=1).max(axis=1)
        data['atr'] = tr.rolling(14).mean()
        data['atr_pct'] = data['atr'] / data['close']
        
        # === ADX ===
        up = data['high'] - data['high'].shift()
        down = data['low'].shift() - data['low']
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        plus_di = pd.Series(plus_dm, index=data.index).rolling(14).mean() / data['atr'] * 100
        minus_di = pd.Series(minus_dm, index=data.index).rolling(14).mean() / data['atr'] * 100
        
        data['adx'] = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100).rolling(14).mean()
        data['di_diff'] = plus_di - minus_di
        
        # === BOLLINGER BANDS ===
        data['bb_mid'] = data['close'].rolling(20).mean()
        data['bb_std'] = data['close'].rolling(20).std()
        data['bb_pos'] = (data['close'] - data['bb_mid']) / (2 * data['bb_std'] + 1e-10)
        data['bb_width'] = data['bb_std'] * 4 / data['bb_mid']
        
        # === VOLUME ===
        data['vol_ratio'] = data['volume'] / (data['volume'].rolling(20).mean() + 1)
        
        # === STRUCTURE ===
        high_24 = data['high'].rolling(24).max()
        low_24 = data['low'].rolling(24).min()
        data['range_pos_24'] = (data['close'] - low_24) / (high_24 - low_24 + 1e-10)
        
        # === MTF ALIGNMENT ===
        data['mtf_align'] = sum([
            (data[f'mom_{p}'] > 0).astype(int) for p in [4, 8, 12, 24, 48]
        ]) / 5
        
        return data[self.FEATURE_NAMES].copy()
    
    def get_regime(self, features: pd.DataFrame) -> np.ndarray:
        """
        Detect market regime.
        
        Returns:
            1 = trending (ADX > 25)
            0 = ranging (ADX <= 25)
        """
        return (features['adx'] > 25).astype(int).values


# Backward compatibility
def build_h1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function."""
    return H1Features().build_features(df)
