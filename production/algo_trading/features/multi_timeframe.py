"""
Multi-Timeframe Features Module

Thêm features từ nhiều khung thời gian khác nhau để:
- Nhận diện trend chính xác hơn
- Giảm false signals
- Timing entry tốt hơn
- Phát hiện divergence sớm
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

# Import indicators từ hệ thống
try:
    from algo_trading.indicators.core import rsi, macd, bb_width, atr
    HAS_INDICATORS = True
except ImportError:
    HAS_INDICATORS = False

try:
    from algo_trading.indicators.ict import (
        detect_order_blocks,
        ob_confluence_signal,
        fib_levels_from_swing,
        fib_features,
    )
    HAS_ICT_FEATURES = True
except Exception:
    HAS_ICT_FEATURES = False


class MultiTimeframeFeatureGenerator:
    """
    Tạo features từ nhiều khung thời gian
    """

    def __init__(
        self,
        base_timeframe: str = '5T',  # 5 phút
        multi_timeframes: List[str] = None,
        indicators: List[str] = None
    ):
        """
        Args:
            base_timeframe: Khung thời gian chính
            multi_timeframes: Danh sách khung thời gian phụ (ví dụ: ['15T', '30T', '1H', '4H'])
            indicators: Danh sách indicators cần tính (default: ['rsi', 'macd', 'bb', 'atr'])
        """
        self.base_timeframe = base_timeframe
        self.multi_timeframes = multi_timeframes or ['15T', '30T', '1H', '4H']
        self.indicators = indicators or ['rsi', 'macd', 'bb', 'atr', 'volume']

        # Map timeframe string sang pandas offset
        self.tf_map = {
            '15T': '15T', '15min': '15T',
            '30T': '30T', '30min': '30T',
            '1H': '1H', '1h': '1H',
            '2H': '2H', '2h': '2H',
            '4H': '4H', '4h': '4H',
            '6H': '6H', '6h': '6H',
            '12H': '12H', '12h': '12H',
            '1D': '1D', '1d': '1D',
        }

    def resample_to_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:

        tf = self.tf_map.get(timeframe, timeframe)

        o = df['open'].resample(tf).first()
        h = df['high'].resample(tf).max()
        l = df['low'].resample(tf).min()
        c = df['close'].resample(tf).last()
        v = df['volume'].resample(tf).sum() if 'volume' in df.columns else None

        out = pd.concat([o, h, l, c], axis=1)
        if v is not None:
            out['volume'] = v

        out = out.dropna(how='any')
        return out

    def calculate_rsi_features(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        if not HAS_INDICATORS:
            # Fallback implementation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        return rsi(df['close'], period)

    def calculate_macd_features(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        if not HAS_INDICATORS:
            # Fallback implementation
            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram

        return macd(df['close'], fast, slow, signal)

    def calculate_bb_features(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        if not HAS_INDICATORS:
            # Fallback implementation
            middle = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            width = (upper - lower) / middle
            return upper, lower, width

        return bb_width(df['close'], period, std_dev)

    def calculate_atr_features(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """Calculate ATR"""
        if not HAS_INDICATORS:
            # Fallback implementation
            high = df['high']
            low = df['low']
            close = df['close']

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(period).mean()

        return atr(df, period)

    def _add_ict_fibo_features(self, df_tf: pd.DataFrame, tf: str) -> pd.DataFrame:
        """
        Tính và gắn các feature ICT & Fibonacci cho khung thời gian `tf`.
        Trả về DataFrame chỉ chứa các cột mới (để merge vào df chính).
        """
        if not HAS_ICT_FEATURES:
            return pd.DataFrame(index=df_tf.index)

        ob_raw = detect_order_blocks(df_tf, lookback=40, min_body_pct=0.004)
        if isinstance(ob_raw, pd.DataFrame):
            ob_df = ob_raw.copy()
        else:
            ob_df = pd.DataFrame(ob_raw, index=df_tf.index)

        ob_bull = ob_df.get('ob_bull_level', pd.Series(np.nan, index=df_tf.index))
        ob_bear = ob_df.get('ob_bear_level', pd.Series(np.nan, index=df_tf.index))

        confluence_raw = ob_confluence_signal(df_tf, ob_bull=ob_bull, ob_bear=ob_bear)
        if isinstance(confluence_raw, pd.DataFrame):
            confluence_df = confluence_raw.copy()
        else:
            confluence_df = pd.DataFrame(confluence_raw, index=df_tf.index)

        fib_df = fib_features(df_tf, lookback=180)

        confluence_df['ob_confluence'] = confluence_df[
            [c for c in ['ob_long_zone', 'ob_short_zone'] if c in confluence_df.columns]
        ].max(axis=1).fillna(0)
        confluence_df['fib_confluence'] = (1.0 - np.clip(
            fib_df.get('fib_dist_nearest', pd.Series(np.nan, index=df_tf.index)).fillna(1.0) / 0.02,
            0,
            1,
        ))

        result = pd.concat([ob_df, confluence_df, fib_df], axis=1)
        return result.add_suffix(f'_{tf}')

    def add_multi_timeframe_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add features từ nhiều khung thời gian

        Args:
            df: DataFrame với OHLCV data (base timeframe)

        Returns:
            DataFrame với thêm multi-timeframe features
        """
        df = df.copy()

        for tf in self.multi_timeframes:
            # Resample to higher timeframe
            df_tf = self.resample_to_timeframe(df, tf)

            # Calculate RSI
            if 'rsi' in self.indicators:
                df[f'rsi_{tf}'] = self.calculate_rsi_features(df_tf, 14)
                df[f'rsi_{tf}_momentum'] = df[f'rsi_{tf}'].diff()

            # Calculate MACD
            if 'macd' in self.indicators:
                macd_line, signal_line, histogram = self.calculate_macd_features(df_tf)
                df[f'macd_{tf}_line'] = macd_line
                df[f'macd_{tf}_signal'] = signal_line
                df[f'macd_{tf}_histogram'] = histogram
                df[f'macd_{tf}_cross'] = (macd_line > signal_line).astype(int)

            # Calculate BB
            if 'bb' in self.indicators:
                _, _, bb_width_tf = self.calculate_bb_features(df_tf)
                df[f'bb_width_{tf}'] = bb_width_tf

            # Calculate ATR
            if 'atr' in self.indicators:
                atr_tf = self.calculate_atr_features(df_tf)
                df[f'atr_{tf}'] = atr_tf
                df[f'atr_ratio_{tf}'] = atr_tf / df_tf['close']

            # Calculate Volume features
            if 'volume' in self.indicators:
                if 'volume' in df_tf.columns:
                    df[f'volume_{tf}'] = df_tf['volume']
                    df[f'volume_ratio_{tf}'] = df_tf['volume'] / df_tf['volume'].rolling(20).mean()

            # ICT & Fibonacci features
            ict_fibo_df = self._add_ict_fibo_features(df_tf, tf)
            df = df.join(ict_fibo_df, how='left')

        # Add cross-timeframe features
        df = self._add_cross_timeframe_features(df)

        # Forward fill to align with base timeframe
        df = df.ffill().bfill()

        return df

    def _add_cross_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features so sánh giữa các khung thời gian
        """
        # RSI divergence detection
        for tf1, tf2 in [('15T', '1H'), ('30T', '4H')]:
            if f'rsi_{tf1}' in df.columns and f'rsi_{tf2}' in df.columns:
                # Divergence: RSI trên các khung khác nhau đi ngược hướng
                df[f'rsi_divergence_{tf1}_{tf2}'] = (
                    (df[f'rsi_{tf1}'].diff() > 0) & (df[f'rsi_{tf2}'].diff() < 0)
                ).astype(int) + (
                    (df[f'rsi_{tf1}'].diff() < 0) & (df[f'rsi_{tf2}'].diff() > 0)
                ).astype(int)

                # Agreement: RSI trên các khung cùng hướng
                df[f'rsi_agreement_{tf1}_{tf2}'] = (
                    (df[f'rsi_{tf1}'].diff() > 0) & (df[f'rsi_{tf2}'].diff() > 0)
                ).astype(int) + (
                    (df[f'rsi_{tf1}'].diff() < 0) & (df[f'rsi_{tf2}'].diff() < 0)
                ).astype(int)

        # Trend alignment
        for tf in self.multi_timeframes:
            if f'macd_{tf}_cross' in df.columns:
                df[f'trend_aligned_{tf}'] = (
                    (df['macd_1H_cross'] == 1) & (df[f'macd_{tf}_cross'] == 1)
                ).astype(int) if 'macd_1H_cross' in df.columns else 0

        return df


def add_multi_timeframe_features(
    df: pd.DataFrame,
    base_timeframe: str = '5T',
    multi_timeframes: List[str] = None
) -> pd.DataFrame:
    """
    Convenience function để thêm multi-timeframe features

    Args:
        df: DataFrame với OHLCV data
        base_timeframe: Khung thời gian chính
        multi_timeframes: Danh sách khung thời gian phụ

    Returns:
        DataFrame với thêm multi-timeframe features
    """
    generator = MultiTimeframeFeatureGenerator(
        base_timeframe=base_timeframe,
        multi_timeframes=multi_timeframes
    )
    return generator.add_multi_timeframe_features(df)
