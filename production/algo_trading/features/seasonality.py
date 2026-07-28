"""
Seasonality Features Module

Thêm features dựa trên thời gian để:
- Tận dụng patterns theo mùa
- Tránh giao dịch trong giờ không tốt
- Tối ưu timing entry/exit
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings


class SeasonalityFeatureGenerator:
    """
    Tạo seasonality features
    """

    def __init__(
        self,
        include_trading_sessions: bool = True,
        include_weekend_effect: bool = True,
        include_hourly_patterns: bool = True
    ):
        """
        Args:
            include_trading_sessions: Có thêm trading session features không
            include_weekend_effect: Có thêm weekend effect không
            include_hourly_patterns: Có thêm hourly pattern features không
        """
        self.include_trading_sessions = include_trading_sessions
        self.include_weekend_effect = include_weekend_effect
        self.include_hourly_patterns = include_hourly_patterns

    def add_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add seasonality features vào DataFrame

        Args:
            df: DataFrame với DatetimeIndex

        Returns:
            DataFrame với thêm seasonality features
        """
        df = df.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        # Convert to UTC if needed
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif df.index.tz != 'UTC':
            df.index = df.index.tz_convert('UTC')

        # Hour features
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week features
        df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Month features
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['is_quarter_end'] = ((df['month'] % 3 == 0) & (df.index.day >= 25)).astype(int)

        # Trading sessions (UTC time)
        if self.include_trading_sessions:
            df = self._add_trading_sessions(df)

        # Weekend effect
        if self.include_weekend_effect:
            df = self._add_weekend_effect(df)

        # Hourly patterns
        if self.include_hourly_patterns:
            df = self._add_hourly_patterns(df)

        return df

    def _add_trading_sessions(self, df: pd.DataFrame) -> pd.DataFrame:

        # Asia session: 00:00-08:00 UTC (08:00-16:00 Asia/Shanghai)
        df['is_asia_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)

        # Europe session: 07:00-15:00 UTC (08:00-16:00 Europe/London)
        df['is_europe_session'] = ((df.index.hour >= 7) & (df.index.hour < 15)).astype(int)

        # US session: 13:00-21:00 UTC (08:00-16:00 US/Eastern)
        df['is_us_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)

        # Overlap sessions
        df['europe_us_overlap'] = ((df.index.hour >= 13) & (df.index.hour < 15)).astype(int)
        df['asia_europe_overlap'] = ((df.index.hour >= 7) & (df.index.hour < 8)).astype(int)

        return df

    def _add_weekend_effect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add weekend effect features
        """
        # Friday afternoon effect (high volatility before weekend)
        df['is_friday_afternoon'] = (
            (df['day_of_week'] == 4) & (df.index.hour >= 15)
        ).astype(int)

        # Monday morning effect (gap after weekend)
        df['is_monday_morning'] = (
            (df['day_of_week'] == 0) & (df.index.hour <= 10)
        ).astype(int)

        # Weekend proximity (Friday + Saturday + Sunday)
        df['is_weekend_proximity'] = (
            (df['day_of_week'].isin([4, 5, 6])) |
            ((df['day_of_week'] == 0) & (df.index.hour <= 12))
        ).astype(int)

        return df

    def _add_hourly_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add hourly pattern features based on crypto market behavior
        """
        # High volatility hours (typically US session)
        df['is_high_volatility_hour'] = (
            (df.index.hour >= 13) & (df.index.hour <= 21)
        ).astype(int)

        # Low volatility hours (typically Asia night)
        df['is_low_volatility_hour'] = (
            (df.index.hour >= 22) | (df.index.hour <= 6)
        ).astype(int)

        # Asian market opening (00:00-01:00 UTC)
        df['is_asian_opening'] = (df.index.hour == 0).astype(int)

        # European market opening (07:00-08:00 UTC)
        df['is_europe_opening'] = (df.index.hour == 7).astype(int)

        # US market opening (13:00-14:00 UTC)
        df['is_us_opening'] = (df.index.hour == 13).astype(int)

        # US market closing (21:00-22:00 UTC)
        df['is_us_closing'] = (df.index.hour == 21).astype(int)

        return df


def add_seasonality_features(
    df: pd.DataFrame,
    include_trading_sessions: bool = True,
    include_weekend_effect: bool = True,
    include_hourly_patterns: bool = True
) -> pd.DataFrame:
    generator = SeasonalityFeatureGenerator(
        include_trading_sessions=include_trading_sessions,
        include_weekend_effect=include_weekend_effect,
        include_hourly_patterns=include_hourly_patterns
    )
    return generator.add_seasonality_features(df)