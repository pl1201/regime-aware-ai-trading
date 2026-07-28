
import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class M15MicrostructureFeatures:
    """
    Generate microstructure features optimized for M15 trading.
    
    Focus areas:
    1. Fast momentum (short lookbacks)
    2. Structure breaks (swing violations)
    3. Volume analysis
    4. Candle patterns
    5. Volatility microstructure
    """
    
    def __init__(self):
        # M15-optimized parameters
        self.fast_period = 9   # 2.25 hours
        self.medium_period = 21  # 5.25 hours  
        self.slow_period = 50    # 12.5 hours
        self.swing_lookback = 12  # 3 hours
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all M15 microstructure features.
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            DataFrame with microstructure features
        """
        if len(df) < 100:
            raise ValueError("Need at least 100 bars of data")
        
        features = pd.DataFrame(index=df.index)
        
        # 1. MOMENTUM FEATURES (fast-moving for M15)
        features['momentum_3'] = self._momentum(df, 3)   # 45 min
        features['momentum_6'] = self._momentum(df, 6)   # 1.5h
        features['momentum_9'] = self._momentum(df, 9)   # 2.25h
        features['momentum_acceleration'] = self._momentum_acceleration(df)
        features['momentum_slope_3'] = self._momentum_slope(df, 3)
        
        # 2. STRUCTURE FEATURES
        features['swing_high_break'] = self._swing_high_break(df, self.swing_lookback)
        features['swing_low_break'] = self._swing_low_break(df, self.swing_lookback)
        features['higher_highs_count'] = self._count_higher_highs(df, self.fast_period)
        features['lower_lows_count'] = self._count_lower_lows(df, self.fast_period)
        features['swing_high_level'] = self._get_swing_high(df, self.swing_lookback)
        features['swing_low_level'] = self._get_swing_low(df, self.swing_lookback)
        
        # 3. VOLUME FEATURES
        features['volume_spike'] = self._volume_spike(df, threshold=1.5)
        features['volume_trend'] = self._volume_trend(df, window=12)
        features['volume_ma_ratio'] = self._volume_ma_ratio(df, window=12)
        features['volume_price_divergence'] = self._volume_price_divergence(df)
        
        # 4. CANDLE PATTERNS (last 3 bars)
        features['engulfing_bull'] = self._detect_engulfing(df, 'bull')
        features['engulfing_bear'] = self._detect_engulfing(df, 'bear')
        features['pin_bar_bull'] = self._detect_pin_bar(df, 'bull')
        features['pin_bar_bear'] = self._detect_pin_bar(df, 'bear')
        features['doji'] = self._detect_doji(df)
        features['strong_close'] = self._strong_close(df)
        
        # 5. VOLATILITY MICROSTRUCTURE
        features['true_range_ma_ratio'] = self._tr_ma_ratio(df, period=9)
        features['candle_body_pct'] = self._candle_body_percentage(df)
        features['wick_ratio'] = self._wick_ratio(df)
        features['body_position'] = self._body_position(df)
        features['atr_9'] = self._calculate_atr(df, 9)
        features['atr_percentile'] = self._atr_percentile(df, 9, 100)
        
        # 6. PRICE LEVELS
        features['near_round_number'] = self._near_round_number(df)
        features['distance_from_open'] = self._distance_from_open(df)
        
        # 7. FAST TECHNICALS (M15-optimized)
        features['rsi_9'] = self._calculate_rsi(df, 9)
        features['rsi_slope'] = self._rsi_slope(df, 9)
        features['stoch_k'] = self._stochastic(df, 9, 3)['k']
        features['bb_position'] = self._bb_position(df, 20)
        
        return features.fillna(0)
    
    # ========== MOMENTUM FEATURES ==========
    
    def _momentum(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Rate of Change (ROC) over period."""
        return df['close'].pct_change(period)
    
    def _momentum_acceleration(self, df: pd.DataFrame) -> pd.Series:
        """2nd derivative of price (momentum of momentum)."""
        mom3 = self._momentum(df, 3)
        return mom3.diff()
    
    def _momentum_slope(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Slope of momentum (is momentum increasing?)."""
        mom = self._momentum(df, period)
        return mom.diff()
    
    # ========== STRUCTURE FEATURES ==========
    
    def _swing_high_break(self, df: pd.DataFrame, lookback: int) -> pd.Series:
        """Detect swing high breakouts."""
        swing_high = df['high'].rolling(lookback).max()
        return (df['close'] > swing_high.shift(1)).astype(int)
    
    def _swing_low_break(self, df: pd.DataFrame, lookback: int) -> pd.Series:
        """Detect swing low breakdowns."""
        swing_low = df['low'].rolling(lookback).min()
        return (df['close'] < swing_low.shift(1)).astype(int)
    
    def _get_swing_high(self, df: pd.DataFrame, lookback: int) -> pd.Series:
        """Get current swing high level."""
        return df['high'].rolling(lookback).max()
    
    def _get_swing_low(self, df: pd.DataFrame, lookback: int) -> pd.Series:
        """Get current swing low level."""
        return df['low'].rolling(lookback).min()
    
    def _count_higher_highs(self, df: pd.DataFrame, lookback: int) -> pd.Series:
        """Count consecutive higher highs."""
        high = df['high']
        higher = (high > high.shift(1)).astype(int)
        
        # Count consecutive
        consecutive = higher.rolling(lookback).sum()
        return consecutive
    
    def _count_lower_lows(self, df: pd.DataFrame, lookback: int) -> pd.Series:
        """Count consecutive lower lows."""
        low = df['low']
        lower = (low < low.shift(1)).astype(int)
        
        consecutive = lower.rolling(lookback).sum()
        return consecutive
    
    # ========== VOLUME FEATURES ==========
    
    def _volume_spike(self, df: pd.DataFrame, threshold: float = 1.5) -> pd.Series:
        """Detect volume spikes (> threshold * MA)."""
        vol_ma = df['volume'].rolling(12).mean()
        return (df['volume'] > threshold * vol_ma).astype(int)
    
    def _volume_trend(self, df: pd.DataFrame, window: int = 12) -> pd.Series:
        """Volume trend: positive if increasing, negative if decreasing."""
        vol_ma_short = df['volume'].rolling(window // 2).mean()
        vol_ma_long = df['volume'].rolling(window).mean()
        return ((vol_ma_short > vol_ma_long).astype(int) * 2 - 1)  # -1 or +1
    
    def _volume_ma_ratio(self, df: pd.DataFrame, window: int = 12) -> pd.Series:
        """Current volume / MA ratio."""
        vol_ma = df['volume'].rolling(window).mean()
        return df['volume'] / (vol_ma + 1e-10)
    
    def _volume_price_divergence(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect volume-price divergence.
        Price up + Volume down = bearish divergence (-1)
        Price down + Volume up = bullish divergence (+1)
        """
        price_change = df['close'].pct_change(3)
        volume_change = df['volume'].pct_change(3)
        
        # Divergence conditions
        bearish_div = (price_change > 0) & (volume_change < 0)
        bullish_div = (price_change < 0) & (volume_change > 0)
        
        divergence = pd.Series(0, index=df.index)
        divergence[bullish_div] = 1
        divergence[bearish_div] = -1
        
        return divergence
    
    # ========== CANDLE PATTERNS ==========
    
    def _detect_engulfing(self, df: pd.DataFrame, direction: str) -> pd.Series:
        """Detect bullish/bearish engulfing patterns."""
        body = abs(df['close'] - df['open'])
        body_prev = abs(df['close'].shift(1) - df['open'].shift(1))
        
        if direction == 'bull':
            # Green candle engulfs previous red candle
            engulfing = (
                (df['close'] > df['open']) &  # Current green
                (df['close'].shift(1) < df['open'].shift(1)) &  # Previous red
                (body > body_prev * 1.2) &  # Bigger body
                (df['close'] > df['open'].shift(1)) &  # Engulfs
                (df['open'] < df['close'].shift(1))
            )
        else:  # bear
            # Red candle engulfs previous green candle
            engulfing = (
                (df['close'] < df['open']) &  # Current red
                (df['close'].shift(1) > df['open'].shift(1)) &  # Previous green
                (body > body_prev * 1.2) &
                (df['close'] < df['open'].shift(1)) &
                (df['open'] > df['close'].shift(1))
            )
        
        return engulfing.astype(int)
    
    def _detect_pin_bar(self, df: pd.DataFrame, direction: str) -> pd.Series:
        """Detect pin bar (hammer/shooting star) patterns."""
        body = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        
        if direction == 'bull':
            # Hammer: long lower wick, small body at top
            lower_wick = df[['open', 'close']].min(axis=1) - df['low']
            upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
            
            pin_bar = (
                (lower_wick > body * 2) &  # Lower wick 2x body
                (upper_wick < body * 0.5) &  # Small upper wick
                (body < total_range * 0.3)  # Small body
            )
        else:  # bear
            # Shooting star: long upper wick, small body at bottom
            upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
            lower_wick = df[['open', 'close']].min(axis=1) - df['low']
            
            pin_bar = (
                (upper_wick > body * 2) &
                (lower_wick < body * 0.5) &
                (body < total_range * 0.3)
            )
        
        return pin_bar.astype(int)
    
    def _detect_doji(self, df: pd.DataFrame) -> pd.Series:
        """Detect doji candles (indecision)."""
        body = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        
        doji = (body < total_range * 0.1)  # Body < 10% of range
        return doji.astype(int)
    
    def _strong_close(self, df: pd.DataFrame) -> pd.Series:
        """
        Strong close near high (bullish) or low (bearish).
        +1 = closed near high, -1 = closed near low
        """
        total_range = df['high'] - df['low']
        close_position = (df['close'] - df['low']) / (total_range + 1e-10)
        
        # Strong if close in top/bottom 20%
        strong = pd.Series(0, index=df.index)
        strong[close_position > 0.8] = 1   # Bullish
        strong[close_position < 0.2] = -1  # Bearish
        
        return strong
    
    # ========== VOLATILITY MICROSTRUCTURE ==========
    
    def _tr_ma_ratio(self, df: pd.DataFrame, period: int = 9) -> pd.Series:
        """True Range / MA ratio (volatility expansion)."""
        tr = self._calculate_true_range(df)
        tr_ma = tr.rolling(period).mean()
        return tr / (tr_ma + 1e-10)
    
    def _candle_body_percentage(self, df: pd.DataFrame) -> pd.Series:
        """Body size as % of total range."""
        body = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        return body / (total_range + 1e-10)
    
    def _wick_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Upper wick / Lower wick ratio."""
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        return upper_wick / (lower_wick + 1e-10)
    
    def _body_position(self, df: pd.DataFrame) -> pd.Series:
        """Body position within range (0=bottom, 1=top)."""
        body_mid = (df['open'] + df['close']) / 2
        total_range = df['high'] - df['low']
        position = (body_mid - df['low']) / (total_range + 1e-10)
        return position
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR."""
        tr = self._calculate_true_range(df)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def _atr_percentile(self, df: pd.DataFrame, period: int, lookback: int) -> pd.Series:
        """ATR percentile over lookback period."""
        atr = self._calculate_atr(df, period)
        percentile = atr.rolling(lookback).apply(
            lambda x: (x.iloc[-1] >= x).sum() / len(x) * 100 if len(x) > 0 else 50
        )
        return percentile
    
    # ========== PRICE LEVELS ==========
    
    def _near_round_number(self, df: pd.DataFrame) -> pd.Series:
        """Distance to nearest round number (for BTC: 100, 500, 1000)."""
        close = df['close']
        
        # Find nearest round number
        round_500 = (close / 500).round() * 500
        distance = abs(close - round_500) / close
        
        # Near if within 0.2%
        near = (distance < 0.002).astype(int)
        return near
    
    def _distance_from_open(self, df: pd.DataFrame) -> pd.Series:
        """Distance from session open (can be day open on M15)."""
        # Use daily open as reference
        daily_open = df['open'].resample('D').first().reindex(df.index, method='ffill')
        distance = (df['close'] - daily_open) / daily_open
        return distance
    
    # ========== FAST TECHNICALS ==========
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 9) -> pd.Series:
        """Calculate RSI with given period."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _rsi_slope(self, df: pd.DataFrame, period: int = 9) -> pd.Series:
        """RSI slope (is RSI rising or falling)."""
        rsi = self._calculate_rsi(df, period)
        return rsi.diff()
    
    def _stochastic(self, df: pd.DataFrame, k_period: int = 9, 
                   d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()
        
        k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        d = k.rolling(d_period).mean()
        
        return {'k': k, 'd': d}
    
    def _bb_position(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Position within Bollinger Bands (0=lower, 1=upper)."""
        middle = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = middle + 2 * std
        lower = middle - 2 * std
        
        position = (df['close'] - lower) / (upper - lower + 1e-10)
        return position.clip(0, 1)


def test_microstructure():
    """Test M15 Microstructure Features."""
    print("Testing M15 Microstructure Features...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=300, freq='15min')
    
    price = 50000 + np.cumsum(np.random.randn(300) * 20)
    df = pd.DataFrame({
        'open': price + np.random.randn(300) * 10,
        'high': price + abs(np.random.randn(300) * 30),
        'low': price - abs(np.random.randn(300) * 30),
        'close': price + np.random.randn(300) * 15,
        'volume': np.random.uniform(100, 1000, 300)
    }, index=dates)
    
    # Build features
    builder = M15MicrostructureFeatures()
    features = builder.build_features(df)
    
    print(f"\n=== Features Generated: {len(features.columns)} ===")
    print("\nSample features (last row):")
    sample = features.iloc[-1]
    print(f"  momentum_3: {sample['momentum_3']:.4f}")
    print(f"  momentum_6: {sample['momentum_6']:.4f}")
    print(f"  swing_high_break: {sample['swing_high_break']}")
    print(f"  swing_low_break: {sample['swing_low_break']}")
    print(f"  volume_spike: {sample['volume_spike']}")
    print(f"  engulfing_bull: {sample['engulfing_bull']}")
    print(f"  pin_bar_bull: {sample['pin_bar_bull']}")
    print(f"  rsi_9: {sample['rsi_9']:.2f}")
    print(f"  atr_9: {sample['atr_9']:.2f}")
    print(f"  candle_body_pct: {sample['candle_body_pct']:.2f}")
    
    print(f"\n✅ Generated {len(features.columns)} features successfully!")
    print(f"   Feature list: {', '.join(features.columns[:10])}...")


if __name__ == '__main__':
    test_microstructure()
