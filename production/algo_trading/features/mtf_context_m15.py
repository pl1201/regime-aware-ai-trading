
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class MTFContextBuilder:

    
    def __init__(self):
        self.timeframes = ['15m', '1h', '4h']
        
    def build_context(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, 
                     df_4h: pd.DataFrame) -> Dict:
        """
        Build trading context từ 3 timeframes.
        
        Args:
            df_15m: M15 dataframe (for current state)
            df_1h: 1H dataframe (for trend confirmation)
            df_4h: 4H dataframe (for regime detection)
            
        Returns:
            context: Dict containing regime, direction, tradeable, etc.
        """
        # Ensure all dataframes have required columns
        for df, name in [(df_15m, '15m'), (df_1h, '1h'), (df_4h, '4h')]:
            if not self._validate_dataframe(df):
                raise ValueError(f"Invalid dataframe for {name}")
        
        # 1. REGIME DETECTION (4H primary)
        regime_4h = self._detect_regime(df_4h)
        regime_1h = self._detect_regime(df_1h)
        
        # 2. TREND ANALYSIS
        trend_4h = self._detect_trend(df_4h)
        trend_1h = self._detect_trend(df_1h)
        trend_15m = self._detect_trend(df_15m)
        
        # 3. VOLATILITY STATE
        vol_4h = self._volatility_state(df_4h)
        vol_1h = self._volatility_state(df_1h)
        
        # 4. ALIGNMENT CHECK
        trend_aligned = self._check_trend_alignment(trend_4h, trend_1h, trend_15m)
        
        # 5. TRADEABLE CONDITIONS
        tradeable = self._is_tradeable(regime_4h, trend_4h, vol_4h, trend_aligned)
        
        # 6. RISK MODE
        risk_mode = self._determine_risk_mode(regime_4h, vol_4h, trend_aligned)
        
        return {
            'regime': regime_4h['regime'],
            'regime_confidence': regime_4h['confidence'],
            'direction': trend_4h['direction'],
            'trend_strength': trend_1h['strength'],  # Use 1H for execution
            'trend_aligned': trend_aligned,
            'volatility': vol_4h,
            'tradeable': tradeable,
            'risk_mode': risk_mode,
            # Additional info
            'regime_1h': regime_1h['regime'],
            'adx_4h': trend_4h.get('adx', 0),
            'adx_1h': trend_1h.get('adx', 0),
        }
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate dataframe has required columns."""
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        return required_cols.issubset(df.columns) and len(df) >= 100
    
    def _detect_regime(self, df: pd.DataFrame) -> Dict:
        # Calculate ADX
        adx = self._calculate_adx(df, period=14).iloc[-1]
        
        # Calculate Bollinger Band width
        bb_width = self._calculate_bb_width(df, period=20, std=2).iloc[-1]
        
        # Calculate ATR percentile
        atr = self._calculate_atr(df, period=14)
        atr_current = atr.iloc[-1]
        atr_percentile = (atr.rank(pct=True).iloc[-1]) * 100
        
        # Regime classification
        if adx > 30:
            regime = 'strong_trend'
            confidence = min(0.95, 0.5 + (adx - 30) / 100)  # Higher ADX = higher confidence
        elif adx > 20:
            regime = 'weak_trend'
            confidence = 0.3 + (adx - 20) / 50
        elif bb_width < 0.02:  # BB width < 2%
            regime = 'range'
            confidence = min(0.8, 0.5 + (0.02 - bb_width) * 10)
        elif atr_percentile > 85:
            regime = 'volatile'
            confidence = 0.4 + (atr_percentile - 85) / 50
        else:
            # Default: weak trend
            regime = 'weak_trend'
            confidence = 0.3
        
        return {
            'regime': regime,
            'confidence': confidence,
            'adx': adx,
            'bb_width': bb_width,
            'atr_percentile': atr_percentile
        }
    
    def _detect_trend(self, df: pd.DataFrame) -> Dict:
        """
        Detect trend direction and strength.
        
        Uses:
        - ADX for strength
        - Linear regression slope for direction
        - EMA crossovers for confirmation
        
        Returns:
            {
                'direction': 'bull'/'bear'/'neutral',
                'strength': float (0-1),
                'adx': float,
                'slope': float
            }
        """
        close = df['close'].values
        
        # ADX for strength
        adx = self._calculate_adx(df, period=14).iloc[-1]
        
        # Linear regression slope (last 20 bars)
        x = np.arange(20)
        y = close[-20:]
        slope = np.polyfit(x, y, 1)[0] / y[-1]  # Normalized slope
        
        # EMAs for confirmation
        ema_fast = df['close'].ewm(span=21, adjust=False).mean().iloc[-1]
        ema_slow = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        current_price = close[-1]
        
        # Determine direction
        if slope > 0.001 and ema_fast > ema_slow and current_price > ema_fast:
            direction = 'bull'
            direction_confidence = min(1.0, slope * 500)  # Scale slope
        elif slope < -0.001 and ema_fast < ema_slow and current_price < ema_fast:
            direction = 'bear'
            direction_confidence = min(1.0, abs(slope) * 500)
        else:
            direction = 'neutral'
            direction_confidence = 0.3
        
        # Trend strength (0-1) based on ADX
        strength = min(1.0, adx / 50)
        
        return {
            'direction': direction,
            'strength': strength * direction_confidence,  # Combined score
            'adx': adx,
            'slope': slope,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow
        }
    
    def _volatility_state(self, df: pd.DataFrame) -> Dict:
        """
        Analyze volatility state.
        
        Returns:
            {
                'state': 'low'/'medium'/'high'/'extreme',
                'percentile': float (0-100),
                'atr': float,
                'expanding': bool
            }
        """
        # Calculate ATR
        atr = self._calculate_atr(df, period=14)
        atr_current = atr.iloc[-1]
        
        # ATR percentile (over last 100 bars)
        atr_percentile = (atr.iloc[-100:].rank(pct=True).iloc[-1]) * 100
        
        # ATR trend (expanding or contracting)
        atr_ma = atr.rolling(10).mean()
        expanding = atr_current > atr_ma.iloc[-1]
        
        # Classify state
        if atr_percentile < 25:
            state = 'low'
        elif atr_percentile < 60:
            state = 'medium'
        elif atr_percentile < 85:
            state = 'high'
        else:
            state = 'extreme'
        
        return {
            'state': state,
            'percentile': atr_percentile,
            'atr': atr_current,
            'expanding': expanding,
            'atr_ma': atr_ma.iloc[-1]
        }
    
    def _check_trend_alignment(self, trend_4h: Dict, trend_1h: Dict, 
                               trend_15m: Dict) -> bool:
        """
        Check if trends are aligned across timeframes.
        
        Aligned = same direction on all TFs (or at least 2/3)
        """
        directions = [
            trend_4h['direction'],
            trend_1h['direction'],
            trend_15m['direction']
        ]
        
        # Count bull/bear/neutral
        bull_count = directions.count('bull')
        bear_count = directions.count('bear')
        
        # Aligned if at least 2/3 agree and not all neutral
        aligned = (bull_count >= 2 or bear_count >= 2) and directions.count('neutral') < 2
        
        return aligned
    
    def _is_tradeable(self, regime_4h: Dict, trend_4h: Dict, 
                      vol_4h: Dict, trend_aligned: bool) -> bool:
        """
        Determine if market conditions are tradeable.
        
        DO NOT TRADE if:
        - Extreme volatility (>90th percentile) and volatile regime
        - Very weak trend (strength < 0.2) unless in range regime
        - Not trend-aligned (exception: range regime)
        - Regime confidence too low
        """
        # Rule 1: Extreme volatility filter
        if regime_4h['regime'] == 'volatile' and vol_4h['percentile'] > 90:
            return False
        
        # Rule 2: Very weak trend (unless ranging)
        if trend_4h['strength'] < 0.2 and regime_4h['regime'] not in ['range']:
            return False
        
        # Rule 3: Trend alignment (except range)
        if not trend_aligned and regime_4h['regime'] not in ['range', 'weak_trend']:
            return False
        
        # Rule 4: Regime confidence
        if regime_4h['confidence'] < 0.3:
            return False
        
        return True
    
    def _determine_risk_mode(self, regime_4h: Dict, vol_4h: Dict, 
                             trend_aligned: bool) -> str:
        """
        Determine risk mode: aggressive/normal/defensive.
        
        Aggressive: Strong trend + aligned + normal vol
        Normal: Most conditions
        Defensive: Volatile, unaligned, or uncertain regime
        """
        # Aggressive conditions
        if (regime_4h['regime'] == 'strong_trend' and
            trend_aligned and
            vol_4h['state'] in ['medium', 'high'] and
            regime_4h['confidence'] > 0.7):
            return 'aggressive'
        
        # Defensive conditions
        if (regime_4h['regime'] in ['volatile'] or
            vol_4h['state'] == 'extreme' or
            not trend_aligned or
            regime_4h['confidence'] < 0.4):
            return 'defensive'
        
        # Default: normal
        return 'normal'
    
    # ========== TECHNICAL INDICATOR HELPERS ==========
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up = high - high.shift(1)
        down = low.shift(1) - low
        
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        # Smooth
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx.fillna(0)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR (Average True Range)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr.fillna(method='bfill')
    
    def _calculate_bb_width(self, df: pd.DataFrame, period: int = 20, 
                           std: float = 2.0) -> pd.Series:
        """
        Calculate Bollinger Band width as percentage.
        
        Width = (upper - lower) / middle
        """
        close = df['close']
        middle = close.rolling(period).mean()
        std_dev = close.rolling(period).std()
        
        upper = middle + std * std_dev
        lower = middle - std * std_dev
        
        width = (upper - lower) / middle
        return width.fillna(0)


def test_mtf_context():
    """Test function for MTF Context Builder."""
    print("Testing MTF Context Builder...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='4h')
    
    df_4h = pd.DataFrame({
        'open': 50000 + np.cumsum(np.random.randn(200) * 100),
        'high': 50000 + np.cumsum(np.random.randn(200) * 100) + 100,
        'low': 50000 + np.cumsum(np.random.randn(200) * 100) - 100,
        'close': 50000 + np.cumsum(np.random.randn(200) * 100),
        'volume': np.random.uniform(100, 1000, 200)
    }, index=dates)
    
    df_1h = df_4h.copy()  # Simplified
    df_15m = df_4h.copy()
    
    # Build context
    builder = MTFContextBuilder()
    context = builder.build_context(df_15m, df_1h, df_4h)
    
    print("\n=== MTF Context ===")
    print(f"Regime: {context['regime']} (confidence: {context['regime_confidence']:.2f})")
    print(f"Direction: {context['direction']} (strength: {context['trend_strength']:.2f})")
    print(f"Trend Aligned: {context['trend_aligned']}")
    print(f"Volatility: {context['volatility']['state']} ({context['volatility']['percentile']:.1f}%ile)")
    print(f"Tradeable: {context['tradeable']}")
    print(f"Risk Mode: {context['risk_mode']}")
    print(f"ADX 4H: {context['adx_4h']:.1f}")
    print(f"ADX 1H: {context['adx_1h']:.1f}")
    print("\n✅ Test passed!")


if __name__ == '__main__':
    test_mtf_context()
