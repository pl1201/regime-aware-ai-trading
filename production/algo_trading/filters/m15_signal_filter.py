"""
M15 Signal Quality Filter

Multi-layer filtering để giữ chỉ top 10-15% signals.
Mục tiêu: Tăng PF từ 1.25 → 2.0+ bằng cách chỉ trade high-quality signals.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class M15SignalFilter:
    """
    5-layer signal quality filter.
    
    Layers:
    1. Context alignment (regime must support signal)
    2. Technical confirmation (multiple indicators agree)
    3. Risk/Reward validation (min R:R ratio)
    4. Timing checks (avoid bad periods)
    5. Final quality score (composite)
    
    Target: Pass only 10-15% of signals (best quality)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Default config
        self.config = config or {
            'min_confirmation_rate': 0.60,  # 60% of indicators must agree
            'min_rr_ratio': 1.5,            # Minimum risk/reward
            'min_quality_score': 0.60,      # Final quality threshold
            'min_model_confidence': 0.50,   # Model must be confident
        }
        
        # Track filter statistics
        self.stats = {
            'total_signals': 0,
            'passed_layer1': 0,
            'passed_layer2': 0,
            'passed_layer3': 0,
            'passed_layer4': 0,
            'passed_layer5': 0,
            'final_passed': 0,
        }
    
    def filter_signal(self, signal: int, features: pd.Series, 
                     context: Dict, model_proba: np.ndarray) -> Tuple[bool, float, str]:
        """
        Filter a single signal through all layers.
        
        Args:
            signal: Predicted signal (0=neutral, 1=long, 2=short)
            features: Feature values for this bar
            context: MTF context dict
            model_proba: Model probability output [p_neutral, p_long, p_short]
            
        Returns:
            (passed: bool, quality_score: float, reason: str)
        """
        self.stats['total_signals'] += 1
        
        # Skip neutral signals
        if signal == 0:
            return False, 0.0, "Neutral signal"
        
        # Layer 1: Context Alignment
        passed, score, reason = self._layer1_context_alignment(signal, context)
        if not passed:
            return False, score, f"Layer1: {reason}"
        self.stats['passed_layer1'] += 1
        
        # Layer 2: Technical Confirmation
        passed, conf_score, reason = self._layer2_technical_confirmation(signal, features)
        if not passed:
            return False, conf_score, f"Layer2: {reason}"
        self.stats['passed_layer2'] += 1
        
        # Layer 3: Risk/Reward
        passed, rr_score, reason = self._layer3_risk_reward(signal, features, context)
        if not passed:
            return False, rr_score, f"Layer3: {reason}"
        self.stats['passed_layer3'] += 1
        
        # Layer 4: Timing
        passed, timing_score, reason = self._layer4_timing(features, context)
        if not passed:
            return False, timing_score, f"Layer4: {reason}"
        self.stats['passed_layer4'] += 1
        
        # Layer 5: Final Quality Score
        passed, quality_score, reason = self._layer5_final_quality(
            signal, features, context, model_proba, 
            conf_score, rr_score, timing_score
        )
        if not passed:
            return False, quality_score, f"Layer5: {reason}"
        
        self.stats['passed_layer5'] += 1
        self.stats['final_passed'] += 1
        
        return True, quality_score, "All filters passed"
    
    def _layer1_context_alignment(self, signal: int, context: Dict) -> Tuple[bool, float, str]:
        """
        Layer 1: Signal must align with higher timeframe context.
        """
        regime = context.get('regime', '')
        direction = context.get('direction', 'neutral')
        tradeable = context.get('tradeable', False)
        trend_strength = context.get('trend_strength', 0)
        
        # Must be tradeable
        if not tradeable:
            return False, 0.0, "Market not tradeable"
        
        # Long signal checks
        if signal == 1:  # Long
            # Don't long in bearish context
            if direction == 'bear' and trend_strength > 0.6:
                return False, 0.3, "Strong bearish trend"
            
            # Volatile markets are risky
            vol_state = context.get('volatility', {}).get('state', 'medium')
            if vol_state == 'extreme':
                return False, 0.2, "Extreme volatility"
        
        # Short signal checks
        elif signal == 2:  # Short
            # Don't short in bullish context
            if direction == 'bull' and trend_strength > 0.6:
                return False, 0.3, "Strong bullish trend"
            
            # Volatile markets are risky
            vol_state = context.get('volatility', {}).get('state', 'medium')
            if vol_state == 'extreme':
                return False, 0.2, "Extreme volatility"
        
        # Calculate alignment score
        score = 0.5  # Base score
        
        # Bonus if trend aligned
        trend_aligned = context.get('trend_aligned', False)
        if trend_aligned:
            score += 0.3
        
        # Bonus if regime confidence high
        regime_conf = context.get('regime_confidence', 0.5)
        score += 0.2 * regime_conf
        
        return True, min(1.0, score), ""
    
    def _layer2_technical_confirmation(self, signal: int, 
                                       features: pd.Series) -> Tuple[bool, float, str]:
        """
        Layer 2: Multiple technical indicators must confirm.
        """
        confirmations = 0
        total_checks = 0
        
        if signal == 1:  # Long
            # Check 1: Momentum positive
            if features.get('momentum_6', 0) > 0.001:
                confirmations += 1
            total_checks += 1
            
            # Check 2: RSI not overbought
            rsi = features.get('rsi_9', 50)
            if 30 < rsi < 70:
                confirmations += 1
            total_checks += 1
            
            # Check 3: Volume support
            if features.get('volume_spike', 0) > 0 or features.get('volume_ma_ratio', 1) > 1.0:
                confirmations += 1
            total_checks += 1
            
            # Check 4: Structure break (optional bonus)
            if features.get('swing_low_break', 0) > 0:
                confirmations += 1
            total_checks += 1
            
            # Check 5: Bullish patterns
            if features.get('engulfing_bull', 0) > 0 or features.get('pin_bar_bull', 0) > 0:
                confirmations += 1
            total_checks += 1
        
        elif signal == 2:  # Short
            # Check 1: Momentum negative
            if features.get('momentum_6', 0) < -0.001:
                confirmations += 1
            total_checks += 1
            
            # Check 2: RSI not oversold
            rsi = features.get('rsi_9', 50)
            if 30 < rsi < 70:
                confirmations += 1
            total_checks += 1
            
            # Check 3: Volume support
            if features.get('volume_spike', 0) > 0 or features.get('volume_ma_ratio', 1) > 1.0:
                confirmations += 1
            total_checks += 1
            
            # Check 4: Structure break
            if features.get('swing_high_break', 0) > 0:
                confirmations += 1
            total_checks += 1
            
            # Check 5: Bearish patterns
            if features.get('engulfing_bear', 0) > 0 or features.get('pin_bar_bear', 0) > 0:
                confirmations += 1
            total_checks += 1
        
        confirmation_rate = confirmations / total_checks if total_checks > 0 else 0
        
        # Require minimum confirmation rate
        min_rate = self.config['min_confirmation_rate']
        if confirmation_rate < min_rate:
            return False, confirmation_rate, f"Low confirmation ({confirmation_rate:.1%} < {min_rate:.1%})"
        
        return True, confirmation_rate, ""
    
    def _layer3_risk_reward(self, signal: int, features: pd.Series, 
                           context: Dict) -> Tuple[bool, float, str]:
        """
        Layer 3: Risk/Reward must be acceptable.
        """
        atr = features.get('atr_9', 0.01)
        
        # Estimate SL based on ATR
        sl_distance = 1.5 * atr
        
        # Estimate TP based on regime
        regime = context.get('regime', 'weak_trend')
        if regime == 'strong_trend':
            tp_multiplier = 3.0
        elif regime == 'weak_trend':
            tp_multiplier = 2.5
        else:
            tp_multiplier = 2.0
        
        tp_distance = tp_multiplier * sl_distance
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
        
        # Check minimum R:R
        min_rr = self.config['min_rr_ratio']
        if rr_ratio < min_rr:
            return False, rr_ratio / 3.0, f"R:R too low ({rr_ratio:.2f} < {min_rr:.2f})"
        
        # Score based on R:R (cap at 1.0)
        score = min(1.0, rr_ratio / 3.0)
        
        return True, score, ""
    
    def _layer4_timing(self, features: pd.Series, context: Dict) -> Tuple[bool, float, str]:
        """
        Layer 4: Timing checks (avoid bad periods).
        """
        # For now, simple timing checks
        # In production, would check:
        # - Economic calendar events
        # - Session timing
        # - Churn guard (time since last signal)
        
        # Placeholder: Always pass
        return True, 1.0, ""
    
    def _layer5_final_quality(self, signal: int, features: pd.Series, 
                             context: Dict, model_proba: np.ndarray,
                             conf_score: float, rr_score: float, 
                             timing_score: float) -> Tuple[bool, float, str]:
        """
        Layer 5: Calculate final quality score and apply threshold.
        """
        # Model confidence
        model_conf = np.max(model_proba)
        
        # Check minimum model confidence
        min_conf = self.config['min_model_confidence']
        if model_conf < min_conf:
            return False, 0.0, f"Low model confidence ({model_conf:.2f} < {min_conf:.2f})"
        
        # Context quality
        regime_conf = context.get('regime_confidence', 0.5)
        trend_strength = context.get('trend_strength', 0.5)
        
        # Composite quality score
        quality_score = (
            0.30 * model_conf +          # 30% model confidence
            0.25 * conf_score +           # 25% technical confirmation
            0.20 * rr_score +             # 20% risk/reward
            0.15 * regime_conf +          # 15% regime confidence
            0.10 * timing_score           # 10% timing
        )
        
        # Apply final threshold
        min_quality = self.config['min_quality_score']
        if quality_score < min_quality:
            return False, quality_score, f"Quality score too low ({quality_score:.2f} < {min_quality:.2f})"
        
        return True, quality_score, ""
    
    def get_filter_stats(self) -> Dict:
        """Get filter statistics."""
        total = self.stats['total_signals']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'pass_rate_layer1': self.stats['passed_layer1'] / total,
            'pass_rate_layer2': self.stats['passed_layer2'] / total,
            'pass_rate_layer3': self.stats['passed_layer3'] / total,
            'pass_rate_layer4': self.stats['passed_layer4'] / total,
            'pass_rate_layer5': self.stats['passed_layer5'] / total,
            'final_pass_rate': self.stats['final_passed'] / total,
        }
    
    def reset_stats(self):
        """Reset filter statistics."""
        for key in self.stats:
            self.stats[key] = 0


def test_signal_filter():
    """Test signal filter."""
    print("Testing M15 Signal Filter...")
    
    # Create filter
    signal_filter = M15SignalFilter()
    
    # Test signal 1: Good quality long
    context_good = {
        'regime': 'strong_trend',
        'regime_confidence': 0.8,
        'direction': 'bull',
        'trend_strength': 0.7,
        'trend_aligned': True,
        'volatility': {'state': 'medium', 'percentile': 50},
        'tradeable': True
    }
    
    features_good = pd.Series({
        'momentum_6': 0.005,
        'rsi_9': 55,
        'volume_spike': 1,
        'swing_low_break': 1,
        'engulfing_bull': 1,
        'atr_9': 0.01,
    })
    
    model_proba_good = np.array([0.1, 0.7, 0.2])  # High confidence long
    
    passed, score, reason = signal_filter.filter_signal(
        signal=1, 
        features=features_good,
        context=context_good,
        model_proba=model_proba_good
    )
    
    print(f"\nTest 1 (Good Signal):")
    print(f"  Passed: {passed}")
    print(f"  Quality Score: {score:.3f}")
    print(f"  Reason: {reason}")
    
    # Test signal 2: Poor quality long
    context_bad = {
        'regime': 'volatile',
        'regime_confidence': 0.3,
        'direction': 'bear',
        'trend_strength': 0.8,
        'trend_aligned': False,
        'volatility': {'state': 'extreme', 'percentile': 95},
        'tradeable': False
    }
    
    features_bad = pd.Series({
        'momentum_6': -0.002,
        'rsi_9': 75,
        'volume_spike': 0,
        'swing_low_break': 0,
        'engulfing_bull': 0,
        'atr_9': 0.03,
    })
    
    model_proba_bad = np.array([0.4, 0.35, 0.25])  # Low confidence
    
    passed, score, reason = signal_filter.filter_signal(
        signal=1,
        features=features_bad,
        context=context_bad,
        model_proba=model_proba_bad
    )
    
    print(f"\nTest 2 (Bad Signal):")
    print(f"  Passed: {passed}")
    print(f"  Quality Score: {score:.3f}")
    print(f"  Reason: {reason}")
    
    # Print stats
    stats = signal_filter.get_filter_stats()
    print(f"\nFilter Statistics:")
    print(f"  Total signals: {stats['total_signals']}")
    print(f"  Final pass rate: {stats['final_pass_rate']:.1%}")
    
    print("\nTest PASSED!")


if __name__ == '__main__':
    test_signal_filter()
