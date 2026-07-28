

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import logging

from algo_trading.strategies.base import BaseStrategy, StrategyResult
from algo_trading.strategies import (
    SMAEMACrossStrategy,
    RSIDivergenceStrategy,
    MACDMomentumStrategy,
    BollingerBreakoutStrategy,
    VWAPMeanReversionStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class IndicatorSignal:
    """Signal từ một indicator."""
    name: str
    signal: pd.Series  # -1, 0, 1
    weight: float = 1.0
    confidence: Optional[pd.Series] = None  # 0-1


class IndicatorCombiner(BaseStrategy):
    """
    Kết hợp nhiều indicator để tạo signal tổng hợp.
    """
    name = "Indicator Combiner"
    
    def __init__(
        self,
        indicators: List[Tuple[BaseStrategy, Dict[str, Any], float]],
        combination_method: str = "weighted_vote",
        min_agreement: float = 0.5,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.indicators = indicators
        self.combination_method = combination_method
        self.min_agreement = min_agreement
    
    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """Tạo signal từ kết hợp các indicator."""
        signals_list = []
        weights_list = []
        
        for strategy_class, params, weight in self.indicators:
            try:
                strategy = strategy_class(**params)
                result = strategy.generate_signals(df)
                signal = result.signals
                
                # Align với df index
                signal = signal.reindex(df.index, method='ffill').fillna(0)
                
                signals_list.append(signal)
                weights_list.append(weight)
            except Exception as e:
                logger.warning(f"Lỗi khi chạy {strategy_class.__name__}: {e}")
                continue
        
        if not signals_list:
            return StrategyResult(
                signals=pd.Series(0, index=df.index),
                meta={'error': 'No valid indicators'}
            )
        
        # Kết hợp signals
        combined_signal = self._combine_signals(signals_list, weights_list)
        
        return StrategyResult(
            signals=combined_signal,
            meta={
                'method': self.combination_method,
                'num_indicators': len(signals_list),
            }
        )
    
    def _combine_signals(
        self,
        signals_list: List[pd.Series],
        weights_list: List[float]
    ) -> pd.Series:
        if self.combination_method == "weighted_vote":
            return self._weighted_vote(signals_list, weights_list)
        elif self.combination_method == "majority":
            return self._majority_vote(signals_list)
        elif self.combination_method == "consensus":
            return self._consensus(signals_list, weights_list)
        elif self.combination_method == "ensemble":
            return self._ensemble(signals_list, weights_list)
        else:
            return self._weighted_vote(signals_list, weights_list)
    
    def _weighted_vote(
        self,
        signals_list: List[pd.Series],
        weights_list: List[float]
    ) -> pd.Series:
        """Bỏ phiếu có trọng số."""
        # Normalize weights
        total_weight = sum(weights_list)
        if total_weight == 0:
            weights_list = [1.0] * len(weights_list)
            total_weight = len(weights_list)
        
        normalized_weights = [w / total_weight for w in weights_list]
        
        # Tính weighted sum
        combined = pd.Series(0.0, index=signals_list[0].index)
        for signal, weight in zip(signals_list, normalized_weights):
            combined += signal * weight
        
        combined = np.sign(combined)
        return combined.fillna(0)
    
    def _majority_vote(self, signals_list: List[pd.Series]) -> pd.Series:
        """Bỏ phiếu đa số."""
        combined = pd.Series(0.0, index=signals_list[0].index)
        for signal in signals_list:
            combined += np.sign(signal)
        
        # Đa số quyết định
        combined = np.sign(combined)
        return combined.fillna(0)
    
    def _consensus(
        self,
        signals_list: List[pd.Series],
        weights_list: List[float]
    ) -> pd.Series:
        """Đồng thuận: cần đủ % indicators đồng ý."""
        # Tính weighted agreement
        total_weight = sum(weights_list)
        if total_weight == 0:
            weights_list = [1.0] * len(weights_list)
            total_weight = len(weights_list)
        
        normalized_weights = [w / total_weight for w in weights_list]
        
        # Tính weighted sum
        combined = pd.Series(0.0, index=signals_list[0].index)
        for signal, weight in zip(signals_list, normalized_weights):
            combined += signal * weight
        
        # Chỉ tạo signal nếu đạt min_agreement
        threshold = self.min_agreement
        result = pd.Series(0, index=combined.index)
        result[combined >= threshold] = 1
        result[combined <= -threshold] = -1
        
        return result.fillna(0)
    
    def _ensemble(
        self,
        signals_list: List[pd.Series],
        weights_list: List[float]
    ) -> pd.Series:
        """Ensemble: kết hợp với điều kiện bổ sung."""
        # Weighted vote
        weighted = self._weighted_vote(signals_list, weights_list)
        
        # Thêm điều kiện: cần ít nhất 2 indicators đồng ý
        agreement_count = pd.Series(0, index=weighted.index)
        for signal in signals_list:
            agreement_count += (np.sign(signal) == np.sign(weighted)).astype(int)
        
        min_agreement = max(2, len(signals_list) // 2)
        result = weighted.copy()
        result[agreement_count < min_agreement] = 0
        
        return result.fillna(0)


# Các preset kết hợp indicator tối ưu
class OptimizedCombinations:
    """Các cách kết hợp indicator đã được tối ưu."""
    
    @staticmethod
    def trend_momentum_combo() -> IndicatorCombiner:
        """Kết hợp Trend + Momentum indicators."""
        return IndicatorCombiner(
            indicators=[
                (SMAEMACrossStrategy, {'fast': 20, 'slow': 50, 'ma_type': 'ema'}, 1.0),
                (MACDMomentumStrategy, {'fast': 12, 'slow': 26, 'signal': 9}, 1.2),
                (RSIDivergenceStrategy, {'period': 14, 'overbought': 70, 'oversold': 30}, 0.8),
            ],
            combination_method="weighted_vote",
            min_agreement=0.4,
        )
    
    @staticmethod
    def mean_reversion_combo() -> IndicatorCombiner:
        """Kết hợp Mean Reversion indicators."""
        return IndicatorCombiner(
            indicators=[
                (VWAPMeanReversionStrategy, {'thr': 1.5}, 1.0),
                (BollingerBreakoutStrategy, {'window': 20, 'k': 2.0}, 0.8),
                (RSIDivergenceStrategy, {'period': 14, 'overbought': 75, 'oversold': 25}, 0.6),
            ],
            combination_method="consensus",
            min_agreement=0.5,
        )
    
    @staticmethod
    def balanced_combo() -> IndicatorCombiner:
        """Kết hợp cân bằng Trend + Momentum + Mean Reversion."""
        return IndicatorCombiner(
            indicators=[
                (SMAEMACrossStrategy, {'fast': 20, 'slow': 50, 'ma_type': 'ema'}, 1.0),
                (MACDMomentumStrategy, {'fast': 12, 'slow': 26, 'signal': 9}, 1.0),
                (RSIDivergenceStrategy, {'period': 14, 'overbought': 70, 'oversold': 30}, 0.8),
                (BollingerBreakoutStrategy, {'window': 20, 'k': 2.0}, 0.8),
                (VWAPMeanReversionStrategy, {'thr': 1.5}, 0.6),
            ],
            combination_method="ensemble",
            min_agreement=0.4,
        )
    
    @staticmethod
    def aggressive_trend_combo() -> IndicatorCombiner:
        """Kết hợp aggressive cho trend following."""
        return IndicatorCombiner(
            indicators=[
                (SMAEMACrossStrategy, {'fast': 10, 'slow': 30, 'ma_type': 'sma'}, 1.2),
                (MACDMomentumStrategy, {'fast': 8, 'slow': 21, 'signal': 5}, 1.2),
                (RSIDivergenceStrategy, {'period': 14, 'overbought': 70, 'oversold': 30}, 0.8),
            ],
            combination_method="weighted_vote",
            min_agreement=0.3,
        )
    
    @staticmethod
    def conservative_combo() -> IndicatorCombiner:
        """Kết hợp conservative với consensus cao."""
        return IndicatorCombiner(
            indicators=[
                (SMAEMACrossStrategy, {'fast': 20, 'slow': 50, 'ma_type': 'ema'}, 1.0),
                (MACDMomentumStrategy, {'fast': 12, 'slow': 26, 'signal': 9}, 1.0),
                (RSIDivergenceStrategy, {'period': 21, 'overbought': 75, 'oversold': 25}, 0.8),
                (BollingerBreakoutStrategy, {'window': 20, 'k': 2.0}, 0.8),
            ],
            combination_method="consensus",
            min_agreement=0.6,  # Cần 60% đồng thuận
        )
    
    @staticmethod
    def momentum_focused_combo() -> IndicatorCombiner:
        """Tập trung vào momentum indicators."""
        return IndicatorCombiner(
            indicators=[
                (MACDMomentumStrategy, {'fast': 12, 'slow': 26, 'signal': 9}, 1.5),
                (RSIDivergenceStrategy, {'period': 14, 'overbought': 70, 'oversold': 30}, 1.2),
                (BollingerBreakoutStrategy, {'window': 15, 'k': 2.5}, 1.0),
            ],
            combination_method="weighted_vote",
            min_agreement=0.4,
        )


def create_custom_combination(
    indicator_configs: List[Dict[str, Any]],
    method: str = "weighted_vote",
    min_agreement: float = 0.5,
) -> IndicatorCombiner:
    """
    Tạo kết hợp tùy chỉnh từ config.
    
    Args:
        indicator_configs: List of {
            'strategy': StrategyClass,
            'params': dict,
            'weight': float
        }
        method: combination method
        min_agreement: min agreement ratio
    """
    indicators = [
        (cfg['strategy'], cfg['params'], cfg.get('weight', 1.0))
        for cfg in indicator_configs
    ]
    
    return IndicatorCombiner(
        indicators=indicators,
        combination_method=method,
        min_agreement=min_agreement,
    )


# Mapping các preset
PRESET_COMBINATIONS = {
    'trend_momentum': OptimizedCombinations.trend_momentum_combo,
    'mean_reversion': OptimizedCombinations.mean_reversion_combo,
    'balanced': OptimizedCombinations.balanced_combo,
    'aggressive_trend': OptimizedCombinations.aggressive_trend_combo,
    'conservative': OptimizedCombinations.conservative_combo,
    'momentum_focused': OptimizedCombinations.momentum_focused_combo,
}

