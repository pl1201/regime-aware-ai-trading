"""
Strategies module - organized by category

All strategies are re-exported here for backward compatibility.
"""
from .base import BaseStrategy, StrategyResult, cross_over

# Import from subpackages
from .trend import (
    SMAEMACrossStrategy,
    RenkoTrendStrategy,
    KalmanFilterForecastStrategy,
    ARIMAStrategy,
)
from .momentum import (
    RSIDivergenceStrategy,
    MACDMomentumStrategy,
    BollingerBreakoutStrategy,
    VolumeProfileImbalanceStrategy,
)
from .mean_reversion import (
    VWAPMeanReversionStrategy,
    OUProcessMeanReversionStrategy,
    StatArbCointegrationStrategy,
)
from .ml import (
    LSTMTransformerStrategy,
    RegimeEnsembleStrategy,
    RegimeEnsembleBanditStrategy,
    RegimeEnsembleHybridStrategy,
)
from .volatility import (
    GARCHVolatilityStrategy,
)

__all__ = [
    # Base
    'BaseStrategy', 'StrategyResult', 'cross_over',
    # Trend
    'SMAEMACrossStrategy', 'RenkoTrendStrategy', 'KalmanFilterForecastStrategy', 'ARIMAStrategy',
    # Momentum
    'RSIDivergenceStrategy', 'MACDMomentumStrategy', 'BollingerBreakoutStrategy', 'VolumeProfileImbalanceStrategy',
    # Mean Reversion
    'VWAPMeanReversionStrategy', 'OUProcessMeanReversionStrategy', 'StatArbCointegrationStrategy',
    # ML
    'LSTMTransformerStrategy', 'RegimeEnsembleStrategy', 'RegimeEnsembleBanditStrategy', 'RegimeEnsembleHybridStrategy',
    # Volatility
    'GARCHVolatilityStrategy',
]
