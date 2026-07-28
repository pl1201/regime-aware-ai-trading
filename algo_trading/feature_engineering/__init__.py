"""Feature engineering compatibility module.

This package keeps backward-compatible imports for legacy scripts that
expect `algo_trading.feature_engineering.feature_generator`.
"""

from .feature_generator import FeatureGenerator, add_multi_timeframe_features

__all__ = ["FeatureGenerator", "add_multi_timeframe_features"]
