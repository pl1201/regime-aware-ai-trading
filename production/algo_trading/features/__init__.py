"""
Feature Engineering Module

Provides feature builders for different timeframes:
- H1Features: Optimized for hourly trading (recommended)
- M15 features: Legacy, use H1 instead
"""
from .h1_features import H1Features, build_h1_features

__all__ = [
    'H1Features',
    'build_h1_features',
]
