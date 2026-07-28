"""
ML Module - Tầng 2: Inference/Learning
Bao gồm các AI models để học conditional probabilities từ market state
"""
from .features import create_features, FeatureEngineer
try:
    from .sequence_extractor import SequenceFeatureExtractor, SequenceExtractorConfig
except Exception:
    SequenceFeatureExtractor = None
    SequenceExtractorConfig = None

# Keep package import lightweight for non-transformer workflows.
try:
    from .models.transformer_distribution import TransformerDistributionModel
except Exception:
    TransformerDistributionModel = None

# Import H1 Enhanced Model
try:
    from .h1_enhanced_model import H1EnhancedModel, backtest_enhanced
except Exception as e:
    H1EnhancedModel = None
    backtest_enhanced = None
    print(f"Warning: Could not import H1EnhancedModel: {e}")

__all__ = [
    'create_features',
    'FeatureEngineer',
    'TransformerDistributionModel',
    'SequenceFeatureExtractor',
    'SequenceExtractorConfig',
    'H1EnhancedModel',
    'backtest_enhanced',
]

