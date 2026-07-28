"""
ML Module - Machine Learning Models

Includes:
- H1HybridModel: Recommended model for hourly trading (PF=1.35-1.58)
- H1EnhancedModel: HMM regime detection + MTF confirmation model (PF=8.24)
- DynamicMOE_v3_HMM_MTF: MOE with HMM regime (PF=1.31)
"""
# Import only existing modules
try:
    from .h1_hybrid_model import H1HybridModel, get_h1_model
except ImportError:
    H1HybridModel = None
    get_h1_model = None

try:
    from .h1_enhanced_model import H1EnhancedModel, backtest_enhanced
except ImportError:
    H1EnhancedModel = None
    backtest_enhanced = None

try:
    from .dynamic_moe_v3_hmm_mtf import DynamicMOE_v3_HMM_MTF
except ImportError:
    DynamicMOE_v3_HMM_MTF = None

__all__ = [
    'H1HybridModel',
    'get_h1_model',
    'H1EnhancedModel',
    'backtest_enhanced',
    'DynamicMOE_v3_HMM_MTF',
]