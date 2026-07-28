"""
Train Ensemble Modules - Tách từ train_regime_ensemble_models_advanced.py

Các module con:
- feature_engineering: Tính indicators và xây dựng feature matrix
- class_imbalance: Xử lý cân bằng class
- data_quality: Kiểm tra chất lượng dữ liệu
- labeling: Tạo nhãn triple-barrier
- threshold_scoring: Tối ưu threshold trading
"""

from train.ensemble.feature_engineering import (
    calculate_indicators_enhanced,
    detect_regime_optimized,
    build_feature_matrix_enhanced,
)
from train.ensemble.class_imbalance import (
    calculate_class_weights,
    handle_class_imbalance,
)
from train.ensemble.data_quality import (
    data_quality_report,
    build_feature_contract,
    check_feature_skew_against_contract,
)
from train.ensemble.labeling import (
    generate_labels_triple_barrier,
)
from train.ensemble.threshold_scoring import (
    _score_trading_threshold,
    _score_trading_threshold_constrained,
    _score_trading_threshold_with_density,
    optimize_threshold_trading_objective,
    optimize_threshold_walk_forward,
)
