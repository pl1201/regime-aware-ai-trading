import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional, Tuple
from algo_trading.ml.dynamic_moe_v2 import DynamicMOE_v2

try:
    from algo_trading.filters.signal_quality_filter import apply_signal_filter, enhanced_signal_scoring
    HAS_SIGNAL_FILTER = True
except Exception:
    HAS_SIGNAL_FILTER = False

def predict_with_moe_v2(
    X_latest: np.ndarray,
    feature_names: list,
    model_path: str = "models/dynamic_moe_v2.pkl",
    neutral_band: tuple = (0.45, 0.55),
    min_confidence: float = 0.6
) -> Dict[str, any]:
    """
    Dự đoán tín hiệu giao dịch thực thời gian thực với MOE v2.

    Args:
        X_latest: np.ndarray - đặc trưng của thanh nến mới nhất (1 mẫu)
        feature_names: list - tên các đặc trưng tương ứng với X_latest
        model_path: str - đường dẫn đến mô hình MOE v2
        neutral_band: tuple - khoảng không giao dịch (tránh giao dịch khi xác suất trung tính)
        min_confidence: float - ngưỡng tối thiểu để chấp nhận tín hiệu

    Returns:
        dict: {
            'signal': 1 (long) | -1 (short) | 0 (neutral),
            'probability': float,  # xác suất long
            'expert_selected': str,  # expert được chọn: 'trend', 'range', 'volatility'
            'regime_detected': str,  # chế độ được nhận diện: 'trending', 'ranging', 'volatile', 'calm'
            'confidence': float  # độ tin cậy của dự đoán
        }
    """

    # Tải mô hình
    try:
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict) and 'model' in loaded:
            moe_model = loaded['model']
        else:
            moe_model = loaded
    except FileNotFoundError:
        return {
            'signal': 0,
            'probability': 0.5,
            'expert_selected': 'none',
            'regime_detected': 'unknown',
            'confidence': 0.0,
            'error': f'Model not found at {model_path}'
        }

    # Chuẩn bị dữ liệu đầu vào
    if isinstance(X_latest, list):
        X_latest = np.array(X_latest)

    # Nếu chỉ có 1 mẫu, reshape
    if len(X_latest.shape) == 1:
        X_latest = X_latest.reshape(1, -1)

    # Tạo DataFrame để dễ xử lý
    X_df = pd.DataFrame(X_latest, columns=feature_names)

    # Dự đoán xác suất từ MOE v2 (hỗ trợ cả output 1D lẫn 2D)
    raw = moe_model.predict_proba(X_df.values)
    if isinstance(raw, np.ndarray) and raw.ndim == 2:
        proba = float(raw[0, 1] if raw.shape[1] > 1 else raw[0, 0])
    elif isinstance(raw, np.ndarray) and raw.ndim == 1:
        proba = float(raw[0])
    else:
        proba = float(raw)

    # Áp dụng quality score nếu có features phù hợp
    if HAS_SIGNAL_FILTER:
        try:
            scored = enhanced_signal_scoring(np.asarray([proba], dtype=float), X_df)
            proba = float(scored[0])
        except Exception:
            pass

    # Tính độ tin cậy
    confidence = float(abs(proba - 0.5) * 2.0)
    expert_selected = 'moe_weighted'
    regime_detected = 'auto'

    # Xác định tín hiệu giao dịch
    signal = 0  # neutral

    if proba >= neutral_band[1] and confidence >= min_confidence:
        signal = 1  # long
    elif proba <= neutral_band[0] and confidence >= min_confidence:
        signal = -1  # short

    return {
        'signal': int(signal),
        'probability': float(proba),
        'expert_selected': expert_selected,
        'regime_detected': regime_detected,
        'confidence': float(confidence),
        'regime_features': {}
    }

# Hàm tiện ích để kiểm tra mô hình

def test_moe_v2_prediction():
    """
    Kiểm tra hàm predict_with_moe_v2 với dữ liệu mẫu
    """
    # Dữ liệu mẫu - cần khớp với feature_names
    sample_features = [
        0.018,  # atr_14
        0.045,  # bb_upper
        0.035,  # bb_lower
        0.040,  # bb_middle
        1.8,    # volume
        58.0,   # rsi_14
        0.005,  # macd_hist
        0.02,   # close
        0.015,  # low_20
        0.025,  # high_20
    ]

    feature_names = [
        'atr_14', 'bb_upper', 'bb_lower', 'bb_middle', 'volume',
        'rsi_14', 'macd_hist', 'close', 'low_20', 'high_20'
    ]

    result = predict_with_moe_v2(sample_features, feature_names)
    print("🧪 Test Prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result

# Nếu chạy trực tiếp, kiểm tra
if __name__ == "__main__":
    test_moe_v2_prediction()