"""
Script để kiểm tra và fix feature dimension mismatch giữa model và FeatureEngineer.
"""
import sys
import os
from pathlib import Path
import torch

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from algo_trading.ml.models.transformer_distribution import TransformerDistributionWrapper


def check_model_input_dim(model_path: str):
    """Kiểm tra input_dim của model"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        input_dim = checkpoint.get('input_dim', None)
        model_config = checkpoint.get('model_config', {})
        
        print(f"📁 Model: {model_path}")
        print(f"   Input dim: {input_dim}")
        print(f"   Model config: {model_config}")
        
        return input_dim, model_config
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        return None, None


if __name__ == "__main__":
    print("=" * 80)
    print("🔍 KIỂM TRA MODEL INPUT DIMENSION")
    print("=" * 80)
    
    # Tìm tất cả models
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("regime_transformer_*.pth"))
        if model_files:
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            print(f"\n📋 Tìm thấy {len(model_files)} models:\n")
            
            for i, model_file in enumerate(model_files, 1):
                input_dim, config = check_model_input_dim(str(model_file))
                if input_dim:
                    print(f"   ✅ Model {i}: {model_file.name}")
                    print(f"      → Input dim: {input_dim}")
                else:
                    print(f"   ❌ Model {i}: {model_file.name} - Không thể đọc")
                print()
            
            # Model mới nhất
            latest_model = model_files[0]
            input_dim, config = check_model_input_dim(str(latest_model))
            
            if input_dim:
                print("=" * 80)
                print("💡 GIẢI PHÁP")
                print("=" * 80)
                print(f"\nModel mới nhất expect {input_dim} features.")
                print("\nNếu gặp lỗi 'shapes cannot be multiplied':")
                print("1. Đảm bảo FeatureEngineer config khớp với training:")
                print("   - sequence_length: 20")
                print("   - use_lags: True")
                print("   - n_lags: 5")
                print("   - use_rolling_stats: True")
                print("   - rolling_windows: [5, 10, 20]")
                print("   - indicators: ['RSI', 'MACD', 'BB', 'ATR', 'VWAP', 'SMA', 'EMA']")
                print("\n2. Hoặc re-train model với FeatureEngineer config hiện tại")
                print("\n3. Hoặc sửa FeatureEngineer trong strategy để match với training")
        else:
            print("❌ Không tìm thấy model nào")
    else:
        print("❌ Không tìm thấy thư mục 'models'")



























