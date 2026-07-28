
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Import backtest function
from backtest_regime_transformer import backtest_regime_transformer

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 BACKTEST REGIME TRANSFORMER MODEL")
    print("=" * 80)
    
    # Tìm model mới nhất
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("regime_transformer_*.pth"))
        if model_files:
            # Sắp xếp theo thời gian, lấy file mới nhất
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_model = str(model_files[0])
            print(f"\n📁 Tìm thấy model mới nhất: {latest_model}")
            print(f"   Tổng số models: {len(model_files)}")
            
            # Hiển thị tất cả models
            print("\n📋 Tất cả models có sẵn:")
            for i, m in enumerate(model_files, 1):
                mtime = datetime.fromtimestamp(m.stat().st_mtime)
                marker = " ← MỚI NHẤT" if m == model_files[0] else ""
                print(f"   {i}. {m.name} ({mtime.strftime('%Y-%m-%d %H:%M:%S')}){marker}")
        else:
            print("\n❌ Không tìm thấy model nào trong thư mục 'models'")
            print("   Vui lòng train model trước.")
            latest_model = None
    else:
        print("\n❌ Không tìm thấy thư mục 'models'")
        latest_model = None
    
    if latest_model:
        print("\n" + "=" * 80)
        print("⚙️ CẤU HÌNH BACKTEST")
        print("=" * 80)
        print("   Model: Model mới nhất")
        print("   Data: yfinance BTC-USD 1h (90 ngày gần nhất)")
        print("   SL: 2% | TP: 4%")
        print("   EV Threshold: 0.0001 (để có nhiều signals)")
        print("   Allowed Regimes: Tất cả (trending, ranging, volatile, calm)")
        print("=" * 80)
        
        # Chạy backtest
        result = backtest_regime_transformer(
            model_path=latest_model,
            source='yfinance',
            ticker='BTC-USD',
            interval='1h',
            start=None,  # Tự động: 90 ngày gần nhất
            end=None,   # Tự động: hiện tại
            sl_pct=0.02,  # 2% stop loss
            tp_pct=0.04,  # 4% take profit
            leverage=1.0,
            commission=0.0005,
            max_trades=100,
            ev_threshold=0.0001,  # Giảm để có nhiều signals
            position_sizing='fixed',
            risk_per_trade=0.02,
            allowed_regimes=None,  # Tất cả regimes
        )
        
        if result:
            print("\n" + "=" * 80)
            print("✅ BACKTEST HOÀN TẤT!")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("❌ BACKTEST THẤT BẠI")
            print("=" * 80)
            print("\n💡 Gợi ý:")
            print("   1. Kiểm tra model có phù hợp với timeframe không")
            print("   2. Giảm EV threshold xuống 0.00001")
            print("   3. Kiểm tra dữ liệu có đủ không (ít nhất 50 bars)")
    else:
        print("\n💡 Cách sử dụng:")
        print("   1. Train model trong Streamlit UI (tab Regime Transformer)")
        print("   2. Chạy lại script này: python run_regime_backtest.py")
        print("   3. Hoặc chỉ định model cụ thể trong code")

