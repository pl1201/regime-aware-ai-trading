"""
Script để test và xem data được load như thế nào khi chạy strategy evaluation.

Chạy script này để xem:
1. Data được load từ đâu
2. Các tham số nào được truyền vào
3. Data có shape như thế nào
4. Có gặp lỗi gì không

Cách chạy:
    cd D:\Bot_Trading
    python scripts/test_data_loading.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algo_trading.data_loader.loader import load_data
from algo_trading.ui.utils import load_df_from_sidebar_config
import pandas as pd

def test_yfinance_loading():
    """Test load data từ Yahoo Finance"""
    print("=" * 80)
    print("TEST: Load Data từ Yahoo Finance")
    print("=" * 80)
    
    # Config giống như trong UI
    config = {
        'source': 'yfinance',
        'ticker': 'BTC-USD',
        'interval': '1h',
        'start': '2024-01-01',  # Dùng date gần đây để tránh lỗi 730 days
        'end': '2025-01-01',
    }
    
    print(f"\n📋 Config:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        # Load data (giống như trong UI)
        print(f"\n🔄 Đang load data...")
        df = load_df_from_sidebar_config(**config)
        
        print(f"\n✅ Load thành công!")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Index type: {type(df.index)}")
        print(f"   Date range: {df.index[0]} → {df.index[-1]}")
        print(f"   Number of bars: {len(df)}")
        
        print(f"\n📊 Sample data (first 5 rows):")
        print(df.head())
        
        print(f"\n📊 Sample data (last 5 rows):")
        print(df.tail())
        
        return df
        
    except Exception as e:
        print(f"\n❌ Lỗi khi load data:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_binance_loading():
    """Test load data từ Binance"""
    print("\n" + "=" * 80)
    print("TEST: Load Data từ Binance")
    print("=" * 80)
    
    config = {
        'source': 'binance',
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'start': '2024-01-01',
        'end': '2025-01-01',
        'market': 'spot',
    }
    
    print(f"\n📋 Config:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        print(f"\n🔄 Đang load data...")
        df = load_df_from_sidebar_config(**config)
        
        print(f"\n✅ Load thành công!")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df.index[0]} → {df.index[-1]}")
        
        return df
        
    except Exception as e:
        print(f"\n❌ Lỗi khi load data:")
        print(f"   {type(e).__name__}: {e}")
        return None


def test_yfinance_historical_limit():
    """Test giới hạn 730 ngày của Yahoo Finance"""
    print("\n" + "=" * 80)
    print("TEST: Yahoo Finance 730 Days Limit")
    print("=" * 80)
    
    # Thử load data từ 2020 (sẽ fail)
    config = {
        'source': 'yfinance',
        'ticker': 'BTC-USD',
        'interval': '1h',
        'start': '2020-01-01',
        'end': '2021-01-01',
    }
    
    print(f"\n📋 Config (sẽ fail vì quá 730 ngày):")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        print(f"\n🔄 Đang load data...")
        df = load_df_from_sidebar_config(**config)
        print(f"\n⚠️ Không có lỗi (có thể đã được auto-adjust)")
        
    except ValueError as e:
        print(f"\n✅ Đúng như mong đợi - ValueError:")
        print(f"   {str(e)[:200]}...")
    except Exception as e:
        print(f"\n❌ Lỗi khác:")
        print(f"   {type(e).__name__}: {e}")


def show_data_flow():
    """Hiển thị flow hoàn chỉnh"""
    print("\n" + "=" * 80)
    print("DATA LOADING FLOW")
    print("=" * 80)
    print("""
1. User nhập config trong UI sidebar:
   - source: 'yfinance' hoặc 'binance'
   - ticker/symbol: 'BTC-USD' hoặc 'BTCUSDT'
   - interval: '1h', '1d', etc.
   - start/end: dates

2. UI gọi: load_df_from_sidebar_config(**config)
   → File: algo_trading/ui/utils.py

3. Utils gọi: load_dataframe(source, **kwargs)
   → File: algo_trading/ui/utils.py

4. Loader router: load_data(source, **kwargs)
   → File: algo_trading/data_loader/loader.py
   → Route đến load_yfinance() hoặc load_binance()

5. Download từ API:
   - yfinance: yf.download(ticker, start, end, interval)
   - binance: Binance API call

6. Process data:
   - Flatten columns
   - Rename (Open→open, Close→close, etc.)
   - Add indicators (SMA, EMA, RSI, etc.)
   - Sort by datetime index

7. Lưu vào session_state['df']

8. StrategyEvaluator nhận df:
   evaluator = StrategyEvaluator(df=df, ...)

9. Mỗi strategy nhận df:
   strategy.generate_signals(df)
    """)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TEST DATA LOADING - Xem Data Được Load Như Thế Nào")
    print("=" * 80)
    
    # Show flow
    show_data_flow()
    
    # Test yfinance (với date gần đây)
    df_yf = test_yfinance_loading()
    
    # Test binance
    df_binance = test_binance_loading()
    
    # Test historical limit
    test_yfinance_historical_limit()
    
    print("\n" + "=" * 80)
    print("KẾT LUẬN")
    print("=" * 80)
    print("""
Khi chạy strategy evaluation trong UI:

1. Data được load từ sidebar config
2. Nếu source='yfinance' và interval='1h':
   - Chỉ có thể load trong 730 ngày gần nhất
   - Nếu request > 730 ngày → sẽ raise ValueError
3. Data được lưu vào st.session_state['df']
4. StrategyEvaluator nhận df này và truyền cho mỗi strategy
5. Tất cả strategies dùng cùng một df → so sánh công bằng

Để xem data thực tế, chạy script này với config của bạn!
    """)

