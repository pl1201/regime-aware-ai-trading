"""
Script để trace một strategy execution từ đầu đến cuối.

Chạy script này để xem:
1. Data được load như thế nào
2. Strategy được tạo như thế nào
3. Signals được generate như thế nào
4. Backtest được chạy như thế nào
5. Metrics được tính như thế nào

Cách chạy:
    cd D:\Bot_Trading
    python scripts/trace_strategy_execution.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algo_trading.data_loader.loader import load_yfinance
from algo_trading.strategies.trend.sma_ema_cross import SMAEMACrossStrategy
from algo_trading.core.backtest_vectorized import vectorized_pnl, BacktestConfig
from algo_trading.core.metrics import performance_summary

def trace_strategy_execution():
    """Trace một strategy execution từ đầu đến cuối"""
    
    print("=" * 80)
    print("TRACE STRATEGY EXECUTION - Từng Bước Chi Tiết")
    print("=" * 80)
    print()
    
    # ========================================================================
    # BƯỚC 1: Load Data
    # ========================================================================
    print("📊 BƯỚC 1: Load Data")
    print("-" * 80)
    print("File: algo_trading/data_loader/loader.py → load_yfinance()")
    print()
    
    try:
        # Load data (dùng date gần đây để tránh lỗi 730 days)
        # Yahoo Finance chỉ cho phép hourly data trong 730 ngày gần nhất
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # 60 ngày gần nhất
        
        print("🔄 Đang load data từ Yahoo Finance...")
        print(f"   Config: ticker='BTC-USD', interval='1h'")
        print(f"   Start: {start_date.strftime('%Y-%m-%d')}")
        print(f"   End: {end_date.strftime('%Y-%m-%d')}")
        print(f"   (Dùng 60 ngày gần nhất để tránh lỗi 730 days limit)")
        
        df = load_yfinance(
            ticker='BTC-USD',
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1h',
            add_features_flag=True
        )
        
        print(f"✅ Load thành công!")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df.index[0]} → {df.index[-1]}")
        print(f"   Number of bars: {len(df)}")
        print()
        
        # Show sample
        print("📋 Sample data (first 3 rows):")
        print(df.head(3))
        print()
        
    except Exception as e:
        print(f"❌ Lỗi khi load data: {e}")
        return
    
    # ========================================================================
    # BƯỚC 2: Create Strategy
    # ========================================================================
    print("=" * 80)
    print("📈 BƯỚC 2: Create Strategy Instance")
    print("-" * 80)
    print("File: algo_trading/strategies/trend/sma_ema_cross.py → SMAEMACrossStrategy")
    print()
    
    strategy_params = {'fast': 10, 'slow': 30, 'ma_type': 'sma'}
    print(f"🔄 Tạo strategy với params: {strategy_params}")
    
    strategy = SMAEMACrossStrategy(**strategy_params)
    print(f"✅ Strategy created: {strategy.name}")
    print(f"   Class: {strategy.__class__.__name__}")
    print(f"   Params: {strategy.params}")
    print()
    
    # ========================================================================
    # BƯỚC 3: Generate Signals
    # ========================================================================
    print("=" * 80)
    print("🎯 BƯỚC 3: Generate Signals")
    print("-" * 80)
    print("File: algo_trading/strategies/trend/sma_ema_cross.py → generate_signals()")
    print()
    
    print("🔄 Đang generate signals...")
    result = strategy.generate_signals(df)
    signals = result.signals
    
    print(f"✅ Signals generated!")
    print(f"   Signals type: {type(signals)}")
    print(f"   Signals shape: {signals.shape}")
    print(f"   Signals unique values: {signals.unique()}")
    print(f"   Signals value counts:")
    print(signals.value_counts())
    print()
    
    # Show sample signals
    print("📋 Sample signals (first 10):")
    print(signals.head(10))
    print()
    
    # ========================================================================
    # BƯỚC 4: Backtest
    # ========================================================================
    print("=" * 80)
    print("💰 BƯỚC 4: Run Backtest")
    print("-" * 80)
    print("File: algo_trading/core/backtest_vectorized.py → vectorized_pnl()")
    print()
    
    cfg = BacktestConfig(
        initial_capital=10000.0,
        commission=0.001,
        allow_short=True,
    )
    
    print(f"🔄 Đang chạy backtest...")
    print(f"   Config: initial_capital={cfg.initial_capital}, commission={cfg.commission}")
    
    equity, returns = vectorized_pnl(df, signals, cfg)
    
    print(f"✅ Backtest hoàn tất!")
    print(f"   Equity shape: {equity.shape}")
    print(f"   Returns shape: {returns.shape}")
    print(f"   Equity start: ${equity.iloc[0]:,.2f}")
    print(f"   Equity end: ${equity.iloc[-1]:,.2f}")
    print(f"   Total return: {(equity.iloc[-1]/equity.iloc[0]-1)*100:.2f}%")
    print()
    
    # Show equity curve sample
    print("📋 Equity curve (first 5, last 5):")
    print("First 5:")
    print(equity.head(5))
    print("Last 5:")
    print(equity.tail(5))
    print()
    
    # ========================================================================
    # BƯỚC 5: Calculate Metrics
    # ========================================================================
    print("=" * 80)
    print("📊 BƯỚC 5: Calculate Metrics")
    print("-" * 80)
    print("File: algo_trading/core/metrics.py → performance_summary()")
    print()
    
    print("🔄 Đang tính metrics...")
    perf = performance_summary(equity, returns, freq='1h')
    
    print(f"✅ Metrics calculated!")
    print()
    print("📋 Performance Metrics:")
    for key, value in perf.items():
        if isinstance(value, float):
            if 'Return' in key or 'Drawdown' in key or 'CAGR' in key:
                print(f"   {key:20s}: {value*100:8.2f}%")
            else:
                print(f"   {key:20s}: {value:8.4f}")
        else:
            print(f"   {key:20s}: {value}")
    print()
    
    # ========================================================================
    # BƯỚC 6: Analysis
    # ========================================================================
    print("=" * 80)
    print("🔍 BƯỚC 6: Analysis")
    print("-" * 80)
    print()
    
    # Signal analysis
    signal_changes = (signals.diff() != 0).sum()
    long_signals = (signals > 0).sum()
    short_signals = (signals < 0).sum()
    neutral_signals = (signals == 0).sum()
    
    print("📊 Signal Analysis:")
    print(f"   Total signal changes: {signal_changes}")
    print(f"   Long signals: {long_signals} ({long_signals/len(signals)*100:.1f}%)")
    print(f"   Short signals: {short_signals} ({short_signals/len(signals)*100:.1f}%)")
    print(f"   Neutral signals: {neutral_signals} ({neutral_signals/len(signals)*100:.1f}%)")
    print()
    
    # Returns analysis
    print("📊 Returns Analysis:")
    print(f"   Returns mean: {returns.mean():.6f} ({returns.mean()*100:.4f}%)")
    print(f"   Returns std: {returns.std():.6f} ({returns.std()*100:.4f}%)")
    print(f"   Returns min: {returns.min():.6f} ({returns.min()*100:.4f}%)")
    print(f"   Returns max: {returns.max():.6f} ({returns.max()*100:.4f}%)")
    print()
    
    # Equity analysis
    print("📊 Equity Analysis:")
    print(f"   Equity min: ${equity.min():,.2f}")
    print(f"   Equity max: ${equity.max():,.2f}")
    print(f"   Equity ratio (max/min): {equity.max()/equity.min():.2f}x")
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("✅ SUMMARY - Flow Hoàn Chỉnh")
    print("=" * 80)
    print("""
1. ✅ Load Data: Yahoo Finance → DataFrame
2. ✅ Create Strategy: SMAEMACrossStrategy với params
3. ✅ Generate Signals: Strategy logic → signals Series
4. ✅ Run Backtest: vectorized_pnl() → equity, returns
5. ✅ Calculate Metrics: performance_summary() → metrics dict
6. ✅ Analysis: Signal stats, returns stats, equity stats

Bạn đã trace được toàn bộ flow từ đầu đến cuối!
    """)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')  # Fix encoding for Windows
    try:
        trace_strategy_execution()
    except Exception as e:
        import traceback
        print(f"❌ Lỗi: {e}")
        traceback.print_exc()

