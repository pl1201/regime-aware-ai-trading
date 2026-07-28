"""
Script đơn giản để xem code thực tế load data khi chạy strategy evaluation.

Chạy: python scripts/show_data_loading_code.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("CODE THỰC TẾ KHI CHẠY STRATEGY EVALUATION")
print("=" * 80)
print()

print("📋 BƯỚC 1: Load Data từ Sidebar Config")
print("-" * 80)
print("""
# File: algo_trading/ui/backtest_tab.py (dòng 47-58)

def _load_df_from_config(sidebar_config: Dict[str, Any]) -> pd.DataFrame:
    return load_df_from_sidebar_config(
        source=sidebar_config['source'],      # 'yfinance'
        ticker=sidebar_config['ticker'],      # 'BTC-USD'
        symbol=sidebar_config['symbol'],      # None
        interval=sidebar_config['interval'], # '1h'
        start=sidebar_config['start'],        # '2020-01-01'
        end=sidebar_config['end'],            # '2021-01-01'
        market=sidebar_config['market'],      # 'spot'
        path=sidebar_config['path']           # None
    )
""")

print("\n📋 BƯỚC 2: Utils Wrapper")
print("-" * 80)
print("""
# File: algo_trading/ui/utils.py (dòng 24-28)

def load_df_from_sidebar_config(source, ticker=None, symbol=None,
                                interval=None, start=None, end=None,
                                market='spot', path=None):
    kwargs = get_load_kwargs(source, ticker, symbol, interval, start, end, market, path)
    return load_dataframe(source, **kwargs)  # → load_data()
""")

print("\n📋 BƯỚC 3: Loader Router")
print("-" * 80)
print("""
# File: algo_trading/data_loader/loader.py (dòng 338-353)

def load_data(source: str, **kwargs) -> pd.DataFrame:
    source = source.lower()
    if source == 'yfinance':
        return load_yfinance(
            kwargs['ticker'],              # 'BTC-USD'
            start=kwargs.get('start'),      # '2020-01-01'
            end=kwargs.get('end'),          # '2021-01-01'
            interval=kwargs.get('interval','1d'),  # '1h'
            add_features_flag=True,
            normalize=None
        )
""")

print("\n📋 BƯỚC 4: Yahoo Finance Loader")
print("-" * 80)
print("""
# File: algo_trading/data_loader/loader.py (dòng 132-256)

def load_yfinance(ticker, start, end, interval='1d', ...):
    # Parse dates
    start_date = pd.to_datetime(start)  # '2020-01-01'
    end_date = pd.to_datetime(end)      # '2021-01-01'
    
    # CRITICAL: Kiểm tra giới hạn 730 ngày cho hourly
    if interval == '1h':
        days_from_today = (pd.Timestamp.now() - end_date).days
        if days_from_today > 730:
            raise ValueError("Không thể tải hourly data > 730 ngày gần nhất")
    
    # Download từ Yahoo Finance
    df = yf.download(
        ticker, 
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval=interval,
        auto_adjust=True
    )
    
    # Process: flatten columns, rename, add indicators
    return df
""")

print("\n📋 BƯỚC 5: Lưu vào Session State")
print("-" * 80)
print("""
# File: algo_trading/ui/backtest_tab.py (dòng 170-183)

if clicked_load or 'df' not in st.session_state:
    df = _load_df_from_config(sidebar_config)  # ← Load từ config
    st.session_state['df'] = df                 # ← Lưu vào session
else:
    df = st.session_state['df']                 # ← Lấy từ session
""")

print("\n📋 BƯỚC 6: Tạo StrategyEvaluator")
print("-" * 80)
print("""
# File: algo_trading/ui/backtest_tab.py (dòng 698-705)

evaluator = StrategyEvaluator(
    df=df,                      # ← DataFrame đã load
    initial_capital=10000.0,
    commission=0.001,
    use_stops=True,
    sl_pct=0.02,
    tp_pct=0.04,
)
""")

print("\n📋 BƯỚC 7: Strategy Generate Signals")
print("-" * 80)
print("""
# File: algo_trading/live/strategy_evaluator.py (dòng 220-222)

strategy = strategy_class(**strategy_params)
result = strategy.generate_signals(self.df)  # ← self.df từ evaluator
signals = result.signals
""")

print("\n" + "=" * 80)
print("TÓM TẮT: Data Flow")
print("=" * 80)
print("""
1. User nhập config trong sidebar (source, ticker, interval, start, end)
2. Click "Load" → _load_df_from_config(sidebar_config)
3. → load_df_from_sidebar_config() trong utils.py
4. → load_data() trong loader.py
5. → load_yfinance() hoặc load_binance()
6. → yf.download() hoặc Binance API
7. → Process data (flatten, rename, add indicators)
8. → Lưu vào st.session_state['df']
9. → StrategyEvaluator(df=df, ...)
10. → strategy.generate_signals(df)
""")

print("\n💡 LƯU Ý QUAN TRỌNG:")
print("-" * 80)
print("""
- Yahoo Finance hourly data: CHỈ trong 730 ngày gần nhất
- Nếu request 2020-2021 với interval='1h' → SẼ LỖI
- Giải pháp: Dùng interval='1d' (daily) hoặc điều chỉnh date range
- Data được cache trong st.session_state['df'] để tái sử dụng
""")





































