"""Sidebar configuration for Streamlit UI"""
import json
import streamlit as st
from pathlib import Path
from algo_trading.ui.config import STRATEGY_MAP
from algo_trading.live.indicator_combiner import PRESET_COMBINATIONS

try:
    from algo_trading.viz.tradingview import (
        prepare_tradingview_data,
        create_tradingview_html,
        create_tradingview_pinescript,
    )
    HAS_TRADINGVIEW = True
except ImportError:
    HAS_TRADINGVIEW = False
    prepare_tradingview_data = None
    create_tradingview_html = None
    create_tradingview_pinescript = None


def render_sidebar():
    """Render the sidebar with all configuration options"""
    st.header("📊 Dữ liệu")
    st.caption("Dữ liệu giá lịch sử (Open, High, Low, Close, Volume) để backtest strategies")
    source = st.selectbox("Nguồn", ["yfinance","binance","csv","parquet"], index=1)
    interval = st.text_input("Interval", value="5m", help="Ví dụ: 1m, 5m, 15m, 1h, 4h, 1d")
    start = st.text_input("Start (YYYY-MM-DD)", value="", help="Ngày bắt đầu (để trống = lấy mới nhất)")
    end = st.text_input("End (YYYY-MM-DD)", value="", help="Ngày kết thúc (để trống = hiện tại)")

    path = None
    ticker = None
    symbol = None
    market = 'spot'
    if source == 'yfinance':
        ticker = st.text_input("Ticker (yfinance)", value="BTC-USD")
    elif source == 'binance':
        symbol = st.text_input("Symbol (Binance)", value="BTCUSDT")
        market = st.selectbox("Thị trường", ["spot","futures"], index=0)
    elif source in ('csv','parquet'):
        # Ưu tiên cho user chọn data test trong folder `data/` của project
        data_dir = Path("data")
        candidates = []
        if data_dir.exists():
            if source == "csv":
                candidates = sorted([str(p) for p in data_dir.glob("*.csv")])
            else:
                candidates = sorted([str(p) for p in data_dir.glob("*.parquet")])

        if candidates:
            selected = st.selectbox(
                "Chọn file trong folder data/",
                candidates,
                index=0,
                help="Dùng data test trong `D:/Bot_Trading/data` để backtest/regime snapshot.",
                key="sidebar_data_file_select",
            )
            # Cho phép override path nếu cần
            path = st.text_input("Đường dẫn file", value=selected, key="sidebar_data_file_path")
        else:
            path = st.text_input(
                "Đường dẫn file",
                value=str(data_dir / ("CRYPTO_BTCUSD, 1D.csv" if source == "csv" else "")),
                help="Không tìm thấy file trong `data/`. Bạn nhập đường dẫn tuyệt đối hoặc tương đối.",
                key="sidebar_data_file_path",
            )

    st.markdown("---")
    st.header("Chiến lược")
    
    # Strategy type selection
    strategy_type = st.radio(
        "Loại chiến lược",
        ["Strategy đơn lẻ", "Indicator Combination", "So sánh nhiều strategies"],
        index=0,
        help="Chọn loại chiến lược để backtest"
    )
    
    # TradingView option
    use_tradingview = False
    if HAS_TRADINGVIEW:
        use_tradingview = st.checkbox("Dùng TradingView Chart", value=False, help="Hiển thị chart với TradingView thay vì matplotlib")
    
    StrategyCls = None
    params = {}
    strat_name = ""
    combo_strategy = None
    compare_strategies = False
    
    if strategy_type == "Strategy đơn lẻ":
        strat_name = st.selectbox("Chọn chiến lược", list(STRATEGY_MAP.keys()), index=2)
        _, StrategyCls, default_params = STRATEGY_MAP[strat_name]
        params_json = st.text_area("Tham số (JSON)", value=json.dumps(default_params), height=120)
        try:
            params = json.loads(params_json) if params_json.strip() else {}
        except Exception:
            st.warning("JSON tham số không hợp lệ. Dùng mặc định.")
            params = default_params
    
    elif strategy_type == "Indicator Combination":
        combo_mode = st.radio(
            "Chế độ",
            ["Preset Combinations", "Custom Combination"],
            horizontal=True,
            key="combo_mode_sidebar"
        )
        
        if combo_mode == "Preset Combinations":
            preset_options = {
                'Trend + Momentum': 'trend_momentum',
                'Mean Reversion': 'mean_reversion',
                'Balanced': 'balanced',
                'Aggressive Trend': 'aggressive_trend',
                'Conservative': 'conservative',
                'Momentum Focused': 'momentum_focused',
            }
            selected_preset = st.selectbox("Chọn preset", list(preset_options.keys()), key="preset_select_sidebar")
            preset_key = preset_options[selected_preset]
            strat_name = f"Combination: {selected_preset}"
            
            if st.button("Tạo Combination", key="btn_create_combo"):
                try:
                    combo_func = PRESET_COMBINATIONS[preset_key]
                    combo_strategy = combo_func()
                    st.session_state['combo_strategy'] = combo_strategy
                    st.success("✅ Đã tạo combination!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
        
        else:  # Custom
            st.info("💡 Tạo custom combination trong tab Backtest")
            if 'combo_strategy' in st.session_state:
                combo_strategy = st.session_state['combo_strategy']
                strat_name = "Custom Combination"
    
    else:  # So sánh nhiều strategies
        compare_strategies = True
        st.info("💡 Chọn các strategies để so sánh trong tab Backtest")
        strat_name = "Multiple Strategies Comparison"

    st.markdown("---")
    st.header("Backtest")
    mode = st.selectbox("Chế độ", ["vectorized","event"], index=0)
    allow_short = st.checkbox("Cho phép short", value=True)
    use_next_open = st.checkbox("Vào/ra tại open kế tiếp", value=True)
    leverage = st.number_input("Leverage", value=1.0, min_value=0.0, step=0.1)
    commission = st.number_input("Commission (tỷ lệ)", value=0.0005, step=0.0001, format="%f")
    slippage_bps = st.number_input("Slippage (bps)", value=1.0, step=0.5)

    st.markdown("---")
    st.header("Risk (tuỳ chọn)")
    col1, col2, col3 = st.columns(3)
    with col1:
        sl_pct = st.number_input("SL %", value=0.0, step=0.001, format="%f")
        sl_atr_k = st.number_input("SL k*ATR", value=0.0, step=0.1)
    with col2:
        tp_pct = st.number_input("TP %", value=0.0, step=0.001, format="%f")
        tp_atr_k = st.number_input("TP k*ATR", value=0.0, step=0.1)
    with col3:
        trailing_pct = st.number_input("Trailing %", value=0.0, step=0.001, format="%f")
        trailing_atr_k = st.number_input("Trailing k*ATR", value=0.0, step=0.1)
    atr_col = st.text_input("ATR column", value="ATR14")

    st.markdown("---")
    st.header("Phân tích phiên/giờ")
    analysis_mode = st.selectbox(
        "Chế độ phân tích",
        ["None", "Session only", "Hour only", "Session + Hour"],
        index=0,
        help="Phân tích return theo phiên (Asia/Europe/US) và/hoặc theo giờ trong ngày",
    )

    # Initialize session state for buttons if not exists
    if 'clicked_load' not in st.session_state:
        st.session_state.clicked_load = False
    if 'clicked_run' not in st.session_state:
        st.session_state.clicked_run = False
    if 'clicked_compare' not in st.session_state:
        st.session_state.clicked_compare = False
    if 'clicked_analysis' not in st.session_state:
        st.session_state.clicked_analysis = False
    
    # Buttons - update session state when clicked and reset others
    if st.button("1) Tải dữ liệu", help="Tải dữ liệu giá lịch sử (OHLCV) để xem trước"):
        st.session_state.clicked_load = True
        st.session_state.clicked_run = False
        st.session_state.clicked_compare = False
        st.session_state.clicked_analysis = False
    clicked_load = st.session_state.clicked_load
    
    if st.button("2) Chạy backtest", help="Chạy backtest với strategy đã chọn (sẽ tự động tải dữ liệu nếu chưa có)"):
        st.session_state.clicked_load = False
        st.session_state.clicked_run = True
        st.session_state.clicked_compare = False
        st.session_state.clicked_analysis = False
    clicked_run = st.session_state.clicked_run
    
    if st.button("3) So sánh nhiều strategies", help="Chạy backtest cho nhiều strategies và tạo báo cáo so sánh (sẽ tự động tải dữ liệu nếu chưa có)"):
        st.session_state.clicked_load = False
        st.session_state.clicked_run = False
        st.session_state.clicked_compare = True
        st.session_state.clicked_analysis = False
    clicked_compare = st.session_state.clicked_compare
    
    if st.button("4) Phân tích phiên/giờ", help="Phân tích return theo phiên giao dịch và giờ trong ngày"):
        st.session_state.clicked_load = False
        st.session_state.clicked_run = False
        st.session_state.clicked_compare = False
        st.session_state.clicked_analysis = True
    clicked_analysis = st.session_state.clicked_analysis
    
    # Return all sidebar values as a dictionary
    return {
        'source': source,
        'interval': interval,
        'start': start,
        'end': end,
        'path': path,
        'ticker': ticker,
        'symbol': symbol,
        'market': market,
        'strategy_type': strategy_type,
        'use_tradingview': use_tradingview,
        'StrategyCls': StrategyCls,
        'params': params,
        'strat_name': strat_name,
        'combo_strategy': combo_strategy,
        'compare_strategies': compare_strategies,
        'mode': mode,
        'allow_short': allow_short,
        'use_next_open': use_next_open,
        'leverage': leverage,
        'commission': commission,
        'slippage_bps': slippage_bps,
        'sl_pct': sl_pct,
        'sl_atr_k': sl_atr_k,
        'tp_pct': tp_pct,
        'tp_atr_k': tp_atr_k,
        'trailing_pct': trailing_pct,
        'trailing_atr_k': trailing_atr_k,
        'atr_col': atr_col,
        'analysis_mode': analysis_mode,
        'clicked_load': clicked_load,
        'clicked_run': clicked_run,
        'clicked_compare': clicked_compare,
        'clicked_analysis': clicked_analysis,
    }

