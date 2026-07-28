from __future__ import annotations
import os
import sys
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Import config and utilities
from algo_trading.ui.sidebar import render_sidebar
# Import tab renderers from separate modules
from algo_trading.ui.backtest_tab import render_backtest_tab as render_backtest_tab_module
from algo_trading.ui.live_trading_tab import render_live_trading_tabs
from algo_trading.ui.regime_transformer_tab import render_regime_transformer_tab

# Import advanced ML tab
try:
    from algo_trading.ui.regime_ensemble_advanced_tab import render_regime_ensemble_advanced_tab
    HAS_ADVANCED_TAB = True
except ImportError:
    HAS_ADVANCED_TAB = False
    import warnings
    warnings.warn("⚠️ Advanced ML tab không tìm thấy. Bỏ qua tab Advanced ML.")

st.set_page_config(page_title="Algo Trading UI", layout="wide")
st.title("Algo Trading – Giao diện chạy nhanh")

# Main tabs
tab_names = [
    "📊 Backtest",
    "🤖 Live Trading",
    "🎯 Regime Transformer"
]

if HAS_ADVANCED_TAB:
    tab_names.append("🚀 Advanced ML")

main_tabs = st.tabs(tab_names)

# Sidebar - shared across all tabs
with st.sidebar:
    sidebar_config = render_sidebar()
    

# Render tabs using modules
with main_tabs[0]:
    render_backtest_tab_module(sidebar_config)

with main_tabs[1]:
    render_live_trading_tabs()

with main_tabs[2]:
    render_regime_transformer_tab(sidebar_config)

if HAS_ADVANCED_TAB and len(main_tabs) > 3:
    with main_tabs[3]:
        render_regime_ensemble_advanced_tab(sidebar_config)

