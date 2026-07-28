"""Module chứa các hàm render cho tab Live Trading."""
from __future__ import annotations
from datetime import datetime
from typing import Dict, Any
import os
import time
import pandas as pd
import streamlit as st

from algo_trading.ui.config import STRATEGY_MAP


def render_live_trading_tabs() -> None:
    """Render toàn bộ giao diện Live Trading với các tabs."""
    st.markdown("---")
    st.header("🤖 Live Trading Bot")

    tab_config, tab_chart, tab_status, tab_orders, tab_logs = st.tabs([
        "⚙️ Cấu hình & Khởi động",
        "📈 Chart Real-time",
        "📊 Trạng thái",
        "📋 Lệnh & Số dư",
        "📝 Log"
    ])

    with tab_config:
        _render_config_tab()

    with tab_chart:
        _render_chart_tab()

    with tab_status:
        _render_status_tab()

    with tab_orders:
        _render_orders_tab()

    with tab_logs:
        _render_logs_tab()


def _render_config_tab() -> None:
    """Render tab cấu hình bot."""
    st.subheader("Cấu hình Bot")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Chế độ & API")
        live_mode = st.selectbox(
            "Chế độ",
            ["paper", "testnet", "live"],
            index=0,
            help="paper: mô phỏng, testnet: Binance testnet, live: THẬT (cẩn thận!)"
        )

        live_symbol = st.text_input("Symbol", value="BTCUSDT", help="Ví dụ: BTCUSDT, ETHUSDT")
        live_interval = st.text_input("Interval", value="5m", help="Ví dụ: 1m, 5m, 15m, 1h")

        api_key = st.text_input(
            "Binance API Key",
            type="password",
            help="Bắt buộc nếu mode=testnet hoặc live"
        )
        api_secret = st.text_input(
            "Binance API Secret",
            type="password",
            help="Bắt buộc nếu mode=testnet hoặc live"
        )

    with col2:
        st.markdown("#### Strategy & Risk")
        live_strategy = st.selectbox(
            "Strategy",
            list(STRATEGY_MAP.keys()),
            index=0,
            help="Chọn strategy để bot sử dụng"
        )

        live_risk = st.number_input(
            "Risk per Trade (%)",
            value=0.1,
            min_value=0.01,
            max_value=1.0,
            step=0.01,
            help="% số dư quote cho mỗi lệnh"
        )

        live_sl_pct = st.number_input(
            "Stop Loss (%)",
            value=0.02,
            min_value=0.0,
            step=0.001,
            format="%f"
        )

        live_tp_pct = st.number_input(
            "Take Profit (%)",
            value=0.04,
            min_value=0.0,
            step=0.001,
            format="%f"
        )

    st.markdown("---")

    # Nút khởi động/dừng bot
    col_start, col_stop, col_status = st.columns(3)

    with col_start:
        start_bot = st.button("🚀 Khởi động Bot", type="primary", use_container_width=True)

    with col_stop:
        stop_bot = st.button("⏹️ Dừng Bot", use_container_width=True)

    with col_status:
        if 'bot_running' not in st.session_state:
            st.session_state.bot_running = False

        if st.session_state.bot_running:
            st.success("🟢 Bot đang chạy")
        else:
            st.info("⚪ Bot đã dừng")

    # Xử lý start/stop
    if start_bot:
        if live_mode in ("testnet", "live") and (not api_key or not api_secret):
            st.error("❌ Cần API Key và Secret cho chế độ testnet/live!")
        else:
            try:
                # Lưu config vào session state
                st.session_state.live_config = {
                    "mode": live_mode,
                    "symbol": live_symbol.upper(),
                    "interval": live_interval,
                    "strategy": STRATEGY_MAP[live_strategy][0],
                    "strategy_params": STRATEGY_MAP[live_strategy][2],
                    "risk_per_trade": live_risk,
                    "sl_pct": live_sl_pct if live_sl_pct > 0 else None,
                    "tp_pct": live_tp_pct if live_tp_pct > 0 else None,
                    "api_key": api_key,
                    "api_secret": api_secret,
                }
                st.session_state.bot_running = True
                st.success("✅ Bot đã được khởi động! (Lưu ý: Bot chạy trong background process)")
                st.info("💡 Để bot chạy thực sự, bạn cần chạy: `python -m algo_trading.live.universal_bot` trong terminal riêng")
            except Exception as e:
                st.error(f"Lỗi khởi động bot: {e}")

    if stop_bot:
        st.session_state.bot_running = False
        st.warning("⚠️ Bot đã được dừng (trong session này). Để dừng bot thực sự, dùng Ctrl+C trong terminal chạy bot")


def _render_chart_tab() -> None:
    """Render tab chart real-time."""
    st.subheader("📈 Chart Real-time")

    if 'live_config' not in st.session_state:
        st.info("ℹ️ Chưa cấu hình bot. Vào tab 'Cấu hình & Khởi động' để thiết lập.")
        return

    config = st.session_state.live_config

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        chart_limit = st.number_input("Số nến hiển thị", min_value=50, max_value=500, value=100, step=50)
    with col2:
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=True)
    with col3:
        refresh_interval = st.slider("Interval (giây)", 1, 60, 5)

    try:
        # Lấy dữ liệu từ Binance
        from binance.client import Client

        client = None
        if config["mode"] in ("testnet", "live") and config.get("api_key"):
            client = Client(config["api_key"], config["api_secret"])
            if config["mode"] == "testnet":
                client.API_URL = "https://testnet.binance.vision/api"
        else:
            # Paper mode - dùng public API
            client = Client()

        # Lấy klines
        klines = client.get_klines(
            symbol=config["symbol"],
            interval=config["interval"],
            limit=chart_limit
        )

        # Convert to DataFrame
        df_chart = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        # Convert types
        df_chart["open_time"] = pd.to_datetime(df_chart["open_time"], unit="ms")
        df_chart.set_index("open_time", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df_chart[col] = df_chart[col].astype(float)

        # Tính signals từ strategy
        StrategyCls = STRATEGY_MAP[list(STRATEGY_MAP.keys())[0]][1]  # Default
        for name, (key, cls, params) in STRATEGY_MAP.items():
            if key == config["strategy"]:
                StrategyCls = cls
                break

        strategy = StrategyCls(**(config["strategy_params"] or {}))
        sig_result = strategy.generate_signals(df_chart[["open", "high", "low", "close", "volume"]])
        signals = sig_result.signals

        # Tính indicators cho overlay
        overlays = {}
        if config["strategy"] == "sma_ema":
            from algo_trading.indicators.moving_averages import sma, ema
            params = config["strategy_params"] or {}
            fast = params.get("fast", 20)
            slow = params.get("slow", 50)
            ma_type = params.get("ma_type", "ema")
            if ma_type == "ema":
                overlays[f"EMA{fast}"] = ema(df_chart["close"], fast)
                overlays[f"EMA{slow}"] = ema(df_chart["close"], slow)
            else:
                overlays[f"SMA{fast}"] = sma(df_chart["close"], fast)
                overlays[f"SMA{slow}"] = sma(df_chart["close"], slow)

        # Vẽ chart với plotly
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df_chart.index,
                open=df_chart["open"],
                high=df_chart["high"],
                low=df_chart["low"],
                close=df_chart["close"],
                name="Price"
            ))

            # Overlays
            for name, series in overlays.items():
                fig.add_trace(go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=name,
                    mode='lines',
                    line=dict(width=1.5)
                ))

            # Signals
            if signals is not None and len(signals) > 0:
                sig_aligned = signals.reindex(df_chart.index).fillna(0)
                sig_changes = sig_aligned.diff().fillna(sig_aligned)

                # Buy signals
                buy_signals = sig_changes > 0
                if buy_signals.any():
                    buy_prices = df_chart.loc[buy_signals, "close"]
                    fig.add_trace(go.Scatter(
                        x=buy_prices.index,
                        y=buy_prices.values,
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(
                            symbol='triangle-up',
                            size=15,
                            color='green',
                            line=dict(width=2, color='darkgreen')
                        )
                    ))

                # Sell signals
                sell_signals = sig_changes < 0
                if sell_signals.any():
                    sell_prices = df_chart.loc[sell_signals, "close"]
                    fig.add_trace(go.Scatter(
                        x=sell_prices.index,
                        y=sell_prices.values,
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='red',
                            line=dict(width=2, color='darkred')
                        )
                    ))

            # Lấy entry/exit từ orders nếu có
            if config["mode"] in ("testnet", "live") and config.get("api_key"):
                try:
                    # Lấy orders gần đây
                    recent_orders = client.get_all_orders(symbol=config["symbol"], limit=20)
                    for order in recent_orders:
                        if order["status"] == "FILLED":
                            order_time = pd.to_datetime(order["time"], unit="ms")
                            order_price = float(order.get("price", 0)) or float(order.get("avgPrice", 0))
                            side = order["side"]

                            if order_time in df_chart.index or (df_chart.index.min() <= order_time <= df_chart.index.max()):
                                color = "green" if side == "BUY" else "red"
                                fig.add_trace(go.Scatter(
                                    x=[order_time],
                                    y=[order_price],
                                    mode='markers',
                                    name=f'{side} Order',
                                    marker=dict(
                                        symbol='circle',
                                        size=12,
                                        color=color,
                                        line=dict(width=2, color='white')
                                    ),
                                    showlegend=False
                                ))
                except:
                    pass

            # SL/TP levels (nếu có position)
            if config.get("sl_pct") or config.get("tp_pct"):
                current_price = df_chart["close"].iloc[-1]

                if config.get("sl_pct"):
                    sl_price = current_price * (1 - config["sl_pct"]) if config["sl_pct"] else None
                    if sl_price:
                        fig.add_hline(
                            y=sl_price,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"SL: ${sl_price:,.2f}",
                            annotation_position="right"
                        )

                if config.get("tp_pct"):
                    tp_price = current_price * (1 + config["tp_pct"]) if config["tp_pct"] else None
                    if tp_price:
                        fig.add_hline(
                            y=tp_price,
                            line_dash="dash",
                            line_color="green",
                            annotation_text=f"TP: ${tp_price:,.2f}",
                            annotation_position="right"
                        )

            # Layout
            fig.update_layout(
                title=f"{config['symbol']} - {config['interval']} | Strategy: {config['strategy']}",
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_dark",
                height=600,
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Thông tin thêm
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Giá hiện tại", f"${df_chart['close'].iloc[-1]:,.2f}")
            with col2:
                st.metric("Signal hiện tại", f"{int(signals.iloc[-1]) if len(signals) > 0 else 0}")
            with col3:
                st.metric("Số nến", len(df_chart))
            with col4:
                st.metric("Cập nhật", datetime.now().strftime("%H:%M:%S"))

            if auto_refresh:
                time.sleep(refresh_interval)
                st.rerun()

        except ImportError:
            st.error("⚠️ Cần cài đặt plotly: `pip install plotly`")
            st.info("Hoặc dùng matplotlib chart...")

            # Fallback với matplotlib
            from algo_trading.viz.plots import plot_candlestick
            fig = plot_candlestick(
                df_chart[["open", "high", "low", "close", "volume"]],
                overlays=overlays,
                signals=signals,
                title=f"{config['symbol']} - {config['interval']}",
                use_plotly=False
            )
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Lỗi vẽ chart: {e}")
        import traceback
        st.code(traceback.format_exc())

    # Nút refresh manual
    if st.button("🔄 Refresh Chart"):
        st.rerun()


def _render_status_tab() -> None:
    """Render tab trạng thái bot."""
    st.subheader("Trạng thái Bot Real-time")

    if 'live_config' not in st.session_state:
        st.info("ℹ️ Chưa cấu hình bot. Vào tab 'Cấu hình & Khởi động' để thiết lập.")
        return

    config = st.session_state.live_config

    # Hiển thị config hiện tại
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mode", config["mode"].upper())
        st.metric("Symbol", config["symbol"])
    with col2:
        st.metric("Interval", config["interval"])
        st.metric("Strategy", config["strategy"])
    with col3:
        st.metric("Risk/Trade", f"{config['risk_per_trade']*100:.1f}%")
        st.metric("Status", "🟢 Running" if st.session_state.bot_running else "⚪ Stopped")

    # Nút refresh
    if st.button("🔄 Refresh Status"):
        st.rerun()

    # Thử kết nối Binance API để lấy thông tin real-time
    if config["mode"] in ("testnet", "live") and config.get("api_key"):
        try:
            from binance.client import Client

            client = Client(config["api_key"], config["api_secret"])
            if config["mode"] == "testnet":
                client.API_URL = "https://testnet.binance.vision/api"

            # Lấy giá hiện tại
            ticker = client.get_symbol_ticker(symbol=config["symbol"])
            current_price = float(ticker["price"])

            st.markdown("---")
            st.subheader("📊 Thông tin thị trường")
            st.metric("💰 Giá hiện tại", f"${current_price:,.2f}")

            # Lấy 24h stats
            stats = client.get_ticker(symbol=config["symbol"])
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("24h Change", f"{float(stats['priceChangePercent']):+.2f}%")
            with col2:
                st.metric("24h High", f"${float(stats['highPrice']):,.2f}")
            with col3:
                st.metric("24h Low", f"${float(stats['lowPrice']):,.2f}")
            with col4:
                st.metric("24h Volume", f"{float(stats['volume']):,.2f}")

        except Exception as e:
            st.warning(f"Không thể kết nối Binance API: {e}")
            st.info("💡 Đảm bảo API keys đúng và bot đang chạy")


def _render_orders_tab() -> None:
    """Render tab lệnh & số dư."""
    st.subheader("Lệnh & Số dư")

    if 'live_config' not in st.session_state:
        st.info("ℹ️ Chưa cấu hình bot.")
        return

    config = st.session_state.live_config

    if config["mode"] in ("testnet", "live") and config.get("api_key"):
        try:
            from binance.client import Client

            client = Client(config["api_key"], config["api_secret"])
            if config["mode"] == "testnet":
                client.API_URL = "https://testnet.binance.vision/api"

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 💵 Số dư")
                account = client.get_account()
                balances = []
                for bal in account.get("balances", []):
                    free = float(bal.get("free", 0))
                    locked = float(bal.get("locked", 0))
                    if free > 0 or locked > 0:
                        balances.append({
                            "Asset": bal["asset"],
                            "Free": f"{free:.8f}",
                            "Locked": f"{locked:.8f}",
                            "Total": f"{free + locked:.8f}"
                        })

                if balances:
                    df_balances = pd.DataFrame(balances)
                    st.dataframe(df_balances, use_container_width=True, hide_index=True)
                else:
                    st.info("Không có số dư")

            with col2:
                st.markdown("#### 📋 Lệnh đang mở")
                open_orders = client.get_open_orders(symbol=config["symbol"])
                if open_orders:
                    df_open = pd.DataFrame(open_orders)
                    # Chọn các cột quan trọng
                    display_cols = ["side", "type", "origQty", "price", "status", "time"]
                    available_cols = [col for col in display_cols if col in df_open.columns]
                    if available_cols:
                        df_display = df_open[available_cols].copy()
                        if "time" in df_display.columns:
                            df_display["time"] = pd.to_datetime(df_display["time"], unit="ms")
                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                else:
                    st.info("Không có lệnh đang mở")

            st.markdown("---")
            st.markdown("#### 📜 Lệnh gần đây (20 lệnh)")
            recent_orders = client.get_all_orders(symbol=config["symbol"], limit=20)
            if recent_orders:
                df_recent = pd.DataFrame(recent_orders)
                # Sắp xếp theo thời gian
                if "time" in df_recent.columns:
                    df_recent = df_recent.sort_values("time", ascending=False)
                    df_recent["time"] = pd.to_datetime(df_recent["time"], unit="ms")

                display_cols = ["side", "type", "executedQty", "price", "status", "time"]
                available_cols = [col for col in display_cols if col in df_recent.columns]
                if available_cols:
                    st.dataframe(df_recent[available_cols], use_container_width=True, hide_index=True)
            else:
                st.info("Chưa có lệnh nào")

            if st.button("🔄 Refresh Orders"):
                st.rerun()

        except Exception as e:
            st.error(f"Lỗi lấy thông tin: {e}")
            st.info("💡 Kiểm tra API keys và kết nối")
    else:
        st.info("ℹ️ Chế độ Paper không có lệnh thật. Chuyển sang testnet/live để xem lệnh.")


def _render_logs_tab() -> None:
    """Render tab logs."""
    st.subheader("📝 Log Bot")

    log_file = "live_trading.log"

    if os.path.exists(log_file):
        # Đọc log file
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Hiển thị số dòng
            st.info(f"📄 Tổng cộng {len(lines)} dòng log")

            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                num_lines = st.slider("Số dòng hiển thị", 10, 200, 50)
            with col2:
                filter_text = st.text_input("🔍 Lọc log", placeholder="VD: ERROR, BUY, SELL")

            # Lọc và hiển thị
            display_lines = lines[-num_lines:] if len(lines) > num_lines else lines

            if filter_text:
                display_lines = [line for line in display_lines if filter_text.upper() in line.upper()]

            # Hiển thị log
            log_text = "".join(display_lines)
            st.text_area("Log content", log_text, height=400, key="log_display")

            if st.button("🔄 Refresh Log"):
                st.rerun()

        except Exception as e:
            st.error(f"Lỗi đọc log: {e}")
    else:
        st.warning(f"⚠️ Không tìm thấy file log: {log_file}")
        st.info("💡 Log sẽ được tạo khi bot chạy lần đầu")

        # Hướng dẫn
        st.markdown("""
        ### 📖 Hướng dẫn xem log:

        1. **Chạy bot trong terminal:**
           ```bash
           python -m algo_trading.live.universal_bot
           ```

        2. **Log sẽ được ghi vào:** `live_trading.log`

        3. **Xem log real-time trong terminal:**
           ```bash
           tail -f live_trading.log
           ```
        """)

    # Link đến Binance
    st.markdown("---")
    if 'live_config' in st.session_state:
        config = st.session_state.live_config
        if config["mode"] == "testnet":
            st.markdown("🔗 [Mở Binance Testnet Dashboard](https://testnet.binance.vision/)")
        elif config["mode"] == "live":
            st.markdown("🔗 [Mở Binance Web](https://www.binance.com/)")

