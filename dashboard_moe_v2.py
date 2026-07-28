
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json

# Cấu hình trang
st.set_page_config(
    page_title="MOE v2 Enhanced Dashboard",
    page_icon="📈",
    layout="wide"
)

# Tiêu đề
st.title("📈 MOE v2 Enhanced Trading Bot Dashboard")

# Sidebar
st.sidebar.header("Cài đặt dashboard")

# Load dữ liệu từ log
@st.cache_data(ttl=300)  # Cache 5 phút
def load_trading_data():
    """Load dữ liệu giao dịch từ log file"""
    log_file = "trading_bot.log"
    if not os.path.exists(log_file):
        return pd.DataFrame()

    # Đọc log file với xử lý lỗi encoding
    trades = []
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings_to_try:
        try:
            with open(log_file, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        # Nếu tất cả các encoding đều thất bại, dùng ignore
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

    # Phân tích các dòng log
    for line in lines:
        if "✅ Đã thực hiện lệnh" in line:
            # Extract thông tin lệnh
            parts = line.split("✅ Đã thực hiện lệnh ")
            if len(parts) > 1:
                trade_info = parts[1]
                if "buy" in trade_info:
                    side = "Buy"
                elif "sell" in trade_info:
                    side = "Sell"
                else:
                    side = "Unknown"

                # Extract thời gian
                timestamp = line.split(" - ")[0]
                try:
                    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f")
                except:
                    dt = datetime.now()

                trades.append({
                    "timestamp": dt,
                    "side": side,
                    "type": "trade"
                })

        elif "🎯 Có tín hiệu giao dịch" in line:
            # Extract tín hiệu
            timestamp = line.split(" - ")[0]
            try:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f")
            except:
                dt = datetime.now()

            trades.append({
                "timestamp": dt,
                "side": "Signal",
                "type": "signal"
            })

    df = pd.DataFrame(trades)
    if len(df) > 0:
        df = df.sort_values("timestamp")
    return df

# Load dữ liệu
trading_df = load_trading_data()

# Hiển thị thông tin tổng quan
if len(trading_df) > 0:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_trades = len(trading_df[trading_df["type"] == "trade"])
        st.metric("Tổng giao dịch", total_trades)

    with col2:
        win_trades = len(trading_df[(trading_df["type"] == "trade") & (trading_df["side"] == "Buy")])
        st.metric("Giao dịch mua", win_trades)

    with col3:
        loss_trades = len(trading_df[(trading_df["type"] == "trade") & (trading_df["side"] == "Sell")])
        st.metric("Giao dịch bán", loss_trades)

    with col4:
        total_signals = len(trading_df[trading_df["type"] == "signal"])
        st.metric("Tổng tín hiệu", total_signals)

    # Biểu đồ số lượng giao dịch theo thời gian
    st.subheader("📊 Số lượng giao dịch theo thời gian")

    # Tạo dữ liệu theo ngày
    if len(trading_df) > 0:
        trading_df["date"] = trading_df["timestamp"].dt.date
        daily_trades = trading_df[trading_df["type"] == "trade"].groupby("date").size().reset_index(name="count")
        daily_signals = trading_df[trading_df["type"] == "signal"].groupby("date").size().reset_index(name="count")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_trades["date"],
            y=daily_trades["count"],
            name="Giao dịch",
            mode="lines+markers",
            line=dict(color="green")
        ))
        fig.add_trace(go.Scatter(
            x=daily_signals["date"],
            y=daily_signals["count"],
            name="Tín hiệu",
            mode="lines+markers",
            line=dict(color="blue")
        ))

        fig.update_layout(
            title="Số lượng giao dịch và tín hiệu theo ngày",
            xaxis_title="Ngày",
            yaxis_title="Số lượng",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Biểu đồ phân bố side
    st.subheader("⚖️ Phân bố giao dịch (Buy/Sell)")

    if len(trading_df) > 0:
        side_counts = trading_df[trading_df["type"] == "trade"]["side"].value_counts()
        fig = px.pie(
            values=side_counts.values,
            names=side_counts.index,
            title="Phân bố giao dịch Buy/Sell",
            color_discrete_sequence=["#2ecc71", "#e74c3c"]
        )
        st.plotly_chart(fig, use_container_width=True)

    # Bảng giao dịch gần đây
    st.subheader("📋 Giao dịch gần đây")

    if len(trading_df) > 0:
        recent_trades = trading_df[trading_df["type"] == "trade"].tail(10)
        if len(recent_trades) > 0:
            st.dataframe(
                recent_trades["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").to_frame().join(
                    recent_trades["side"].to_frame()
                ),
                use_container_width=True
            )

    # Tín hiệu gần đây
    st.subheader("🎯 Tín hiệu gần đây")

    if len(trading_df) > 0:
        recent_signals = trading_df[trading_df["type"] == "signal"].tail(10)
        if len(recent_signals) > 0:
            st.dataframe(
                recent_signals["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").to_frame(),
                use_container_width=True
            )
else:
    st.warning("Chưa có dữ liệu giao dịch. Hãy chạy bot trong vài giờ để thu thập dữ liệu.")

# Thêm thông tin về mô hình
st.sidebar.subheader("Thông tin mô hình")
st.sidebar.info(
    """
    - Model: MOE v2 Enhanced
    - Strategy: Weighted OR Signal Quality Filter
    - Threshold: > 0.3
    - Risk per trade: 2%
    - TP/SL Ratio: 3.0
    - Trailing Stop: 1.5%
    """
)

# Footer
st.markdown("---")
st.markdown("*Dashboard theo dõi hiệu suất MOE v2 Enhanced Trading Bot - Cập nhật mỗi 5 phút*")