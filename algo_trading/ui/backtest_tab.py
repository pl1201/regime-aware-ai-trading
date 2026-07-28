from __future__ import annotations
from datetime import datetime
from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

from algo_trading.ui.config import STRATEGY_MAP
from algo_trading.ui.utils import load_df_from_sidebar_config
from algo_trading.backtest.vectorized import run_backtest, BacktestConfig, RiskConfig
from algo_trading.backtest.event_driven import run_event_backtest, EventConfig
from algo_trading.visualization.plots import (
    plot_candlestick,
    plot_equity_curve,
    plot_drawdown,
    plot_trade_pnl_distribution,
    plot_trade_timeline,
    plot_cumulative_pnl,
    plot_winrate_metrics,
)
from algo_trading.utils.trade_stats import (
    calculate_trade_stats,
    get_winning_trades,
    get_losing_trades,
    get_trade_summary_table,
)
from algo_trading.utils.trade_formatter import format_trades_csv
try:
    from algo_trading.viz.tradingview import (
        prepare_tradingview_data,
        create_tradingview_html,
        create_tradingview_pinescript,
    )
    HAS_TRADINGVIEW = True
except ImportError:
    HAS_TRADINGVIEW = False
from algo_trading.utils.session_analysis import (
    session_return_stats,
    hour_of_day_return_stats,
)
from algo_trading.live.strategy_evaluator import StrategyEvaluator
from algo_trading.live.indicator_combiner import PRESET_COMBINATIONS
import logging

logger = logging.getLogger(__name__)


def _load_df_from_config(sidebar_config: Dict[str, Any]) -> pd.DataFrame:
    """Load DataFrame từ sidebar config."""
    return load_df_from_sidebar_config(
        source=sidebar_config['source'],
        ticker=sidebar_config['ticker'],
        symbol=sidebar_config['symbol'],
        interval=sidebar_config['interval'],
        start=sidebar_config['start'],
        end=sidebar_config['end'],
        market=sidebar_config['market'],
        path=sidebar_config['path']
    )


def render_backtest_tab(sidebar_config: Dict[str, Any]) -> None:
    """Render toàn bộ giao diện tab Backtest."""
    # Extract config values
    clicked_load = sidebar_config.get('clicked_load', False)
    clicked_compare = sidebar_config.get('clicked_compare', False)
    clicked_run = sidebar_config.get('clicked_run', False)
    clicked_analysis = sidebar_config.get('clicked_analysis', False)
    analysis_mode = sidebar_config.get('analysis_mode', 'None')
    
    st.info(
        "💡 **Lưu ý:** Dữ liệu ở đây là **giá lịch sử** từ Binance/yfinance để backtest strategies. "
        "Bạn có thể tải trước bằng nút '1) Tải dữ liệu' hoặc hệ thống sẽ tự động tải khi chạy backtest/so sánh."
    )

    # Tải dữ liệu nếu bấm (tùy chọn - để xem trước)
    if clicked_load:
        try:
            df = _load_df_from_config(sidebar_config)
            st.session_state['df'] = df
            st.success(f"✅ Đã tải dữ liệu: {len(df)} dòng")
            st.markdown("### 📋 Xem trước dữ liệu")
            st.dataframe(df.head(20))
            if not df.empty:
                st.caption(f"Dữ liệu từ {df.index[0]} đến {df.index[-1]}")
            # Reset sau khi xử lý xong
            st.session_state.clicked_load = False
        except Exception as e:
            st.error(f"❌ Lỗi tải dữ liệu: {e}")
            st.info("💡 Kiểm tra lại cấu hình trong sidebar (symbol, interval, API keys nếu dùng Binance)")
            # Reset ngay cả khi có lỗi
            st.session_state.clicked_load = False

    # Chức năng so sánh nhiều strategies
    if clicked_compare:
        try:
            _render_strategy_comparison(sidebar_config)
            # Không reset clicked_compare để giữ UI so sánh hiển thị
            # Sẽ tự động reset khi người dùng nhấn nút khác trong sidebar
        except Exception as e:
            st.error(f"❌ Lỗi khi so sánh strategies: {e}")
            logger.exception("Lỗi khi so sánh strategies")
            # Không reset khi có lỗi để người dùng có thể xem lỗi

    # Chạy backtest nếu bấm
    if clicked_run:
        try:
            _render_single_backtest(sidebar_config)
            # Reset sau khi xử lý xong
            st.session_state.clicked_run = False
        except Exception as e:
            st.error(f"❌ Lỗi khi chạy backtest: {e}")
            logger.exception("Lỗi khi chạy backtest")
            # Reset khi có lỗi
            st.session_state.clicked_run = False

    # Phân tích phiên/giờ
    if clicked_analysis:
        try:
            _render_session_analysis(sidebar_config)
            # Reset sau khi xử lý xong
            st.session_state.clicked_analysis = False
        except Exception as e:
            st.error(f"❌ Lỗi khi phân tích phiên/giờ: {e}")
            logger.exception("Lỗi khi phân tích phiên/giờ")
            # Reset khi có lỗi
            st.session_state.clicked_analysis = False
    
    # Hiển thị hướng dẫn nếu chưa có action nào
    if not (clicked_load or clicked_compare or clicked_run or clicked_analysis):
        st.markdown("---")
        st.markdown("### 📖 Hướng dẫn sử dụng")
        st.markdown("""
        1. **Tải dữ liệu** (tùy chọn): Nhấn nút "1) Tải dữ liệu" trong sidebar để xem trước dữ liệu
        2. **Chạy backtest**: Chọn strategy trong sidebar và nhấn nút "2) Chạy backtest"
        3. **So sánh strategies**: Nhấn nút "3) So sánh nhiều strategies" để so sánh nhiều strategies cùng lúc
        4. **Phân tích phiên/giờ**: Nhấn nút "4) Phân tích phiên/giờ" để xem thống kê theo phiên hoặc giờ
        """)


def _render_strategy_comparison(sidebar_config: Dict[str, Any]) -> None:
    """Render phần so sánh nhiều strategies."""
    
    # Tự động load dữ liệu nếu chưa có
    if 'df' not in st.session_state:
        with st.spinner("🔄 Đang tải dữ liệu..."):
            try:
                df = _load_df_from_config(sidebar_config)
                st.session_state['df'] = df
                st.success(f"✅ Đã tải dữ liệu: {len(df)} dòng")
            except Exception as e:
                st.error(f"❌ Lỗi tải dữ liệu: {e}")
                logger.exception("Lỗi tải dữ liệu")
                return
    else:
        df = st.session_state['df']

    if df.empty or len(df) < 50:
        st.error("⚠️ Không đủ dữ liệu (cần ít nhất 50 klines). Vui lòng kiểm tra cấu hình dữ liệu trong sidebar.")
        return

    st.header("📊 So sánh nhiều Strategies")

    compare_mode = st.radio(
        "Chế độ so sánh",
        ["So sánh nhanh (chọn strategies)", "Đánh giá tất cả strategies"],
        horizontal=True,
        help="So sánh nhanh: chọn strategies cụ thể. Đánh giá tất cả: chạy tất cả strategies với nhiều tham số",
    )

    # Thêm preset combinations (dùng cho cả 2 chế độ)
    include_combinations = st.checkbox("Bao gồm Indicator Combinations", value=True)
    selected_combos: list[str] = []
    combo_options: Dict[str, str] = {}
    if include_combinations:
        combo_options = {
            'Trend + Momentum': 'trend_momentum',
            'Mean Reversion': 'mean_reversion',
            'Balanced': 'balanced',
            'Aggressive Trend': 'aggressive_trend',
            'Conservative': 'conservative',
            'Momentum Focused': 'momentum_focused',
        }
        selected_combos = st.multiselect(
            "Chọn Combinations",
            list(combo_options.keys()),
            default=['Balanced', 'Trend + Momentum'],
        )

    if compare_mode == "So sánh nhanh (chọn strategies)":
        try:
            _render_quick_comparison(df, sidebar_config, selected_combos, combo_options)
        except Exception as e:
            st.error(f"❌ Lỗi khi so sánh nhanh: {e}")
            logger.exception("Lỗi khi so sánh nhanh")
    else:
        try:
            _render_full_evaluation(df, sidebar_config)
        except Exception as e:
            st.error(f"❌ Lỗi khi đánh giá tất cả strategies: {e}")
            logger.exception("Lỗi khi đánh giá tất cả strategies")


def _render_quick_comparison(
    df: pd.DataFrame,
    sidebar_config: Dict[str, Any],
    selected_combos: list[str],
    combo_options: Dict[str, str]
) -> None:
    """Render phần so sánh nhanh các strategies."""
    
    selected_strategies = st.multiselect(
        "Chọn strategies để so sánh",
        list(STRATEGY_MAP.keys()),
        default=list(STRATEGY_MAP.keys())[:5],
        help="Chọn nhiều strategies để so sánh hiệu quả",
    )
    
    if st.button("🚀 Chạy so sánh", type="primary"):
        with st.spinner("🔄 Đang chạy backtest cho tất cả strategies... (có thể mất vài phút)"):
            try:
                results_list: list[Dict[str, Any]] = []
                interval = sidebar_config['interval']
                mode = sidebar_config['mode']
                leverage = sidebar_config['leverage']
                allow_short = sidebar_config['allow_short']
                commission = sidebar_config['commission']
                slippage_bps = sidebar_config['slippage_bps']
                use_next_open = sidebar_config['use_next_open']
                sl_pct = sidebar_config['sl_pct']
                tp_pct = sidebar_config['tp_pct']
                trailing_pct = sidebar_config['trailing_pct']
                sl_atr_k = sidebar_config['sl_atr_k']
                tp_atr_k = sidebar_config['tp_atr_k']
                trailing_atr_k = sidebar_config['trailing_atr_k']
                atr_col = sidebar_config['atr_col']
                
                freq = '1H' if 'h' in interval.lower() else ('1D' if 'd' in interval.lower() else None)
                risk = None
                if any([sl_pct > 0, tp_pct > 0, trailing_pct > 0, sl_atr_k > 0, tp_atr_k > 0, trailing_atr_k > 0]):
                    risk = RiskConfig(
                        sl_pct=sl_pct or None,
                        tp_pct=tp_pct or None,
                        trailing_pct=trailing_pct or None,
                        sl_atr_k=sl_atr_k or None,
                        tp_atr_k=tp_atr_k or None,
                        trailing_atr_k=trailing_atr_k or None,
                        atr_col=atr_col,
                    )

                # Test từng strategy
                progress_bar = st.progress(0)
                total_tests = len(selected_strategies) + len(selected_combos)

                for idx, strat_name_key in enumerate(selected_strategies):
                    progress_bar.progress((idx + 1) / max(total_tests, 1))
                    try:
                        _, StrategyClsLocal, default_params = STRATEGY_MAP[strat_name_key]
                        strat = StrategyClsLocal(**default_params)
                        sig = strat.generate_signals(df).signals

                        if mode == 'vectorized':
                            cfg = BacktestConfig(
                                initial_capital=1.0,
                                leverage=leverage,
                                allow_short=allow_short,
                                commission=commission,
                                slippage_bps=slippage_bps,
                                use_next_open=use_next_open,
                                freq=freq,
                            )
                            res = run_backtest(df, sig, cfg=cfg, risk=risk, max_trades=100)
                        else:
                            cfg = EventConfig(
                                initial_cash=10000.0,
                                leverage=leverage,
                                allow_short=allow_short,
                                commission=commission,
                                slippage_bps=slippage_bps,
                                use_next_open=use_next_open,
                                price_col='close',
                                open_col='open',
                                high_col='high',
                                low_col='low',
                                freq=freq,
                            )
                            res = run_event_backtest(df, sig, cfg=cfg, risk=risk, max_trades=100)

                        results_list.append({
                            'strategy_name': strat_name_key,
                            'type': 'Strategy',
                            'summary': res['summary'],
                            'equity': res['equity'],
                            'returns': res['returns'],
                        })
                    except Exception as e:
                        st.warning(f"Lỗi với {strat_name_key}: {e}")

                # Test combinations
                for idx, combo_name in enumerate(selected_combos):
                    progress_bar.progress((len(selected_strategies) + idx + 1) / max(total_tests, 1))
                    try:
                        combo_key = combo_options[combo_name]
                        combo_func = PRESET_COMBINATIONS[combo_key]
                        combo_strategy = combo_func()
                        sig = combo_strategy.generate_signals(df).signals

                        if mode == 'vectorized':
                            cfg = BacktestConfig(
                                initial_capital=1.0,
                                leverage=leverage,
                                allow_short=allow_short,
                                commission=commission,
                                slippage_bps=slippage_bps,
                                use_next_open=use_next_open,
                                freq=freq,
                            )
                            res = run_backtest(df, sig, cfg=cfg, risk=risk, max_trades=100)
                        else:
                            cfg = EventConfig(
                                initial_cash=10000.0,
                                leverage=leverage,
                                allow_short=allow_short,
                                commission=commission,
                                slippage_bps=slippage_bps,
                                use_next_open=use_next_open,
                                price_col='close',
                                open_col='open',
                                high_col='high',
                                low_col='low',
                                freq=freq,
                            )
                            res = run_event_backtest(df, sig, cfg=cfg, risk=risk, max_trades=100)

                        results_list.append({
                            'strategy_name': f"Combination: {combo_name}",
                            'type': 'Combination',
                            'summary': res['summary'],
                            'equity': res['equity'],
                            'returns': res['returns'],
                        })
                    except Exception as e:
                        st.warning(f"Lỗi với {combo_name}: {e}")

                progress_bar.empty()

                if results_list:
                    _display_comparison_results(results_list, sidebar_config)
            except Exception as e:
                st.error(f"❌ Lỗi so sánh: {e}")
                logger.exception("Lỗi so sánh")


def _display_comparison_results(results_list: list[Dict[str, Any]], sidebar_config: Dict[str, Any]) -> None:
    """Hiển thị kết quả so sánh."""
    st.success(f"✅ Đã hoàn tất so sánh {len(results_list)} strategies!")

    # Tạo DataFrame so sánh
    comparison_data: list[Dict[str, Any]] = []
    for r in results_list:
        summary = r['summary']
        comparison_data.append({
            'Strategy': r['strategy_name'],
            'Type': r['type'],
            'Total Return (%)': summary.get('TotalReturn', 0) * 100,
            'Sharpe': summary.get('Sharpe', 0),
            'Sortino': summary.get('Sortino', 0),
            'CAGR (%)': summary.get('CAGR', 0) * 100,
            'Max Drawdown (%)': summary.get('MaxDrawdown', 0) * 100,
            'Calmar': summary.get('Calmar', 0),
            'Volatility': summary.get('Volatility', 0),
        })

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Total Return (%)', ascending=False)

    st.markdown("### 📊 Bảng so sánh")
    st.dataframe(df_comparison, use_container_width=True)

    # Export CSV
    csv_comparison = df_comparison.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Tải bảng so sánh (CSV)",
        data=csv_comparison,
        file_name=f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    # Export PDF
    _export_comparison_pdf(df_comparison, results_list, sidebar_config)

    # Charts comparison
    _display_comparison_charts(results_list)


def _export_comparison_pdf(
    df_comparison: pd.DataFrame,
    results_list: list[Dict[str, Any]],
    sidebar_config: Dict[str, Any]
) -> None:
    """Export PDF cho comparison."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Table,
            TableStyle,
            Paragraph,
            Spacer,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from io import BytesIO

        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        story.append(Paragraph("BÁO CÁO SO SÁNH STRATEGIES", title_style))
        story.append(Spacer(1, 0.2 * inch))

        # Info
        symbol = sidebar_config.get('symbol', '')
        ticker = sidebar_config.get('ticker', '')
        interval = sidebar_config.get('interval', '')
        info_text = f"""
        <b>Ngày tạo:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Symbol:</b> {symbol if symbol else (ticker if ticker else 'N/A')}<br/>
        <b>Interval:</b> {interval}<br/>
        <b>Số strategies:</b> {len(results_list)}
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))

        # Table
        table_data = [['Strategy', 'Type', 'Return %', 'Sharpe', 'Sortino', 'CAGR %', 'Max DD %', 'Calmar']]
        for _, row in df_comparison.iterrows():
            table_data.append([
                row['Strategy'],
                row['Type'],
                f"{row['Total Return (%)']:.2f}",
                f"{row['Sharpe']:.3f}",
                f"{row['Sortino']:.3f}",
                f"{row['CAGR (%)']:.2f}",
                f"{row['Max Drawdown (%)']:.2f}",
                f"{row['Calmar']:.3f}",
            ])

        table = Table(
            table_data,
            colWidths=[2 * inch, 0.8 * inch, 0.7 * inch, 0.6 * inch, 0.6 * inch, 0.7 * inch, 0.7 * inch, 0.6 * inch],
        )
        table.setStyle(
            TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ])
        )
        story.append(table)

        doc.build(story)
        pdf_buffer.seek(0)

        st.download_button(
            "📄 Tải báo cáo so sánh (PDF)",
            data=pdf_buffer.getvalue(),
            file_name=f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )
    except ImportError:
        st.warning("⚠️ Cần cài đặt reportlab để export PDF: `pip install reportlab`")
    except Exception as e:
        st.warning(f"⚠️ Lỗi tạo PDF: {e}")


def _display_comparison_charts(results_list: list[Dict[str, Any]]) -> None:
    """Hiển thị biểu đồ so sánh."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=('Equity Curves Comparison', 'Returns Distribution'),
            vertical_spacing=0.1,
        )

        # Equity curves
        for r in results_list:
            fig.add_trace(
                go.Scatter(
                    x=r['equity'].index,
                    y=r['equity'].values,
                    name=r['strategy_name'],
                    mode='lines',
                ),
                row=1,
                col=1,
            )

        # Returns distribution
        for r in results_list:
            fig.add_trace(
                go.Histogram(
                    x=r['returns'].values,
                    name=r['strategy_name'],
                    opacity=0.6,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        fig.update_layout(height=800, title_text="So sánh Strategies", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("💡 Cài đặt plotly để xem biểu đồ: `pip install plotly`")


def _render_full_evaluation(df: pd.DataFrame, sidebar_config: Dict[str, Any]) -> None:
    st.subheader("Đánh giá tất cả Strategies")
    st.info("💡 Chức năng này sẽ đánh giá tất cả strategies với nhiều bộ tham số khác nhau")

    col1, col2 = st.columns(2)
    with col1:
        eval_initial_capital = st.number_input(
            "Vốn ban đầu",
            value=10000.0,
            min_value=100.0,
            step=100.0,
            key="eval_capital_main",
        )
        eval_commission = (
            st.number_input(
                "Phí giao dịch (%)",
                value=0.1,
                min_value=0.0,
                step=0.01,
                key="eval_comm_main",
            )
            / 100
        )
    with col2:
        eval_use_stops = st.checkbox("Sử dụng SL/TP", value=True, key="eval_stops_main")
        eval_sl_pct = (
            st.number_input(
                "Stop Loss (%)",
                value=2.0,
                min_value=0.0,
                step=0.1,
                key="eval_sl_main",
            )
            / 100
            if eval_use_stops
            else None
        )
        eval_tp_pct = (
            st.number_input(
                "Take Profit (%)",
                value=4.0,
                min_value=0.0,
                step=0.1,
                key="eval_tp_main",
            )
            / 100
            if eval_use_stops
            else None
        )

    eval_top_n = st.slider("Hiển thị top N strategies", 5, 20, 10, key="eval_topn_main")

    # Nút tạo báo cáo chi tiết (độc lập, có thể dùng sau khi đã chạy đánh giá)
    if st.button("📊 Tạo báo cáo chi tiết (PnL, RR, Entry Signals, Time Analysis)", key="btn_detailed_report"):
        # Kiểm tra xem đã có evaluator trong session state chưa
        if 'evaluator' in st.session_state and st.session_state['evaluator'] is not None:
            evaluator = st.session_state['evaluator']
            eval_top_n = st.session_state.get('eval_config', {}).get('top_n', eval_top_n)
            
            with st.spinner("🔄 Đang tạo báo cáo chi tiết..."):
                try:
                    detailed_report_file = f"strategy_detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    detailed_report = evaluator.generate_detailed_report(
                        output_file=detailed_report_file,
                        top_n=eval_top_n
                    )
                    
                    if detailed_report and len(detailed_report) > 0:
                        st.success(f"✅ Đã tạo báo cáo chi tiết! Báo cáo đã được lưu vào: {detailed_report_file}")
                        
                        # Display detailed report
                        st.markdown("---")
                        st.subheader("📊 Báo cáo chi tiết")
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.text_area("Nội dung báo cáo chi tiết", detailed_report, height=500, key=f"report_text_detailed_{timestamp}")
                        
                        # Download detailed report TXT
                        st.download_button(
                            "📥 Tải báo cáo chi tiết (TXT)",
                            data=detailed_report.encode('utf-8'),
                            file_name=detailed_report_file,
                            mime="text/plain",
                            key=f"dl_detailed_report_txt_{timestamp}",
                        )
                    else:
                        st.error("❌ Báo cáo chi tiết trống. Có thể không có dữ liệu để phân tích.")
                except Exception as e:
                    st.error(f"❌ Lỗi khi tạo báo cáo chi tiết: {e}")
                    logger.exception("Lỗi khi tạo báo cáo chi tiết")
        else:
            st.warning("⚠️ Vui lòng chạy đánh giá tất cả strategies trước khi tạo báo cáo chi tiết!")

    st.markdown("---")

    if st.button("🚀 Chạy đánh giá tất cả strategies", type="primary", key="btn_eval_all"):
        with st.spinner("🔄 Đang đánh giá tất cả strategies... (có thể mất vài phút)"):
            try:
                evaluator = StrategyEvaluator(
                    df=df,
                    initial_capital=eval_initial_capital,
                    commission=eval_commission,
                    use_stops=eval_use_stops,
                    sl_pct=eval_sl_pct,
                    tp_pct=eval_tp_pct,
                )
                
                # Lưu evaluator vào session state để dùng sau
                st.session_state['evaluator'] = evaluator
                st.session_state['eval_df'] = df
                st.session_state['eval_config'] = {
                    'initial_capital': eval_initial_capital,
                    'commission': eval_commission,
                    'use_stops': eval_use_stops,
                    'sl_pct': eval_sl_pct,
                    'tp_pct': eval_tp_pct,
                    'top_n': eval_top_n,
                }

                # Generate report
                report_file = f"strategy_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                report = evaluator.generate_comparison_report(output_file=report_file, top_n=eval_top_n)

                st.success(f"✅ Đã hoàn tất đánh giá! Báo cáo đã được lưu vào: {report_file}")

                # Display report
                st.markdown("---")
                st.subheader("📊 Báo cáo so sánh")
                st.text_area("Nội dung báo cáo", report, height=400, key="report_text_main")

                # Download report TXT
                st.download_button(
                    "📥 Tải báo cáo (TXT)",
                    data=report.encode('utf-8'),
                    file_name=report_file,
                    mime="text/plain",
                    key="dl_report_txt",
                )

                # Get results DataFrame
                st.info("🔄 Đang đánh giá từng strategy...")
                try:
                    df_results = evaluator.evaluate_all()

                    if df_results is None or df_results.empty:
                        st.warning("⚠️ Không có kết quả đánh giá. Có thể do lỗi trong quá trình đánh giá.")
                    else:
                        # Xử lý lỗi an toàn hơn
                        try:
                            if 'error' in df_results.columns:
                                df_results_clean = df_results[~df_results['error'].astype(bool)]
                            else:
                                df_results_clean = df_results
                        except Exception as e:
                            st.warning(f"⚠️ Lỗi khi lọc kết quả: {e}")
                            df_results_clean = df_results

                        if df_results_clean.empty:
                            st.warning("⚠️ Tất cả strategies đều có lỗi. Vui lòng kiểm tra lại dữ liệu và cấu hình.")
                            # Hiển thị lỗi chi tiết nếu có
                            if 'error' in df_results.columns:
                                error_df = df_results[df_results['error'].astype(bool)]
                                if not error_df.empty:
                                    st.error("📋 Chi tiết lỗi:")
                                    st.dataframe(error_df[['strategy_name', 'error']] if 'strategy_name' in error_df.columns else error_df)
                        else:
                            try:
                                _display_evaluation_results(
                                    df_results_clean,
                                    eval_top_n,
                                    eval_initial_capital,
                                    eval_commission,
                                    eval_use_stops,
                                    sidebar_config
                                )
                            except Exception as e:
                                st.error(f"❌ Lỗi khi hiển thị kết quả: {e}")
                                logger.exception("Lỗi khi hiển thị kết quả")
                                # Vẫn hiển thị DataFrame thô nếu có thể
                                if not df_results_clean.empty:
                                    st.dataframe(df_results_clean)
                except Exception as e:
                    st.error(f"❌ Lỗi khi đánh giá strategies: {e}")
                    logger.exception("Lỗi khi đánh giá strategies")
            except Exception as e:
                st.error(f"❌ Lỗi đánh giá: {e}")
                logger.exception("Lỗi đánh giá")


def _display_evaluation_results(
    df_results_clean: pd.DataFrame,
    eval_top_n: int,
    eval_initial_capital: float,
    eval_commission: float,
    eval_use_stops: bool,
    sidebar_config: Dict[str, Any]
) -> None:
    """Hiển thị kết quả đánh giá."""
    st.markdown("---")
    st.subheader("📈 Bảng kết quả chi tiết")

    # Kiểm tra các cột có tồn tại không
    available_cols = df_results_clean.columns.tolist()
    
    # Format for display - chỉ lấy các cột có sẵn
    display_cols = [
        'strategy_name',
        'category',
        'total_return',
        'sharpe',
        'sortino',
        'cagr',
        'max_drawdown',
        'win_rate',
        'composite_score',
    ]
    display_cols = [col for col in display_cols if col in available_cols]
    
    if not display_cols:
        st.warning("⚠️ Không có cột nào để hiển thị. Hiển thị tất cả các cột có sẵn:")
        st.dataframe(df_results_clean)
        return
    
    display_df = df_results_clean[display_cols].copy()
    
    # Format các cột số nếu có
    if 'total_return' in display_df.columns:
        display_df['total_return'] = (display_df['total_return'] * 100).round(2)
    if 'cagr' in display_df.columns:
        display_df['cagr'] = (display_df['cagr'] * 100).round(2)
    if 'max_drawdown' in display_df.columns:
        display_df['max_drawdown'] = (display_df['max_drawdown'] * 100).round(2)
    if 'win_rate' in display_df.columns:
        display_df['win_rate'] = (display_df['win_rate'] * 100).round(1)
    if 'sharpe' in display_df.columns:
        display_df['sharpe'] = display_df['sharpe'].round(3)
    if 'sortino' in display_df.columns:
        display_df['sortino'] = display_df['sortino'].round(3)
    if 'composite_score' in display_df.columns:
        display_df['composite_score'] = display_df['composite_score'].round(2)

    # Sort by composite score nếu có, nếu không thì sort theo total_return hoặc cột đầu tiên
    if 'composite_score' in display_df.columns:
        display_df = display_df.sort_values('composite_score', ascending=False)
    elif 'total_return' in display_df.columns:
        display_df = display_df.sort_values('total_return', ascending=False)
    elif len(display_df.columns) > 0:
        display_df = display_df.sort_values(display_df.columns[0], ascending=False)

    st.dataframe(display_df, use_container_width=True, height=400)

    # Download CSV
    csv_results = df_results_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Tải kết quả (CSV)",
        data=csv_results,
        file_name=f"strategy_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="dl_csv_eval",
    )

    # Export PDF
    _export_evaluation_pdf(display_df, eval_top_n, eval_initial_capital, eval_commission, eval_use_stops, sidebar_config)

    # Charts
    _display_evaluation_charts(display_df, df_results_clean, eval_top_n)


def _export_evaluation_pdf(
    display_df: pd.DataFrame,
    eval_top_n: int,
    eval_initial_capital: float,
    eval_commission: float,
    eval_use_stops: bool,
    sidebar_config: Dict[str, Any]
) -> None:
    """Export PDF cho evaluation."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Table,
            TableStyle,
            Paragraph,
            Spacer,
            PageBreak,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from io import BytesIO

        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        story.append(Paragraph("BÁO CÁO ĐÁNH GIÁ TẤT CẢ STRATEGIES", title_style))
        story.append(Spacer(1, 0.2 * inch))

        # Info
        symbol = sidebar_config.get('symbol', '')
        ticker = sidebar_config.get('ticker', '')
        interval = sidebar_config.get('interval', '')
        symbol_name = symbol if symbol else (ticker if ticker else 'N/A')
        info_text = f"""
        <b>Ngày tạo:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Symbol:</b> {symbol_name}<br/>
        <b>Interval:</b> {interval}<br/>
        <b>Vốn ban đầu:</b> ${eval_initial_capital:,.2f}<br/>
        <b>Phí giao dịch:</b> {eval_commission*100:.2f}%<br/>
        <b>SL/TP:</b> {'Có' if eval_use_stops else 'Không'}
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))

        # Summary table
        story.append(Paragraph("TOP STRATEGIES", styles['Heading2']))
        story.append(Spacer(1, 0.2 * inch))

        top_df = display_df.head(eval_top_n)
        table_data = [
            ['#', 'Strategy', 'Category', 'Return %', 'Sharpe', 'CAGR %', 'Max DD %', 'Score']
        ]
        for idx, (_, row) in enumerate(top_df.iterrows(), 1):
            table_data.append([
                str(idx),
                row['strategy_name'],
                row['category'],
                f"{row['total_return']:.2f}",
                f"{row['sharpe']:.3f}",
                f"{row['cagr']:.2f}",
                f"{row['max_drawdown']:.2f}",
                f"{row['composite_score']:.2f}",
            ])

        table = Table(
            table_data,
            colWidths=[
                0.3 * inch,
                2 * inch,
                1 * inch,
                0.7 * inch,
                0.6 * inch,
                0.7 * inch,
                0.7 * inch,
                0.6 * inch,
            ],
        )
        table.setStyle(
            TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ])
        )
        story.append(table)

        story.append(PageBreak())

        # Full table
        story.append(
            Paragraph("BẢNG TỔNG HỢP TẤT CẢ STRATEGIES", styles['Heading2'])
        )
        story.append(Spacer(1, 0.2 * inch))

        full_table_data = [
            [
                'Strategy',
                'Category',
                'Return %',
                'Sharpe',
                'Sortino',
                'CAGR %',
                'Max DD %',
                'Win Rate %',
                'Score',
            ]
        ]
        for _, row in display_df.iterrows():
            full_table_data.append([
                str(row['strategy_name'])[:30],
                str(row['category']),
                f"{row['total_return']:.2f}",
                f"{row['sharpe']:.3f}",
                f"{row['sortino']:.3f}",
                f"{row['cagr']:.2f}",
                f"{row['max_drawdown']:.2f}",
                f"{row['win_rate']:.1f}",
                f"{row['composite_score']:.2f}",
            ])

        full_table = Table(
            full_table_data,
            colWidths=[
                1.5 * inch,
                0.8 * inch,
                0.6 * inch,
                0.5 * inch,
                0.5 * inch,
                0.6 * inch,
                0.6 * inch,
                0.6 * inch,
                0.5 * inch,
            ],
        )
        full_table.setStyle(
            TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 7),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 6),
            ])
        )
        story.append(full_table)

        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)

        st.download_button(
            "📄 Tải báo cáo đánh giá (PDF)",
            data=pdf_buffer.getvalue(),
            file_name=f"strategy_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="dl_pdf_eval",
        )
    except ImportError:
        st.warning("⚠️ Cần cài đặt reportlab để export PDF: `pip install reportlab`")
    except Exception as e:
        st.warning(f"⚠️ Lỗi tạo PDF: {e}")


def _display_evaluation_charts(
    display_df: pd.DataFrame,
    df_results_clean: pd.DataFrame,
    eval_top_n: int
) -> None:
    """Hiển thị biểu đồ đánh giá."""
    try:
        import plotly.express as px

        top_df = display_df.head(eval_top_n)
        
        # Kiểm tra composite_score có tồn tại không
        if 'composite_score' not in top_df.columns:
            # Nếu không có composite_score, dùng total_return hoặc cột đầu tiên
            y_col = 'total_return' if 'total_return' in top_df.columns else top_df.columns[0]
            st.warning(f"⚠️ Không có cột 'composite_score', sử dụng '{y_col}' thay thế.")
        else:
            y_col = 'composite_score'
        
        fig1 = px.bar(
            top_df,
            x='strategy_name',
            y=y_col,
            color='category',
            title=f'Top {eval_top_n} Strategies - {"Composite Score" if y_col == "composite_score" else y_col.replace("_", " ").title()}',
            labels={
                y_col: 'Composite Score' if y_col == 'composite_score' else y_col.replace('_', ' ').title(),
                'strategy_name': 'Strategy',
            },
        )
        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)

        # Sharpe vs Drawdown scatter (fix negative marker sizes)
        plot_df = df_results_clean.copy()

        # Convert percentage-style fields for readability and sizing
        x_col = 'max_drawdown'
        if 'max_drawdown' in plot_df.columns:
            plot_df['max_drawdown_pct'] = (plot_df['max_drawdown'] * 100).round(2)
            x_col = 'max_drawdown_pct'

        size_col = None
        if 'total_return' in plot_df.columns:
            plot_df['total_return_pct'] = (plot_df['total_return'] * 100).round(2)
            # Marker size must be non-negative; use absolute % return with a small floor
            safe_size = plot_df['total_return_pct'].abs().replace([np.inf, -np.inf], np.nan).fillna(0)
            plot_df['bubble_size'] = safe_size.clip(lower=0) + 1  # avoid zero-size markers
            size_col = 'bubble_size'

        hover_fields = ['strategy_name'] if 'strategy_name' in plot_df.columns else []
        for col in ['total_return_pct', 'win_rate', 'max_drawdown_pct']:
            if col in plot_df.columns:
                hover_fields.append(col)

        fig2 = px.scatter(
            plot_df,
            x=x_col,
            y='sharpe' if 'sharpe' in plot_df.columns else plot_df.columns[0],
            color='category' if 'category' in plot_df.columns else None,
            size=size_col,
            hover_data=hover_fields or None,
            title='Sharpe Ratio vs Max Drawdown',
            labels={
                x_col: 'Max Drawdown (%)' if x_col.endswith('_pct') else 'Max Drawdown',
                'sharpe': 'Sharpe Ratio',
            },
        )
        st.plotly_chart(fig2, use_container_width=True)

    except ImportError:
        st.info("💡 Cài đặt plotly để xem biểu đồ: `pip install plotly`")


def _render_single_backtest(sidebar_config: Dict[str, Any]) -> None:
    """Render phần chạy backtest cho một strategy."""
    try:
        strategy_type = sidebar_config['strategy_type']
        StrategyCls = sidebar_config['StrategyCls']
        params = sidebar_config['params']
        strat_name = sidebar_config['strat_name']
        interval = sidebar_config['interval']
        mode = sidebar_config['mode']
        leverage = sidebar_config['leverage']
        allow_short = sidebar_config['allow_short']
        commission = sidebar_config['commission']
        slippage_bps = sidebar_config['slippage_bps']
        use_next_open = sidebar_config['use_next_open']
        sl_pct = sidebar_config['sl_pct']
        tp_pct = sidebar_config['tp_pct']
        trailing_pct = sidebar_config['trailing_pct']
        sl_atr_k = sidebar_config['sl_atr_k']
        tp_atr_k = sidebar_config['tp_atr_k']
        trailing_atr_k = sidebar_config['trailing_atr_k']
        atr_col = sidebar_config['atr_col']
        use_tradingview = sidebar_config['use_tradingview']
        symbol = sidebar_config.get('symbol', '')
        ticker = sidebar_config.get('ticker', '')

        # Kiểm tra strategy type TRƯỚC khi load dữ liệu hoặc hiển thị loading
        if strategy_type == "So sánh nhiều strategies":
            st.warning(
                "⚠️ **Lưu ý:** Bạn đã chọn 'So sánh nhiều strategies' nhưng đang nhấn nút '2) Chạy backtest'."
            )
            st.info(
                "💡 **Hướng dẫn:** Để so sánh nhiều strategies, vui lòng nhấn nút **'3) So sánh nhiều strategies'** ở sidebar."
            )
            return

        # Tự động load dữ liệu nếu chưa có
        if 'df' not in st.session_state:
            with st.spinner("🔄 Đang tải dữ liệu..."):
                try:
                    df = _load_df_from_config(sidebar_config)
                    st.session_state['df'] = df
                    st.success(f"✅ Đã tải dữ liệu: {len(df)} dòng")
                except Exception as e:
                    st.error(f"❌ Lỗi tải dữ liệu: {e}")
                    logger.exception("Lỗi tải dữ liệu")
                    return
        else:
            df = st.session_state['df']

        if df.empty:
            st.error("⚠️ DataFrame rỗng. Vui lòng kiểm tra cấu hình dữ liệu trong sidebar.")
            return

        # Hiển thị thông báo đang chạy backtest
        MAX_TRADES = 100
        st.info(f"🔄 Đang chạy backtest với giới hạn {MAX_TRADES} lệnh...")

        # Tạo progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # tạo chiến lược
        status_text.text("📊 Đang tạo chiến lược...")
        progress_bar.progress(10)

        # Xử lý strategy type
        if strategy_type == "Indicator Combination":
            if 'combo_strategy' in st.session_state:
                combo_strategy = st.session_state['combo_strategy']
                sig = combo_strategy.generate_signals(df).signals
                strat = combo_strategy
            else:
                st.error("⚠️ Chưa tạo combination. Vui lòng tạo combination trước.")
                return
        else:
            # Strategy đơn lẻ
            if StrategyCls is None:
                st.error("⚠️ Chưa chọn strategy. Vui lòng chọn strategy trong sidebar.")
                return
            try:
                strat = StrategyCls(**params)
                sig = strat.generate_signals(df).signals
            except Exception as e:
                st.error(f"❌ Lỗi tạo strategy: {e}")
                logger.exception("Lỗi tạo strategy")
                return

        freq = '1H' if 'h' in interval.lower() else ('1D' if 'd' in interval.lower() else None)

        # risk
        status_text.text("⚙️ Đang thiết lập risk management...")
        progress_bar.progress(20)
        risk = None
        if any([sl_pct > 0, tp_pct > 0, trailing_pct > 0, sl_atr_k > 0, tp_atr_k > 0, trailing_atr_k > 0]):
            risk = RiskConfig(
                sl_pct=sl_pct or None,
                tp_pct=tp_pct or None,
                trailing_pct=trailing_pct or None,
                sl_atr_k=sl_atr_k or None,
                tp_atr_k=tp_atr_k or None,
                trailing_atr_k=trailing_atr_k or None,
                atr_col=atr_col,
            )

        status_text.text(f"🚀 Đang chạy backtest (tối đa {MAX_TRADES} lệnh)...")
        progress_bar.progress(40)

        if mode == 'vectorized':
            cfg = BacktestConfig(
                initial_capital=1.0,
                leverage=leverage,
                allow_short=allow_short,
                commission=commission,
                slippage_bps=slippage_bps,
                use_next_open=use_next_open,
                freq=freq,
            )
            res = run_backtest(df, sig, cfg=cfg, risk=risk, max_trades=MAX_TRADES)
        else:
            cfg = EventConfig(
                initial_cash=10000.0,
                leverage=leverage,
                allow_short=allow_short,
                commission=commission,
                slippage_bps=slippage_bps,
                use_next_open=use_next_open,
                price_col='close',
                open_col='open',
                high_col='high',
                low_col='low',
                freq=freq,
            )
            res = run_event_backtest(df, sig, cfg=cfg, risk=risk, max_trades=MAX_TRADES)

        progress_bar.progress(90)
        status_text.text("✅ Đang tính toán kết quả...")

        actual_trades = 0
        if 'trades' in res and isinstance(res['trades'], pd.DataFrame) and not res['trades'].empty:
            actual_trades = len(res['trades'])

        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

        st.success(f"✅ Backtest hoàn tất! Đã thực hiện {actual_trades}/{MAX_TRADES} lệnh")
        st.subheader("Kết quả – Metrics")
        st.json(res['summary'])

        # Winrate và Trade Statistics
        if 'trades' in res and isinstance(res['trades'], pd.DataFrame) and not res['trades'].empty:
            _display_trade_statistics(res['trades'], df, symbol, ticker, interval)

        _display_backtest_exports(res, df, sig, strat_name, symbol, ticker, interval, StrategyCls, use_tradingview, sidebar_config)

    except Exception as e:
        st.error(f"Lỗi backtest: {e}")


def _display_trade_statistics(trades: pd.DataFrame, df: pd.DataFrame, symbol: str, ticker: str, interval: str) -> None:
    """Hiển thị thống kê trades."""
    st.markdown("---")
    st.subheader("📊 Thống kê Winrate & Trades")

    # Tính toán stats
    trade_stats = calculate_trade_stats(trades)

    # Hiển thị metrics chính
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Winrate", f"{trade_stats.get('winrate', 0):.2f}%")
    with col2:
        st.metric("Tổng Trades", trade_stats.get('total_trades', 0))
    with col3:
        st.metric(
            "Profit Factor",
            f"{trade_stats.get('profit_factor', 0):.2f}"
            if trade_stats.get('profit_factor', 0) != float('inf')
            else "∞",
        )
    with col4:
        st.metric("Expectancy", f"{trade_stats.get('expectancy', 0):.4f}")

    # Thêm các metrics khác
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Trades Thắng", trade_stats.get('winning_trades', 0))
    with col6:
        st.metric("Trades Thua", trade_stats.get('losing_trades', 0))
    with col7:
        st.metric("Avg Win", f"{trade_stats.get('avg_win', 0):.4f}")
    with col8:
        st.metric("Avg Loss", f"{trade_stats.get('avg_loss', 0):.4f}")

    # Biểu đồ winrate
    try:
        fig_winrate = plot_winrate_metrics(trade_stats, title="Phân bố Win/Loss/Breakeven")
        st.pyplot(fig_winrate)
    except Exception as e:
        st.warning(f"Không vẽ được biểu đồ winrate: {e}")

    # Biểu đồ phân bố PnL
    try:
        fig_pnl_dist = plot_trade_pnl_distribution(
            trades,
            title="Phân bố PnL của Trades",
        )
        st.pyplot(fig_pnl_dist)
    except Exception as e:
        st.warning(f"Không vẽ được biểu đồ phân bố PnL: {e}")

    # Biểu đồ timeline
    try:
        fig_timeline = plot_trade_timeline(
            trades,
            title="Timeline Trades (Win/Loss)",
        )
        st.pyplot(fig_timeline)
    except Exception as e:
        st.warning(f"Không vẽ được biểu đồ timeline: {e}")

    # Biểu đồ cumulative PnL
    try:
        fig_cum = plot_cumulative_pnl(trades, title="Cumulative PnL")
        st.pyplot(fig_cum)
    except Exception as e:
        st.warning(f"Không vẽ được biểu đồ cumulative PnL: {e}")

    # Bảng chi tiết trades với phân loại
    st.markdown("#### 📋 Chi tiết Trades (Win/Loss)")
    trade_summary = get_trade_summary_table(trades)

    # Tabs để xem tất cả, chỉ win, chỉ loss
    tab_all, tab_win, tab_loss = st.tabs(["Tất cả Trades", "Trades Thắng", "Trades Thua"])

    with tab_all:
        if not trade_summary.empty and 'status' in trade_summary.columns:
            def style_row(row):
                status_val = row.get('status', '')
                color = (
                    '#d4edda'
                    if status_val == 'Win'
                    else ('#f8d7da' if status_val == 'Loss' else '#e2e3e5')
                )
                return [f'background-color: {color}'] * len(row)

            st.dataframe(
                trade_summary.style.apply(style_row, axis=1),
                use_container_width=True,
                height=400,
            )
        else:
            st.dataframe(trade_summary, use_container_width=True, height=400)

    with tab_win:
        winning_trades = get_winning_trades(trades)
        if not winning_trades.empty:
            def style_win_row(row):
                return ['background-color: #d4edda'] * len(row)

            st.dataframe(
                winning_trades.style.apply(style_win_row, axis=1),
                use_container_width=True,
                height=400,
            )
            pnl_cols = [
                c
                for c in winning_trades.columns
                if 'pnl' in c.lower() or 'profit' in c.lower()
            ]
            if pnl_cols:
                total_pnl = winning_trades[pnl_cols[0]].sum()
                st.info(
                    f"Tổng cộng {len(winning_trades)} trades thắng, Tổng PnL: {total_pnl:.4f}"
                )
            else:
                st.info(f"Tổng cộng {len(winning_trades)} trades thắng")
        else:
            st.info("Không có trades thắng")

    with tab_loss:
        losing_trades = get_losing_trades(trades)
        if not losing_trades.empty:
            def style_loss_row(row):
                return ['background-color: #f8d7da'] * len(row)

            st.dataframe(
                losing_trades.style.apply(style_loss_row, axis=1),
                use_container_width=True,
                height=400,
            )
            pnl_cols = [
                c
                for c in losing_trades.columns
                if 'pnl' in c.lower() or 'profit' in c.lower()
            ]
            if pnl_cols:
                total_pnl = losing_trades[pnl_cols[0]].sum()
                st.info(
                    f"Tổng cộng {len(losing_trades)} trades thua, Tổng PnL: {total_pnl:.4f}"
                )
            else:
                st.info(f"Tổng cộng {len(losing_trades)} trades thua")
        else:
            st.info("Không có trades thua")

    # Export CSV - Trades (formatted chuẩn)
    symbol_name = symbol if symbol else (ticker if ticker else "UNKNOWN")
    formatted_trades = format_trades_csv(
        trades,
        df,
        symbol=symbol_name,
        timeframe=interval,
    )
    csv_trades_formatted = formatted_trades.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Tải CSV Trades (Định dạng chuẩn)",
        data=csv_trades_formatted,
        file_name=f"trades_{symbol_name}_{interval}.csv",
        mime="text/csv",
    )

    # Export CSV - Trades (có phân loại Win/Loss)
    csv_trades = trade_summary.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Tải Trades CSV (có phân loại Win/Loss)",
        data=csv_trades,
        file_name="trades_with_status.csv",
        mime="text/csv",
    )


def _display_backtest_exports(
    res: Dict[str, Any],
    df: pd.DataFrame,
    sig: pd.Series,
    strat_name: str,
    symbol: str,
    ticker: str,
    interval: str,
    StrategyCls: Any,
    use_tradingview: bool,
    sidebar_config: Dict[str, Any]
) -> None:
    """Hiển thị các phần export và biểu đồ."""
    st.markdown("---")
    st.subheader(" Export dữ liệu Backtest")

    symbol_name = symbol if symbol else (ticker if ticker else "UNKNOWN")

    # Export CSV Trades (formatted) - luôn có nếu có trades
    if 'trades' in res and isinstance(res['trades'], pd.DataFrame) and not res['trades'].empty:
        formatted_trades = format_trades_csv(
            res['trades'],
            df,
            symbol=symbol_name,
            timeframe=interval,
        )
        csv_trades_formatted = formatted_trades.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Tải CSV Trades",
            data=csv_trades_formatted,
            file_name=f"trades_{symbol_name}_{interval}.csv",
            mime="text/csv",
        )

        # Export CSV Backtest đầy đủ
        csv_full = formatted_trades.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Tải CSV Backtest đầy đủ",
            data=csv_full,
            file_name=f"backtest_full_{symbol_name}_{interval}.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "Không có trades để export. Vui lòng chạy backtest với risk management (SL/TP) để có trades."
        )

    # Export CSV Equity Curve
    equity_df = pd.DataFrame({
        'timestamp': res['equity'].index,
        'equity': res['equity'].values,
        'returns': res['returns'].values,
    })
    csv_equity = equity_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Tải CSV Equity Curve",
        data=csv_equity,
        file_name=f"equity_curve_{strat_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    # Export CSV Summary Metrics
    summary_df = pd.DataFrame([res['summary']])
    csv_summary = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Tải CSV Summary Metrics",
        data=csv_summary,
        file_name=f"summary_metrics_{strat_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    # Export PDF Report
    _export_backtest_pdf(res, strat_name, symbol, ticker, interval)

    # Plots
    st.markdown("---")
    st.subheader("Biểu đồ")
    overlays = {
        k: df[k]
        for k in ["SMA20", "EMA20", "BB_MID", "BB_UPPER", "BB_LOWER", "VWAP"]
        if k in df.columns
    }

    # TradingView Chart
    if HAS_TRADINGVIEW and use_tradingview:
        try:
            tv_data = prepare_tradingview_data(df, signals=sig, indicators=overlays)
            symbol_name = symbol if symbol else (ticker if ticker else "BTCUSDT")
            tv_html = create_tradingview_html(
                tv_data,
                symbol=symbol_name,
                interval=interval,
                theme="dark",
                height=600,
            )
            st.markdown("### 📊 TradingView Chart")
            st.components.v1.html(tv_html, height=620, scrolling=False)

            # Export Pine Script option
            with st.expander("📝 Export Pine Script cho TradingView"):
                strategy_display_name = (
                    [k for k, v in STRATEGY_MAP.items() if v[1] == StrategyCls][0]
                    if StrategyCls
                    else "Custom Strategy"
                )
                pinescript = create_tradingview_pinescript(
                    strategy_name=strategy_display_name,
                    indicators=list(overlays.keys()),
                    buy_condition="ta.crossover(close, sma20)",  # Example
                    sell_condition="ta.crossunder(close, sma20)",  # Example
                )
                st.code(pinescript, language="javascript")
                st.download_button(
                    "Tải Pine Script",
                    data=pinescript,
                    file_name=f"{strategy_display_name.lower().replace(' ', '_')}.pine",
                    mime="text/plain",
                )
        except Exception as e:
            st.error(f"Lỗi khi hiển thị TradingView chart: {str(e)}")
            st.info("Đang fallback sang matplotlib...")
            logger.exception("Lỗi khi hiển thị TradingView chart")
            use_tradingview = False

    # Matplotlib fallback
    if not (HAS_TRADINGVIEW and use_tradingview):
        try:
            fig1 = plot_candlestick(
                df,
                overlays=overlays,
                signals=sig,
                title="Giá & Tín hiệu",
                use_plotly=False,
            )
            st.pyplot(fig1)
        except Exception as e:
            st.warning(f"Không vẽ được candlestick: {e}")
    try:
        fig2 = plot_equity_curve(res['equity'], title='Equity Curve')
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Không vẽ được equity: {e}")
    try:
        fig3 = plot_drawdown(res['equity'], title='Drawdown')
        st.pyplot(fig3)
    except Exception as e:
        st.warning(f"Không vẽ được drawdown: {e}")


def _export_backtest_pdf(
    res: Dict[str, Any],
    strat_name: str,
    symbol: str,
    ticker: str,
    interval: str
) -> None:
    """Export PDF cho backtest."""
    st.markdown("---")
    st.subheader("📄 Export PDF")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Table,
            TableStyle,
            Paragraph,
            Spacer,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from io import BytesIO

        if st.button("📄 Tạo báo cáo PDF", key="btn_pdf"):
            with st.spinner("🔄 Đang tạo PDF..."):
                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
                story = []
                styles = getSampleStyleSheet()

                # Title
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=18,
                    textColor=colors.HexColor('#1f77b4'),
                    spaceAfter=20,
                    alignment=TA_CENTER,
                )
                story.append(Paragraph(f"BÁO CÁO BACKTEST: {strat_name}", title_style))
                story.append(Spacer(1, 0.2 * inch))

                # Info
                symbol_name = symbol if symbol else (ticker if ticker else "N/A")
                info_text = f"""
                <b>Ngày tạo:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
                <b>Symbol:</b> {symbol_name}<br/>
                <b>Interval:</b> {interval}<br/>
                <b>Strategy:</b> {strat_name}
                """
                story.append(Paragraph(info_text, styles['Normal']))
                story.append(Spacer(1, 0.3 * inch))

                # Metrics table
                metrics_data = [['Metric', 'Value']]
                for key, value in res['summary'].items():
                    if isinstance(value, float):
                        if 'Return' in key or 'CAGR' in key or 'Drawdown' in key:
                            metrics_data.append([key, f"{value*100:.2f}%"])
                        else:
                            metrics_data.append([key, f"{value:.4f}"])
                    else:
                        metrics_data.append([key, str(value)])

                table = Table(metrics_data, colWidths=[2.5 * inch, 2 * inch])
                table.setStyle(
                    TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ])
                )
                story.append(table)

                # Build PDF
                doc.build(story)
                pdf_buffer.seek(0)

                st.download_button(
                    "📄 Tải báo cáo PDF",
                    data=pdf_buffer.getvalue(),
                    file_name=f"backtest_report_{strat_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )
    except ImportError:
        st.info("💡 Cài đặt reportlab để export PDF: `pip install reportlab`")
    except Exception as e:
        st.warning(f"⚠️ Lỗi tạo PDF: {e}")


def _render_session_analysis(sidebar_config: Dict[str, Any]) -> None:
    """Render phần phân tích phiên/giờ."""
    try:
        analysis_mode = sidebar_config['analysis_mode']
        
        if analysis_mode == "None":
            st.info("Hãy chọn chế độ phân tích (Session/Hour) trong sidebar trước.")
            return

        if 'df' not in st.session_state:
            st.session_state['df'] = _load_df_from_config(sidebar_config)
        df = st.session_state['df']
        
        if df.empty:
            st.error("DataFrame rỗng")
            return

        st.subheader("Phân tích theo phiên/giờ")
        tz = "UTC"  # Có thể thêm lựa chọn timezone nếu cần

        if analysis_mode in ("Session only", "Session + Hour"):
            st.markdown("#### Thống kê theo phiên (Asia / Europe / US / Other)")
            sess_stats = session_return_stats(df, price_col="close", tz=tz)
            st.dataframe(sess_stats.style.format("{:.6f}"))

        if analysis_mode in ("Hour only", "Session + Hour"):
            st.markdown("#### Thống kê theo giờ trong ngày (0–23)")
            hour_stats = hour_of_day_return_stats(df, price_col="close", tz=tz)
            st.dataframe(hour_stats.style.format("{:.6f}"))
    except Exception as e:
        st.error(f"Lỗi phân tích phiên/giờ: {e}")

