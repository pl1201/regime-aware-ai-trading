from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import pathlib
import numpy as np
import pandas as pd
import streamlit as st

from algo_trading.ui.utils import load_df_from_sidebar_config
from algo_trading.backtest.vectorized import run_backtest, BacktestConfig, RiskConfig
from algo_trading.core.backtest_event import run_event_backtest, EventConfig
from algo_trading.strategies.ml import RegimeEnsembleStrategy, RegimeEnsembleBanditStrategy
from algo_trading.core.risk_exit_engine import RiskExitEngineConfig, compute_trend_consensus
from algo_trading.market_models import detect_regime_hmm
from algo_trading.visualization.plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_trade_pnl_distribution,
    plot_trade_timeline,
    plot_cumulative_pnl,
    plot_winrate_metrics,
)
from algo_trading.utils.trade_stats import calculate_trade_stats
from algo_trading.utils.trade_formatter import format_trades_csv


def render_regime_transformer_tab(sidebar_config: Dict[str, Any]) -> None:
    """Main entry for the Regime Transformer tab."""
    st.header("🎯 Regime Transformer – ML Training & Backtest")
    st.caption(
        "Tab chuyên cho Regime Ensemble / Regime Transformer: "
        "train models với script tối ưu, rồi backtest với event-driven Risk/Exit Engine."
    )

    tab_train, tab_bt = st.tabs(["🎓 Training (Optimized)", "📊 Backtest (ML + Risk/Exit)"])
    
    with tab_train:
        _render_training_tab(sidebar_config)
    
    with tab_bt:
        _render_backtest_tab(sidebar_config)


# ============================================================
# TRAINING
# ============================================================


def _render_training_tab(sidebar_config: Dict[str, Any]) -> None:
    st.subheader("🎓 Training với script `train_regime_ensemble_optimized.py`")
    st.info(
        "Phần này wrap lại script training đã có (multi-timeframe, ICT, regime-specific...).\n"
        "Bạn có thể chỉnh một số tham số cao cấp, còn chi tiết sâu giữ nguyên trong script."
    )

    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input(
            "Symbol (Binance)",
            value=(sidebar_config.get("symbol") or "BTCUSDT"),
            key="rt_train_symbol",
        ).upper()
        timeframe = st.text_input(
            "Timeframe",
            value=(sidebar_config.get("interval") or "1h"),
            key="rt_train_timeframe",
        )
        test_size = st.slider("Test size (%)", 10, 40, 20, 5, key="rt_train_test_size") / 100

    with col2:
        use_multi_timeframe = st.checkbox(
            "Use multi-timeframe features (4h, 1d)", value=True, key="rt_train_use_mtf"
        )
        use_regime_specific = st.checkbox(
            "Train regime-specific models", value=True, key="rt_train_use_regime_specific"
        )
        use_ict = st.checkbox("Use ICT features", value=True, key="rt_train_use_ict")
        optimize_hyperparams = st.checkbox(
            "Optuna hyperparameter optimization (tất cả models)", value=True, key="rt_train_opt_hp"
        )
        n_trials = st.number_input(
            "Optuna trials", min_value=10, max_value=300, value=80, step=10, key="rt_train_n_trials"
        )

    train_stacking_only = st.checkbox(
        "Chỉ train Stacking Ensemble (dùng features/models đã có)",
            value=False,
        key="rt_train_stacking_only",
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Train & Save ML Models", type="primary", key="rt_train_btn"):
            with st.spinner("🔄 Training... (có thể mất khá lâu)"):
                try:
                    from train_regime_ensemble_optimized import train_optimized_models

                    results = train_optimized_models(
                        symbol=symbol,
                        timeframe=timeframe,
                        test_size=test_size,
                        optimize_hyperparams=optimize_hyperparams,
                        use_multi_timeframe=use_multi_timeframe,
                        use_regime_specific=use_regime_specific,
                        use_ict=use_ict,
                        n_trials=int(n_trials),
                        train_stacking_only=train_stacking_only,
                    )
                    st.success("✅ Training hoàn tất. Models đã lưu trong thư mục `models/`.")
                    st.session_state["rt_last_training_result"] = results

                    models_dir = pathlib.Path("models")
                    if models_dir.exists():
                        pkl_files = sorted(p.name for p in models_dir.glob("*.pkl"))
                        if pkl_files:
                            st.markdown("#### 📦 Models hiện có trong `models/`")
                            st.code("\n".join(pkl_files))
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
                    import traceback as _tb
                    st.code(_tb.format_exc())

    with col_btn2:
        if st.button("🔥 Train FULL pipeline (5m + LSTM seq + RegimeSpecific)", type="secondary", key="rt_train_full_btn"):
            with st.spinner("🔄 Training FULL pipeline 5m... (LSTM extractor + optimized models)"):
                try:
                    from train_regime_ensemble_optimized import train_full_pipeline

                    full_results = train_full_pipeline(
                        symbol=symbol,
                        timeframe="5m",  # cố định 5m cho pipeline này
                        test_size=test_size,
                        optimize_hyperparams=optimize_hyperparams,
                        use_multi_timeframe=use_multi_timeframe,
                        use_regime_specific=use_regime_specific,
                        use_ict=use_ict,
                        n_trials=int(n_trials),
                        train_stacking_only=train_stacking_only,
                        lstm_epochs=20,
                        lstm_seq_len=64,
                        lstm_horizon=5,
                        lstm_device="cpu",
                    )
                    st.success("✅ FULL pipeline training hoàn tất (LSTM extractor + models).")
                    st.session_state["rt_last_training_full"] = full_results

                    models_dir = pathlib.Path("models")
                    if models_dir.exists():
                        pkl_files = sorted(p.name for p in models_dir.glob("*.pkl"))
                        if pkl_files:
                            st.markdown("#### 📦 Models hiện có trong `models/`")
                            st.code("\n".join(pkl_files))
                        st.markdown("#### 📦 Seq extractor")
                        st.code(full_results.get("extractor_path", "models/seq_lstm_extractor.pt"))
                except Exception as e:
                    st.error(f"❌ FULL pipeline training failed: {e}")
                    import traceback as _tb
                    st.code(_tb.format_exc())


# ============================================================
# BACKTEST
# ============================================================


def _load_rt_df(sidebar_config: Dict[str, Any]) -> pd.DataFrame:
    return load_df_from_sidebar_config(
        source=sidebar_config["source"],
        ticker=sidebar_config.get("ticker"),
        symbol=sidebar_config.get("symbol"),
        interval=sidebar_config.get("interval"),
        start=sidebar_config.get("start"),
        end=sidebar_config.get("end"),
        market=sidebar_config.get("market", "spot"),
        path=sidebar_config.get("path"),
    )


def _render_backtest_tab(sidebar_config: Dict[str, Any]) -> None:
    st.subheader("📊 Backtest Regime ML + Risk/Exit")

    # ---------------- 0) DATA ----------------
    st.markdown("### 0) Dữ liệu")
    col0a, col0b = st.columns([1, 2])
    with col0a:
        if st.button("🔄 Load Data", key="rt_bt_load_data"):
            with st.spinner("Đang tải dữ liệu..."):
                df = _load_rt_df(sidebar_config)
                st.session_state["rt_df"] = df

    df: Optional[pd.DataFrame] = st.session_state.get("rt_df")
    with col0b:
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.success(f"✅ Loaded {len(df)} bars | {df.index[0]} → {df.index[-1]}")
        else:
            st.info("Chưa có dữ liệu. Hãy bấm 'Load Data'.")

    if not isinstance(df, pd.DataFrame) or df.empty:
        st.stop()

    with st.expander("Xem trước dữ liệu"):
        st.dataframe(df.head(30), use_container_width=True)

    # ---------------- 1) CURRENT REGIME (HMM SNAPSHOT) ----------------
    st.markdown("### 1) Current Regime – HMM snapshot (theo tài liệu)")
    col_reg1, col_reg2 = st.columns([1, 2])

    with col_reg1:
        if st.button("🔍 Cập nhật current regime (HMM)", key="rt_bt_update_regime"):
            try:
                # Sử dụng đúng hàm detect_regime_hmm mô tả trong PHUONG_PHAP_LUONG_HOA.md
                reg_info = detect_regime_hmm(df)
            except Exception as e:
                reg_info = {"error": str(e)}
            st.session_state["rt_regime_info"] = reg_info

    reg_info = st.session_state.get("rt_regime_info")
    with col_reg2:
        if isinstance(reg_info, dict) and reg_info.get("current_regime") is not None:
            st.success(f"Current regime (HMM): {reg_info.get('current_regime')}")

            probs = reg_info.get("regime_probabilities")
            if isinstance(probs, pd.DataFrame) and not probs.empty:
                # Lấy hàng cuối cùng → Series (probabilities cho từng regime)
                last_probs = probs.iloc[-1]
                # Chuyển thành DataFrame chuẩn cho Altair: index = regime, cột = probability
                probs_df = last_probs.to_frame(name="probability")
                probs_df.index.name = "regime"
                st.caption("Xác suất regime tại bar mới nhất (HMM):")
                st.bar_chart(probs_df)
        else:
            st.caption("Chưa có snapshot HMM. Bấm nút bên trái để tính current regime đúng theo HMM.")

    # ---------------- 2) MODEL / STRATEGY ----------------
    st.markdown("### 2) Chọn model/strategy ML")
    strategy_mode = st.radio(
        "Chế độ",
        ["Regime Ensemble (ML)", "Regime Ensemble (Bandit)", "Regime-Specific Models"],
        index=0,
        horizontal=True,
        key="rt_bt_strategy_mode",
    )

    proba_threshold = st.slider(
        "Proba threshold",
        0.30,
        0.90,
        0.45,
        0.01,
        key="rt_bt_proba_threshold",
    )

    allowed_regimes = st.multiselect(
        "Allowed regimes",
        ["trending", "ranging", "volatile", "calm"],
        default=["trending", "ranging", "calm"],
        key="rt_bt_allowed_regimes",
    )

    use_seq_features = st.checkbox(
        "Use sequence features (LSTM extractor seq_score)",
        value=True,
        help="Dùng LSTM extractor để tạo seq_score/seq_vol/seq_trend đưa vào model cuối (LGBM/XGB).",
        key="rt_bt_use_seq_features",
    )

    use_dynamic_threshold = st.checkbox(
        "Use Dynamic Threshold (Volatility-based)",
        value=True,
        help="Giảm ngưỡng Entry khi volatility cao (BB Width), giúp bắt trend tốt hơn.",
        key="rt_bt_use_dynamic_threshold",
    )

    # ICT filter
    st.markdown("#### ICT Filter (optional)")
    col_ict1, col_ict2, col_ict3 = st.columns(3)
    with col_ict1:
        use_ict_filter = st.checkbox("Use ICT filter", value=False, key="rt_bt_use_ict_filter")
    with col_ict2:
        ict_ob_tolerance_pct = (
            st.number_input(
                "OB tolerance (%)",
                min_value=0.01,
                max_value=1.0,
                value=0.20,
                step=0.01,
                key="rt_bt_ict_ob_tol",
            )
            / 100.0
        )
    with col_ict3:
        ict_fib_max_dist = (
            st.number_input(
                "Fib max dist (%)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                key="rt_bt_ict_fib_dist",
            )
            / 100.0
        )

    # Discover models in models/ folder
    models_dir = pathlib.Path("models")
    ensemble_models = []
    bandit_models_rf = []
    bandit_models_gb = []
    bandit_models_logit = []
    regime_specific_models = []
    if models_dir.exists():
        for p in models_dir.glob("*.pkl"):
            name = p.name
            if "regime_ensemble" in name and "bandit" not in name:
                ensemble_models.append(name)
            if "regime_bandit_rf" in name:
                bandit_models_rf.append(name)
            if "regime_bandit_gb" in name:
                bandit_models_gb.append(name)
            if "regime_bandit_logit" in name:
                bandit_models_logit.append(name)
            if "regime_specific_models" in name:
                regime_specific_models.append(name)

    model_path: Optional[str] = None
    regime_specific_model_path: Optional[str] = None
    bandit_model_paths: Dict[str, str] = {}

    if strategy_mode == "Regime Ensemble (ML)":
        st.markdown("**Ensemble model**")
        if ensemble_models:
            selected_name = st.selectbox(
                "Chọn ensemble model trong models/",
                ensemble_models,
                index=0,
                key="rt_bt_ensemble_model_sel",
            )
            model_path = str(models_dir / selected_name)
        else:
            model_path = st.text_input(
                "Model path (không tìm thấy file trong models/, nhập tay):",
                value="models/regime_ensemble_optimized.pkl",
                key="rt_bt_model_path",
            )
    elif strategy_mode == "Regime-Specific Models":
        st.markdown("**Regime-specific models**")
        if regime_specific_models:
            selected_name = st.selectbox(
                "Chọn regime-specific model",
                regime_specific_models,
                index=0,
                key="rt_bt_regime_specific_sel",
            )
            regime_specific_model_path = str(models_dir / selected_name)
        else:
            regime_specific_model_path = st.text_input(
                "Regime-specific model path:",
                value="models/regime_specific_models_optimized.pkl",
                key="rt_bt_regime_specific_path",
            )
    elif strategy_mode == "Regime Ensemble (Bandit)":
        st.markdown("**Bandit arms (multi-model)**")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            rf_name = (
                st.selectbox(
                    "RF bandit model",
                    ["<none>"] + bandit_models_rf,
                    index=1 if bandit_models_rf else 0,
                    key="rt_bt_bandit_rf_sel",
                )
                if bandit_models_rf
                else "<none>"
            )
        with col_b2:
            gb_name = (
                st.selectbox(
                    "GB bandit model",
                    ["<none>"] + bandit_models_gb,
                    index=1 if bandit_models_gb else 0,
                    key="rt_bt_bandit_gb_sel",
                )
                if bandit_models_gb
                else "<none>"
            )
        with col_b3:
            logit_name = (
                st.selectbox(
                    "Logit bandit model",
                    ["<none>"] + bandit_models_logit,
                    index=1 if bandit_models_logit else 0,
                    key="rt_bt_bandit_logit_sel",
                )
                if bandit_models_logit
                else "<none>"
            )
        if rf_name and rf_name != "<none>":
            bandit_model_paths["rf"] = str(models_dir / rf_name)
        if gb_name and gb_name != "<none>":
            bandit_model_paths["gb"] = str(models_dir / gb_name)
        if logit_name and logit_name != "<none>":
            bandit_model_paths["logit"] = str(models_dir / logit_name)

    # ---------------- 3) GENERATE SIGNALS ----------------
    st.markdown("### 3) Generate Signals (từ ML model)")
    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        if st.button("📡 Generate Signals", key="rt_bt_generate"):
            try:
                if strategy_mode == "Regime Ensemble (Bandit)":
                    if not bandit_model_paths:
                        raise ValueError("Chưa chọn bandit models.")
                    strat = RegimeEnsembleBanditStrategy(
                        model_paths=bandit_model_paths,
                        proba_threshold=proba_threshold,
                        allowed_regimes=allowed_regimes,
                        bandit_type="ucb",
                        epsilon=0.1,
                        reward_mode="direction",
                    )
                elif strategy_mode == "Regime-Specific Models":
                    if not regime_specific_model_path:
                        raise ValueError("Thiếu regime-specific model path.")
                    strat = RegimeEnsembleStrategy(
                        model_path=None,
                        use_regime_specific=True,
                        regime_specific_model_path=regime_specific_model_path,
                        proba_threshold=proba_threshold,
                        allowed_regimes=allowed_regimes,
                        use_direction_output=False,
                        use_ict_filter=use_ict_filter,
                        ict_ob_tolerance_pct=ict_ob_tolerance_pct,
                        ict_fib_max_dist=ict_fib_max_dist,
                        use_sequence_features=use_seq_features,
                        sequence_model_path="models/seq_lstm_extractor.pt",
                        sequence_len=64,
                        use_dynamic_threshold=use_dynamic_threshold,
                    )
                else:
                    if not model_path:
                        raise ValueError("Thiếu ensemble model path.")
                    strat = RegimeEnsembleStrategy(
                        model_path=model_path,
                        proba_threshold=proba_threshold,
                        allowed_regimes=allowed_regimes,
                        use_direction_output=False,
                        use_ict_filter=use_ict_filter,
                        ict_ob_tolerance_pct=ict_ob_tolerance_pct,
                        ict_fib_max_dist=ict_fib_max_dist,
                        use_sequence_features=use_seq_features,
                        sequence_model_path="models/seq_lstm_extractor.pt",
                        sequence_len=64,
                        use_dynamic_threshold=use_dynamic_threshold,
                    )

                result = strat.generate_signals(df)
                st.session_state["rt_signals"] = result.signals
                st.session_state["rt_signals_meta"] = result.meta
                st.session_state["rt_strategy_snapshot"] = {
                    "strategy_mode": strategy_mode,
                    "model_path": model_path,
                    "regime_specific_model_path": regime_specific_model_path,
                    "bandit_model_paths": bandit_model_paths,
                    "proba_threshold": proba_threshold,
                    "allowed_regimes": allowed_regimes,
                    "use_ict_filter": use_ict_filter,
                    "ict_ob_tolerance_pct": ict_ob_tolerance_pct,
                    "ict_fib_max_dist": ict_fib_max_dist,
                    "ict_fib_max_dist": ict_fib_max_dist,
                    "use_sequence_features": use_seq_features,
                    "use_dynamic_threshold": use_dynamic_threshold,
                }
                sig = result.signals
                st.success(f"✅ Generated {int((sig != 0).sum())} non-zero signals.")
            except Exception as e:
                st.error(f"❌ Generate signals failed: {e}")
                import traceback

                st.code(traceback.format_exc())

    signals: Optional[pd.Series] = st.session_state.get("rt_signals")
    meta: Dict[str, Any] = st.session_state.get("rt_signals_meta", {}) or {}
    reg_info = st.session_state.get("rt_regime_info") or {}
    canonical_regime = reg_info.get("current_regime") or meta.get("current_regime", "N/A")
    with col_g2:
        if isinstance(signals, pd.Series) and not signals.empty:
            non_zero = int((signals != 0).sum())
            st.info(
                f"Signals ready: {non_zero} / {len(signals)} non-zero | "
                f"current_regime={canonical_regime}"
            )
            # Nếu không có lệnh nào, show thêm meta debug để dễ hiểu lý do
            if non_zero == 0:
                reason = meta.get("reason") or meta.get("error")
                if reason:
                    st.warning(f"Lý do không có lệnh: {reason}")
                # Nếu có thêm debug trong meta (ví dụ từ RegimeEnsembleStrategy)
                debug_keys = [
                    "max_p_long",
                    "max_p_short",
                    "max_p_neutral",
                    "mean_p_long",
                    "mean_p_short",
                    "mean_p_neutral",
                    "samples_above_threshold_long",
                    "samples_above_threshold_short",
                ]
                debug_info = {k: meta.get(k) for k in debug_keys if k in meta}
                if debug_info:
                    with st.expander("Chi tiết debug signals (proba, threshold...)"):
                        st.json(debug_info)
        else:
            st.caption("Chưa có signals. Hãy bấm 'Generate Signals'.")

    # ---------------- 4) RISK / EXIT CONFIG ----------------
    st.markdown("### 4) Risk / Exit Config")

    bt_mode = st.selectbox(
        "Backtest mode", ["vectorized", "event-driven"], index=1, key="rt_bt_mode"
    )

    col_r0, col_r1, col_r2, col_r3 = st.columns(4)
    with col_r0:
        leverage = st.number_input("Leverage", 1.0, 5.0, 1.0, 0.5, key="rt_bt_leverage")
    with col_r1:
        commission = (
            st.number_input("Commission (%)", 0.0, 1.0, 0.1, 0.01, key="rt_bt_commission") / 100.0
        )
    with col_r2:
        slippage_bps = st.number_input("Slippage (bps)", 0.0, 20.0, 1.0, 0.5, key="rt_bt_slippage_bps")
    with col_r3:
        max_trades = st.number_input("Max trades", 10, 5000, 1000, 50, key="rt_bt_max_trades")

    use_next_open = st.checkbox(
        "Use next open for entry/exit", value=False, key="rt_bt_use_next_open"
    )

    st.markdown("#### Legacy SL/TP (optional)")
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        # Mặc định tắt SL/TP % để ưu tiên regime-based ATR từ Advanced Risk/Exit Engine
        sl_pct = st.number_input("Stop Loss (%)", 0.0, 10.0, 0.0, 0.1, key="rt_bt_sl_pct") / 100.0
    with col_l2:
        tp_pct = st.number_input("Take Profit (%)", 0.0, 20.0, 0.0, 0.1, key="rt_bt_tp_pct") / 100.0
    with col_l3:
        trailing_pct = (
            st.number_input("Trailing (%)", 0.0, 10.0, 0.0, 0.1, key="rt_bt_trailing_pct") / 100.0
        )

    st.markdown("#### Advanced Risk/Exit Engine (event-driven only)")
    enable_adv = st.checkbox(
        "Enable Advanced Risk/Exit Engine",
        value=False,
        key="rt_bt_enable_adv",
        disabled=(bt_mode != "event-driven"),
    )

    adv_cfg: Optional[RiskExitEngineConfig] = None
    if enable_adv and bt_mode == "event-driven":
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            trailing_atr_k = st.number_input("Trailing ATR k", 0.0, 10.0, 2.0, 0.1, key="rt_bt_trailing_atr_k")
        with col_b:
            be_trigger = st.number_input("Breakeven trigger (ATR)", 0.0, 10.0, 1.0, 0.1, key="rt_bt_be_trigger")
        with col_c:
            be_buffer = st.number_input("Breakeven buffer (ATR)", 0.0, 5.0, 0.0, 0.1, key="rt_bt_be_buffer")

        col_d, col_e, col_f = st.columns(3)
        with col_d:
            exit_on_regime_change = st.checkbox(
                "Exit on regime change", True, key="rt_bt_exit_regime_change"
            )
        with col_e:
            exit_on_trend_consensus = st.checkbox(
                "Exit on trend consensus", False, key="rt_bt_exit_trend"
            )
        with col_f:
            tc_min_long = st.number_input("TC min (LONG)", 0.0, 1.0, 0.55, 0.01, key="rt_bt_tc_min_long")
            tc_max_short = st.number_input("TC max (SHORT)", 0.0, 1.0, 0.45, 0.01, key="rt_bt_tc_max_short")

        adv_cfg = RiskExitEngineConfig(
            trailing_atr_k=trailing_atr_k if trailing_atr_k > 0 else None,
            breakeven_trigger_atr=be_trigger if be_trigger > 0 else None,
            breakeven_buffer_atr=be_buffer,
            exit_on_regime_change=exit_on_regime_change,
            exit_on_trend_consensus=exit_on_trend_consensus,
            trend_consensus_min_long=tc_min_long,
            trend_consensus_max_short=tc_max_short,
        )

    # ---------------- 4) RUN BACKTEST ----------------
    st.markdown("### 4) Run Backtest")
    can_run = isinstance(signals, pd.Series) and not signals.empty
    if st.button("🚀 Run Backtest", type="primary", key="rt_bt_run", disabled=not can_run):
        with st.spinner("Đang chạy backtest..."):
            try:
                freq = "1H" if "h" in str(sidebar_config.get("interval", "1h")).lower() else "1D"

                # Legacy RiskConfig
                risk = None
                if (sl_pct and sl_pct > 0) or (tp_pct and tp_pct > 0) or (trailing_pct and trailing_pct > 0):
                    risk = RiskConfig(
                        sl_pct=sl_pct if sl_pct > 0 else None,
                        tp_pct=tp_pct if tp_pct > 0 else None,
                        trailing_pct=trailing_pct if trailing_pct > 0 else None,
                        atr_col="ATR14",
                    )

                if bt_mode == "vectorized":
                    cfg = BacktestConfig(
                        initial_capital=10000.0,
                        leverage=leverage,
                        allow_short=True,
                        commission=commission,
                        slippage_bps=slippage_bps,
                        use_next_open=use_next_open,
                        freq=freq,
                    )
                    res = run_backtest(df, signals, cfg=cfg, risk=risk, max_trades=int(max_trades))
                else:
                    cfg = EventConfig(
                        initial_cash=10000.0,
                        leverage=leverage,
                        allow_short=True,
                        commission=commission,
                        slippage_bps=slippage_bps,
                        use_next_open=use_next_open,
                        price_col="close",
                        open_col="open",
                        high_col="high",
                        low_col="low",
                        freq=freq,
                    )

                    regime_series = None
                    trend_cons = None
                    if adv_cfg is not None:
                        # Build regime series via RegimeEnsembleStrategy snapshot (best-effort)
                        snap = st.session_state.get("rt_strategy_snapshot", {}) or {}
                        try:
                            if snap.get("strategy_mode") == "Regime Ensemble (Bandit)":
                                tmp = RegimeEnsembleBanditStrategy(
                                    model_paths=snap.get("bandit_model_paths", {}),
                                    proba_threshold=snap.get("proba_threshold", 0.55),
                                    allowed_regimes=snap.get("allowed_regimes", ["trending", "ranging", "calm"]),
                                    bandit_type="ucb",
                                    epsilon=0.1,
                                    reward_mode="direction",
                                )
                            elif snap.get("strategy_mode") == "Regime-Specific Models":
                                tmp = RegimeEnsembleStrategy(
                                    model_path=None,
                                    use_regime_specific=True,
                                    regime_specific_model_path=snap.get("regime_specific_model_path"),
                                    proba_threshold=snap.get("proba_threshold", 0.55),
                                    allowed_regimes=snap.get("allowed_regimes", ["trending", "ranging", "calm"]),
                                    use_direction_output=False,
                                    use_ict_filter=snap.get("use_ict_filter", False),
                                    ict_ob_tolerance_pct=snap.get("ict_ob_tolerance_pct", 0.002),
                                    ict_fib_max_dist=snap.get("ict_fib_max_dist", 0.02),
                                    use_dynamic_threshold=snap.get("use_dynamic_threshold", False),
                                )
                            else:
                                tmp = RegimeEnsembleStrategy(
                                    model_path=snap.get("model_path"),
                                    proba_threshold=snap.get("proba_threshold", 0.55),
                                    allowed_regimes=snap.get("allowed_regimes", ["trending", "ranging", "calm"]),
                                    use_direction_output=False,
                                    use_ict_filter=snap.get("use_ict_filter", False),
                                    ict_ob_tolerance_pct=snap.get("ict_ob_tolerance_pct", 0.002),
                                    ict_fib_max_dist=snap.get("ict_fib_max_dist", 0.02),
                                    use_dynamic_threshold=snap.get("use_dynamic_threshold", False),
                                )
                            inds = tmp._calculate_indicators(df)  # type: ignore[attr-defined]
                            reg_info = tmp._detect_regime(df, inds)  # type: ignore[attr-defined]
                            regime_series = reg_info.get("regime", None)
                            if isinstance(regime_series, pd.Series):
                                regime_series = regime_series.reindex(df.index, method="ffill").fillna("trending")
                            else:
                                regime_series = pd.Series("trending", index=df.index)
                        except Exception:
                            regime_series = pd.Series("trending", index=df.index)

                        try:
                            trend_cons = compute_trend_consensus(df).reindex(df.index, method="ffill").fillna(0.5)
                        except Exception:
                            trend_cons = pd.Series(0.5, index=df.index)

                    res = run_event_backtest(
                        df,
                        signals,
                        cfg=cfg,
                        risk=None if adv_cfg is not None else risk,
                        risk_exit=adv_cfg,
                        regime_series=regime_series,
                        trend_consensus=trend_cons,
                        max_trades=int(max_trades),
                    )

                st.session_state["rt_bt_last_res"] = res
                st.session_state["rt_bt_last_meta"] = meta
                
                # Lưu kết quả vào file để Telegram bot có thể đọc
                try:
                    import json
                    from pathlib import Path
                    from algo_trading.utils.trade_stats import calculate_trade_stats
                    
                    # Chuẩn bị dữ liệu để lưu
                    summary = res.get("summary", {}) or {}
                    
                    # Tính stats từ trades nếu không có
                    stats = res.get("stats", {}) or {}
                    trades = res.get("trades")
                    if (not stats or not stats.get("winrate")) and isinstance(trades, pd.DataFrame) and not trades.empty:
                        stats = calculate_trade_stats(trades)
                    
                    backtest_results = {
                        "total_return": float(summary.get("TotalReturn", 0)) * 100,
                        "sharpe": float(summary.get("Sharpe", 0)),
                        "max_drawdown": float(summary.get("MaxDrawdown", 0)) * 100,
                        "cagr": float(summary.get("CAGR", 0)) * 100,
                        "profit_factor": float(summary.get("ProfitFactor", 0)) if summary.get("ProfitFactor") is not None else 0.0,
                        "winrate": float(stats.get("winrate", 0)) if stats.get("winrate") is not None else 0.0,
                        "total_trades": int(stats.get("total_trades", 0)),
                        "winning_trades": int(stats.get("winning_trades", 0)),
                        "losing_trades": int(stats.get("losing_trades", 0)),
                        "expectancy": float(stats.get("expectancy", 0)) if stats.get("expectancy") is not None else 0.0,
                        "avg_win": float(stats.get("avg_win", 0)) if stats.get("avg_win") is not None else 0.0,
                        "avg_loss": float(stats.get("avg_loss", 0)) if stats.get("avg_loss") is not None else 0.0,
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "meta": meta,
                    }
                    
                    # Lưu vào file
                    output_file = pathlib.Path("backtest_results.json")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(backtest_results, f, indent=2, ensure_ascii=False)
                    
                    st.info(f"💾 Đã lưu kết quả vào {output_file}")
                except Exception as save_err:
                    st.warning(f"⚠️ Không thể lưu kết quả vào file: {save_err}")
                    import traceback
                    st.code(traceback.format_exc())
                
                st.success("✅ Backtest completed.")
            except Exception as e:
                st.error(f"❌ Backtest failed: {e}")
                import traceback

                st.code(traceback.format_exc())

    # ---------------- 5) DISPLAY RESULTS ----------------
    res = st.session_state.get("rt_bt_last_res")
    if isinstance(res, dict) and "summary" in res:
        _display_results(res, df, signals, sidebar_config, st.session_state.get("rt_bt_last_meta", meta))


def _display_results(
    res: Dict[str, Any],
    df: pd.DataFrame,
    signals: Optional[pd.Series],
    sidebar_config: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    st.markdown("---")
    st.subheader("📈 Kết quả Backtest")

    summary = res.get("summary", {}) or {}
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Return", f"{float(summary.get('TotalReturn', 0))*100:.2f}%")
    with col2:
        st.metric("Sharpe", f"{float(summary.get('Sharpe', 0)):.3f}")
    with col3:
        st.metric("Max DD", f"{float(summary.get('MaxDrawdown', 0))*100:.2f}%")
    with col4:
        st.metric("CAGR", f"{float(summary.get('CAGR', 0))*100:.2f}%")
    with col5:
        pf = summary.get("ProfitFactor", None)
        if pf is not None:
            st.metric("Profit Factor", f"{float(pf):.3f}")
    
    if meta:
        with st.expander("Strategy meta"):
            st.json(meta)

    equity = res.get("equity")
    trades = res.get("trades")

    if isinstance(equity, pd.Series) and not equity.empty:
        try:
            st.pyplot(plot_equity_curve(equity, title="Equity Curve"))
        except Exception:
            pass
        try:
            st.pyplot(plot_drawdown(equity, title="Drawdown"))
        except Exception:
            pass

    if isinstance(trades, pd.DataFrame) and not trades.empty:
        # === Bảng thống kê Winrate chi tiết (format giống bản cũ) ===
        stats = calculate_trade_stats(trades)
        st.markdown("### 📊 Bảng Thống Kê Winrate Chi Tiết")

        # Hàng 1: Winrate, Total Trades, Winning Trades, Losing Trades, Profit Factor, Expectancy
        r1c1, r1c2, r1c3, r1c4, r1c5, r1c6 = st.columns(6)
        with r1c1:
            st.metric("Winrate", f"{stats.get('winrate', 0):.2f}%")
        with r1c2:
            st.metric("Total Trades", int(stats.get("total_trades", 0)))
        with r1c3:
            st.metric("Winning Trades", int(stats.get("winning_trades", 0)))
        with r1c4:
            st.metric("Losing Trades", int(stats.get("losing_trades", 0)))
        with r1c5:
            pf2 = stats.get("profit_factor", 0.0)
            pf_text = "∞" if pf2 == float("inf") else f"{pf2:.2f}"
            st.metric("Profit Factor", pf_text)
        with r1c6:
            st.metric("Expectancy", f"{stats.get('expectancy', 0):.4f}")

        # Hàng 2: Avg Win, Avg Loss, Largest Win, Largest Loss
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            st.metric("Avg Win", f"{stats.get('avg_win', 0):.4f}")
        with r2c2:
            st.metric("Avg Loss", f"{stats.get('avg_loss', 0):.4f}")
        with r2c3:
            st.metric("Largest Win", f"{stats.get('largest_win', 0):.4f}")
        with r2c4:
            st.metric("Largest Loss", f"{stats.get('largest_loss', 0):.4f}")

        try:
            st.pyplot(plot_winrate_metrics(stats, title="Win/Loss/Breakeven"))
        except Exception:
            pass
        try:
            st.pyplot(plot_trade_pnl_distribution(trades, title="PnL distribution"))
        except Exception:
            pass
        try:
            st.pyplot(plot_trade_timeline(trades, title="Trade timeline"))
        except Exception:
            pass
        try:
            st.pyplot(plot_cumulative_pnl(trades, title="Cumulative PnL"))
        except Exception:
            pass

        with st.expander("📋 Trades table"):
            st.dataframe(trades, use_container_width=True, height=350)

        with st.expander("⬇️ Export trades.csv"):
            formatted = format_trades_csv(
                trades,
                df,
                symbol=sidebar_config.get("symbol") or sidebar_config.get("ticker"),
                timeframe=sidebar_config.get("interval"),
            )
            csv = formatted.to_csv(index=False).encode("utf-8")
        st.download_button(
                "Download trades.csv",
                data=csv,
                file_name="trades.csv",
                mime="text/csv",
                key="rt_bt_dl_trades",
            )

 