"""
Tab cho Regime Ensemble Advanced ML
Bao gồm:
- Advanced Training với XGBoost, LightGBM, CatBoost, Stacking
- Feature Engineering nâng cao
- Hyperparameter Optimization với Optuna
- Advanced Backtest với Validation
- Dynamic Threshold
- Regime-Specific Parameters
- Feature Importance Visualization
"""
from __future__ import annotations
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import warnings
import sys
import os

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from algo_trading.ui.utils import load_df_from_sidebar_config
from algo_trading.backtest.vectorized import run_backtest, BacktestConfig, RiskConfig
from algo_trading.strategies.ml.regime_ensemble_strategy import RegimeEnsembleStrategy
from algo_trading.visualization.plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_trade_pnl_distribution,
)
from algo_trading.utils.trade_stats import calculate_trade_stats
from algo_trading.core.metrics import (
    safe_total_return,
    performance_summary,
)

# Import advanced training functions
try:
    from train_regime_ensemble_models_advanced import (
        calculate_indicators_enhanced,
        detect_regime_optimized,
        build_feature_matrix_enhanced,
        handle_class_imbalance,
        select_features,
        optimize_hyperparameters,
        train_advanced_models,
    )
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False
    warnings.warn("⚠️ Advanced training functions không tìm thấy. Sử dụng basic training.")


# Import advanced backtest functions
try:
    from backtest_regime_ensemble_advanced import (
        validate_equity_curve as validate_equity_curve_func,
        calculate_dynamic_threshold,
        get_regime_specific_params,
        backtest_regime_ensemble_advanced,
    )
except ImportError:
    # Fallback functions
    def validate_equity_curve_func(equity):
        return True, []
    def calculate_dynamic_threshold(*args, **kwargs):
        return 0.55
    def get_regime_specific_params(regime, base_params):
        return base_params

from joblib import dump, load as joblib_load
import matplotlib.pyplot as plt
import seaborn as sns

# Check for advanced ML libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from imblearn.over_sampling import SMOTE, SMOTEN
    HAS_SMOTE = True
    HAS_SMOTEN = True
except ImportError:
    HAS_SMOTE = False
    HAS_SMOTEN = False


def render_regime_ensemble_advanced_tab(sidebar_config: Dict[str, Any]) -> None:
    """Render tab Advanced ML với training và backtest."""
    st.header("🚀 Regime Ensemble Advanced ML")
    st.info(
        "💡 **Advanced ML Features:**\n"
        "- XGBoost, LightGBM, CatBoost, Stacking Ensemble\n"
        "- Feature Engineering nâng cao (200+ features)\n"
        "- Hyperparameter Optimization với Optuna\n"
        "- Class Imbalance Handling (SMOTE)\n"
        "- Feature Selection\n"
        "- Dynamic Threshold\n"
        "- Regime-Specific Parameters\n"
        "- Equity Curve Validation"
    )
    
    # Check dependencies
    missing_deps = []
    if not HAS_XGB:
        missing_deps.append("xgboost")
    if not HAS_LGB:
        missing_deps.append("lightgbm")
    if not HAS_CAT:
        missing_deps.append("catboost")
    if not HAS_OPTUNA:
        missing_deps.append("optuna")
    if not HAS_SMOTE:
        missing_deps.append("imbalanced-learn")
    
    if missing_deps:
        st.warning(
            f"⚠️ **Thiếu dependencies:** {', '.join(missing_deps)}\n\n"
            f"Chạy: `pip install {' '.join(missing_deps)}`"
        )
    
    # Tabs
    tab_train, tab_backtest, tab_analysis = st.tabs([
        "🎓 Advanced Training",
        "📊 Advanced Backtest",
        "📈 Analysis & Visualization"
    ])
    
    with tab_train:
        _render_advanced_training_tab(sidebar_config)
    
    with tab_backtest:
        _render_advanced_backtest_tab(sidebar_config)
    
    with tab_analysis:
        _render_analysis_tab(sidebar_config)


def _render_advanced_training_tab(sidebar_config: Dict[str, Any]) -> None:
    """Tab training advanced models."""
    st.subheader("🎓 Advanced ML Training")
    
    st.markdown("""
    **Tính năng:**
    - ✅ XGBoost, LightGBM, CatBoost với early stopping
    - ✅ Stacking Ensemble (kết hợp nhiều models)
    - ✅ Feature Engineering nâng cao (200+ features)
    - ✅ Hyperparameter Optimization với Optuna
    - ✅ Class Imbalance Handling (SMOTE)
    - ✅ Feature Selection (Mutual Information)
    - ✅ Train/Val/Test Split với time-series
    """)
    
    # Load data
    st.markdown("### 📥 Dữ liệu Training")
    
    if st.button("🔄 Load Data từ Sidebar Config", type="secondary", key="advanced_load_data"):
        with st.spinner("🔄 Đang tải dữ liệu..."):
            try:
                df = load_df_from_sidebar_config(
                    source=sidebar_config['source'],
                    ticker=sidebar_config['ticker'],
                    symbol=sidebar_config['symbol'],
                    interval=sidebar_config['interval'],
                    start=sidebar_config['start'],
                    end=sidebar_config['end'],
                    market=sidebar_config['market'],
                    path=sidebar_config['path']
                )
                st.session_state['advanced_training_df'] = df
                st.success(f"✅ Đã tải dữ liệu: {len(df)} dòng từ {df.index[0]} đến {df.index[-1]}")
            except Exception as e:
                st.error(f"❌ Lỗi tải dữ liệu: {e}")
                return
    
    if 'advanced_training_df' not in st.session_state:
        st.warning("⚠️ Vui lòng load dữ liệu trước khi train.")
        return
    
    df = st.session_state['advanced_training_df']
    
    if len(df) < 1000:
        st.error(f"⚠️ Cần ít nhất 1000 bars để train advanced models. Hiện tại có {len(df)} bars.")
        return
    
    # Configuration
    st.markdown("### ⚙️ Cấu hình Training")
    
    col1, col2 = st.columns(2)
    with col1:
        use_xgb = st.checkbox("Train XGBoost", value=HAS_XGB, disabled=not HAS_XGB, key="advanced_use_xgb")
        use_lgb = st.checkbox("Train LightGBM", value=HAS_LGB, disabled=not HAS_LGB, key="advanced_use_lgb")
        use_cat = st.checkbox("Train CatBoost", value=HAS_CAT, disabled=not HAS_CAT, key="advanced_use_cat")
        use_rf = st.checkbox("Train Random Forest", value=True, key="advanced_use_rf")
        use_gb = st.checkbox("Train Gradient Boosting", value=True, key="advanced_use_gb")
        use_logit = st.checkbox("Train Logistic Regression", value=True, key="advanced_use_logit")
        use_stacking = st.checkbox("Train Stacking Ensemble", value=True, 
                                  help="Kết hợp tất cả models đã train", key="advanced_use_stacking")
    
    with col2:
        validation_split = st.slider("Validation Split (%)", 10, 40, 20, 5, key="advanced_val_split") / 100
        use_smoten = st.checkbox("Sử dụng SMOTEN (Multi-class)", value=HAS_SMOTEN, disabled=not HAS_SMOTEN,
                               help="Xử lý class imbalance với SMOTEN cho multi-class (-1, 0, 1)", key="advanced_use_smoten")
        use_feature_selection = st.checkbox("Feature Selection", value=True,
                                           help="Chọn 100 features quan trọng nhất", key="advanced_use_feat_sel")
        n_features = st.number_input("Số Features (nếu dùng Feature Selection)", 
                                     min_value=50, max_value=200, value=100, step=10, key="advanced_n_features")
        optimize_hp = st.checkbox("Hyperparameter Optimization (Optuna)", 
                                 value=HAS_OPTUNA, disabled=not HAS_OPTUNA,
                                 help="Tối ưu hyperparameters (mất nhiều thời gian)", key="advanced_optimize_hp")
        n_trials = st.number_input("Số Trials (Optuna)", min_value=10, max_value=100, 
                                  value=30, step=10, disabled=not optimize_hp, key="advanced_n_trials")
    
    # Train button
    if st.button("🚀 Train Advanced Models", type="primary", key="advanced_train_button"):
        with st.spinner("🔄 Đang train advanced models... (có thể mất 30-60 phút)"):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Calculate indicators
                status_text.text("📊 Bước 1/7: Tính indicators...")
                progress_bar.progress(1/7)
                
                # Use local variable to avoid UnboundLocalError
                use_advanced_features = HAS_ADVANCED
                
                if use_advanced_features:
                    try:
                        indicators = calculate_indicators_enhanced(df)
                        regime_info = detect_regime_optimized(df, indicators, lookback_window=500)
                        X = build_feature_matrix_enhanced(df, indicators, regime_info)
                    except Exception as e:
                        st.warning(f"⚠️ Lỗi khi dùng advanced features, fallback về basic: {e}")
                        use_advanced_features = False
                
                if not use_advanced_features:
                    # Fallback to basic
                    from algo_trading.indicators import rsi, macd, bollinger_bands, atr
                    indicators = {}
                    close = df["close"]
                    indicators["rsi"] = rsi(close, 14)
                    macd_line, macd_signal, macd_hist = macd(close)
                    indicators["macd_hist"] = macd_hist
                    bb_upper, bb_middle, bb_lower = bollinger_bands(close)
                    indicators["bb_width"] = (bb_upper - bb_lower) / bb_middle
                    regime_info = {"regime": pd.Series(0, index=df.index), "current_regime": "trending"}
                    X = pd.DataFrame({
                        "ret_1": close.pct_change().fillna(0),
                        "ind_rsi": indicators["rsi"],
                        "ind_macd_hist": indicators["macd_hist"],
                        "ind_bb_width": indicators["bb_width"],
                    })
                
                X = X.reindex(df.index).ffill().bfill()
                
                # Convert to numeric, coercing errors to NaN
                X = X.apply(pd.to_numeric, errors='coerce')
                
                # Fill remaining NaN values with 0 (sau khi đã convert to numeric)
                # GIẢI THÍCH: Đảm bảo không có NaN trước khi check isfinite
                X = X.fillna(0)
                
                st.success(f"Đã tạo {X.shape[1]} features")
                
                status_text.text("Bước 2/7: Tạo labels...")
                progress_bar.progress(2/7)
                
                future_ret = df["close"].pct_change().shift(-1).fillna(0)
                y = np.sign(future_ret.values).astype(float)
                
                # Check isfinite (sau khi đã fillna)
                mask = np.isfinite(X.values).all(axis=1) & np.isfinite(y)
                
                # Validate: Đảm bảo có ít nhất một số samples
                if mask.sum() == 0:
                    st.error("❌ Không có samples hợp lệ sau khi clean! Kiểm tra lại dữ liệu.")
                    st.stop()
                
                X_clean = X.values[mask]
                y_clean = y[mask]
                X_df_clean = X.iloc[mask]
                
                st.info(f"✅ Sau khi clean: {len(X_clean)} samples (từ {len(X)} samples ban đầu)")
                
                # Step 3: Feature Selection
                status_text.text("📊 Bước 3/7: Feature Selection...")
                progress_bar.progress(3/7)
                
                if use_feature_selection and use_advanced_features:
                    try:
                        X_selected, selected_features = select_features(
                            X_df_clean, y_clean, method="mutual_info", k=n_features
                        )
                        X_clean = X_selected.values
                        st.info(f"✅ Đã chọn {len(selected_features)} features")
                    except Exception as e:
                        st.warning(f"⚠️ Lỗi feature selection, bỏ qua: {e}")
                        selected_features = list(X_df_clean.columns)
                else:
                    selected_features = list(X_df_clean.columns)
                
                status_text.text("📊 Bước 4/7: Split data...")
                progress_bar.progress(4/7)
                
                n_total = len(X_clean)
                if n_total == 0:
                    st.error("❌ Không có dữ liệu để train! Kiểm tra lại feature engineering.")
                    st.stop()
                
                n_train = int(n_total * (1 - validation_split))
                n_val = int(n_total * validation_split)
                n_train = max(1, n_train)
                
                X_train = X_clean[:n_train]
                y_train = y_clean[:n_train]
                X_val = X_clean[n_train:n_train+n_val] if n_val > 0 else X_clean[n_train:n_train+1]
                y_val = y_clean[n_train:n_train+n_val] if n_val > 0 else y_clean[n_train:n_train+1]
                X_test = X_clean[n_train+n_val:]
                y_test = y_clean[n_train+n_val:]
                
                st.info(f"✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
                
                # Validate train set
                if len(X_train) == 0:
                    st.error("❌ Train set rỗng! Không thể train model.")
                    st.stop()
                
                # Step 5: Handle Class Imbalance
                status_text.text("📊 Bước 5/7: Handle class imbalance...")
                progress_bar.progress(5/7)
                
                if use_smoten and HAS_SMOTEN and use_advanced_features:
                    try:
                        X_train_balanced, y_train_balanced = handle_class_imbalance(
                            X_train, y_train, method="smoten"
                        )
                        st.info(f"✅ After SMOTE: {len(X_train_balanced)} samples")
                        
                        # Validate sau SMOTE
                        if len(X_train_balanced) == 0:
                            st.warning("⚠️ SMOTE trả về 0 samples, dùng data gốc")
                            X_train_balanced, y_train_balanced = X_train, y_train
                    except Exception as e:
                        st.warning(f"⚠️ Lỗi SMOTE, bỏ qua: {e}")
                        X_train_balanced, y_train_balanced = X_train, y_train
                else:
                    X_train_balanced, y_train_balanced = X_train, y_train
                
                # Final validation trước khi train
                if len(X_train_balanced) == 0:
                    st.error("❌ Không có dữ liệu để train sau khi xử lý!")
                    st.stop()
                
                # Step 6: Train Models
                status_text.text("📊 Bước 6/7: Train models...")
                progress_bar.progress(6/7)
                
                # Import training function
                if use_advanced_features:
                    try:
                        models, scalers = train_advanced_models(
                            X_train_balanced, y_train_balanced, X_val, y_val, optimize=optimize_hp
                        )
                    except Exception as e:
                        st.warning(f"⚠️ Lỗi advanced training, fallback về basic: {e}")
                        use_advanced_features = False
                
                if not use_advanced_features:
                    # Fallback to basic training
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.preprocessing import RobustScaler
                    
                    # Validate trước khi scale
                    if len(X_train_balanced) == 0:
                        st.error("❌ Không có dữ liệu để train (fallback mode)!")
                        st.stop()
                    
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train_balanced)
                    X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else X_train_scaled[:1]
                    
                    models = {}
                    scalers = {"scaler": scaler}
                    
                    if use_rf:
                        rf = RandomForestClassifier(n_estimators=300, max_depth=10, 
                                                   random_state=42, class_weight="balanced")
                        rf.fit(X_train_balanced, y_train_balanced)
                        models["rf"] = rf
                    
                    if use_logit:
                        from sklearn.linear_model import LogisticRegression
                        logit = LogisticRegression(max_iter=5000, class_weight="balanced", 
                                                  random_state=42)
                        y_train_binary = (y_train_balanced > 0).astype(int)
                        logit.fit(X_train_scaled, y_train_binary)
                        models["logit"] = logit
                
                # Step 7: Save Models
                status_text.text("📊 Bước 7/7: Lưu models...")
                progress_bar.progress(7/7)
                
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)
                
                model_paths = {}
                
                # Save main ensemble
                if "stacking" in models:
                    dump({"model": models["stacking"], "scaler": scalers["scaler"], "feature_names": selected_features}, 
                         models_dir / "regime_ensemble_advanced.pkl")
                    model_paths["regime_ensemble_advanced.pkl"] = str(models_dir / "regime_ensemble_advanced.pkl")
                elif "xgb" in models:
                    dump({"model": models["xgb"], "scaler": scalers["scaler"], "feature_names": selected_features}, 
                         models_dir / "regime_ensemble_advanced.pkl")
                    model_paths["regime_ensemble_advanced.pkl"] = str(models_dir / "regime_ensemble_advanced.pkl")
                elif "rf" in models:
                    # RF model có thể không có scaler
                    if "scaler" in scalers:
                        dump({"model": models["rf"], "scaler": scalers["scaler"], "feature_names": selected_features}, 
                             models_dir / "regime_ensemble_advanced.pkl")
                    else:
                        dump({"model": models["rf"], "feature_names": selected_features}, 
                             models_dir / "regime_ensemble_advanced.pkl")
                    model_paths["regime_ensemble_advanced.pkl"] = str(models_dir / "regime_ensemble_advanced.pkl")
                
                # Save individual models for bandit
                for name, model in models.items():
                    if name == "scaler":
                        continue
                    if name in ["xgb", "lgb", "cat", "logit", "stacking"]:
                        # Lưu với tên _advanced cho backward compatibility
                        dump({"model": model, "scaler": scalers["scaler"], "feature_names": selected_features}, 
                             models_dir / f"regime_bandit_{name}_advanced.pkl")
                        model_paths[f"regime_bandit_{name}_advanced.pkl"] = str(models_dir / f"regime_bandit_{name}_advanced.pkl")
                        
                        # Lưu thêm với tên không có _advanced cho bandit (xgb, lgb, cat, stacking)
                        if name in ["xgb", "lgb", "cat", "stacking"]:
                            dump({"model": model, "scaler": scalers["scaler"], "feature_names": selected_features}, 
                                 models_dir / f"regime_bandit_{name}.pkl")
                            model_paths[f"regime_bandit_{name}.pkl"] = str(models_dir / f"regime_bandit_{name}.pkl")
                
                
                # Save feature names separately for backward compatibility
                dump(selected_features, models_dir / "regime_ensemble_features.pkl")
                
                progress_bar.progress(1.0)
                status_text.text("✅ Hoàn tất!")
                
                st.success("✅ Training hoàn tất!")
                
                # Display results
                st.markdown("### 📁 Models đã được lưu:")
                for model_name, model_path in model_paths.items():
                    st.info(f"✅ **{model_name}**: `{model_path}`")
                
                # Evaluate on test set
                st.markdown("### 📊 Test Set Evaluation:")
                
                from sklearn.metrics import f1_score, precision_score, recall_score
                
                eval_results = []
                for name, model in models.items():
                    if name == "scaler":
                        continue
                    
                    if name in ["xgb", "lgb", "cat", "logit", "stacking"]:
                        X_test_scaled = scalers["scaler"].transform(X_test)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        y_pred = model.predict(X_test)
                    
                    # Multi-class evaluation
                    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
                    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
                    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
                    accuracy = (y_test == y_pred).mean()
                    
                    eval_results.append({
                        "Model": name.upper(),
                        "Accuracy": accuracy,
                        "F1-Score (macro)": f1_macro,
                        "F1-Score (weighted)": f1_weighted,
                        "Precision (macro)": precision_macro,
                        "Recall (macro)": recall_macro,
                    })
                
                eval_df = pd.DataFrame(eval_results)
                st.dataframe(eval_df, use_container_width=True)
                
                # Store in session state
                st.session_state['advanced_model_paths'] = model_paths
                st.session_state['advanced_selected_features'] = selected_features
                
            except Exception as e:
                st.error(f"❌ Lỗi khi train: {e}")
                import traceback
                st.code(traceback.format_exc())


def _render_advanced_backtest_tab(sidebar_config: Dict[str, Any]) -> None:
    """Tab advanced backtest với validation."""
    st.subheader("📊 Advanced Backtest với Validation")
    
    st.markdown("""
    **Tính năng:**
    - ✅ Train/Validation Split
    - ✅ Dynamic Threshold
    - ✅ Regime-Specific Parameters
    - ✅ Equity Curve Validation
    - ✅ Comprehensive Metrics
    """)
    
    # Model selection
    st.markdown("### 📦 Model Selection")
    
    models_dir = Path("models")
    advanced_model = models_dir / "regime_ensemble_advanced.pkl"
    
    if advanced_model.exists():
        model_path = str(advanced_model)
        st.success(f"✅ Tìm thấy advanced model: `{model_path}`")
    else:
        model_path = st.text_input(
            "Đường dẫn model (.pkl)",
            value="models/regime_ensemble_advanced.pkl",
            help="Nhập path đến advanced model",
            key="advanced_backtest_model_path"
        )
    
    # Configuration
    st.markdown("### ⚙️ Cấu hình Backtest")
    
    col1, col2 = st.columns(2)
    with col1:
        use_dynamic_threshold = st.checkbox("Dynamic Threshold", value=True,
                                           help="Threshold adapt với performance", key="advanced_use_dynamic_thresh")
        base_threshold = st.number_input("Base Threshold", min_value=0.3, max_value=0.9, 
                                        value=0.55, step=0.05, key="advanced_base_threshold")
        use_regime_specific = st.checkbox("Regime-Specific Parameters", value=True,
                                         help="Parameters khác nhau cho mỗi regime", key="advanced_use_regime_specific")
        validation_split = st.slider("Validation Split (%)", 10, 30, 20, 5, key="advanced_backtest_val_split") / 100
    
    with col2:
        sl_pct = st.number_input("Stop Loss (%)", min_value=0.5, max_value=10.0, 
                                 value=2.0, step=0.5, key="advanced_sl_pct") / 100
        tp_pct = st.number_input("Take Profit (%)", min_value=1.0, max_value=20.0, 
                                 value=4.0, step=0.5, key="advanced_tp_pct") / 100
        leverage = st.number_input("Leverage", min_value=1.0, max_value=10.0, 
                                  value=1.0, step=0.5, key="advanced_leverage")
        commission = st.number_input("Commission (%)", min_value=0.0, max_value=1.0, 
                                    value=0.1, step=0.01, key="advanced_commission") / 100
    
    allowed_regimes = st.multiselect(
        "Allowed Regimes",
        ["trending", "ranging", "volatile", "calm"],
        default=["trending", "ranging", "calm"],
        key="advanced_allowed_regimes"
    )
    
    # Run backtest
    if st.button("🚀 Chạy Advanced Backtest", type="primary", key="advanced_backtest_button"):
        with st.spinner("🔄 Đang chạy advanced backtest..."):
            try:
                # Load data
                df = load_df_from_sidebar_config(
                    source=sidebar_config['source'],
                    ticker=sidebar_config['ticker'],
                    symbol=sidebar_config['symbol'],
                    interval=sidebar_config['interval'],
                    start=sidebar_config['start'],
                    end=sidebar_config['end'],
                    market=sidebar_config['market'],
                    path=sidebar_config['path']
                )
                
                if Path("backtest_regime_ensemble_advanced.py").exists():
                    result = backtest_regime_ensemble_advanced(
                        model_path=model_path,
                        source=sidebar_config['source'],
                        ticker=sidebar_config['ticker'],
                        symbol=sidebar_config['symbol'],
                        interval=sidebar_config['interval'],
                        start=sidebar_config['start'],
                        end=sidebar_config['end'],
                        sl_pct=sl_pct,
                        tp_pct=tp_pct,
                        leverage=leverage,
                        commission=commission,
                        proba_threshold=base_threshold,
                        use_dynamic_threshold=use_dynamic_threshold,
                        use_regime_specific_params=use_regime_specific,
                        allowed_regimes=allowed_regimes,
                        validation_split=validation_split,
                    )
                    
                    if result:
                        _display_advanced_backtest_results(result, sidebar_config)
                else:
                    # Fallback to basic backtest
                    st.warning("⚠️ Advanced backtest script không tìm thấy. Sử dụng basic backtest.")
                    
                    strategy = RegimeEnsembleStrategy(
                        model_path=model_path,
                        proba_threshold=base_threshold,
                        allowed_regimes=allowed_regimes,
                        use_direction_output=False,
                        use_dynamic_threshold=use_dynamic_threshold,
                    )
                    
                    signals = strategy.generate_signals(df).signals
                    
                    freq = '1H' if 'h' in sidebar_config['interval'].lower() else '1D'
                    cfg = BacktestConfig(
                        initial_capital=10000.0,
                        leverage=leverage,
                        allow_short=True,
                        commission=commission,
                        freq=freq,
                    )
                    
                    risk = RiskConfig(sl_pct=sl_pct, tp_pct=tp_pct)
                    res = run_backtest(df, signals, cfg=cfg, risk=risk, max_trades=1000)
                    
                    st.success("✅ Backtest hoàn tất!")
                    _display_basic_backtest_results(res, df, signals, sidebar_config)
                
            except Exception as e:
                st.error(f"❌ Lỗi khi chạy backtest: {e}")
                import traceback
                st.code(traceback.format_exc())


def _display_advanced_backtest_results(result: Dict[str, Any], sidebar_config: Dict[str, Any]) -> None:
    """Hiển thị kết quả advanced backtest."""
    st.markdown("---")
    st.subheader("📈 Kết Quả Backtest")
    
    # Combined metrics (hiển thị chính)
    if "metrics_combined" in result:
        metrics_combined = result["metrics_combined"]
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Return", f"{metrics_combined.get('TotalReturn', 0)*100:.2f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics_combined.get('Sharpe', 0):.3f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics_combined.get('MaxDrawdown', 0)*100:.2f}%")
        with col4:
            st.metric("CAGR", f"{metrics_combined.get('CAGR', 0)*100:.2f}%")
        with col5:
            st.metric("Calmar Ratio", f"{metrics_combined.get('Calmar', 0):.3f}")
    
    # Strategy Information
    st.markdown("### 📊 Strategy Information")
    
    # Lấy meta từ train hoặc validation
    train_res = result.get("train", {})
    val_res = result.get("validation", {})
    meta = train_res.get("meta", {}) or val_res.get("meta", {}) or {}
    
    col1, col2 = st.columns(2)
    with col1:
        if 'current_regime' in meta:
            st.info(f"**Current Regime:** {meta['current_regime']}")
        elif 'regime' in meta:
            st.info(f"**Current Regime:** {meta['regime']}")
    with col2:
        # Tìm proba_threshold từ meta hoặc từ sidebar_config
        proba_threshold = meta.get('proba_threshold') or sidebar_config.get('proba_threshold', 0.30)
        st.info(f"**Proba Threshold:** {proba_threshold:.2f}")
    
    # Bandit Statistics (nếu có)
    if 'selected_model' in meta or 'bandit_counts' in meta:
        st.markdown("#### 🎰 Bandit Statistics")
        col3, col4, col5 = st.columns(3)
        with col3:
            if 'selected_model' in meta:
                st.info(f"**Last Selected Model:** {meta['selected_model']}")
        with col4:
            if 'bandit_counts' in meta:
                counts = meta['bandit_counts']
                total = sum(counts.values()) if isinstance(counts, dict) else sum(counts) if isinstance(counts, (list, tuple)) else 0
                st.info(f"**Total Selections:** {total}")
        with col5:
            if 'bandit_values' in meta and 'bandit_counts' in meta:
                values = meta['bandit_values']
                counts = meta['bandit_counts']
                if isinstance(values, dict) and isinstance(counts, dict):
                    # Chỉ xét các models đã được chọn ít nhất 1 lần
                    models_with_selections = {
                        name: avg_reward 
                        for name, avg_reward in values.items() 
                        if counts.get(name, 0) > 0
                    }
                    if models_with_selections:
                        best_model = max(models_with_selections.items(), key=lambda x: x[1])[0]
                        best_reward = models_with_selections[best_model]
                        st.info(f"**Best Model (Avg Reward):** {best_model} ({best_reward:.4f})")
                    else:
                        st.info("**Best Model:** Chưa có model nào được chọn")
                elif isinstance(values, dict):
                    # Fallback nếu không có counts
                    best_model = max(values.items(), key=lambda x: x[1])[0] if values else "N/A"
                    st.info(f"**Best Model (Avg Reward):** {best_model}")
        
        # Model Selection Distribution
        if 'bandit_counts' in meta and 'bandit_values' in meta:
            st.markdown("**Model Selection Distribution:**")
            counts = meta['bandit_counts']
            values = meta['bandit_values']
            
            if isinstance(counts, dict) and isinstance(values, dict):
                bandit_df = pd.DataFrame({
                    'Model': list(counts.keys()),
                    'Selections': [counts[k] for k in counts.keys()],
                    'Avg Reward': [values.get(k, 0) for k in counts.keys()],
                })
                if bandit_df['Selections'].sum() > 0:
                    bandit_df['Selection %'] = (bandit_df['Selections'] / bandit_df['Selections'].sum() * 100).round(2)
                st.dataframe(bandit_df, use_container_width=True)
    
    # Trade Statistics - Lấy từ combined trades
    train_res = result.get("train", {})
    val_res = result.get("validation", {})
    
    trades_train = train_res.get('trades', pd.DataFrame())
    trades_val = val_res.get('trades', pd.DataFrame())
    
    # Combine trades nếu có
    if not trades_train.empty and not trades_val.empty:
        trades_combined = pd.concat([trades_train, trades_val], ignore_index=True)
    elif not trades_train.empty:
        trades_combined = trades_train
    elif not trades_val.empty:
        trades_combined = trades_val
    else:
        trades_combined = pd.DataFrame()
    
    if not trades_combined.empty:
        st.markdown("---")
        st.subheader("📊 Bảng Thống Kê Winrate Chi Tiết")
        
        # Calculate trade stats
        trade_stats = calculate_trade_stats(trades_combined)
        
        # Cảnh báo nếu quá ít trades
        total_trades = trade_stats.get('total_trades', 0)
        if total_trades < 10:
            st.warning(
                f"⚠️ **CẢNH BÁO: Chỉ có {total_trades} trade(s)!**\n\n"
                f"**Nguyên nhân có thể:**\n"
                f"1. Proba threshold quá cao (hiện tại: {proba_threshold:.2f}) → Giảm threshold xuống 0.3-0.4\n"
                f"2. Allowed regimes quá hạn chế → Thêm 'volatile' vào allowed regimes\n"
                f"3. Model không tạo đủ signals → Kiểm tra lại model training\n"
                f"4. Dữ liệu quá ít → Tăng khoảng thời gian backtest\n\n"
                f"**Khuyến nghị:**\n"
                f"- Giảm proba threshold xuống 0.3-0.4 để có nhiều signals hơn\n"
                f"- Cho phép tất cả regimes (trending, ranging, volatile, calm)\n"
                f"- Kiểm tra lại model có được train đúng không"
            )
        
        # Main metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Winrate", f"{trade_stats.get('winrate', 0):.2f}%")
        with col2:
            st.metric("Total Trades", trade_stats.get('total_trades', 0))
        with col3:
            st.metric("Winning Trades", trade_stats.get('winning_trades', 0))
        with col4:
            st.metric("Losing Trades", trade_stats.get('losing_trades', 0))
        with col5:
            pf = trade_stats.get('profit_factor', 0)
            st.metric("Profit Factor", f"{pf:.2f}" if pf != float('inf') else "∞")
        with col6:
            st.metric("Expectancy", f"{trade_stats.get('expectancy', 0):.4f}")
        
        # Detailed metrics
        col7, col8, col9, col10 = st.columns(4)
        with col7:
            st.metric("Avg Win", f"{trade_stats.get('avg_win', 0):.4f}")
        with col8:
            st.metric("Avg Loss", f"{trade_stats.get('avg_loss', 0):.4f}")
        with col9:
            st.metric("Largest Win", f"{trade_stats.get('largest_win', 0):.4f}")
        with col10:
            st.metric("Largest Loss", f"{trade_stats.get('largest_loss', 0):.4f}")
    else:
        st.error(
            "❌ **KHÔNG CÓ TRADES NÀO!**\n\n"
            "**Nguyên nhân:**\n"
            "1. Proba threshold quá cao → Không có signals nào đạt threshold\n"
            "2. Allowed regimes quá hạn chế → Không có regime nào được phép\n"
            "3. Model không tạo signals → Kiểm tra lại model\n"
            "4. Dữ liệu không hợp lệ → Kiểm tra lại data loading\n\n"
            "**Giải pháp:**\n"
            "- Giảm proba threshold xuống 0.3-0.4\n"
            "- Cho phép tất cả regimes\n"
            "- Kiểm tra lại model và data"
        )
    
    # Train/Validation split metrics (nếu muốn xem chi tiết)
    with st.expander("📊 Chi Tiết Train/Validation Metrics"):
        if "metrics_train" in result:
            st.markdown("### 📊 Train Metrics")
            metrics_train = result["metrics_train"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{metrics_train.get('TotalReturn', 0)*100:.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{metrics_train.get('Sharpe', 0):.3f}")
            with col3:
                st.metric("Max Drawdown", f"{metrics_train.get('MaxDrawdown', 0)*100:.2f}%")
            with col4:
                st.metric("CAGR", f"{metrics_train.get('CAGR', 0)*100:.2f}%")
        
        if "metrics_val" in result:
            st.markdown("### 📊 Validation Metrics")
            metrics_val = result["metrics_val"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{metrics_val.get('TotalReturn', 0)*100:.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{metrics_val.get('Sharpe', 0):.3f}")
            with col3:
                st.metric("Max Drawdown", f"{metrics_val.get('MaxDrawdown', 0)*100:.2f}%")
            with col4:
                st.metric("CAGR", f"{metrics_val.get('CAGR', 0)*100:.2f}%")
        
        # Equity curves
        if "train" in result and "validation" in result:
            train_res = result["train"]
            val_res = result["validation"]
            
            if "equity" in train_res and "equity" in val_res:
                st.markdown("### 📈 Equity Curves")
                
                equity_train = train_res["equity"]
                equity_val = val_res["equity"]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(equity_train.index, equity_train.values, label="Train", linewidth=2)
                ax.plot(equity_val.index, equity_val.values, label="Validation", linewidth=2)
                ax.set_xlabel("Time")
                ax.set_ylabel("Equity")
                ax.set_title("Train vs Validation Equity Curve")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)


def _display_basic_backtest_results(res: Dict[str, Any], df: pd.DataFrame, 
                                    signals: pd.Series, sidebar_config: Dict[str, Any]) -> None:
    """Hiển thị kết quả basic backtest."""
    summary = res.get('summary', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Return", f"{summary.get('TotalReturn', 0)*100:.2f}%")
    with col2:
        st.metric("Sharpe Ratio", f"{summary.get('Sharpe', 0):.3f}")
    with col3:
        st.metric("Max Drawdown", f"{summary.get('MaxDrawdown', 0)*100:.2f}%")
    with col4:
        st.metric("CAGR", f"{summary.get('CAGR', 0)*100:.2f}%")
    with col5:
        st.metric("Calmar Ratio", f"{summary.get('Calmar', 0):.3f}")
    
    # Equity curve
    if "equity" in res:
        equity = res["equity"]
        fig = plot_equity_curve(equity, title="Equity Curve")
        st.pyplot(fig)
        
        # Validate equity curve
        is_valid, issues = validate_equity_curve_func(equity)
        if not is_valid:
            st.warning("⚠️ Equity curve có vấn đề:")
            for issue in issues:
                st.warning(issue)


def _render_analysis_tab(sidebar_config: Dict[str, Any]) -> None:
    """Tab analysis và visualization."""
    st.subheader("📈 Analysis & Visualization")
    
    st.info("💡 Tính năng này sẽ được phát triển thêm trong tương lai.")
    
    # Feature importance
    models_dir = Path("models")
    advanced_model = models_dir / "regime_ensemble_advanced.pkl"
    
    if advanced_model.exists():
        st.markdown("### 🔍 Feature Importance")
        
        try:
            loaded = joblib_load(advanced_model)
            if isinstance(loaded, dict):
                model = loaded["model"]
            else:
                model = loaded
            
            # Get feature names (ưu tiên lấy từ model dict; fallback file riêng để tương thích ngược)
            if isinstance(loaded, dict):
                feature_names = loaded.get("feature_names", None)
                if feature_names is None:
                    features_path = models_dir / "regime_ensemble_features.pkl"
                    if features_path.exists():
                        feature_names = joblib_load(features_path)
                    else:
                        feature_names = None
            else:
                features_path = models_dir / "regime_ensemble_features.pkl"
                if features_path.exists():
                    feature_names = joblib_load(features_path)
                else:
                    feature_names = None
            
            # Get feature importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                
                if feature_names and len(feature_names) == len(importances):
                    importance_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importances
                    }).sort_values("Importance", ascending=False).head(20)
                    
                    st.dataframe(importance_df, use_container_width=True)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(data=importance_df, y="Feature", x="Importance", ax=ax)
                    ax.set_title("Top 20 Feature Importance")
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ Feature names không khớp với importances.")
            else:
                st.info("ℹ️ Model không có feature_importances_ attribute.")
        
        except Exception as e:
            st.warning(f"⚠️ Không thể load feature importance: {e}")
def render_regime_ensemble_advanced_tab_alias(sidebar_config: Dict[str, Any]) -> None:
    return render_regime_ensemble_advanced_tab(sidebar_config)

