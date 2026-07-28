"""
Regime Ensemble Hybrid Strategy

Kết hợp 1 model chính (baseline) với bandit để fine-tune hoặc validate signals.
"""

from __future__ import annotations

import warnings
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from ..base import BaseStrategy, StrategyResult
from .regime_ensemble_strategy import RegimeEnsembleStrategy


class RegimeEnsembleHybridStrategy(BaseStrategy):
    """
    Hybrid Strategy: Kết hợp 1 model chính (baseline) với bandit để fine-tune hoặc validate.
    
    Có 2 modes:
    1. "fine_tune": Dùng bandit để điều chỉnh signals từ model chính
       - Model chính tạo signal
       - Bandit chọn model tốt nhất để validate/confirm signal
       - Chỉ trade khi cả 2 đồng ý
       
    2. "weighted": Kết hợp predictions từ model chính và bandit với weights
       - Model chính: weight cao (0.7)
       - Bandit best model: weight thấp (0.3)
       - Weighted average probabilities
    
    Parameters
    ----------
    main_model_path : str
        Đường dẫn tới model chính (RF lớn).
    bandit_model_paths : dict[str, str]
        Đường dẫn tới các bandit models.
    proba_threshold : float
        Ngưỡng xác suất để vào lệnh.
    allowed_regimes : list[str]
        Danh sách regimes được phép trade.
    hybrid_mode : str
        "fine_tune" hoặc "weighted".
    main_model_weight : float
        Weight cho model chính (nếu dùng weighted mode).
    bandit_type : str
        Loại bandit: 'ucb' hoặc 'eps_greedy'.
    reward_mode : str
        'direction' hoặc 'pnl'.
    """

    name = "Regime Ensemble (Hybrid)"

    def __init__(
        self,
        main_model_path: str,
        bandit_model_paths: Dict[str, str],
        proba_threshold: float = 0.55,
        allowed_regimes: Optional[list[str]] = None,
        hybrid_mode: str = "fine_tune",
        main_model_weight: float = 0.7,
        bandit_type: str = "ucb",
        epsilon: float = 0.1,
        reward_mode: str = "direction",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not main_model_path:
            raise ValueError("RegimeEnsembleHybridStrategy cần `main_model_path`.")
        if not bandit_model_paths:
            raise ValueError("RegimeEnsembleHybridStrategy cần `bandit_model_paths`.")

        self.main_model_path = main_model_path
        self.bandit_model_paths = bandit_model_paths
        self.proba_threshold = proba_threshold
        self.allowed_regimes = allowed_regimes or ["trending", "ranging", "calm"]
        self.hybrid_mode = hybrid_mode.lower()
        self.main_model_weight = main_model_weight
        self.bandit_type = bandit_type.lower()
        self.epsilon = epsilon
        self.reward_mode = reward_mode

        self.indicators_list = ["RSI", "MACD", "BB", "ATR", "VWAP", "SMA", "EMA"]

        # Load main model
        try:
            loaded = joblib_load(main_model_path)
            if isinstance(loaded, dict) and "model" in loaded:
                self.main_model = loaded["model"]
                self.main_model_scaler = loaded.get("scaler", None)
            else:
                self.main_model = loaded
                self.main_model_scaler = None
        except Exception as e:
            raise ImportError(f"Không load được main model từ {main_model_path}: {e}")

        # Load bandit models
        self.bandit_models: Dict[str, Any] = {}
        self.bandit_scalers: Dict[str, Any] = {}
        for name, path in bandit_model_paths.items():
            try:
                loaded = joblib_load(path)
                if isinstance(loaded, dict) and "model" in loaded:
                    self.bandit_models[name] = loaded["model"]
                    self.bandit_scalers[name] = loaded.get("scaler", None)
                else:
                    self.bandit_models[name] = loaded
            except Exception as e:
                warnings.warn(f"Không load được bandit model '{name}' từ {path}: {e}")

        if len(self.bandit_models) < 1:
            raise ValueError("Không load được bandit model nào.")

        # Bandit stats
        self.bandit_names = list(self.bandit_models.keys())
        self.n_bandit_models = len(self.bandit_names)
        self.bandit_counts = np.zeros(self.n_bandit_models, dtype=float)
        self.bandit_values = np.zeros(self.n_bandit_models, dtype=float)

    # Tái sử dụng helpers
    _calculate_indicators = RegimeEnsembleStrategy._calculate_indicators
    _detect_regime = RegimeEnsembleStrategy._detect_regime
    _build_feature_matrix = RegimeEnsembleStrategy._build_feature_matrix

    def _select_bandit_arm_ucb(self, counts: np.ndarray, values: np.ndarray, t: int) -> int:
        """UCB1 selection."""
        n_arms = len(counts)
        for i in range(n_arms):
            if counts[i] == 0:
                return i
        ucb = values + np.sqrt(2 * np.log(t) / counts)
        return int(np.argmax(ucb))

    def _select_bandit_arm_eps_greedy(self, counts: np.ndarray, values: np.ndarray) -> int:
        """Epsilon-greedy selection."""
        n_arms = len(counts)
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, n_arms))
        return int(np.argmax(values))

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        if df is None or df.empty:
            raise ValueError("DataFrame df rỗng.")
        if "close" not in df.columns:
            raise ValueError("DataFrame df cần có cột 'close'.")

        indicators = self._calculate_indicators(df)
        regime_info = self._detect_regime(df, indicators)
        current_regime = regime_info.get("current_regime", "trending")

        if current_regime not in self.allowed_regimes:
            signals = pd.Series(0, index=df.index)
            meta = {
                "reason": f"Regime {current_regime} not in allowed_regimes {self.allowed_regimes}",
                "current_regime": current_regime,
            }
            return StrategyResult(signals=signals, meta=meta)

        X = self._build_feature_matrix(df, indicators, regime_info)
        X = X.reindex(df.index).ffill().bfill()

        # Áp dụng scaler cho main model nếu có
        X_main = X.values
        if self.main_model_scaler is not None:
            X_main = self.main_model_scaler.transform(X_main)

        closes = df["close"].values
        n = len(df)

        # Main model predictions
        try:
            if hasattr(self.main_model, "predict_proba"):
                main_proba = self.main_model.predict_proba(X_main)
                if main_proba.shape[1] == 2:
                    main_p_short = main_proba[:, 0]
                    main_p_long = main_proba[:, 1]
                else:
                    classes = list(self.main_model.classes_)
                    idx_long = classes.index(1) if 1 in classes else None
                    idx_short = classes.index(-1) if -1 in classes else None
                    main_p_long = main_proba[:, idx_long] if idx_long is not None else np.zeros(n)
                    main_p_short = main_proba[:, idx_short] if idx_short is not None else np.zeros(n)
            else:
                main_pred = self.main_model.predict(X_main)
                main_p_long = (main_pred == 1).astype(float)
                main_p_short = (main_pred == -1).astype(float)
        except Exception as e:
            warnings.warn(f"Lỗi khi chạy main model: {e}")
            main_p_long = np.zeros(n)
            main_p_short = np.zeros(n)

        signals_arr = np.zeros(n, dtype=float)

        if self.hybrid_mode == "fine_tune":
            # Fine-tune mode: Bandit validate signals từ main model
            for t in range(n - 1):
                x_t = X.iloc[t : t + 1].values

                # Main model signal
                if main_p_long[t] >= self.proba_threshold and main_p_long[t] > main_p_short[t]:
                    main_signal = 1.0
                elif main_p_short[t] >= self.proba_threshold and main_p_short[t] > main_p_long[t]:
                    main_signal = -1.0
                else:
                    main_signal = 0.0

                # Nếu main model không có signal, skip
                if main_signal == 0.0:
                    signals_arr[t] = 0.0
                    continue

                # Bandit chọn model để validate
                if self.bandit_type == "eps_greedy":
                    arm = self._select_bandit_arm_eps_greedy(self.bandit_counts, self.bandit_values)
                else:
                    arm = self._select_bandit_arm_ucb(self.bandit_counts, self.bandit_values, t + 1)

                model_name = self.bandit_names[arm]
                model = self.bandit_models[model_name]

                # Áp dụng scaler nếu có
                x_t_scaled = x_t
                if model_name in self.bandit_scalers:
                    scaler = self.bandit_scalers[model_name]
                    x_t_scaled = scaler.transform(x_t)

                # Bandit model prediction
                try:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(x_t_scaled)[0]
                        if proba.shape[0] == 2:
                            b_p_short, b_p_long = proba[0], proba[1]
                        else:
                            classes = list(model.classes_)
                            idx_long = classes.index(1) if 1 in classes else None
                            idx_short = classes.index(-1) if -1 in classes else None
                            b_p_long = proba[idx_long] if idx_long is not None else 0.0
                            b_p_short = proba[idx_short] if idx_short is not None else 0.0
                    else:
                        b_pred = model.predict(x_t_scaled)[0]
                        b_p_long = 1.0 if b_pred == 1 else 0.0
                        b_p_short = 1.0 if b_pred == -1 else 0.0

                    # Chỉ trade nếu cả 2 đồng ý
                    if main_signal == 1.0 and b_p_long >= self.proba_threshold and b_p_long > b_p_short:
                        signals_arr[t] = 1.0
                    elif main_signal == -1.0 and b_p_short >= self.proba_threshold and b_p_short > b_p_long:
                        signals_arr[t] = -1.0
                    else:
                        signals_arr[t] = 0.0  # Không đồng ý → không trade
                except Exception as e:
                    warnings.warn(f"Lỗi khi chạy bandit model '{model_name}': {e}")
                    signals_arr[t] = 0.0

                # Update bandit reward
                if t < n - 1:
                    ret_next = (closes[t + 1] / closes[t] - 1.0) if closes[t] != 0 else 0.0
                    if self.reward_mode == "pnl":
                        reward = ret_next * signals_arr[t]
                    else:
                        reward = 1.0 if np.sign(ret_next) == np.sign(signals_arr[t]) and signals_arr[t] != 0 else 0.0

                    self.bandit_counts[arm] += 1.0
                    self.bandit_values[arm] += (reward - self.bandit_values[arm]) / self.bandit_counts[arm]

        else:  # weighted mode
            # Weighted mode: Kết hợp probabilities
            for t in range(n - 1):
                x_t = X.iloc[t : t + 1].values

                # Bandit chọn model tốt nhất
                if self.bandit_type == "eps_greedy":
                    arm = self._select_bandit_arm_eps_greedy(self.bandit_counts, self.bandit_values)
                else:
                    arm = self._select_bandit_arm_ucb(self.bandit_counts, self.bandit_values, t + 1)

                model_name = self.bandit_names[arm]
                model = self.bandit_models[model_name]

                # Áp dụng scaler nếu có
                x_t_scaled = x_t
                if model_name in self.bandit_scalers:
                    scaler = self.bandit_scalers[model_name]
                    x_t_scaled = scaler.transform(x_t)

                # Bandit model prediction
                try:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(x_t_scaled)[0]
                        if proba.shape[0] == 2:
                            b_p_short, b_p_long = proba[0], proba[1]
                        else:
                            classes = list(model.classes_)
                            idx_long = classes.index(1) if 1 in classes else None
                            idx_short = classes.index(-1) if -1 in classes else None
                            b_p_long = proba[idx_long] if idx_long is not None else 0.0
                            b_p_short = proba[idx_short] if idx_short is not None else 0.0
                    else:
                        b_pred = model.predict(x_t_scaled)[0]
                        b_p_long = 1.0 if b_pred == 1 else 0.0
                        b_p_short = 1.0 if b_pred == -1 else 0.0
                except Exception as e:
                    warnings.warn(f"Lỗi khi chạy bandit model '{model_name}': {e}")
                    b_p_long = 0.0
                    b_p_short = 0.0

                # Weighted average
                w_main = self.main_model_weight
                w_bandit = 1.0 - w_main

                combined_p_long = w_main * main_p_long[t] + w_bandit * b_p_long
                combined_p_short = w_main * main_p_short[t] + w_bandit * b_p_short

                # Signal từ combined probabilities
                if combined_p_long >= self.proba_threshold and combined_p_long > combined_p_short:
                    signals_arr[t] = 1.0
                elif combined_p_short >= self.proba_threshold and combined_p_short > combined_p_long:
                    signals_arr[t] = -1.0
                else:
                    signals_arr[t] = 0.0

                # Update bandit reward
                if t < n - 1:
                    ret_next = (closes[t + 1] / closes[t] - 1.0) if closes[t] != 0 else 0.0
                    if self.reward_mode == "pnl":
                        reward = ret_next * signals_arr[t]
                    else:
                        reward = 1.0 if np.sign(ret_next) == np.sign(signals_arr[t]) and signals_arr[t] != 0 else 0.0

                    self.bandit_counts[arm] += 1.0
                    self.bandit_values[arm] += (reward - self.bandit_values[arm]) / self.bandit_counts[arm]

        raw_signals = pd.Series(signals_arr, index=df.index)
        signals = BaseStrategy.validate_signals(raw_signals, df.index)

        # Convert bandit stats
        counts_dict = {self.bandit_names[i]: int(self.bandit_counts[i]) for i in range(self.n_bandit_models)}
        values_dict = {self.bandit_names[i]: float(self.bandit_values[i]) for i in range(self.n_bandit_models)}
        selected_model = self.bandit_names[int(np.argmax(self.bandit_counts))] if self.bandit_counts.sum() > 0 else None

        meta = {
            "current_regime": current_regime,
            "allowed_regimes": self.allowed_regimes,
            "proba_threshold": self.proba_threshold,
            "hybrid_mode": self.hybrid_mode,
            "main_model_path": self.main_model_path,
            "bandit_counts": counts_dict,
            "bandit_values": values_dict,
            "selected_model": selected_model,
        }
        return StrategyResult(signals=signals, meta=meta)
























