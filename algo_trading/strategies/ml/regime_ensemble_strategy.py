from __future__ import annotations

import warnings
import logging
from typing import Dict, Any, Optional, List
from collections import deque

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from pathlib import Path

from ..base import BaseStrategy, StrategyResult
from algo_trading.indicators import (
    rsi,
    macd,
    bollinger_bands,
    atr,
    vwap,
    sma,
    ema,
)
from algo_trading.indicators.ict import (
    detect_order_blocks,
    ob_confluence_signal,
    fib_features,
)

from algo_trading.ml.sequence_extractor import SequenceFeatureExtractor, SequenceExtractorConfig

try:
    from algo_trading.market_models.regime import detect_regime_hmm

    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    detect_regime_hmm = None

try:
    from algo_trading.ml.regime_specific_models import RegimeSpecificModels
    HAS_REGIME_SPECIFIC = True
except ImportError:
    HAS_REGIME_SPECIFIC = False
    RegimeSpecificModels = None

# New improvements imports
try:
    from algo_trading.ml.regime_specific_thresholds import RegimeSpecificThresholds
    HAS_REGIME_THRESHOLDS = True
except ImportError:
    HAS_REGIME_THRESHOLDS = False
    RegimeSpecificThresholds = None

try:
    from algo_trading.ml.regime_confidence_score import RegimeConfidenceScorer
    HAS_REGIME_CONFIDENCE = True
except ImportError:
    HAS_REGIME_CONFIDENCE = False
    RegimeConfidenceScorer = None

try:
    from algo_trading.ml.probability_calibration import ProbabilityCalibrator
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False
    ProbabilityCalibrator = None


class RegimeEnsembleStrategy(BaseStrategy):
    

    name = "Regime Ensemble (ML)"

    def __init__(
        self,
        model_path: Optional[str] = None,
        proba_threshold: float = 0.35,  # Giảm xuống 0.35 để tăng tần suất lệnh
        allowed_regimes: Optional[list[str]] = None,
        use_direction_output: bool = False,
        # ICT filter params
        use_ict_filter: bool = False,
        ict_ob_tolerance_pct: float = 0.002,
        ict_fib_max_dist: float = 0.02,
        # Regime-specific models
        use_regime_specific: bool = False,
        regime_specific_model_path: Optional[str] = None,
        # Sequence features (Option 3: deep feature extractor -> tabular head)
        use_sequence_features: bool = True,
        sequence_model_path: Optional[str] = None,
        sequence_len: int = 64,
        allow_fallback: bool = False,
        # Dynamic threshold
        use_dynamic_threshold: bool = False,
        regime_thresholds: Optional[Dict[int, float]] = None,
        regime_threshold_density_boost: float = 0.0,
        # Quantile threshold mode (alternative to static threshold)
        use_quantile_threshold: bool = True,
        quantile_window: int = 400,
        target_signal_rate: float = 0.08,
        quantile_floor: float = 0.55,
        quantile_ceiling: float = 0.95,
        # Uncertainty-aware adaptive regime detection
        use_uncertainty_aware: bool = False,
        uncertainty_confidence_threshold: float = 0.6,  # Skip if max proba < this
        uncertainty_regime_transition_threshold: float = 0.3,  # Skip if regime prob < this
        **kwargs: Any,
    ) -> None:

        super().__init__(**kwargs)

        # Tự động tìm model trong thư mục `models/` nếu không truyền model_path
        if model_path is None and not use_regime_specific:
            default_candidates = [
                "models/regime_ensemble_advanced.pkl",
                "models/regime_ensemble_optimized.pkl",
                "models/regime_ensemble.pkl",
            ]
            auto_path = None
            for p in default_candidates:
                if Path(p).exists():
                    auto_path = p
                    break
            if auto_path is None:
                raise ValueError(
                    "RegimeEnsembleStrategy cần `model_path` tới model sklearn đã pretrain, "
                    "hoặc set `use_regime_specific=True` với `regime_specific_model_path`. "
                    "Không tìm thấy file model mặc định trong thư mục `models/`."
                )
            model_path = auto_path

        self.model_path = model_path
        self.proba_threshold = proba_threshold
        self.allowed_regimes = allowed_regimes or ["trending", "ranging", "calm"]
        self.use_direction_output = use_direction_output

        # ICT filter settings
        self.use_ict_filter = use_ict_filter
        self.ict_ob_tolerance_pct = ict_ob_tolerance_pct
        self.ict_fib_max_dist = ict_fib_max_dist

        # Regime-specific models settings
        self.use_regime_specific = use_regime_specific
        self.regime_specific_model_path = regime_specific_model_path

        # Sequence feature extractor settings
        self.use_sequence_features = bool(use_sequence_features)
        self.sequence_model_path = sequence_model_path or "models/seq_lstm_extractor.pt"
        self.sequence_len = int(sequence_len)
        self.allow_fallback = bool(allow_fallback)
        self._seq_extractor: Optional[SequenceFeatureExtractor] = None

        # Dynamic threshold settings
        self.use_dynamic_threshold = use_dynamic_threshold
        self.regime_threshold_density_boost = float(regime_threshold_density_boost)
        self.regime_thresholds = regime_thresholds or {
            0: 0.42,  # trending
            1: 0.47,  # ranging
            2: 0.52,  # volatile
            3: 0.46,  # calm
        }
        self.best_thresholds: Dict[str, float] = {}
        self.model_threshold: Optional[float] = None

        # Quantile threshold settings to keep signal density stable under calibration drift.
        self.use_quantile_threshold = bool(use_quantile_threshold)
        self.quantile_window = max(50, int(quantile_window))
        self.target_signal_rate = float(np.clip(target_signal_rate, 0.01, 0.50))
        self.quantile_floor = float(np.clip(quantile_floor, 0.01, 0.99))
        self.quantile_ceiling = float(np.clip(quantile_ceiling, self.quantile_floor, 0.99))
        self._score_history = deque(maxlen=self.quantile_window)

        # Uncertainty-aware settings
        self.use_uncertainty_aware = use_uncertainty_aware
        self.uncertainty_confidence_threshold = uncertainty_confidence_threshold
        self.uncertainty_regime_transition_threshold = uncertainty_regime_transition_threshold

        # Các indicators mặc định (có thể mở rộng sau)
        self.indicators_list = ["RSI", "MACD", "BB", "ATR", "VWAP", "SMA", "EMA"]
        self.selected_features = None
        self.feature_contract: Optional[Dict[str, Any]] = None

        # Load model(s)
        if use_regime_specific and HAS_REGIME_SPECIFIC:
            # Load regime-specific models
            if regime_specific_model_path is None:
                raise ValueError("Cần `regime_specific_model_path` khi `use_regime_specific=True`")
            
            try:
                loaded = joblib_load(regime_specific_model_path)
                if isinstance(loaded, dict) and "regime_models" in loaded:
                    self.regime_models = loaded["regime_models"]
                    # RegimeSpecificModels tự quản lý scaler theo từng regime,
                    # nên không dùng global scaler bên ngoài để tránh mismatch số chiều.
                    self.model_scaler = None
                    self._trained_feature_names = loaded.get("feature_names", None)
                    self.feature_contract = loaded.get("feature_contract", None)
                    loaded_regime_thresholds = loaded.get("regime_thresholds", None)
                    if isinstance(loaded_regime_thresholds, dict):
                        try:
                            self.regime_thresholds = {int(k): float(v) for k, v in loaded_regime_thresholds.items()}
                        except Exception:
                            pass
                    loaded_best = loaded.get("best_thresholds", None)
                    if isinstance(loaded_best, dict):
                        self.best_thresholds = {str(k): float(v) for k, v in loaded_best.items()}
                    if self.feature_contract is None:
                        contract_path = Path(regime_specific_model_path).parent / "regime_feature_contract.pkl"
                        if contract_path.exists():
                            try:
                                self.feature_contract = joblib_load(contract_path)
                            except Exception as e:
                                warnings.warn(f"Không load được feature contract: {e}")
                else:
                    raise ValueError("Regime-specific model file không đúng format")
            except Exception as e:
                raise ImportError(f"Không load được regime-specific models từ {regime_specific_model_path}: {e}")
            
            self.model = None  # Không dùng model chung
        else:
            # Load model sklearn đã pretrain (phương pháp cũ)
            try:
                loaded = joblib_load(model_path)
                # Kiểm tra nếu model được lưu cùng với scaler (dict với keys "model" và "scaler")
                if isinstance(loaded, dict) and "model" in loaded:
                    self.model = loaded["model"]
                    self.model_scaler = loaded.get("scaler", None)  # Có thể có hoặc không
                    # Load feature names từ saved model dict (quan trọng để align features)
                    self._trained_feature_names = loaded.get("feature_names", None)
                    self.feature_contract = loaded.get("feature_contract", None)
                    loaded_regime_thresholds = loaded.get("regime_thresholds", None)
                    if isinstance(loaded_regime_thresholds, dict):
                        try:
                            self.regime_thresholds = {int(k): float(v) for k, v in loaded_regime_thresholds.items()}
                        except Exception:
                            pass
                    loaded_best = loaded.get("best_thresholds", None)
                    if isinstance(loaded_best, dict):
                        self.best_thresholds = {str(k): float(v) for k, v in loaded_best.items()}

                    # Safety: nếu scaler và feature_names không khớp số chiều
                    # (ví dụ dùng model .pkl cũ train trước khi có feature selection fix),
                    # thì bỏ scaler để tránh lỗi "X has N features, but RobustScaler is expecting M".
                    try:
                        n_features_scaler = getattr(self.model_scaler, "n_features_in_", None)
                        n_features_names = len(self._trained_feature_names) if self._trained_feature_names else None
                        if (
                            self.model_scaler is not None
                            and n_features_scaler is not None
                            and n_features_names is not None
                            and n_features_scaler != n_features_names
                        ):
                            warnings.warn(
                                f"Scaler feature dimension ({n_features_scaler}) != feature_names length ({n_features_names}). "
                                "Bỏ qua scaler để tránh lỗi shape mismatch. Nên retrain models để đồng bộ."
                            )
                            self.model_scaler = None
                        # Also check model
                        n_features_model = getattr(self.model, "n_features_in_", None)
                        if (
                            n_features_model is not None
                            and n_features_names is not None
                            and n_features_model != n_features_names
                        ):
                            warnings.warn(
                                f"Model feature dimension ({n_features_model}) != feature_names length ({n_features_names}). "
                                "Setting feature_names to None to avoid mismatch."
                            )
                            self._trained_feature_names = None
                    except Exception:
                        # Nếu check thất bại vì lý do gì đó, không làm crash init
                        pass
                else:
                    self.model = loaded
                    self.model_scaler = None
                    self._trained_feature_names = None
            except Exception as e:
                raise ImportError(f"Không load được ensemble model từ {model_path}: {e}")
            
            # Load selected features
            features_path = Path(model_path).parent / "regime_ensemble_features.pkl"
            if features_path.exists():
                try:
                    self.selected_features = joblib_load(features_path)
                    print(f"✅ Loaded selected features: {len(self.selected_features)} features")
                except Exception as e:
                    warnings.warn(f"Không load được selected features: {e}")
                    self.selected_features = None
            else:
                self.selected_features = None
                warnings.warn(f"Không tìm thấy file selected features: {features_path}")

            # Load feature contract (fallback từ file nếu không có trong model dict)
            if self.feature_contract is None:
                contract_path = Path(model_path).parent / "regime_feature_contract.pkl"
                if contract_path.exists():
                    try:
                        self.feature_contract = joblib_load(contract_path)
                    except Exception as e:
                        warnings.warn(f"Không load được feature contract: {e}")
            
            self.regime_models = None  # Không dùng regime-specific models

        # Infer default threshold cho model hiện tại từ best_thresholds (nếu có)
        if self.model_threshold is None and self.best_thresholds:
            cls_name = type(self.model).__name__.lower() if self.model is not None else ""
            key_priority = [
                "moe",
                "ewa",
                "xgb",
                "lgb",
                "cat",
                "mlp",
                "svm",
                "logit",
                "rf",
                "gb",
            ]
            for k in key_priority:
                if k in self.best_thresholds and k in cls_name:
                    self.model_threshold = float(self.best_thresholds[k])
                    break
            if self.model_threshold is None:
                for k in ["moe", "ewa", "xgb", "lgb", "cat", "mlp", "svm", "logit", "rf", "gb"]:
                    if k in self.best_thresholds:
                        self.model_threshold = float(self.best_thresholds[k])
                        break

    def _apply_feature_contract(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ép X theo feature contract để đảm bảo parity train/inference.
        """
        if not self.feature_contract:
            return X
        feature_names = self.feature_contract.get("feature_names", None)
        if not feature_names:
            return X

        # Giữ đúng thứ tự cột theo train contract; thiếu cột thì fill 0.
        return X.reindex(columns=feature_names, fill_value=0.0).ffill().bfill().fillna(0.0)

    def _feature_skew_summary(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Tính skew nhanh giữa X hiện tại và thống kê train trong contract.
        Dùng mean z-shift để monitor drift thời gian thực.
        """
        if not self.feature_contract:
            return {}
        stats = self.feature_contract.get("stats", {})
        if not stats:
            return {}

        z_shifts: List[float] = []
        shifted = 0
        for col in X.columns:
            st = stats.get(col)
            if not st:
                continue
            train_mean = float(st.get("mean", 0.0))
            train_std = float(st.get("std", 0.0))
            live_mean = float(pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).mean())
            z = float(abs(live_mean - train_mean) / max(train_std, 1e-8))
            z_shifts.append(z)
            if z >= 3.0:
                shifted += 1

        if not z_shifts:
            return {}
        return {
            "feature_contract_enabled": True,
            "feature_contract_n": int(len(X.columns)),
            "feature_mean_zshift_avg": float(np.mean(z_shifts)),
            "feature_mean_zshift_max": float(np.max(z_shifts)),
            "feature_zshift_ge_3_count": int(shifted),
        }

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Tính các indicators enhanced để match với training script."""
        indicators: Dict[str, pd.Series] = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df.get("volume", None)
        
        # === RSI với nhiều periods ===
        indicators["rsi"] = rsi(close, 14)
        indicators["rsi_9"] = rsi(close, 9)
        indicators["rsi_21"] = rsi(close, 21)
        indicators["rsi_50"] = rsi(close, 50)
        
        # === MACD ===
        macd_line, macd_signal, macd_hist = macd(close)
        indicators["macd_line"] = macd_line
        indicators["macd_signal"] = macd_signal
        indicators["macd_hist"] = macd_hist
        
        # === Bollinger Bands ===
        bb_upper, bb_middle, bb_lower = bollinger_bands(close)
        indicators["bb_upper"] = bb_upper
        indicators["bb_lower"] = bb_lower
        indicators["bb_middle"] = bb_middle
        indicators["bb_width"] = (bb_upper - bb_lower) / bb_middle
        indicators["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # === ATR ===
        atr_val = atr(df, 14)
        indicators["atr"] = atr_val
        indicators["atr_ratio"] = atr_val / close
        indicators["atr_20"] = atr(df, 20)
        
        # === Volatility ===
        returns = close.pct_change()
        indicators["volatility_5"] = returns.rolling(5).std()
        indicators["volatility_20"] = returns.rolling(20).std()
        indicators["volatility_ratio"] = indicators["volatility_5"] / indicators["volatility_20"]
        
        # === Moving Averages ===
        indicators["sma_20"] = sma(close, 20)
        indicators["sma_50"] = sma(close, 50)
        indicators["sma_100"] = sma(close, 100)
        indicators["sma_200"] = sma(close, 200)
        indicators["ema_20"] = ema(close, 20)
        indicators["ema_50"] = ema(close, 50)
        indicators["ema_200"] = ema(close, 200)
        
        # === MA Crossovers ===
        indicators["sma_20_50_cross"] = (indicators["sma_20"] > indicators["sma_50"]).astype(float)
        indicators["ema_20_50_cross"] = (indicators["ema_20"] > indicators["ema_50"]).astype(float)
        indicators["price_sma20_ratio"] = close / indicators["sma_20"]
        indicators["price_sma50_ratio"] = close / indicators["sma_50"]
        
        # === Volume ===
        if volume is not None:
            indicators["volume"] = volume
            indicators["volume_ma"] = volume.rolling(20).mean()
            indicators["volume_ratio"] = volume / indicators["volume_ma"]
            vwap_val = vwap(df)
            indicators["vwap"] = vwap_val
            indicators["vwap_distance"] = (close - vwap_val) / vwap_val
        else:
            indicators["volume"] = pd.Series(0, index=df.index)
            indicators["volume_ma"] = pd.Series(0, index=df.index)
            indicators["volume_ratio"] = pd.Series(0, index=df.index)
            indicators["vwap"] = close
            indicators["vwap_distance"] = pd.Series(0, index=df.index)
        
        # === Market Structure ===
        indicators["higher_high"] = (high > high.shift(1).rolling(5).max()).astype(float)
        indicators["lower_low"] = (low < low.shift(1).rolling(5).min()).astype(float)
        indicators["price_position"] = (close - low.rolling(20).min()) / (
            high.rolling(20).max() - low.rolling(20).min()
        )
        
        # === Momentum ===
        indicators["momentum_5"] = close / close.shift(5) - 1
        indicators["momentum_10"] = close / close.shift(10) - 1
        indicators["momentum_20"] = close / close.shift(20) - 1
        indicators["roc_10"] = (close - close.shift(10)) / close.shift(10)
        indicators["roc_20"] = (close - close.shift(20)) / close.shift(20)
        
        # === Cross-Indicator Features ===
        # GIẢI THÍCH: Tương tác giữa indicators giúp capture complex patterns
        indicators["rsi_macd_divergence"] = indicators["rsi"] - (indicators["macd_hist"] * 100)
        indicators["bb_rsi_interaction"] = indicators["bb_position"] * (indicators["rsi"] / 100)

        # === Volume Features ===
        if volume is not None:
            indicators["volume_ma_20"] = volume.rolling(20).mean()
            indicators["volume_trend"] = volume.rolling(5).mean() / indicators["volume_ma_20"]
            # OBV (On-Balance Volume)
            price_change = close.diff()
            obv = (volume * np.sign(price_change)).cumsum()
            indicators["obv"] = obv
            indicators["obv_ma"] = obv.rolling(20).mean()
            indicators["obv_ratio"] = obv / indicators["obv_ma"]
        else:
            indicators["volume_ma_20"] = pd.Series(0, index=df.index)
            indicators["volume_trend"] = pd.Series(0, index=df.index)
            indicators["obv"] = pd.Series(0, index=df.index)
            indicators["obv_ma"] = pd.Series(0, index=df.index)
            indicators["obv_ratio"] = pd.Series(0, index=df.index)

        return indicators

    def _detect_regime(
        self, df: pd.DataFrame, indicators: Dict[str, pd.Series]
    ) -> Dict[str, Any]:

        # Fallback: simple regime rules dùng MACD / BB width / RSI
        def simple_regime() -> Dict[str, Any]:
            macd_hist = indicators.get("macd_hist", pd.Series(0, index=df.index))
            rsi_val = indicators.get("rsi", pd.Series(50, index=df.index))
            bb_width = indicators.get("bb_width", pd.Series(0.02, index=df.index))
            volatility = indicators.get("volatility_20", pd.Series(0.01, index=df.index))
            volume_ratio = indicators.get("volume_ratio", pd.Series(1.0, index=df.index))
            momentum = indicators.get("momentum_20", pd.Series(0.0, index=df.index))

            # Adaptive thresholds (hãy giữ để tránh “dính” regime quá lâu)
            vol_threshold = volatility.rolling(20).quantile(0.5).fillna(0.01)
            macd_threshold = 0.01 * (1 + volatility / (vol_threshold + 1e-9))
            bb_threshold_high = 0.09 * (1 + volatility / (vol_threshold + 1e-9))

            # Regime rules cân bằng giữa momentum + volatility + volume
            trending = (
                (macd_hist > macd_threshold)
                & (rsi_val > 45)
                & (momentum > 0)
                & (volume_ratio > 0.9)
            )

            ranging = (
                (macd_hist.abs() < macd_threshold * 0.5)
                & (bb_width < 0.055)
                & (volatility < vol_threshold * 0.9)
            )

            volatile = (
                (bb_width > bb_threshold_high)
                | (volatility > vol_threshold * 1.4)
                | (volume_ratio > 1.3)
            )

            calm = (
                (bb_width < 0.03)
                & ~trending
                & (volatility < vol_threshold * 0.6)
                & (momentum.abs() < 0.4)
            )

            regime_id = pd.Series(0, index=df.index)  # trending default
            regime_id[ranging] = 1
            regime_id[volatile] = 2
            regime_id[calm] = 3

            names = ["trending", "ranging", "volatile", "calm"]
            current_name = names[int(regime_id.iloc[-1])]

            return {
                "regime": regime_id,
                "current_regime": current_name,
                "current_regime_id": int(regime_id.iloc[-1]),
            }

        # Ensemble voting: multiple regime voters + weighted HMM vote
        idx = df.index
        close = df["close"].astype(float)
        ret = close.pct_change().fillna(0.0)
        n = len(df)

        if n == 0:
            return {"regime": pd.Series(dtype=int), "current_regime": "trending", "current_regime_id": 0}

        macd_hist = indicators.get("macd_hist", pd.Series(0.0, index=idx)).fillna(0.0)
        rsi_val = indicators.get("rsi", pd.Series(50.0, index=idx)).fillna(50.0)
        bb_width = indicators.get("bb_width", pd.Series(0.05, index=idx)).fillna(0.05)
        vol20 = indicators.get("volatility_20", ret.rolling(20).std()).fillna(ret.rolling(20).std().median())
        momentum_20 = indicators.get("momentum_20", close.pct_change(20)).fillna(0.0)
        sma20 = indicators.get("sma_20", close.rolling(20).mean()).fillna(close)
        sma50 = indicators.get("sma_50", close.rolling(50).mean()).fillna(close)

        vol_median = vol20.rolling(50).median().fillna(vol20.median())
        bb_median = bb_width.rolling(50).median().fillna(bb_width.median())

        voters: List[pd.Series] = []

        # Voter 1: trend-momentum
        v1 = pd.Series(1, index=idx, dtype=int)
        v1[(macd_hist > 0.0) & (rsi_val.between(40, 70)) & (momentum_20 > 0)] = 0
        voters.append(v1)

        # Voter 2: volatility state
        v2 = pd.Series(1, index=idx, dtype=int)
        v2[(bb_width > bb_median * 1.35) | (vol20 > vol_median * 1.35)] = 2
        v2[(bb_width < bb_median * 0.75) & (vol20 < vol_median * 0.75)] = 3
        voters.append(v2)

        # Voter 3: range detector
        v3 = pd.Series(0, index=idx, dtype=int)
        range_cond = (macd_hist.abs() < macd_hist.abs().rolling(30).median().fillna(0.0)) & (bb_width < bb_median * 1.05)
        v3[range_cond] = 1
        v3[~range_cond & (bb_width > bb_median * 1.45)] = 2
        voters.append(v3)

        # Voter 4: market structure
        v4 = pd.Series(1, index=idx, dtype=int)
        v4[(close > sma20) & (sma20 > sma50)] = 0
        v4[(close < sma20) & (sma20 < sma50) & (vol20 < vol_median)] = 3
        v4[(vol20 > vol_median * 1.4)] = 2
        voters.append(v4)

        scores = np.zeros((n, 4), dtype=float)
        for voter in voters:
            v = voter.reindex(idx).fillna(1).astype(int).clip(0, 3).values
            scores[np.arange(n), v] += 1.0

        hmm_probs = None
        if HAS_HMM and detect_regime_hmm is not None:
            try:
                hmm_info = detect_regime_hmm(df, indicators=indicators, lookback_window=500)
                hmm_series = hmm_info.get("regime", None) if isinstance(hmm_info, dict) else None
                if isinstance(hmm_series, pd.Series):
                    hmm_ids = hmm_series.reindex(idx).fillna(0).astype(int).clip(0, 3).values
                    scores[np.arange(n), hmm_ids] += 2.0  # HMM weighted vote
                    rp = hmm_info.get("regime_probabilities", None)
                    if isinstance(rp, pd.DataFrame) and len(rp) > 0:
                        hmm_probs = rp.reindex(idx).ffill().bfill()
            except Exception as e:
                warnings.warn(f"HMM regime detection failed, fallback to voters-only: {e}")

        score_df = pd.DataFrame(scores, index=idx, columns=[0, 1, 2, 3])
        score_smooth = score_df.ewm(span=5, adjust=False).mean()

        regime_probabilities = score_smooth.div(score_smooth.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.25)
        regime_probabilities.columns = ["trending", "ranging", "volatile", "calm"]
        if hmm_probs is not None and set(regime_probabilities.columns).issubset(set(hmm_probs.columns)):
            aligned_hmm = hmm_probs[regime_probabilities.columns].reindex(idx).ffill().bfill()
            regime_probabilities = 0.7 * regime_probabilities + 0.3 * aligned_hmm

        # === A. Entropy filter ===
        # H(t) = -Σ P(r|t) * log(P(r|t)), chuẩn hóa về [0, 1] với max = log(4)
        # 0 = voters hoàn toàn đồng thuận, 1 = nhiễu tối đa (mỗi voter bầu khác nhau)
        probs_arr = regime_probabilities.values
        entropy = -np.sum(probs_arr * np.log(np.clip(probs_arr, 1e-9, 1.0)), axis=1)
        entropy_normalized = entropy / np.log(4)
        entropy_series = pd.Series(entropy_normalized, index=idx)

        # === C. Hysteresis (regime stickiness) ===
        # Chỉ đổi regime khi P(r_new) > P(r_cur) + delta để tránh flip liên tục khi nhiễu
        # delta = 0.15: cần dẫn trước 15% xác suất mới được phép chuyển regime
        delta = 0.15
        regime_ids_arr = np.zeros(n, dtype=int)
        regime_ids_arr[0] = int(np.argmax(probs_arr[0]))
        for t in range(1, n):
            cur = regime_ids_arr[t - 1]
            new = int(np.argmax(probs_arr[t]))
            if new != cur and probs_arr[t, new] > probs_arr[t, cur] + delta:
                regime_ids_arr[t] = new
            else:
                regime_ids_arr[t] = cur
        regime_series = pd.Series(regime_ids_arr, index=idx, dtype=int)

        names = ["trending", "ranging", "volatile", "calm"]
        cur_id = int(regime_series.iloc[-1])
        return {
            "regime": regime_series,
            "current_regime": names[cur_id],
            "current_regime_id": cur_id,
            "regime_probabilities": regime_probabilities,
            "entropy": entropy_series,       # 0 = tin tưởng, 1 = hoàn toàn nhiễu
            "entropy_current": float(entropy_series.iloc[-1]),
        }

    # ------------------------------------------------------------------
    # Features for ensemble model
    # ------------------------------------------------------------------

    def _build_feature_matrix(
        self, df: pd.DataFrame, indicators: Dict[str, pd.Series], regime_info: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Xây dựng feature matrix optimized để match với training script (192 features).
        """
        feats: Dict[str, pd.Series] = {}
        close = df["close"]
        volume = df.get("volume", None)
        
        # === Basic Returns ===
        feats["ret_1"] = close.pct_change().fillna(0)
        feats["ret_5"] = close.pct_change(5).fillna(0)
        feats["ret_10"] = close.pct_change(10).fillna(0)
        feats["ret_20"] = close.pct_change(20).fillna(0)
        feats["ret_50"] = close.pct_change(50).fillna(0)
        
        # === Indicators ===
        for k, v in indicators.items():
            feats[f"ind_{k}"] = v
        
        # === Lagged Features ===
        for lag in [1, 2, 3, 5, 10]:
            feats[f"ret_lag{lag}"] = feats["ret_1"].shift(lag).fillna(0)
            if "rsi" in indicators:
                feats[f"rsi_lag{lag}"] = indicators["rsi"].shift(lag).fillna(50)
            if "macd_hist" in indicators:
                feats[f"macd_hist_lag{lag}"] = indicators["macd_hist"].shift(lag).fillna(0)
        
        # === Rolling Statistics ===
        for window in [5, 10, 20, 50]:
            feats[f"ret_ma{window}"] = feats["ret_1"].rolling(window).mean().fillna(0)
            feats[f"ret_std{window}"] = feats["ret_1"].rolling(window).std().fillna(0)
            feats[f"ret_skew{window}"] = feats["ret_1"].rolling(window).skew().fillna(0)
            feats[f"ret_kurt{window}"] = feats["ret_1"].rolling(window).kurt().fillna(0)
        
        # === Regime Features ===
        regime_series = regime_info.get("regime", None)
        if isinstance(regime_series, pd.Series):
            reg_ids = regime_series.astype(int)
            for rid, name in enumerate(["trending", "ranging", "volatile", "calm"]):
                feats[f"regime_{name}"] = (reg_ids == rid).astype(float)
                # Regime persistence
                feats[f"regime_{name}_persist"] = (
                    (reg_ids == rid).astype(int).groupby((reg_ids != rid).cumsum()).cumsum()
                )
        else:
            for name in ["trending", "ranging", "volatile", "calm"]:
                feats[f"regime_{name}"] = 0.0
                feats[f"regime_{name}_persist"] = 0.0
        
        # === Interaction Features ===
        if "ind_rsi" in feats and "ind_macd_hist" in feats:
            feats["rsi_macd_interaction"] = feats["ind_rsi"] * feats["ind_macd_hist"]
            feats["rsi_macd_divergence"] = feats["ind_rsi"] - (feats["ind_macd_hist"] * 100)
        
        if "ind_bb_position" in feats and "ind_rsi" in feats:
            feats["bb_rsi_interaction"] = feats["ind_bb_position"] * (feats["ind_rsi"] / 100)
        
        if "ind_volatility_20" in feats and "ind_atr_ratio" in feats:
            feats["vol_atr_interaction"] = feats["ind_volatility_20"] * feats["ind_atr_ratio"]
        
        # === ICT Features ===
        # Note: ICT features có thể không có trong prediction, nhưng sẽ được fill với 0 nếu thiếu
        try:
            ict_ob = detect_order_blocks(
                df,
                lookback=20,
                min_body_pct=0.005,
            )
            ict_ob_zone = ob_confluence_signal(
                df,
                ict_ob["ob_bull_level"],
                ict_ob["ob_bear_level"],
                tolerance_pct=0.002,
            )
            fib_df = fib_features(df, lookback=100)
            
            for k, v in ict_ob.items():
                feats[f"ict_{k}"] = v
            for k, v in ict_ob_zone.items():
                feats[f"ict_{k}"] = v
            for col in fib_df.columns:
                feats[f"ict_{col}"] = fib_df[col]
        except Exception as e:
            warnings.warn(f"⚠️ Lỗi ICT features (sẽ fill 0 nếu thiếu): {e}")
        
        # === Multi-Timeframe Features ===
        # Note: Multi-timeframe features có thể không có trong prediction
        # Chúng sẽ được fill với 0 nếu thiếu khi align với feature_names

        # === Sequence Features (TCN/LSTM extractor) ===
        if self.use_sequence_features:
            try:
                if self._seq_extractor is None:
                    cfg = SequenceExtractorConfig(
                        enabled=True,
                        model_path=self.sequence_model_path,
                        seq_len=self.sequence_len,
                        device="cpu",
                        use_log_returns=True,
                        fallback=self.allow_fallback,
                    )
                    self._seq_extractor = SequenceFeatureExtractor(cfg)
                seq_df = self._seq_extractor.transform(df)
                if not self.allow_fallback:
                    loaded_arch = self._seq_extractor.loaded_arch
                    if loaded_arch != "lstm":
                        raise ValueError(
                            f"Strict mode yêu cầu LSTM extractor, nhưng loaded_arch={loaded_arch}."
                        )
                for col in seq_df.columns:
                    feats[f"seq_{col}"] = seq_df[col]
            except Exception as e:
                if self.allow_fallback:
                    warnings.warn(f"⚠️ Sequence features failed (skip): {e}")
                else:
                    raise ValueError(f"Sequence extractor strict mode failed: {e}")
        
        X = pd.DataFrame(feats).ffill().bfill().fillna(0)
        # Đảm bảo tất cả columns là numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        return X

    # ------------------------------------------------------------------
    # Dynamic Threshold Calculation
    # ------------------------------------------------------------------

    def _calculate_dynamic_threshold(
        self,
        base_thresholds: np.ndarray,
        volatility_indicator: pd.Series,
    ) -> np.ndarray:
        """
        Tính dynamic threshold dựa trên biến động (volatility).
        
        Logic:
        - Chuẩn hóa volatility score (dựa trên BB Width) về 0-1.
        - High Volatility -> Giảm threshold (dễ vào lệnh hơn).
        - Low Volatility -> Giữ nguyên (hoặc tăng nhẹ).
        """
        # Đảm bảo input không có NaN
        bb_width = volatility_indicator.fillna(0.05).values
        
        # Chuẩn hóa volatility score về 0-1
        # bb_width mặc định ~ 0.05. Cao > 0.10. Thấp < 0.02.
        # Clip để đảm bảo giá trị trong [0, 1]
        vol_score = np.clip((bb_width - 0.02) / (0.10 - 0.02), 0.0, 1.0)
        
        # Điều chỉnh: Giảm tối đa 0.05 (5%) khi vol cao nhất
        # Khi vol thấp (<= 0.02), adj = 0
        adj = vol_score * 0.05
        
        # Dynamic Threshold = Base - Adj
        return base_thresholds - adj

    def _build_adaptive_thresholds(
        self,
        directional_strength: np.ndarray,
        base_thresholds: np.ndarray,
    ) -> np.ndarray:
        """
        Build per-bar thresholds from score quantiles to stabilize signal rate.
        Uses only historical scores (no look-ahead) to remain backtest-safe.
        """
        n = int(len(directional_strength))
        if n == 0:
            return np.asarray(base_thresholds, dtype=float)

        base_arr = np.asarray(base_thresholds, dtype=float)
        if base_arr.ndim == 0:
            base_arr = np.full(n, float(base_arr), dtype=float)

        if (not self.use_quantile_threshold) or (len(base_arr) != n):
            return np.clip(base_arr, 0.01, 0.99)

        target_quantile = float(np.clip(1.0 - self.target_signal_rate, 0.50, 0.995))
        min_hist = max(30, int(0.3 * self.quantile_window))

        hist = deque(self._score_history, maxlen=self.quantile_window)
        adaptive = np.zeros(n, dtype=float)

        for i in range(n):
            if len(hist) >= min_hist:
                hist_arr = np.asarray(hist, dtype=float)
                q_th = float(np.quantile(hist_arr, target_quantile))
                q_th = float(np.clip(q_th, self.quantile_floor, self.quantile_ceiling))
                adaptive[i] = max(float(base_arr[i]), q_th)
            else:
                adaptive[i] = float(base_arr[i])

            hist.append(float(directional_strength[i]))

        self._score_history = hist
        return np.clip(adaptive, 0.01, 0.99)

    # ------------------------------------------------------------------
    # Main generate_signals
    # ------------------------------------------------------------------

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

        # === A. Entropy filter: bỏ qua nếu thị trường đang nhiễu ===
        # entropy_current gần 1.0 → voters hoàn toàn bất đồng, không nên vào lệnh
        entropy_current = regime_info.get("entropy_current", 0.0)
        ENTROPY_SKIP_THRESHOLD = 0.85  # bỏ qua nếu entropy > 85% max entropy
        if entropy_current > ENTROPY_SKIP_THRESHOLD:
            signals = pd.Series(0, index=df.index)
            meta = {
                "reason": f"Entropy quá cao ({entropy_current:.3f} > {ENTROPY_SKIP_THRESHOLD}), thị trường nhiễu",
                "current_regime": current_regime,
                "entropy_current": entropy_current,
            }
            return StrategyResult(signals=signals, meta=meta)

        X = self._build_feature_matrix(df, indicators, regime_info)

        # Align X index with df
        X = X.reindex(df.index).ffill().bfill()

        meta_debug: Dict[str, Any] = {
            "n_bars": int(len(df)),
        }

        # Apply feature contract first (hard alignment to trained schema)
        if self.feature_contract is not None:
            X = self._apply_feature_contract(X)
            meta_debug.update(self._feature_skew_summary(X))

        # Apply feature selection if available
        if self.feature_contract is None and self.selected_features is not None:
            available_features = [f for f in self.selected_features if f in X.columns]
            if len(available_features) != len(self.selected_features):
                missing = set(self.selected_features) - set(X.columns)
                warnings.warn(f"Missing features in X: {missing}")
            X = X[available_features]
            print(f"✅ Applied feature selection: {len(available_features)}/{len(self.selected_features)} features")

        X_values = X.values
        
        if self.model_scaler is not None:
            n_features_scaler = getattr(self.model_scaler, "n_features_in_", None)
            n_features_X = X.shape[1]
            if n_features_scaler is not None and n_features_scaler != n_features_X:
                warnings.warn(f"Scaler expects {n_features_scaler} features but X has {n_features_X}. Disabling scaler.")
                self.model_scaler = None
        
        # Check model feature compatibility
        n_features_model = getattr(self.model, "n_features_in_", None)
        if n_features_model is not None and n_features_model != X.shape[1]:
            warnings.warn(f"Model expects {n_features_model} features but X has {X.shape[1]}. Returning neutral signals.")
            signals = pd.Series(0, index=df.index)
            meta = {"error": f"Feature shape mismatch, expected: {n_features_model}, got {X.shape[1]}"}
            return StrategyResult(signals=signals, meta=meta)
        
        if self.model_scaler is not None:
            X_values = self.model_scaler.transform(X_values)

        try:
            # SỬ DỤNG REGIME-SPECIFIC MODELS NẾU ĐƯỢC BẬT
            if self.use_regime_specific and self.regime_models is not None:
                # Lấy regime IDs và probabilities từ regime_info
                regime_series = regime_info.get("regime", None)
                regime_probabilities = regime_info.get("regime_probabilities", None)
                
                if regime_series is not None:

                    if isinstance(regime_series, pd.Series):
                        regime_series = regime_series.reindex(df.index, method="ffill")
                        # Thay NaN / inf bằng 0 trước khi convert sang int / map
                        regime_series = regime_series.replace([np.inf, -np.inf], 0).fillna(0)
                    
                    # Convert regime names/ids to numeric IDs 0-3 (theo thứ tự trong tài liệu)
                    regime_name_to_id = {"trending": 0, "ranging": 1, "volatile": 2, "calm": 3}
                    if regime_series is not None:
                        if not np.issubdtype(regime_series.dtype, np.number):
                            regime_ids = np.array(
                                [regime_name_to_id.get(str(r), 0) for r in regime_series.values]
                            )
                        else:
                            # Ở nhánh numeric, đảm bảo đã loại NaN/inf ở trên
                            regime_ids = regime_series.astype(int).values
                    
                    if len(X_values) == 0:
                        raise ValueError("X_values is empty, cannot predict")

                    # Convert to DataFrame:
                    # - Nếu dùng regime-specific: align X với feature_names_regime (trước feature selection)
                    # - Nếu dùng ensemble: dùng X_values (sau khi đã align + scale)
                    if self.use_regime_specific and self.regime_models is not None:
                        # Regime-specific models dùng feature_names_regime (trước feature selection)
                        # Cần align X với feature_names_regime từ model
                        regime_feature_names = getattr(self.regime_models, "feature_names", None)
                        if regime_feature_names:
                            # Thêm các cột thiếu với giá trị 0.0
                            missing_cols = [col for col in regime_feature_names if col not in X.columns]
                            if missing_cols:
                                missing_df = pd.DataFrame(0.0, index=X.index, columns=missing_cols)
                                X = pd.concat([X, missing_df], axis=1)
                            # Reorder và chỉ giữ các cột có trong regime_feature_names
                            X_df = X.reindex(columns=regime_feature_names).ffill().bfill().fillna(0.0)
                        else:
                            X_df = X.copy()
                    else:
                        if isinstance(X_values, np.ndarray) and self._trained_feature_names:
                            X_df = pd.DataFrame(X_values, columns=self._trained_feature_names)
                        else:
                            X_df = pd.DataFrame(X_values)
                
                    use_regime_probs = False
                    if regime_probabilities is not None and isinstance(regime_probabilities, pd.DataFrame):
                        # Align regime_probabilities với df.index để tránh lỗi broadcast
                        rp = regime_probabilities.copy()
                        # Nếu index của rp không cùng dtype với df.index (ví dụ RangeIndex vs DatetimeIndex),
                        # align theo vị trí để tránh lỗi "Cannot compare dtypes int64 and datetime64[ns, UTC]"
                        if not isinstance(rp.index, type(df.index)):
                            # Cắt/pad theo độ dài, rồi gán index = df.index (ưu tiên đoạn cuối nếu rp ngắn hơn)
                            if len(rp) >= len(df):
                                rp = rp.iloc[-len(df):]
                            if len(rp) > 0:
                                rp = pd.DataFrame(rp.values, index=df.index[-len(rp):], columns=rp.columns)
                                rp = rp.reindex(df.index).ffill().fillna(0.0)
                            else:
                                # Nếu rp rỗng ngay từ đầu, fallback về regime_ids
                                use_regime_probs = False
                        else:
                            rp = rp.reindex(df.index, method="ffill").fillna(0.0)
                        
                        # Kiểm tra empty sau khi align
                        if len(rp) > 0 and rp.values.size > 0:
                            use_regime_probs = True
                            regime_probs_array = rp.values
                    
                    if use_regime_probs:
                        # Dùng regime_probabilities nếu có và hợp lệ
                        proba = self.regime_models.predict_proba(
                            X=X_df,
                            regime_probabilities=regime_probs_array
                        )
                    else:
                        # Fallback: dùng regime_ids (phương pháp đơn giản hơn, nhưng vẫn hoạt động)
                        if not self.allow_fallback:
                            raise ValueError(
                                "Strict mode: regime_probabilities không hợp lệ, không dùng fallback regime_ids."
                            )
                        if len(regime_ids) == 0:
                            raise ValueError("regime_ids is empty and regime_probabilities is also invalid")
                        proba = self.regime_models.predict_proba(
                            X=X_df,
                            regime_ids=regime_ids
                        )
                    
                    # Process probabilities như bình thường
                    classes = [0, 1]  # Binary classification
                    p_short = proba[:, 0]
                    p_long = proba[:, 1]
                    p_neutral = np.zeros(len(X))
                    
                    # Ngưỡng proba theo từng regime (đã tinh chỉnh để cân bằng giữa entry và noise):
                    # - trending: dễ vào lệnh, ưu tiên trend-following
                    # - ranging: thận trọng, ưu tiên giá ổn định
                    # - volatile: ít vào hơn, ưu tiên những xác suất mạnh
                    # - calm: vừa phải
                    regime_thresholds = dict(self.regime_thresholds)
                    
                    # Chuẩn bị array base thresholds cho từng bar dựa trên regime ID
                    base_thresholds_arr = np.array([regime_thresholds.get(int(rid), self.proba_threshold) for rid in regime_ids])
                    if self.regime_threshold_density_boost > 0:
                        base_thresholds_arr = np.clip(
                            base_thresholds_arr - self.regime_threshold_density_boost,
                            0.30,
                            0.95,
                        )
                    
                    # Nếu dùng Dynamic Threshold
                    if self.use_dynamic_threshold:
                        # Lấy BB Width từ indicators (đã tính ở đầu hàm)
                        # Cần đảm bảo index khớp với X/df
                        bb_width_series = indicators.get("bb_width", pd.Series(0.05, index=df.index))
                        # Align với length của X (vì vòng lặp chạy theo len(X))
                        # Lưu ý: X đã được reindex theo df.index và fillna
                        if len(bb_width_series) != len(X):
                            bb_width_series = bb_width_series.reindex(df.index).ffill().bfill()
                        
                        final_thresholds = self._calculate_dynamic_threshold(base_thresholds_arr, bb_width_series)
                        
                        # Debug: lưu lại threshold trung bình để check
                        meta_debug["avg_dynamic_threshold"] = float(np.mean(final_thresholds))
                    else:
                        final_thresholds = base_thresholds_arr

                    directional_strength = np.maximum(p_long, p_short)
                    final_thresholds = self._build_adaptive_thresholds(
                        directional_strength=directional_strength,
                        base_thresholds=final_thresholds,
                    )
                    meta_debug.update({
                        "use_quantile_threshold": bool(self.use_quantile_threshold),
                        "target_signal_rate": float(self.target_signal_rate),
                        "avg_effective_threshold": float(np.mean(final_thresholds)) if len(final_thresholds) else 0.0,
                        "last_effective_threshold": float(final_thresholds[-1]) if len(final_thresholds) else 0.0,
                    })

                    # Uncertainty-aware filtering for regime-specific
                    if self.use_uncertainty_aware:
                        # Calculate confidence as max probability (binary: long vs short)
                        confidence_scores = np.maximum(p_long, p_short)
                        
                        # Check regime transition probabilities
                        regime_confident = np.ones(len(X), dtype=bool)
                        if use_regime_probs and 'regime_probs_array' in locals():
                            # regime_probs_array is the probability for current regime
                            regime_confident = regime_probs_array >= self.uncertainty_regime_transition_threshold
                        
                        # Combine confidence checks
                        confident_mask = (confidence_scores >= self.uncertainty_confidence_threshold) & regime_confident
                    else:
                        confident_mask = np.ones(len(X), dtype=bool)

                    signals_arr = np.zeros(len(X))
                    signal_count = 0
                    for i in range(len(X)):
                        # reg_id = int(regime_ids[i]) if i < len(regime_ids) else 0 # Đã map vào base_thresholds_arr
                        th = final_thresholds[i]
                        
                        if confident_mask[i] and p_long[i] >= th and (p_long[i] > p_short[i]):
                            signals_arr[i] = 1.0
                            signal_count += 1
                        elif confident_mask[i] and p_short[i] >= th and (p_short[i] > p_long[i]):
                            signals_arr[i] = -1.0
                            signal_count += 1
                        else:
                            signals_arr[i] = 0.0
                    
                    raw_signals = pd.Series(signals_arr, index=df.index)
                    
                    # Add uncertainty info to meta_debug for regime-specific
                    if self.use_uncertainty_aware:
                        meta_debug.update({
                            "uncertainty_aware": True,
                            "uncertainty_filtered_signals": int((~confident_mask).sum()),
                            "avg_confidence": float(np.mean(confidence_scores)),
                            "min_confidence": float(np.min(confidence_scores)),
                            "confidence_threshold": self.uncertainty_confidence_threshold,
                            "regime_transition_threshold": self.uncertainty_regime_transition_threshold,
                        })
                    
                    meta_debug.update({
                        "signal_count": int(signal_count),
                        "using_regime_specific": True
                    })
                else:
                    # Fallback: dùng model chung nếu có
                    if self.allow_fallback and self.model is not None:
                        warnings.warn("Không có regime info, fallback về model chung")
                        # Continue với code cũ bên dưới
                    else:
                        raise ValueError("Không có regime info (strict mode không cho fallback về model chung)")
            
            # PHƯƠNG PHÁP CŨ: Dùng model chung (hoặc fallback từ regime-specific)
            if not (self.use_regime_specific and self.regime_models is not None and 'raw_signals' in locals()):
                if self.model is None:
                    raise ValueError("Không có model nào để predict")
                    
                if self.use_direction_output:
                    y_pred = self.model.predict(X_values)
                    raw_signals = pd.Series(y_pred, index=df.index)
                    signal_count = (raw_signals != 0).sum()
                    meta_debug.update({"signal_count": int(signal_count)})
                else:
                    if hasattr(self.model, "predict_proba"):
                        proba = self.model.predict_proba(X_values)
                        classes = list(getattr(self.model, "classes_", []))
                        p_short = np.zeros(len(X))
                        p_long = np.zeros(len(X))
                        p_neutral = np.zeros(len(X))

                        if proba.shape[1] == 2 and not classes:
                            p_short = proba[:, 0]
                            p_long = proba[:, 1]
                        else:
                            # Prefer mapping via classes_ when available
                            if classes and proba.shape[1] == len(classes):
                                idx_long = classes.index(1) if 1 in classes else None
                                idx_short = classes.index(-1) if -1 in classes else None
                                idx_neutral = classes.index(0) if 0 in classes else None
                                if idx_long is not None:
                                    p_long = proba[:, idx_long]
                                if idx_short is not None:
                                    p_short = proba[:, idx_short]
                                if idx_neutral is not None:
                                    p_neutral = proba[:, idx_neutral]
                            elif proba.shape[1] == 2:
                                # Binary but classes_ exists: try to map by label, else fallback by order
                                if classes and len(classes) == 2:
                                    idx_long = classes.index(1) if 1 in classes else None
                                    idx_short = classes.index(-1) if -1 in classes else None
                                    if idx_long is not None and idx_short is not None:
                                        p_long = proba[:, idx_long]
                                        p_short = proba[:, idx_short]
                                    else:
                                        p_short = proba[:, 0]
                                        p_long = proba[:, 1]
                                else:
                                    p_short = proba[:, 0]
                                    p_long = proba[:, 1]

                        # Uncertainty-aware filtering
                        if self.use_uncertainty_aware:
                            # Calculate confidence as max probability across all classes
                            confidence_scores = np.maximum.reduce([p_long, p_short, p_neutral])
                            
                            # Check regime transition probabilities if available
                            regime_confident = np.ones(len(X), dtype=bool)
                            regime_probabilities = regime_info.get("regime_probabilities", None)
                            if regime_probabilities is not None and isinstance(regime_probabilities, pd.DataFrame):
                                # Get current regime probability
                                current_regime_prob = regime_probabilities.get(current_regime, pd.Series(1.0, index=df.index))
                                current_regime_prob = current_regime_prob.reindex(df.index, method='ffill').fillna(1.0).values[:len(X)]
                                regime_confident = current_regime_prob >= self.uncertainty_regime_transition_threshold
                            
                            # Combine confidence checks
                            confident_mask = (confidence_scores >= self.uncertainty_confidence_threshold) & regime_confident
                        else:
                            confident_mask = np.ones(len(X), dtype=bool)

                        effective_threshold = float(self.model_threshold) if self.model_threshold is not None else self.proba_threshold
                        directional_strength = np.maximum(p_long, p_short)
                        adaptive_thresholds = self._build_adaptive_thresholds(
                            directional_strength=directional_strength,
                            base_thresholds=np.full(len(X), effective_threshold, dtype=float),
                        )
                        signals_arr = np.zeros(len(X))
                        signal_count = 0
                        for i in range(len(X)):
                            th_i = float(adaptive_thresholds[i])
                            # Nếu có neutral (0), yêu cầu directional proba thắng neutral để tránh spam.
                            if confident_mask[i] and p_long[i] >= th_i and (p_long[i] > p_short[i]) and (p_long[i] >= p_neutral[i]):
                                signals_arr[i] = 1.0
                                signal_count += 1
                            elif confident_mask[i] and p_short[i] >= th_i and (p_short[i] > p_long[i]) and (p_short[i] >= p_neutral[i]):
                                signals_arr[i] = -1.0
                                signal_count += 1
                            else:
                                signals_arr[i] = 0.0

                        raw_signals = pd.Series(signals_arr, index=df.index)
                        
                        # Add uncertainty info to meta_debug
                        if self.use_uncertainty_aware:
                            meta_debug.update({
                                "uncertainty_aware": True,
                                "uncertainty_filtered_signals": int((~confident_mask).sum()),
                                "avg_confidence": float(np.mean(confidence_scores)),
                                "min_confidence": float(np.min(confidence_scores)),
                                "confidence_threshold": self.uncertainty_confidence_threshold,
                                "regime_transition_threshold": self.uncertainty_regime_transition_threshold,
                            })
                        
                        # Thêm thông tin debug vào meta nếu không có signals
                        if signal_count == 0:
                            meta_debug.update({
                                "classes": classes,
                                "max_p_long": float(np.max(p_long)) if len(p_long) else 0.0,
                                "max_p_short": float(np.max(p_short)) if len(p_short) else 0.0,
                                "max_p_neutral": float(np.max(p_neutral)) if len(p_neutral) else 0.0,
                                "mean_p_long": float(np.mean(p_long)) if len(p_long) else 0.0,
                                "mean_p_short": float(np.mean(p_short)) if len(p_short) else 0.0,
                                "mean_p_neutral": float(np.mean(p_neutral)) if len(p_neutral) else 0.0,
                                "samples_above_threshold_long": int((p_long >= effective_threshold).sum()),
                                "samples_above_threshold_short": int((p_short >= effective_threshold).sum()),
                            })
                        else:
                            meta_debug.update({
                                "classes": classes,
                                "signal_count": int(signal_count),
                            })
                        meta_debug.update({
                            "use_quantile_threshold": bool(self.use_quantile_threshold),
                            "target_signal_rate": float(self.target_signal_rate),
                            "avg_effective_threshold": float(np.mean(adaptive_thresholds)) if len(adaptive_thresholds) else float(effective_threshold),
                            "last_effective_threshold": float(adaptive_thresholds[-1]) if len(adaptive_thresholds) else float(effective_threshold),
                        })
                    else:
                        # Fallback: dùng predict (giả định -1/0/1)
                        y_pred = self.model.predict(X_values)
                        raw_signals = pd.Series(y_pred, index=df.index)
                        signal_count = (raw_signals != 0).sum()
                        meta_debug.update({"signal_count": int(signal_count)})

        except Exception as e:
            # Tránh reentrant call trong stderr bằng cách dùng logger thay vì warnings.warn
            logger = logging.getLogger(__name__)
            logger.warning(f"Lỗi khi chạy ensemble model: {e}", exc_info=False)
            signals = pd.Series(0, index=df.index)
            return StrategyResult(signals=signals, meta={"error": str(e)})

       
        if self.use_ict_filter:
            try:
                ict_ob = detect_order_blocks(df)
                ict_ob_zone = ob_confluence_signal(
                    df,
                    ict_ob["ob_bull_level"],
                    ict_ob["ob_bear_level"],
                    tolerance_pct=self.ict_ob_tolerance_pct,
                )
                fib_df = fib_features(df, lookback=100)

                long_zone = ict_ob_zone["ob_long_zone"] > 0.5
                short_zone = ict_ob_zone["ob_short_zone"] > 0.5

                fib_ok = fib_df["fib_dist_nearest"] < self.ict_fib_max_dist
                fib_ok = fib_ok.fillna(False)

                filtered = raw_signals.copy()
                # LONG: chỉ khi gần OB bullish + gần mức fib
                filtered[(filtered == 1) & ~(long_zone & fib_ok)] = 0
                # SHORT: chỉ khi gần OB bearish + gần mức fib
                filtered[(filtered == -1) & ~(short_zone & fib_ok)] = 0

                raw_signals = filtered
            except Exception as e:
                warnings.warn(f"⚠️ Lỗi khi áp dụng ICT filter (bỏ qua): {e}")

        signals = BaseStrategy.validate_signals(raw_signals, df.index)

        meta = {
            "current_regime": current_regime,
            "allowed_regimes": self.allowed_regimes,
            "proba_threshold": self.proba_threshold,
            "effective_threshold": float(self.model_threshold) if self.model_threshold is not None else self.proba_threshold,
            "use_quantile_threshold": bool(self.use_quantile_threshold),
            "target_signal_rate": float(self.target_signal_rate),
            "model_path": self.model_path,
            "entropy_current": entropy_current,
        }
        
        # Thêm debug info
        meta.update(meta_debug)
        
        return StrategyResult(signals=signals, meta=meta)


class RegimeEnsembleBanditStrategy(BaseStrategy):
    """
    Regime-aware **Dynamic Ensemble** strategy với Multi-Armed Bandit.

    - Có nhiều base models (ensemble nhỏ, pretrained) → mỗi model là một "arm".
    - Mỗi bar, bandit chọn **một** model để dùng, dựa trên reward quá khứ (PnL / direction).
    - Giới hạn compute: mỗi timestep chỉ chạy 1 model → tối ưu CPU nhưng vẫn tận dụng đa mô hình.

    Parameters
    ----------
    model_paths : dict[str, str]
        Mapping tên model → đường dẫn tới file model sklearn (joblib/pickle).
        Ví dụ:
        {
          "rf":  "models/regime_bandit_rf.pkl",
          "gb":  "models/regime_bandit_gb.pkl",
          "log": "models/regime_bandit_logit.pkl",
        }
    proba_threshold : float
        Ngưỡng xác suất để vào lệnh (giống RegimeEnsembleStrategy).
    allowed_regimes : list[str]
        Danh sách regimes được phép trade.
    bandit_type : str
        Loại bandit: 'ucb' (Upper Confidence Bound) hoặc 'eps_greedy'.
    epsilon : float
        Tham số cho epsilon-greedy (nếu dùng).
    reward_mode : str
        'direction' → thưởng 1 nếu đoán đúng hướng (sign), 0 nếu sai.
        'pnl'       → thưởng bằng return * signal (xấp xỉ PnL một bước).
    """

    name = "Regime Ensemble (Bandit)"

    def __init__(
        self,
        model_paths: Dict[str, str],
        proba_threshold: float = 0.40,  # Đã điều chỉnh từ 0.55 xuống 0.40
        allowed_regimes: Optional[list[str]] = None,
        bandit_type: str = "ucb",
        epsilon: float = 0.1,
        reward_mode: str = "direction",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not model_paths:
            raise ValueError("RegimeEnsembleBanditStrategy cần `model_paths` chứa ít nhất 1 model.")

        self.model_paths = model_paths
        self.proba_threshold = proba_threshold
        self.allowed_regimes = allowed_regimes or ["trending", "ranging", "calm"]
        self.bandit_type = bandit_type.lower()
        self.epsilon = epsilon
        self.reward_mode = reward_mode

        self.indicators_list = ["RSI", "MACD", "BB", "ATR", "VWAP", "SMA", "EMA"]

        # Load tất cả models
        self.models: Dict[str, Any] = {}
        self.model_scalers: Dict[str, Any] = {}              
        self.model_feature_names: Dict[str, List[str]] = {}  # Lưu danh sách features đã train (nếu có)
        self._no_feature_warned: set[str] = set()            # Tránh spam cảnh báo khi không align được feature

        for name, path in model_paths.items():
            try:
                # Kiểm tra file có tồn tại không
                path_obj = Path(path)
                if not path_obj.exists():
                    warnings.warn(
                        f"⚠️ Model '{name}' không tìm thấy tại {path}. "
                        f"Bỏ qua model này. "
                        f"{'Chưa huấn luyện stacking model' if 'stacking' in name.lower() else 'Chưa huấn luyện model này'}."
                    )
                    continue
                
                loaded = joblib_load(path)
                # Nếu lưu dạng dict: {"model": ..., "scaler": ..., "feature_names": [...]}
                if isinstance(loaded, dict) and "model" in loaded:
                    self.models[name] = loaded["model"]
                    if "scaler" in loaded:
                        self.model_scalers[name] = loaded["scaler"]
                    # Hỗ trợ cả key "feature_names" và "features"
                    if "feature_names" in loaded and isinstance(loaded["feature_names"], list):
                        self.model_feature_names[name] = list(loaded["feature_names"])
                    elif "features" in loaded and isinstance(loaded["features"], list):
                        self.model_feature_names[name] = list(loaded["features"])
                else:
                    self.models[name] = loaded
            except FileNotFoundError:
                warnings.warn(
                    f"⚠️ Model '{name}' không tìm thấy tại {path}. "
                    f"Bỏ qua model này. "
                    f"{'Chưa huấn luyện stacking model' if 'stacking' in name.lower() else 'Chưa huấn luyện model này'}."
                )
                continue
            except Exception as e:
                warnings.warn(
                    f"⚠️ Lỗi khi load model '{name}' từ {path}: {e}. "
                    f"Bỏ qua model này."
                )
                continue

        if len(self.models) < 1:
            raise ValueError(
                "Không load được model nào cho bandit. "
                "Vui lòng train ít nhất một model trước khi sử dụng bandit strategy."
            )

    # Tái sử dụng helpers từ RegimeEnsembleStrategy (đã được update với enhanced features)
    _calculate_indicators = RegimeEnsembleStrategy._calculate_indicators
    _detect_regime = RegimeEnsembleStrategy._detect_regime
    _build_feature_matrix = RegimeEnsembleStrategy._build_feature_matrix

    # ------------------------------------------------------------------
    # Bandit helpers
    # ------------------------------------------------------------------

    def _select_arm_ucb(self, counts: np.ndarray, values: np.ndarray, t: int) -> int:
        """
        UCB1: chọn arm với upper confidence bound cao nhất.
        counts: số lần đã chọn mỗi arm
        values: reward trung bình mỗi arm
        t: timestep hiện tại (>= 1)
        """
        n_arms = len(counts)
        # Chọn mỗi arm ít nhất một lần đầu tiên
        for i in range(n_arms):
            if counts[i] == 0:
                return i

        # UCB
        ucb = values + np.sqrt(2 * np.log(t) / counts)
        return int(np.argmax(ucb))

    def _select_arm_eps_greedy(self, counts: np.ndarray, values: np.ndarray) -> int:
        n_arms = len(counts)
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, n_arms))
        return int(np.argmax(values))

    # ------------------------------------------------------------------
    # Main generate_signals
    # ------------------------------------------------------------------

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

        closes = df["close"].values
        n = len(df)

        model_names = list(self.models.keys())
        n_models = len(model_names)

        counts = np.zeros(n_models, dtype=float)
        values = np.zeros(n_models, dtype=float)

        signals_arr = np.zeros(n, dtype=float)

        for t in range(n - 1): 
            x_t_df = X.iloc[t : t + 1].copy() 

            # Chọn arm
            if self.bandit_type == "eps_greedy":
                arm = self._select_arm_eps_greedy(counts, values)
            else:  # default UCB
                arm = self._select_arm_ucb(counts, values, t + 1)

            model_name = model_names[arm]
            model = self.models[model_name]
            x_t_aligned = x_t_df
            
            # Determine expected feature count from model itself (most reliable)
            model_expected_features = None
            if hasattr(model, 'n_features_in_'):
                model_expected_features = model.n_features_in_
            elif hasattr(model, 'feature_name_') and model.feature_name_ is not None:
                model_expected_features = len(model.feature_name_)
            
            # Get saved feature names (after feature selection - 100 features)
            feat_names = self.model_feature_names.get(model_name)

            # CRITICAL: Always align *order* to trained feature_names if present.
            # Previous logic only added missing columns but did not reorder, causing feature mismatch and bad probabilities.
            if feat_names and isinstance(feat_names, list) and len(feat_names) > 0:
                # Keep exactly the trained features in correct order; fill missing with 0.0
                x_t_aligned = x_t_aligned.reindex(columns=feat_names).fillna(0.0)
            
            x_t_values_for_scaling = x_t_aligned.values
            
            if model_name in self.model_scalers:
                scaler = self.model_scalers[model_name]
                if hasattr(scaler, 'n_features_in_'):
                    scaler_expected = scaler.n_features_in_
                    if x_t_values_for_scaling.shape[1] != scaler_expected:
                        if x_t_values_for_scaling.shape[1] < scaler_expected:
                            padding = np.zeros((x_t_values_for_scaling.shape[0], scaler_expected - x_t_values_for_scaling.shape[1]))
                            x_t_values_for_scaling = np.hstack([x_t_values_for_scaling, padding])
                        elif x_t_values_for_scaling.shape[1] > scaler_expected:
                            x_t_values_for_scaling = x_t_values_for_scaling[:, :scaler_expected]
                
                x_t_scaled_arr = scaler.transform(x_t_values_for_scaling)
            else:
                x_t_scaled_arr = x_t_values_for_scaling
            
            # STEP 3: Ensure feature count matches model expectation (pad/truncate only as last resort)
            if model_expected_features is not None and x_t_scaled_arr.shape[1] != model_expected_features:
                if x_t_scaled_arr.shape[1] > model_expected_features:
                    # No feat_names but model expects fewer features, truncate
                    x_t_scaled_arr = x_t_scaled_arr[:, :model_expected_features]
                elif x_t_scaled_arr.shape[1] < model_expected_features:
                    # Pad with zeros
                    padding = np.zeros((x_t_scaled_arr.shape[0], model_expected_features - x_t_scaled_arr.shape[1]))
                    x_t_scaled_arr = np.hstack([x_t_scaled_arr, padding])

            if x_t_scaled_arr.shape[1] == 0:
                if model_name not in self._no_feature_warned:
                    warnings.warn(
                        f"Lỗi khi chạy model '{model_name}' trong bandit: không có feature nào sau align. "
                        f"Model sẽ vẫn được bandit chọn nhưng luôn cho tín hiệu 0 (neutral). "
                        f"Hãy kiểm tra lại feature_names khi train model này."
                    )
                    self._no_feature_warned.add(model_name)
                s_t = 0.0
                signals_arr[t] = s_t
                # Update bandit stats using next-bar info (reward=0 for neutral)
                ret_next = (closes[t + 1] / closes[t] - 1.0) if closes[t] != 0 else 0.0
                reward = 0.0 if self.reward_mode != "pnl" else (ret_next * s_t)
                counts[arm] += 1.0
                values[arm] += (reward - values[arm]) / counts[arm]
                continue

            model_type = type(model).__name__
            model_module = type(model).__module__
            model_name_lower = model_name.lower()
            
            is_lgbm = (
                "lgb" in model_name_lower or
                "lightgbm" in model_name_lower or 
                "lightgbm" in model_module.lower() or
                model_type == "LGBMClassifier" or
                model_type.startswith("LGBM") or  
                hasattr(model, "feature_name_")
            )
            is_xgb = (
                "xgb" in model_name_lower or
                "xgboost" in model_name_lower or 
                "xgboost" in model_module.lower() or
                model_type == "XGBClassifier" or
                model_type == "XGBRFClassifier" or
                model_type.startswith("XGB")  
            )
            
            if hasattr(model, "feature_name_") and model.feature_name_ is not None:
                use_dataframe = True
            else:
                use_dataframe = is_lgbm or is_xgb
            
            if use_dataframe:
                # LightGBM/XGBoost cần DataFrame với feature names
                # Ưu tiên: 1) feature_name_ từ model, 2) feat_names từ saved dict, 3) columns từ DataFrame, 4) generic
                if hasattr(model, "feature_name_") and model.feature_name_ is not None:
                    # Model có feature_name_ → dùng nó (chính xác nhất)
                    feat_cols = list(model.feature_name_)
                elif feat_names:
                    # Có feature names từ saved dict
                    feat_cols = feat_names
                elif isinstance(x_t_aligned, pd.DataFrame) and len(x_t_aligned.columns) == x_t_scaled_arr.shape[1]:
                    # Dùng columns từ aligned DataFrame
                    feat_cols = x_t_aligned.columns.tolist()
                else:
                    # Fallback: tạo tên cột generic nếu không có
                    feat_cols = [f"feature_{i}" for i in range(x_t_scaled_arr.shape[1])]
                
                # Đảm bảo số lượng columns khớp
                if len(feat_cols) != x_t_scaled_arr.shape[1]:
                    # Nếu không khớp, thử dùng feature_name_ từ model nếu có
                    if hasattr(model, "feature_name_") and model.feature_name_ is not None:
                        model_feat_names = list(model.feature_name_)
                        if len(model_feat_names) == x_t_scaled_arr.shape[1]:
                            feat_cols = model_feat_names
                        else:
                            feat_cols = [f"feature_{i}" for i in range(x_t_scaled_arr.shape[1])]
                    else:
                        feat_cols = [f"feature_{i}" for i in range(x_t_scaled_arr.shape[1])]
                
                # Tạo DataFrame với index phù hợp
                if isinstance(x_t_aligned, pd.DataFrame):
                    df_index = x_t_aligned.index
                else:
                    df_index = [0]  # Single row index
                
                x_t_scaled = pd.DataFrame(x_t_scaled_arr, columns=feat_cols, index=df_index)
            else:
                # Sklearn models dùng numpy array
                x_t_scaled = x_t_scaled_arr
            if is_lgbm and not isinstance(x_t_scaled, pd.DataFrame):
                if feat_names and isinstance(feat_names, list) and len(feat_names) == x_t_scaled_arr.shape[1]:
                    feat_cols = feat_names
                elif hasattr(model, "feature_name_") and model.feature_name_ is not None and len(model.feature_name_) == x_t_scaled_arr.shape[1]:
                    feat_cols = list(model.feature_name_)
                else:
                    feat_cols = [f"feature_{i}" for i in range(x_t_scaled_arr.shape[1])]
                x_t_scaled = pd.DataFrame(x_t_scaled_arr, columns=feat_cols, index=x_t_aligned.index)
            try:
                if hasattr(model, "predict_proba"):
                    if is_lgbm:
                        proba = model.predict_proba(x_t_scaled, validate_features=False)[0]
                    else:
                        proba = model.predict_proba(x_t_scaled)[0]
                    classes = list(getattr(model, "classes_", []))
                    p_short = 0.0
                    p_long = 0.0
                    p_neutral = 0.0

                    # Binary legacy
                    if proba.shape[0] == 2 and not classes:
                        p_short, p_long = float(proba[0]), float(proba[1])
                    else:
                        if classes and len(classes) == proba.shape[0]:
                            idx_long = classes.index(1) if 1 in classes else None
                            idx_short = classes.index(-1) if -1 in classes else None
                            idx_neutral = classes.index(0) if 0 in classes else None
                            p_long = float(proba[idx_long]) if idx_long is not None else 0.0
                            p_short = float(proba[idx_short]) if idx_short is not None else 0.0
                            p_neutral = float(proba[idx_neutral]) if idx_neutral is not None else 0.0
                        elif proba.shape[0] == 2:
                            p_short, p_long = float(proba[0]), float(proba[1])

                    # Thresholding: if neutral exists, require directional win neutral (reduces false signals)
                    if (p_long >= self.proba_threshold) and (p_long > p_short) and (p_long >= p_neutral):
                        s_t = 1.0
                    elif (p_short >= self.proba_threshold) and (p_short > p_long) and (p_short >= p_neutral):
                        s_t = -1.0
                    else:
                        s_t = 0.0
                else:
                    # Fallback: model.predict trả ra -1/0/1
                    y_pred = model.predict(x_t_scaled)[0]
                    s_t = float(np.clip(y_pred, -1, 1))
            except Exception as e:
                warnings.warn(f"Lỗi khi chạy model '{model_name}' trong bandit: {e}")
                s_t = 0.0

            signals_arr[t] = s_t

            # Tính reward dùng thông tin ở t+1 (không leak tương lai)
            if t < n - 1:
                ret_next = (closes[t + 1] / closes[t] - 1.0) if closes[t] != 0 else 0.0
                if self.reward_mode == "pnl":
                    reward = ret_next * s_t
                else:  # 'direction'
                    reward = 1.0 if np.sign(ret_next) == np.sign(s_t) and s_t != 0 else 0.0

                # Cập nhật bandit stats
                counts[arm] += 1.0
                # Running average
                values[arm] += (reward - values[arm]) / counts[arm]

        raw_signals = pd.Series(signals_arr, index=df.index)
        signals = BaseStrategy.validate_signals(raw_signals, df.index)

        # Convert counts và values thành dict để dễ hiển thị trong UI
        counts_dict = {model_names[i]: int(counts[i]) for i in range(n_models)}
        values_dict = {model_names[i]: float(values[i]) for i in range(n_models)}
        
        # Tìm model được chọn nhiều nhất
        selected_model = model_names[int(np.argmax(counts))] if counts.sum() > 0 else None

        meta = {
            "current_regime": current_regime,
            "allowed_regimes": self.allowed_regimes,
            "proba_threshold": self.proba_threshold,
            "model_paths": self.model_paths,
            "bandit_type": self.bandit_type,
            "epsilon": self.epsilon,
            "reward_mode": self.reward_mode,
            "bandit_counts": counts_dict,  
            "bandit_values": values_dict, 
            "selected_model": selected_model,
            "n_bars": int(n),
            "signal_count": int((signals != 0).sum()),
        }
        return StrategyResult(signals=signals, meta=meta)


