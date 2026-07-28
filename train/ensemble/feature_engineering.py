"""
Feature Engineering nâng cao cho Regime Ensemble models.

Tách từ train_regime_ensemble_models_advanced.py

Includes:
- calculate_indicators_enhanced: 40+ technical indicators
- detect_regime_optimized: HMM/rule-based regime detection
- build_feature_matrix_enhanced: Feature matrix với lagged, rolling, interaction features
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

from algo_trading.indicators import (
    rsi, macd, bollinger_bands, atr, vwap, sma, ema,
)

try:
    from algo_trading.indicators.ict import (
        detect_order_blocks, ob_confluence_signal, fib_features,
    )
    HAS_ICT = True
except ImportError:
    HAS_ICT = False

try:
    from algo_trading.market_models.regime import detect_regime_hmm
    HAS_HMM = True
except ImportError:
    detect_regime_hmm = None
    HAS_HMM = False


def calculate_indicators_enhanced(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Tính indicators nâng cao với nhiều features hơn.
    
    Returns:
        Dict mapping indicator names to Series.
    """
    indicators: Dict[str, pd.Series] = {}
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    volume = df.get("volume", None)

    # === Basic Indicators ===
    indicators["rsi"] = rsi(close, 14)
    indicators["rsi_9"] = rsi(close, 9)
    indicators["rsi_21"] = rsi(close, 21)

    macd_line, macd_signal, macd_hist = macd(close)
    indicators["macd_line"] = macd_line
    indicators["macd_signal"] = macd_signal
    indicators["macd_hist"] = macd_hist

    bb_upper, bb_middle, bb_lower = bollinger_bands(close)
    indicators["bb_upper"] = bb_upper
    indicators["bb_lower"] = bb_lower
    indicators["bb_width"] = (bb_upper - bb_lower) / bb_middle
    indicators["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)

    atr_val = atr(df, 14)
    indicators["atr"] = atr_val
    indicators["atr_ratio"] = atr_val / close
    indicators["atr_20"] = atr(df, 20)

    if volume is not None:
        vwap_val = vwap(df)
        indicators["vwap"] = vwap_val
        indicators["vwap_distance"] = (close - vwap_val) / vwap_val

    indicators["sma_20"] = sma(close, 20)
    indicators["sma_50"] = sma(close, 50)
    indicators["sma_100"] = sma(close, 100)
    indicators["ema_20"] = ema(close, 20)
    indicators["ema_50"] = ema(close, 50)
    indicators["ema_200"] = ema(close, 200)

    # === Momentum Features ===
    indicators["momentum_5"] = close.pct_change(5)
    indicators["momentum_10"] = close.pct_change(10)
    indicators["momentum_20"] = close.pct_change(20)
    indicators["roc_10"] = (close / close.shift(10) - 1) * 100

    # === Volume Features ===
    if volume is not None:
        indicators["volume_ma_20"] = volume.rolling(20).mean()
        indicators["volume_ratio"] = volume / indicators["volume_ma_20"]
        indicators["volume_trend"] = volume.rolling(5).mean() / indicators["volume_ma_20"]
        price_change = close.diff()
        obv = (volume * np.sign(price_change)).cumsum()
        indicators["obv"] = obv
        indicators["obv_ma"] = obv.rolling(20).mean()
        indicators["obv_ratio"] = obv / indicators["obv_ma"]

    # === Volatility Features ===
    returns = close.pct_change()
    indicators["volatility_5"] = returns.rolling(5).std()
    indicators["volatility_20"] = returns.rolling(20).std()
    indicators["volatility_ratio"] = indicators["volatility_5"] / indicators["volatility_20"]

    # === Fibonacci Retracement Levels ===
    recent_high = high.rolling(50).max()
    recent_low = low.rolling(50).min()
    diff = recent_high - recent_low
    indicators["fib_236"] = recent_high - diff * 0.236
    indicators["fib_382"] = recent_high - diff * 0.382
    indicators["fib_500"] = recent_high - diff * 0.5
    indicators["fib_618"] = recent_high - diff * 0.618
    indicators["fib_786"] = recent_high - diff * 0.786
    indicators["dist_fib_236"] = (close - indicators["fib_236"]) / close
    indicators["dist_fib_382"] = (close - indicators["fib_382"]) / close
    indicators["dist_fib_500"] = (close - indicators["fib_500"]) / close
    indicators["dist_fib_618"] = (close - indicators["fib_618"]) / close
    indicators["dist_fib_786"] = (close - indicators["fib_786"]) / close

    # === Cross-Indicator Features ===
    indicators["rsi_macd_divergence"] = indicators["rsi"] - (indicators["macd_hist"] * 100)
    indicators["bb_rsi_interaction"] = indicators["bb_position"] * (indicators["rsi"] / 100)

    return indicators


def detect_regime_optimized(
    df: pd.DataFrame, indicators: Dict[str, pd.Series], lookback_window: int = 500
) -> Dict:
    """
    Regime detection với parameters được tối ưu.
    Sử dụng HMM nếu có, fallback về rule-based.
    """
    def simple_regime_optimized() -> Dict:
        macd_hist = indicators.get("macd_hist", pd.Series(0, index=df.index))
        rsi_val = indicators.get("rsi", pd.Series(50, index=df.index))
        bb_width = indicators.get("bb_width", pd.Series(0.02, index=df.index))
        vol = indicators.get("volatility_20", pd.Series(0.01, index=df.index))

        vol_threshold = vol.rolling(20).quantile(0.5)
        macd_threshold = 0.01 * (1 + vol / vol_threshold)
        bb_threshold_high = 0.08 * (1 + vol / vol_threshold)

        trending = (macd_hist > macd_threshold) & (rsi_val > 30) & (rsi_val < 70)
        ranging = (macd_hist.abs() < macd_threshold * 0.5) & (bb_width < 0.05)
        volatile = (bb_width > bb_threshold_high) | (vol > vol_threshold * 1.5)
        calm = (bb_width < 0.03) & ~trending & (vol < vol_threshold * 0.5)

        regime_id = pd.Series(0, index=df.index)
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

    if HAS_HMM and detect_regime_hmm is not None:
        try:
            return detect_regime_hmm(df, indicators=indicators, lookback_window=lookback_window)
        except Exception:
            return simple_regime_optimized()
    return simple_regime_optimized()


def build_feature_matrix_enhanced(
    df: pd.DataFrame, indicators: Dict[str, pd.Series], regime_info: Dict
) -> pd.DataFrame:
    """
    Xây dựng feature matrix nâng cao với lagged, rolling, và interaction features.
    """
    feats: Dict[str, pd.Series] = {}
    close = df["close"]

    # === Basic Returns ===
    feats["ret_1"] = close.pct_change().fillna(0)
    feats["ret_5"] = close.pct_change(5).fillna(0)
    feats["ret_10"] = close.pct_change(10).fillna(0)
    feats["ret_20"] = close.pct_change(20).fillna(0)

    # === Indicators ===
    for k, v in indicators.items():
        feats[f"ind_{k}"] = v

    # === Lagged Features ===
    for lag in [1, 2, 3, 5]:
        feats[f"ret_lag{lag}"] = feats["ret_1"].shift(lag).fillna(0)
        if "rsi" in indicators:
            feats[f"rsi_lag{lag}"] = indicators["rsi"].shift(lag).fillna(50)

    # === Rolling Statistics ===
    for window in [5, 10, 20]:
        feats[f"ret_ma{window}"] = feats["ret_1"].rolling(window).mean().fillna(0)
        feats[f"ret_std{window}"] = feats["ret_1"].rolling(window).std().fillna(0)

    # === Regime Features ===
    regime_series = regime_info.get("regime", None)
    if isinstance(regime_series, pd.Series):
        reg_ids = regime_series.astype(int)
        for rid, name in enumerate(["trending", "ranging", "volatile", "calm"]):
            feats[f"regime_{name}"] = (reg_ids == rid).astype(float)
            feats[f"regime_{name}_persist"] = (
                (reg_ids == rid).astype(int).groupby((reg_ids != rid).cumsum()).cumsum()
            )
    else:
        for name in ["trending", "ranging", "volatile", "calm"]:
            feats[f"regime_{name}"] = 1.0 if name == "trending" else 0.0
            feats[f"regime_{name}_persist"] = 0.0

    # === Interaction Features ===
    if "ind_rsi" in feats and "ind_macd_hist" in feats:
        feats["rsi_macd_interaction"] = feats["ind_rsi"] * feats["ind_macd_hist"]
    if "ind_bb_position" in feats and "ind_rsi" in feats:
        feats["bb_rsi_interaction"] = feats["ind_bb_position"] * (feats["ind_rsi"] / 100)

    # === ICT Features ===
    if HAS_ICT:
        try:
            ict_ob = detect_order_blocks(df)
            ict_ob_zone = ob_confluence_signal(
                df, ict_ob["ob_bull_level"], ict_ob["ob_bear_level"], tolerance_pct=0.002,
            )
            fib_df = fib_features(df, lookback=100)
            for k, v in ict_ob.items():
                feats[f"ict_{k}"] = v
            for k, v in ict_ob_zone.items():
                feats[f"ict_{k}"] = v
            for col in fib_df.columns:
                feats[f"ict_{col}"] = fib_df[col]
        except Exception as e:
            warnings.warn(f"⚠️ Lỗi khi tạo ICT features (bỏ qua): {e}")

    X = pd.DataFrame(feats).ffill().bfill()
    return X
