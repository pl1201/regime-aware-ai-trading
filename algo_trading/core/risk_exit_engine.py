from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd


@dataclass
class RegimeATRStops:
    sl_atr_k: float
    tp_atr_k: float


@dataclass
class RiskExitEngineConfig:

    atr_col: str = "ATR14"

    regime_atr: Dict[str, RegimeATRStops] = None  

    # Trailing stop settings (ATR-based)
    trailing_atr_k: Optional[float] = 2.0

    # Breakeven settings (ATR-based)
    breakeven_trigger_atr: Optional[float] = 1.0
    breakeven_buffer_atr: float = 0.0

    # Exit filters
    exit_on_regime_change: bool = True
    exit_on_trend_consensus: bool = False
    trend_consensus_min_long: float = 0.55
    trend_consensus_max_short: float = 0.45

    def __post_init__(self):
        if self.regime_atr is None:

            self.regime_atr = {
                "trending": RegimeATRStops(sl_atr_k=1.5, tp_atr_k=3.0),
                "ranging": RegimeATRStops(sl_atr_k=1.3, tp_atr_k=1.8),
                "volatile": RegimeATRStops(sl_atr_k=2.2, tp_atr_k=3.2),
                "calm": RegimeATRStops(sl_atr_k=1.0, tp_atr_k=1.6),
            }


def compute_trend_consensus(
    df: pd.DataFrame,
    base_tf: str = "1h",
    tf_4h: str = "4h",
    tf_1d: str = "1d",
) -> pd.Series:
    """
    Compute simple multi-timeframe trend consensus in [0, 1].
    1 = all timeframes trending up, 0 = all trending down (or mostly down).

    Implementation:
    - trend_tf = 1 if close_tf > close_tf.shift(1) else 0
    - consensus = mean(trend_1h, trend_4h_aligned, trend_1d_aligned)
    """
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)

    idx = df.index
    close_1h = df["close"]

    # 1h trend
    trend_1h = (close_1h > close_1h.shift(1)).astype(float)

    # 4h trend (resample from 1h)
    close_4h = close_1h.resample("4H", label="right", closed="right").last().ffill()
    trend_4h = (close_4h > close_4h.shift(1)).astype(float)
    trend_4h_aligned = trend_4h.reindex(idx, method="ffill")

    # 1d trend (resample from 1h)
    close_1d = close_1h.resample("1D", label="right", closed="right").last().ffill()
    trend_1d = (close_1d > close_1d.shift(1)).astype(float)
    trend_1d_aligned = trend_1d.reindex(idx, method="ffill")

    consensus = (trend_1h + trend_4h_aligned + trend_1d_aligned) / 3.0
    return consensus.fillna(0.5)


def ensure_atr14(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has ATR14 column; computes a simple ATR(14) if missing."""
    if df is None or df.empty:
        return df
    if "ATR14" in df.columns:
        return df
    if not all(c in df.columns for c in ["high", "low", "close"]):
        df["ATR14"] = 0.0
        return df

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean().fillna(0.0)
    return df


def risk_exit_check_intrabar(
    *,
    broker: Any,
    df: pd.DataFrame,
    idx: int,
    cfg: Any,
    signals: pd.Series,
    regime_series: Optional[pd.Series],
    trend_consensus: Optional[pd.Series],
    engine: RiskExitEngineConfig,
) -> Optional[str]:
    if broker.position == 0 or broker.current_trade is None:
        return None

    # Current bar prices
    high = df[cfg.high_col].iloc[idx]
    low = df[cfg.low_col].iloc[idx]
    close = df[cfg.price_col].iloc[idx]
    ts = df.index[idx]

    # Get ATR
    atr = 0.0
    if engine.atr_col in df.columns:
        atr = float(df[engine.atr_col].iloc[idx] or 0.0)

    # Determine current regime
    regime = None
    if regime_series is not None and idx < len(regime_series):
        regime = str(regime_series.iloc[idx])
    if not regime or regime not in engine.regime_atr:
        regime = "trending"
    regime_params = engine.regime_atr[regime]

    # Initialize trade state on entry
    trade = broker.current_trade
    if trade.get("entry_regime") is None:
        trade["entry_regime"] = regime
    trade["current_regime"] = regime

    # Maintain peak/trough for trailing
    if broker.position > 0:
        trade["peak_price"] = max(float(trade.get("peak_price", broker.entry_price)), float(high))
    else:
        trade["trough_price"] = min(float(trade.get("trough_price", broker.entry_price)), float(low))

    # Base SL/TP from regime ATR multipliers
    if broker.position > 0:
        base_sl = broker.entry_price - regime_params.sl_atr_k * atr
        base_tp = broker.entry_price + regime_params.tp_atr_k * atr
    else:
        base_sl = broker.entry_price + regime_params.sl_atr_k * atr
        base_tp = broker.entry_price - regime_params.tp_atr_k * atr

    # Trailing stop (ATR-based) – uses peak/trough
    trailing_sl = None
    if engine.trailing_atr_k and atr > 0:
        if broker.position > 0:
            peak = float(trade.get("peak_price", broker.entry_price))
            trailing_sl = peak - engine.trailing_atr_k * atr
        else:
            trough = float(trade.get("trough_price", broker.entry_price))
            trailing_sl = trough + engine.trailing_atr_k * atr

    breakeven_sl = None
    if engine.breakeven_trigger_atr and atr > 0:
        if broker.position > 0:
            if (close - broker.entry_price) >= engine.breakeven_trigger_atr * atr:
                breakeven_sl = broker.entry_price + engine.breakeven_buffer_atr * atr
        else:
            if (broker.entry_price - close) >= engine.breakeven_trigger_atr * atr:
                breakeven_sl = broker.entry_price - engine.breakeven_buffer_atr * atr

    # Final SL = max of candidates for long; min of candidates for short
    sl_candidates = [c for c in [base_sl, trailing_sl, breakeven_sl] if c is not None and not np.isnan(c)]
    if broker.position > 0:
        final_sl = max(sl_candidates) if sl_candidates else None
    else:
        final_sl = min(sl_candidates) if sl_candidates else None

    trade["stop_loss"] = float(final_sl) if final_sl is not None else None
    trade["take_profit"] = float(base_tp) if base_tp is not None else None

    # Exit on regime change
    if engine.exit_on_regime_change:
        entry_regime = str(trade.get("entry_regime", regime))
        if regime != entry_regime:
            broker.exit_position(close, idx, ts, reason="regime_change")
            return "regime_change"

    # Exit on trend-consensus deterioration
    if engine.exit_on_trend_consensus and trend_consensus is not None and idx < len(trend_consensus):
        tc = float(trend_consensus.iloc[idx])
        if broker.position > 0 and tc < engine.trend_consensus_min_long:
            broker.exit_position(close, idx, ts, reason="trend_consensus_exit")
            return "trend_consensus_exit"
        if broker.position < 0 and tc > engine.trend_consensus_max_short:
            broker.exit_position(close, idx, ts, reason="trend_consensus_exit")
            return "trend_consensus_exit"

    # Intrabar SL/TP checks
    sl_price = trade.get("stop_loss", None)
    tp_price = trade.get("take_profit", None)

    if broker.position > 0:
        if sl_price is not None and low <= sl_price:
            broker.exit_position(min(float(sl_price), float(close)), idx, ts, reason="sl")
            return "sl"
        if tp_price is not None and high >= tp_price:
            broker.exit_position(max(float(tp_price), float(close)), idx, ts, reason="tp")
            return "tp"
    else:
        if sl_price is not None and high >= sl_price:
            broker.exit_position(max(float(sl_price), float(close)), idx, ts, reason="sl")
            return "sl"
        if tp_price is not None and low <= tp_price:
            broker.exit_position(min(float(tp_price), float(close)), idx, ts, reason="tp")
            return "tp"

    return None

