"""
Threshold Scoring & Optimization cho trading signals.

Tách từ train_regime_ensemble_models_advanced.py

Includes:
- _score_trading_threshold: Basic trading objective
- _score_trading_threshold_constrained: Constrained objective (min trades, dir accuracy)
- _score_trading_threshold_with_density: Density-regularized scoring
- optimize_threshold_trading_objective: Grid search for best threshold
- optimize_threshold_walk_forward: Walk-forward robust threshold selection
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _score_trading_threshold(
    y_proba: np.ndarray, forward_returns: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Score a probability threshold based on trading PnL."""
    long_mask = y_proba >= threshold
    short_mask = y_proba <= (1.0 - threshold)
    signal = np.zeros(len(y_proba), dtype=float)
    signal[long_mask] = 1.0
    signal[short_mask] = -1.0

    trade_mask = signal != 0
    if trade_mask.sum() == 0:
        return {"score": -1e9, "expectancy": 0.0, "profit_factor": 0.0, "trade_rate": 0.0}

    pnl = signal[trade_mask] * forward_returns[trade_mask]
    pos = pnl[pnl > 0].sum()
    neg = pnl[pnl < 0].sum()
    expectancy = float(np.mean(pnl))
    profit_factor = float(pos / (abs(neg) + 1e-12))
    trade_rate = float(trade_mask.mean())

    score = expectancy * 1000.0 + 0.25 * profit_factor + 0.05 * trade_rate
    return {
        "score": float(score),
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "trade_rate": trade_rate,
    }


def _score_trading_threshold_constrained(
    y_proba: np.ndarray,
    forward_returns: np.ndarray,
    threshold: float,
    cost_per_trade: float = 0.001,
    min_trade_rate: float = 0.03,
    target_trade_rate: float = 0.05,
    max_trade_rate: float = 0.25,
    min_trades: int = 120,
    min_directional_accuracy: float = 0.52,
    pf_cap: float = 5.0,
    min_win_trades: int = 20,
    min_loss_trades: int = 20,
) -> Dict[str, float]:
    """
    Constrained trading objective with hard constraints on trade count,
    directional accuracy, and trade rate.
    """
    long_mask = y_proba >= threshold
    short_mask = y_proba <= (1.0 - threshold)
    signal = np.zeros(len(y_proba), dtype=float)
    signal[long_mask] = 1.0
    signal[short_mask] = -1.0

    trade_mask = signal != 0
    n_trades = int(trade_mask.sum())
    trade_rate = float(trade_mask.mean()) if len(signal) else 0.0
    if n_trades == 0:
        return {
            "score": -1e9, "expectancy": 0.0, "profit_factor": 0.0,
            "trade_rate": 0.0, "n_trades": 0, "dir_acc": 0.0,
            "win_trades": 0, "loss_trades": 0, "feasible": False,
        }

    gross_pnl = signal[trade_mask] * forward_returns[trade_mask]
    net_pnl = gross_pnl - float(cost_per_trade)
    expectancy = float(np.mean(net_pnl))

    win_mask = net_pnl > 0
    loss_mask = net_pnl < 0
    win_trades = int(win_mask.sum())
    loss_trades = int(loss_mask.sum())

    pos = float(net_pnl[win_mask].sum()) if win_trades > 0 else 0.0
    neg = float(net_pnl[loss_mask].sum()) if loss_trades > 0 else 0.0
    profit_factor = float(pos / (abs(neg) + 1e-12)) if loss_trades > 0 else float("inf")
    profit_factor_capped = float(min(profit_factor, pf_cap)) if np.isfinite(profit_factor) else float(pf_cap)

    y_sign = np.where(np.asarray(forward_returns, dtype=float) >= 0.0, 1.0, -1.0)
    dir_acc = float((signal[trade_mask] == y_sign[trade_mask]).mean()) if n_trades > 0 else 0.0

    feasible = True
    if trade_rate < min_trade_rate or trade_rate > max_trade_rate:
        feasible = False
    if n_trades < min_trades:
        feasible = False
    if dir_acc < min_directional_accuracy:
        feasible = False
    if win_trades < min_win_trades or loss_trades < min_loss_trades:
        feasible = False

    score = expectancy * 1000.0 + 0.25 * profit_factor_capped
    score -= 1.0 * abs(trade_rate - target_trade_rate)
    if not feasible:
        score -= 1e6

    return {
        "score": float(score), "expectancy": float(expectancy),
        "profit_factor": float(profit_factor), "trade_rate": float(trade_rate),
        "n_trades": int(n_trades), "dir_acc": float(dir_acc),
        "win_trades": int(win_trades), "loss_trades": int(loss_trades),
        "feasible": bool(feasible),
    }


def _score_trading_threshold_with_density(
    y_proba: np.ndarray,
    forward_returns: np.ndarray,
    threshold: float,
    min_trade_rate: float = 0.01,
    target_trade_rate: float = 0.03,
    max_trade_rate: float = 0.25,
) -> Dict[str, float]:
    """Trading objective with density regularization."""
    base = _score_trading_threshold(y_proba, forward_returns, threshold)
    tr = float(base["trade_rate"])
    score = float(base["score"])

    if tr < min_trade_rate:
        score -= 3.0 * (min_trade_rate - tr)
    score -= 0.8 * abs(tr - target_trade_rate)
    if tr > max_trade_rate:
        score -= 0.6 * (tr - max_trade_rate)

    base["score"] = float(score)
    return base


def optimize_threshold_trading_objective(
    y_proba: np.ndarray,
    forward_returns: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    min_trade_rate: float = 0.03,
    target_trade_rate: float = 0.05,
    max_trade_rate: float = 0.25,
    min_trades: int = 120,
    min_directional_accuracy: float = 0.52,
    cost_per_trade: float = 0.001,
    pf_cap: float = 5.0,
    min_win_trades: int = 20,
    min_loss_trades: int = 20,
) -> Tuple[float, Dict[str, float]]:
    """Grid search for best trading threshold."""
    if thresholds is None:
        thresholds = np.arange(0.42, 0.91, 0.01)

    best_th = 0.5
    best_metrics: Dict[str, float] = {
        "score": -1e9, "expectancy": 0.0, "profit_factor": 0.0,
        "trade_rate": 0.0, "n_trades": 0, "dir_acc": 0.0,
        "win_trades": 0, "loss_trades": 0, "feasible": False,
    }
    best_feasible_th = None
    best_feasible_metrics = None
    best_density_th = None
    best_density_gap = 1e9
    best_density_metrics = None

    for th in thresholds:
        m = _score_trading_threshold_constrained(
            y_proba, forward_returns, float(th),
            cost_per_trade=cost_per_trade,
            min_trade_rate=min_trade_rate,
            target_trade_rate=target_trade_rate,
            max_trade_rate=max_trade_rate,
            min_trades=min_trades,
            min_directional_accuracy=min_directional_accuracy,
            pf_cap=pf_cap,
            min_win_trades=min_win_trades,
            min_loss_trades=min_loss_trades,
        )

        if m["score"] > best_metrics["score"]:
            best_metrics = m
            best_th = float(th)

        if bool(m.get("feasible", False)):
            if best_feasible_metrics is None or m["score"] > best_feasible_metrics["score"]:
                best_feasible_metrics = m
                best_feasible_th = float(th)

        tr = float(m.get("trade_rate", 0.0))
        density_gap = abs(tr - target_trade_rate)
        if density_gap < best_density_gap:
            best_density_gap = density_gap
            best_density_th = float(th)
            best_density_metrics = m

    if best_feasible_metrics is not None and best_feasible_th is not None:
        out = dict(best_feasible_metrics)
        out["selection_mode"] = "feasible"
        return float(best_feasible_th), out

    if best_density_metrics is not None and best_density_th is not None:
        out = dict(best_density_metrics)
        out["selection_mode"] = "density_fallback"
        out["feasible"] = False
        return float(best_density_th), out

    out = dict(best_metrics)
    out["selection_mode"] = "global_fallback"
    return best_th, out


def optimize_threshold_walk_forward(
    y_proba: np.ndarray,
    forward_returns: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    n_windows: int = 4,
    min_trade_rate: float = 0.03,
    target_trade_rate: float = 0.05,
    max_trade_rate: float = 0.25,
    min_trades: int = 120,
    min_directional_accuracy: float = 0.52,
    cost_per_trade: float = 0.001,
    pf_cap: float = 5.0,
    min_win_trades: int = 20,
    min_loss_trades: int = 20,
) -> Tuple[float, Dict[str, float]]:
    """Walk-forward robust threshold selection across multiple windows."""
    if thresholds is None:
        thresholds = np.arange(0.42, 0.91, 0.01)

    n = len(y_proba)
    if n < max(200, n_windows * 40):
        return optimize_threshold_trading_objective(
            y_proba=y_proba, forward_returns=forward_returns,
            thresholds=thresholds, min_trade_rate=min_trade_rate,
            target_trade_rate=target_trade_rate, max_trade_rate=max_trade_rate,
            min_trades=min_trades, min_directional_accuracy=min_directional_accuracy,
            cost_per_trade=cost_per_trade, pf_cap=pf_cap,
            min_win_trades=min_win_trades, min_loss_trades=min_loss_trades,
        )

    bounds = np.linspace(0, n, n_windows + 1, dtype=int)
    best_th = 0.5
    best: Dict[str, float] = {
        "score": -1e9, "expectancy": 0.0, "profit_factor": 0.0,
        "trade_rate": 0.0, "n_trades": 0, "dir_acc": 0.0,
        "win_trades": 0, "loss_trades": 0, "feasible": False,
        "wf_expectancy_median": 0.0, "wf_expectancy_worst": 0.0,
        "wf_feasible_ratio": 0.0,
    }
    best_feasible_th = None
    best_feasible = None

    for th in thresholds:
        window_metrics = []
        for i in range(n_windows):
            s, e = bounds[i], bounds[i + 1]
            yp = y_proba[s:e]
            fr = forward_returns[s:e]
            if len(yp) < 20:
                continue
            local_min_trades = max(20, int(len(yp) * min_trade_rate * 0.7))
            m = _score_trading_threshold_constrained(
                yp, fr, float(th),
                cost_per_trade=cost_per_trade,
                min_trade_rate=min_trade_rate * 0.7,
                target_trade_rate=target_trade_rate,
                max_trade_rate=max_trade_rate,
                min_trades=local_min_trades,
                min_directional_accuracy=min_directional_accuracy,
                pf_cap=pf_cap,
                min_win_trades=max(5, min_win_trades // 3),
                min_loss_trades=max(5, min_loss_trades // 3),
            )
            window_metrics.append(m)

        if not window_metrics:
            continue

        exps = [float(wm["expectancy"]) for wm in window_metrics]
        feas = [bool(wm.get("feasible", False)) for wm in window_metrics]
        med_exp = float(np.median(exps))
        worst_exp = float(min(exps))
        feas_ratio = float(sum(feas) / len(feas))

        wf_score = med_exp * 1000.0
        wf_score += 0.3 * worst_exp * 1000.0
        if worst_exp < -0.0005:
            wf_score -= 500.0
        wf_score += 0.5 * feas_ratio

        candidate = {
            "score": float(wf_score),
            "expectancy": med_exp,
            "profit_factor": float(np.median([wm["profit_factor"] for wm in window_metrics])),
            "trade_rate": float(np.median([wm["trade_rate"] for wm in window_metrics])),
            "n_trades": int(sum(wm.get("n_trades", 0) for wm in window_metrics)),
            "dir_acc": float(np.median([wm.get("dir_acc", 0.5) for wm in window_metrics])),
            "win_trades": int(sum(wm.get("win_trades", 0) for wm in window_metrics)),
            "loss_trades": int(sum(wm.get("loss_trades", 0) for wm in window_metrics)),
            "feasible": feas_ratio >= 0.5,
            "wf_expectancy_median": med_exp,
            "wf_expectancy_worst": worst_exp,
            "wf_feasible_ratio": feas_ratio,
        }

        if candidate["score"] > best["score"]:
            best = candidate
            best_th = float(th)

        if feas_ratio >= 0.5 and worst_exp > -0.0005:
            if best_feasible is None or candidate["score"] > best_feasible["score"]:
                best_feasible = candidate
                best_feasible_th = float(th)

    if best_feasible is not None and best_feasible_th is not None:
        out = dict(best_feasible)
        out["selection_mode"] = "walk_forward_feasible"
        return float(best_feasible_th), out

    out = dict(best)
    out["selection_mode"] = "walk_forward_global"
    return best_th, out
