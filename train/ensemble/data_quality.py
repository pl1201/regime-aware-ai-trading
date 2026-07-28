"""
Data Quality checks và Feature Contract cho drift detection.

Tách từ train_regime_ensemble_models_advanced.py

Includes:
- data_quality_report: NaN, Inf, zero-variance, class distribution checks
- build_feature_contract: Lưu train distribution stats + PSI bins
- check_feature_skew_against_contract: PSI + z-shift detection
"""

from __future__ import annotations

from typing import Dict, List, Any

import numpy as np
import pandas as pd


def _compute_psi_from_edges(train_arr: np.ndarray, live_arr: np.ndarray, edges: np.ndarray) -> float:
    """Population Stability Index theo bins cố định từ train."""
    if edges is None or len(edges) < 2:
        return 0.0
    train_hist, _ = np.histogram(train_arr, bins=edges)
    live_hist, _ = np.histogram(live_arr, bins=edges)
    train_pct = train_hist.astype(float) / max(train_hist.sum(), 1)
    live_pct = live_hist.astype(float) / max(live_hist.sum(), 1)
    eps = 1e-8
    train_pct = np.clip(train_pct, eps, None)
    live_pct = np.clip(live_pct, eps, None)
    return float(np.sum((live_pct - train_pct) * np.log(live_pct / train_pct)))


def data_quality_report(X: pd.DataFrame, y: np.ndarray) -> None:
    """In report chất lượng dữ liệu: NaN, Inf, zero-variance, class distribution."""
    n_samples, n_feats = X.shape
    print("\n" + "=" * 80)
    print("🩺 DATA QUALITY REPORT")
    print("=" * 80)
    print(f"   Samples: {n_samples}, Features: {n_feats}")

    nan_counts = X.isna().sum().sum()
    inf_counts = np.isinf(X.values).sum()
    zero_var = (X.std() == 0).sum()
    print(f"   NaN total: {nan_counts}")
    print(f"   Inf total: {inf_counts}")
    print(f"   Zero-variance features: {zero_var}")

    uniq, cnts = np.unique(y, return_counts=True)
    dist = dict(zip(uniq.astype(int), cnts))
    print(f"   Label distribution: {dist}")
    if len(uniq) < 2:
        print("   ⚠️ Chỉ có 1 class → F1 sẽ = 0 và models khó học.")

    pct_zero = (y == 0).mean()
    if pct_zero > 0.8:
        print(f"   ⚠️ {pct_zero:.1%} samples là class 0.")

    dup_cols = X.columns[X.columns.duplicated()].tolist()
    if dup_cols:
        print(f"   ⚠️ {len(dup_cols)} duplicated feature names: {dup_cols[:5]} ...")


def build_feature_contract(X_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Tạo feature contract để khóa train/inference parity.
    Contract gồm: thứ tự cột, stats train, và bins cho PSI.
    """
    contract: Dict[str, Any] = {
        "version": 1,
        "feature_names": list(X_train.columns),
        "n_features": int(X_train.shape[1]),
        "stats": {},
    }

    quantiles = np.linspace(0.0, 1.0, 11)
    for col in X_train.columns:
        s = pd.to_numeric(X_train[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        q = np.quantile(s.values, quantiles)
        q_unique = np.unique(q)
        if len(q_unique) < 2:
            q_unique = np.array([float(s.min()), float(s.max()) + 1e-6])

        contract["stats"][col] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "median": float(s.median()),
            "iqr": float(np.quantile(s.values, 0.75) - np.quantile(s.values, 0.25)),
            "min": float(s.min()),
            "max": float(s.max()),
            "psi_edges": [float(x) for x in q_unique],
        }
    return contract


def check_feature_skew_against_contract(
    contract: Dict[str, Any],
    X_live: pd.DataFrame,
    split_name: str,
    psi_warn_threshold: float = 0.2,
    zshift_warn_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Kiểm tra skew train vs live/out-of-sample bằng PSI + standardized mean shift.
    
    PSI convention:
    - <0.1: ổn
    - 0.1-0.2: theo dõi
    - >0.2: drift đáng kể
    """
    feature_names = list(contract.get("feature_names", []))
    stats = contract.get("stats", {})
    X_aligned = X_live.reindex(columns=feature_names, fill_value=0.0).copy()

    per_feature: Dict[str, Dict[str, float]] = {}
    psi_values: List[float] = []
    zshift_values: List[float] = []

    for col in feature_names:
        st = stats.get(col, {})
        live_s = pd.to_numeric(X_aligned[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        train_mean = float(st.get("mean", 0.0))
        train_std = float(st.get("std", 0.0))
        live_mean = float(live_s.mean())
        mean_z = float(abs(live_mean - train_mean) / max(train_std, 1e-8))

        edges = np.asarray(st.get("psi_edges", []), dtype=float)
        psi = 0.0
        if len(edges) >= 2:
            pseudo_train = np.concatenate([
                np.full(100, (edges[i] + edges[i + 1]) / 2.0) for i in range(len(edges) - 1)
            ])
            psi = _compute_psi_from_edges(pseudo_train, live_s.values, edges)

        per_feature[col] = {"psi": float(psi), "mean_zshift": float(mean_z)}
        psi_values.append(float(psi))
        zshift_values.append(float(mean_z))

    high_psi = [c for c, m in per_feature.items() if m["psi"] >= psi_warn_threshold]
    high_z = [c for c, m in per_feature.items() if m["mean_zshift"] >= zshift_warn_threshold]

    top_shifted = sorted(
        per_feature.items(), key=lambda kv: (kv[1]["psi"], kv[1]["mean_zshift"]), reverse=True,
    )[:10]

    return {
        "split": split_name,
        "n_features": int(len(feature_names)),
        "mean_psi": float(np.mean(psi_values)) if psi_values else 0.0,
        "median_psi": float(np.median(psi_values)) if psi_values else 0.0,
        "mean_zshift": float(np.mean(zshift_values)) if zshift_values else 0.0,
        "features_psi_ge_0_2": int(len(high_psi)),
        "features_zshift_ge_3": int(len(high_z)),
        "top_shifted_features": [
            {"feature": c, "psi": float(m["psi"]), "mean_zshift": float(m["mean_zshift"])}
            for c, m in top_shifted
        ],
    }
