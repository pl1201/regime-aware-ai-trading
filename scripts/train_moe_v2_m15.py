import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo_trading.filters.signal_quality_filter import signal_quality_filter, enhanced_signal_scoring
from algo_trading.ml.dynamic_moe_v2_enhanced import DynamicMOE_v2_Enhanced, save_moe_v2_enhanced
from algo_trading.ml.enhanced_multi_timeframe import prepare_enhanced_multi_timeframe_data_for_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dynamic MOE v2 Enhanced on M15 data.")
    parser.add_argument("--data-file", default="okx_15m.csv", help="CSV file under data/.")
    parser.add_argument("--out-prefix", default="dynamic_moe_v2_enhanced_m15", help="Output model prefix under models/.")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--min-signals", type=int, default=40)
    return parser.parse_args()


def _extract_directional_probs(predictions: np.ndarray, classes):
    pred = np.asarray(predictions)
    if pred.ndim == 1:
        p_long = np.clip(pred.astype(float), 0.0, 1.0)
        p_short = np.clip(1.0 - p_long, 0.0, 1.0)
        p_neutral = np.zeros_like(p_long)
        return p_short, p_neutral, p_long

    cls = list(classes) if classes is not None else None
    if cls is None and pred.shape[1] == 3:
        return pred[:, 0], pred[:, 1], pred[:, 2]
    if cls is None and pred.shape[1] == 2:
        return pred[:, 0], np.zeros(pred.shape[0]), pred[:, 1]

    p_short = pred[:, cls.index(-1)] if -1 in cls else np.zeros(pred.shape[0])
    p_neutral = pred[:, cls.index(0)] if 0 in cls else np.zeros(pred.shape[0])
    p_long = pred[:, cls.index(1)] if 1 in cls else np.zeros(pred.shape[0])
    return p_short, p_neutral, p_long


def build_trade_signals(predictions, features_df, threshold=0.55, classes=None):
    p_short, p_neutral, p_long = _extract_directional_probs(predictions, classes)
    quality_mask = signal_quality_filter(features_df)
    directional_strength = np.maximum(p_long, p_short)
    score = enhanced_signal_scoring(directional_strength, features_df)

    if "multi_tf_trend_consensus" in features_df.columns:
        consensus = pd.to_numeric(features_df["multi_tf_trend_consensus"], errors="coerce").fillna(0.0).values
    else:
        consensus = np.zeros(len(features_df), dtype=float)

    if "volatility_normalized" in features_df.columns:
        vol = pd.to_numeric(features_df["volatility_normalized"], errors="coerce").fillna(0.0)
        lo, hi = float(vol.quantile(0.20)), float(vol.quantile(0.80))
        vol_ok = ((vol >= lo) & (vol <= hi)).values
    else:
        vol_ok = np.ones(len(features_df), dtype=bool)

    base_pass = quality_mask & vol_ok & (score >= threshold) & (directional_strength >= threshold)
    winner_long = (p_long > p_short) & (p_long >= p_neutral)
    winner_short = (p_short > p_long) & (p_short >= p_neutral)

    signals = np.zeros(len(features_df), dtype=int)
    signals[base_pass & winner_long & (consensus >= 0)] = 1
    signals[base_pass & winner_short & (consensus <= 0)] = -1
    return signals


def trade_metrics_from_signals(y_true, signals, returns_next):
    y_true = np.asarray(y_true).astype(int)
    signals = np.asarray(signals).astype(int)
    returns_next = np.asarray(returns_next, dtype=float)

    trade_mask = signals != 0
    valid_mask = trade_mask & (~np.isnan(returns_next))

    if valid_mask.sum() == 0:
        return {
            "trade_accuracy": 0.0,
            "expectancy": 0.0,
            "winrate": 0.0,
            "rr_ratio": 0.0,
            "profit_factor": 0.0,
            "coverage": float(trade_mask.mean()) if len(trade_mask) else 0.0,
            "total_signals": int(trade_mask.sum()),
        }

    directional_true = np.sign(y_true[valid_mask])
    directional_pred = np.sign(signals[valid_mask])
    trade_accuracy = float((directional_true == directional_pred).mean())

    signal_returns = returns_next[valid_mask] * signals[valid_mask]
    wins = signal_returns[signal_returns > 0]
    losses = signal_returns[signal_returns < 0]

    winrate = float(len(wins) / max(1, len(wins) + len(losses)))
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    rr_ratio = float(avg_win / avg_loss) if avg_loss > 0 else 0.0

    total_wins = float(wins.sum()) if len(wins) else 0.0
    total_losses = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = float(total_wins / total_losses) if total_losses > 0 else 0.0

    expectancy = float(signal_returns.mean()) if len(signal_returns) else 0.0

    return {
        "trade_accuracy": trade_accuracy,
        "expectancy": expectancy,
        "winrate": winrate,
        "rr_ratio": rr_ratio,
        "profit_factor": profit_factor,
        "coverage": float(trade_mask.mean()),
        "total_signals": int(trade_mask.sum()),
    }


def optimize_threshold_by_pf(predictions, classes, y_val, df_val, min_signals=40):
    search_space = np.arange(0.35, 0.71, 0.01)
    returns_next = pd.to_numeric(df_val["close"], errors="coerce").pct_change().shift(-1).values
    rows = []

    for th in search_space:
        sig = build_trade_signals(predictions, df_val, threshold=float(th), classes=classes)
        m = trade_metrics_from_signals(y_val, sig, returns_next)
        total_signals = int(m.get("total_signals", 0))
        pf = float(m.get("profit_factor", 0.0))
        objective_score = float(pf * np.log1p(total_signals)) if total_signals >= int(min_signals) else 0.0
        rows.append({"threshold": float(th), "objective_score": objective_score, "valid_threshold": bool(total_signals >= int(min_signals)), **m})

    table = pd.DataFrame(rows).sort_values(
        ["objective_score", "profit_factor", "expectancy", "total_signals"],
        ascending=[False, False, False, False],
    )
    if (table["valid_threshold"] == False).all():
        table["objective_score"] = table["profit_factor"].astype(float) * np.log1p(table["total_signals"].astype(float))
        table = table.sort_values(["objective_score", "profit_factor", "expectancy", "total_signals"], ascending=[False, False, False, False])

    best = table.iloc[0].to_dict()
    return float(best["threshold"]), best, table


def derive_regime_ids(df: pd.DataFrame, n_experts: int = 3) -> np.ndarray:
    vol = pd.to_numeric(df["close"], errors="coerce").pct_change().rolling(96).std().bfill().fillna(0.0)
    try:
        bins = pd.qcut(vol.rank(method="first"), q=n_experts, labels=False)
        return bins.astype(int).values
    except Exception:
        return (np.arange(len(df)) % max(1, n_experts)).astype(int)


def main():
    args = parse_args()

    data_path = ROOT / "data" / args.data_file
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    X, y, feature_names, aligned_df = prepare_enhanced_multi_timeframe_data_for_training(
        df,
        target_lookahead=1,
        return_dataframe=True,
        label_mode="ternary",
        move_threshold=0.001,
        neutral_quantile=0.35,
    )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    valid_mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask].astype(int)
    aligned_df = aligned_df.iloc[valid_mask].copy()

    n = len(X)
    train_end = int(n * (1.0 - args.val_size - args.test_size))
    val_end = int(n * (1.0 - args.test_size))

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    df_train = aligned_df.iloc[:train_end].copy()
    df_val = aligned_df.iloc[train_end:val_end].copy()
    df_test = aligned_df.iloc[val_end:].copy()

    model = DynamicMOE_v2_Enhanced()
    model.feature_names = list(feature_names)
    regime_ids_train = derive_regime_ids(df_train, n_experts=3)
    model.fit(X_train, y_train, regime_ids=regime_ids_train)

    pred_val = model.predict_proba(X_val)
    classes = getattr(model, "classes_", None)
    best_th, best_val, th_table = optimize_threshold_by_pf(pred_val, classes, y_val, df_val, min_signals=args.min_signals)

    pred_test = model.predict_proba(X_test)
    sig_test = build_trade_signals(pred_test, df_test, threshold=best_th, classes=classes)
    ret_test = pd.to_numeric(df_test["close"], errors="coerce").pct_change().shift(-1).values
    test_trade_metrics = trade_metrics_from_signals(y_test, sig_test, ret_test)

    if isinstance(pred_test, np.ndarray) and pred_test.ndim == 2:
        y_pred_class = classes[np.argmax(pred_test, axis=1)]
    else:
        y_pred_class = np.where(np.asarray(pred_test) > 0.5, 1, -1)

    overall_acc = float(accuracy_score(y_test, y_pred_class))
    macro_f1 = float(f1_score(y_test, y_pred_class, average="macro", zero_division=0))

    model_path = models_dir / f"{args.out_prefix}.pkl"
    features_path = models_dir / f"{args.out_prefix}_features.pkl"
    artifact_path = models_dir / f"{args.out_prefix}_artifact.pkl"
    threshold_grid_path = models_dir / f"{args.out_prefix}_threshold_grid.csv"
    results_path = models_dir / f"{args.out_prefix}_training_results.json"

    save_moe_v2_enhanced(model, str(model_path))
    joblib.dump(feature_names, features_path)
    th_table.to_csv(threshold_grid_path, index=False)

    payload = {
        "best_threshold": best_th,
        "val_best_profit_factor": float(best_val.get("profit_factor", 0.0)),
        "val_best_expectancy": float(best_val.get("expectancy", 0.0)),
        "val_best_objective_score": float(best_val.get("objective_score", 0.0)),
        "overall_accuracy": overall_acc,
        "macro_f1": macro_f1,
        **test_trade_metrics,
    }

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "threshold": best_th,
        "created_at": datetime.now().isoformat(),
        "n_samples": int(len(X)),
        "n_features": int(len(feature_names)),
        "class_labels": getattr(model, "classes_", np.array([-1, 0, 1])).tolist(),
        "train_payload": payload,
        "timeframe": "15min",
        "data_file": args.data_file,
    }
    joblib.dump(artifact, artifact_path)

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("M15_TRAINING_DONE")
    print(f"samples={len(X)} features={len(feature_names)}")
    print(f"best_threshold={best_th}")
    print(f"test_trade_accuracy={test_trade_metrics.get('trade_accuracy', 0.0)}")
    print(f"test_profit_factor={test_trade_metrics.get('profit_factor', 0.0)}")
    print(f"test_total_signals={test_trade_metrics.get('total_signals', 0)}")
    print(f"SavedModel={model_path}")
    print(f"SavedArtifact={artifact_path}")
    print(f"SavedThresholdGrid={threshold_grid_path}")
    print(f"SavedResults={results_path}")


if __name__ == "__main__":
    main()
