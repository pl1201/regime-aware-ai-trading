import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo_trading.ml.dynamic_moe_v2_enhanced import load_moe_v2_enhanced
from algo_trading.ml.enhanced_multi_timeframe import prepare_enhanced_multi_timeframe_data_for_training
from algo_trading.core.backtest_event import EventConfig, run_event_backtest
from algo_trading.core.backtest_vectorized import RiskConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run event-driven backtest for current MOE model.")
    parser.add_argument("--data-file", default="okx_1h.csv", help="CSV file under data/ directory.")
    parser.add_argument("--freq", default="1h", help="Bar frequency label for backtest summary (e.g. 1h, 15min).")
    parser.add_argument("--model-file", default="dynamic_moe_v2_enhanced_final.pkl", help="Model file under models/ directory.")
    parser.add_argument("--artifact-file", default="dynamic_moe_v2_enhanced_final_artifact.pkl", help="Artifact file under models/ directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = ROOT / "data" / args.data_file
    model_path = ROOT / "models" / args.model_file
    artifact_path = ROOT / "models" / args.artifact_file

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

    df_raw = pd.read_csv(data_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce", utc=True)
    df_raw = df_raw.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    model = load_moe_v2_enhanced(str(model_path))
    artifact = joblib.load(artifact_path)
    threshold = float(artifact.get("threshold", 0.55))

    X, y, _, aligned_df = prepare_enhanced_multi_timeframe_data_for_training(
        df_raw,
        return_dataframe=True,
        label_mode="ternary",
        move_threshold=0.001,
        neutral_quantile=0.35,
    )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
    aligned_df = aligned_df.iloc[mask].copy()

    feature_names = artifact.get("feature_names", None)
    if feature_names:
        X_df = aligned_df.reindex(columns=feature_names, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X = X_df.to_numpy(dtype=float)
    else:
        X = X[mask]
    y = y[mask].astype(int)

    probs = model.predict_proba(X)
    classes = np.array(getattr(model, "classes_", [-1, 0, 1]))
    pred_cls = classes[np.argmax(probs, axis=1)]
    confidence = np.max(probs, axis=1)

    # Keep only high-confidence signals; otherwise neutral.
    signals_np = np.where(confidence >= threshold, pred_cls, 0)
    signals = pd.Series(signals_np, index=aligned_df.index, dtype=float)

    cfg = EventConfig(
        initial_cash=10000.0,
        leverage=1.0,
        allow_short=True,
        commission=0.0005,
        slippage_bps=3.0,
        use_next_open=True,
        price_col="close",
        open_col="open",
        high_col="high",
        low_col="low",
        freq=args.freq,
    )
    risk = RiskConfig(
        sl_atr_k=2.5,
        tp_atr_k=3.0,
        trailing_atr_k=None,
        atr_col="ATR14",
    )

    result = run_event_backtest(
        df=aligned_df,
        signals=signals,
        cfg=cfg,
        risk=risk,
        max_trades=2000,
    )

    summary = dict(result.get("summary", {}))
    trades = result.get("trades", pd.DataFrame())

    print("EVENT_BACKTEST_SUMMARY")
    keys = [
        "TotalReturn",
        "CAGR",
        "Sharpe",
        "Sortino",
        "MaxDrawdown",
        "Calmar",
        "TotalTrades",
        "WinRate",
        "ProfitFactor",
        "freq",
        "has_sufficient_data",
    ]
    for k in keys:
        if k in summary:
            print(f"{k}={summary[k]}")

    if not trades.empty and "reason" in trades.columns:
        reason_counts = trades["reason"].value_counts(dropna=False).to_dict()
        print(f"ExitReasonCounts={reason_counts}")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = args.freq.replace("/", "_").replace(" ", "_")
    out_json = out_dir / f"event_backtest_moe_current_summary_{safe_tag}.json"
    out_csv = out_dir / f"event_backtest_moe_current_trades_{safe_tag}.csv"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    if isinstance(trades, pd.DataFrame) and not trades.empty:
        trades.to_csv(out_csv, index=False)

    print(f"SavedSummary={out_json}")
    print(f"SavedTrades={out_csv}")


if __name__ == "__main__":
    main()
