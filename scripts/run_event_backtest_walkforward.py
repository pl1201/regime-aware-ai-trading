import json
import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo_trading.core.backtest_event import EventConfig, run_event_backtest
from algo_trading.core.backtest_vectorized import RiskConfig
from algo_trading.ml.dynamic_moe_v2_enhanced import load_moe_v2_enhanced
from algo_trading.ml.enhanced_multi_timeframe import prepare_enhanced_multi_timeframe_data_for_training


def _bars_per_day(freq: str) -> int:
    freq_norm = str(freq).strip().lower()
    mapping = {
        "1h": 24,
        "60min": 24,
        "h": 24,
        "15m": 96,
        "15min": 96,
        "m15": 96,
        "30m": 48,
        "30min": 48,
        "4h": 6,
        "240min": 6,
        "1d": 1,
        "d": 1,
    }
    return mapping.get(freq_norm, 24)


def _prepare_data_and_signals(data_file: str, model_file: str, artifact_file: str):
    data_path = ROOT / "data" / data_file
    model_path = ROOT / "models" / model_file
    artifact_path = ROOT / "models" / artifact_file

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

    probs = model.predict_proba(X)
    classes = np.array(getattr(model, "classes_", [-1, 0, 1]))
    pred_cls = classes[np.argmax(probs, axis=1)]
    confidence = np.max(probs, axis=1)

    signals = np.where(confidence >= threshold, pred_cls, 0)
    signals = pd.Series(signals, index=aligned_df.index, dtype=float)
    return aligned_df, signals, threshold


def _run_period_backtest(df_period: pd.DataFrame, sig_period: pd.Series, freq: str):
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
        freq=freq,
    )
    risk = RiskConfig(
        sl_atr_k=2.5,
        tp_atr_k=3.0,
        trailing_atr_k=None,
        atr_col="ATR14",
    )

    result = run_event_backtest(
        df=df_period,
        signals=sig_period,
        cfg=cfg,
        risk=risk,
        max_trades=2000,
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run walk-forward event-driven backtest for current MOE model.")
    parser.add_argument("--data-file", default="okx_1h.csv", help="CSV file under data/ directory.")
    parser.add_argument("--freq", default="1h", help="Bar frequency label (e.g. 1h, 15min).")
    parser.add_argument("--threshold", type=float, default=None, help="Override confidence threshold. Default uses artifact threshold.")
    parser.add_argument("--model-file", default="dynamic_moe_v2_enhanced_final.pkl", help="Model file under models/ directory.")
    parser.add_argument("--artifact-file", default="dynamic_moe_v2_enhanced_final_artifact.pkl", help="Artifact file under models/ directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    aligned_df, signals, threshold = _prepare_data_and_signals(args.data_file, args.model_file, args.artifact_file)
    effective_threshold = threshold if args.threshold is None else float(args.threshold)
    if args.threshold is not None:
        model_path = ROOT / "models" / args.model_file
        artifact_path = ROOT / "models" / args.artifact_file
        model = load_moe_v2_enhanced(str(model_path))
        artifact = joblib.load(artifact_path)

        X, y, _, aligned_df2 = prepare_enhanced_multi_timeframe_data_for_training(
            pd.read_csv(ROOT / "data" / args.data_file).assign(
                timestamp=lambda x: pd.to_datetime(x["timestamp"], errors="coerce", utc=True)
            ).dropna(subset=["timestamp"]).set_index("timestamp").sort_index(),
            return_dataframe=True,
            label_mode="ternary",
            move_threshold=0.001,
            neutral_quantile=0.35,
        )
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        aligned_df = aligned_df2.iloc[mask].copy()
        feature_names = artifact.get("feature_names", None)
        if feature_names:
            X_df = aligned_df.reindex(columns=feature_names, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X = X_df.to_numpy(dtype=float)
        else:
            X = X[mask]

        probs = model.predict_proba(X)
        classes = np.array(getattr(model, "classes_", [-1, 0, 1]))
        pred_cls = classes[np.argmax(probs, axis=1)]
        confidence = np.max(probs, axis=1)
        signals = pd.Series(np.where(confidence >= effective_threshold, pred_cls, 0), index=aligned_df.index, dtype=float)

    tmp = aligned_df.copy()
    tmp["signal"] = signals
    tmp["period_q"] = tmp.index.to_period("Q").astype(str)
    tmp["period_y"] = tmp.index.to_period("Y").astype(str)

    rows_quarter = []
    quarter_groups = tmp.groupby("period_q", sort=True)
    min_bars = _bars_per_day(args.freq) * 20

    for period, g in quarter_groups:
        if len(g) < min_bars:  # At least ~20 days of bars by selected frequency
            continue
        df_p = g.drop(columns=["signal", "period_q", "period_y"])
        sig_p = g["signal"]
        res = _run_period_backtest(df_p, sig_p, args.freq)
        summary = dict(res.get("summary", {}))
        rows_quarter.append(
            {
                "period": period,
                "bars": int(len(g)),
                "signal_rate": float((sig_p != 0).mean()),
                "TotalReturn": summary.get("TotalReturn"),
                "Sharpe": summary.get("Sharpe"),
                "Sortino": summary.get("Sortino"),
                "MaxDrawdown": summary.get("MaxDrawdown"),
                "Calmar": summary.get("Calmar"),
                "TotalTrades": summary.get("TotalTrades"),
                "WinRate": summary.get("WinRate"),
                "ProfitFactor": summary.get("ProfitFactor"),
                "has_sufficient_data": summary.get("has_sufficient_data"),
            }
        )

    q_df = pd.DataFrame(rows_quarter).sort_values("period").reset_index(drop=True)
    if q_df.empty:
        raise RuntimeError("No quarterly walk-forward windows produced results.")

    q_df["year"] = q_df["period"].str.slice(0, 4)
    y_df = (
        q_df.groupby("year", as_index=False)
        .agg(
            periods=("period", "count"),
            mean_return=("TotalReturn", "mean"),
            median_return=("TotalReturn", "median"),
            mean_sharpe=("Sharpe", "mean"),
            worst_drawdown=("MaxDrawdown", "min"),
            total_trades=("TotalTrades", "sum"),
            mean_winrate=("WinRate", "mean"),
            mean_pf=("ProfitFactor", "mean"),
        )
        .sort_values("year")
    )

    stability = {
        "threshold": effective_threshold,
        "num_quarters": int(len(q_df)),
        "positive_quarters": int((q_df["TotalReturn"] > 0).sum()),
        "negative_quarters": int((q_df["TotalReturn"] <= 0).sum()),
        "positive_ratio": float((q_df["TotalReturn"] > 0).mean()),
        "mean_quarter_return": float(q_df["TotalReturn"].mean()),
        "median_quarter_return": float(q_df["TotalReturn"].median()),
        "worst_quarter_return": float(q_df["TotalReturn"].min()),
        "best_quarter_return": float(q_df["TotalReturn"].max()),
        "mean_quarter_sharpe": float(q_df["Sharpe"].mean()),
        "worst_quarter_drawdown": float(q_df["MaxDrawdown"].min()),
    }

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_tag = args.freq.replace("/", "_").replace(" ", "_")
    q_csv = out_dir / f"event_backtest_walkforward_quarterly_{safe_tag}.csv"
    y_csv = out_dir / f"event_backtest_walkforward_yearly_{safe_tag}.csv"
    j_out = out_dir / f"event_backtest_walkforward_summary_{safe_tag}.json"

    q_df.to_csv(q_csv, index=False)
    y_df.to_csv(y_csv, index=False)
    with j_out.open("w", encoding="utf-8") as f:
        json.dump(stability, f, indent=2)

    print("WALKFORWARD_STABILITY")
    for k, v in stability.items():
        print(f"{k}={v}")

    print(f"SavedQuarterly={q_csv}")
    print(f"SavedYearly={y_csv}")
    print(f"SavedSummary={j_out}")


if __name__ == "__main__":
    main()
