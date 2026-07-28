import argparse
import json
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


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize threshold by net expectancy and PF stability across quarters.")
    parser.add_argument("--data-file", default="okx_15m.csv")
    parser.add_argument("--model-file", default="dynamic_moe_v2_enhanced_m15.pkl")
    parser.add_argument("--artifact-file", default="dynamic_moe_v2_enhanced_m15_artifact.pkl")
    parser.add_argument("--freq", default="15min")
    parser.add_argument("--th-start", type=float, default=0.40)
    parser.add_argument("--th-end", type=float, default=0.70)
    parser.add_argument("--th-step", type=float, default=0.02)
    parser.add_argument("--pf-floor", type=float, default=1.2)
    parser.add_argument("--out-prefix", default="m15_net_threshold_optimization")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = ROOT / "data" / args.data_file
    model_path = ROOT / "models" / args.model_file
    artifact_path = ROOT / "models" / args.artifact_file

    df_raw = pd.read_csv(data_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce", utc=True)
    df_raw = df_raw.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    model = load_moe_v2_enhanced(str(model_path))
    artifact = joblib.load(artifact_path)

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
    risk = RiskConfig(sl_atr_k=2.5, tp_atr_k=3.0, trailing_atr_k=None, atr_col="ATR14")

    tmp = aligned_df.copy()
    tmp["pred_cls"] = pred_cls
    tmp["confidence"] = confidence
    tmp["period_q"] = tmp.index.to_period("Q").astype(str)

    rows = []
    thresholds = np.arange(args.th_start, args.th_end + 1e-9, args.th_step)

    for th in thresholds:
        q_metrics = []
        for period, g in tmp.groupby("period_q", sort=True):
            if len(g) < 96 * 20:
                continue

            sig = np.where(g["confidence"].values >= float(th), g["pred_cls"].values, 0)
            sig_s = pd.Series(sig, index=g.index, dtype=float)
            df_p = g.drop(columns=["pred_cls", "confidence", "period_q"])

            res = run_event_backtest(df=df_p, signals=sig_s, cfg=cfg, risk=risk, max_trades=4000)
            s = dict(res.get("summary", {}))

            q_metrics.append(
                {
                    "period": period,
                    "TotalReturn": float(s.get("TotalReturn", 0.0)),
                    "ProfitFactor": float(s.get("ProfitFactor", 0.0)),
                    "TotalTrades": int(s.get("TotalTrades", 0)),
                }
            )

        if not q_metrics:
            continue

        q_df = pd.DataFrame(q_metrics)
        mean_ret = float(q_df["TotalReturn"].mean())
        std_ret = float(q_df["TotalReturn"].std(ddof=0)) if len(q_df) > 1 else 0.0
        mean_pf = float(q_df["ProfitFactor"].mean())
        pf_stability_penalty = float(np.maximum(0.0, args.pf_floor - q_df["ProfitFactor"]).mean())

        objective = mean_ret - 0.5 * std_ret - 0.25 * pf_stability_penalty

        rows.append(
            {
                "threshold": float(th),
                "quarters": int(len(q_df)),
                "mean_quarter_return": mean_ret,
                "std_quarter_return": std_ret,
                "mean_pf": mean_pf,
                "pf_stability_penalty": pf_stability_penalty,
                "mean_trades": float(q_df["TotalTrades"].mean()),
                "objective": float(objective),
            }
        )

    grid = pd.DataFrame(rows).sort_values("objective", ascending=False)
    if grid.empty:
        raise RuntimeError("No threshold candidates produced valid quarterly results.")

    best = grid.iloc[0].to_dict()

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / f"{args.out_prefix}_grid.csv"
    best_path = out_dir / f"{args.out_prefix}_best.json"

    grid.to_csv(grid_path, index=False)
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"SavedGrid={grid_path}")
    print(f"SavedBest={best_path}")
    print(f"BestThreshold={best['threshold']}")
    print(f"BestObjective={best['objective']}")


if __name__ == "__main__":
    main()
