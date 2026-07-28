import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo_trading.ml.enhanced_multi_timeframe import prepare_enhanced_multi_timeframe_data_for_training


def parse_args():
    parser = argparse.ArgumentParser(description="Audit label/target quality for M15 training data.")
    parser.add_argument("--data-file", default="okx_15m.csv")
    parser.add_argument("--out", default="label_target_audit_m15.json")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = ROOT / "data" / args.data_file
    out_path = ROOT / "results" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    fwd_ret = pd.to_numeric(aligned_df["close"], errors="coerce").pct_change().shift(-1)
    fwd_ret = fwd_ret.fillna(0.0)

    cls_vals, cls_cnt = np.unique(y, return_counts=True)
    class_dist = {str(int(v)): int(c) for v, c in zip(cls_vals, cls_cnt)}

    leakage_scan = []
    if feature_names and len(feature_names) == X.shape[1]:
        X_df = pd.DataFrame(X, index=aligned_df.index, columns=feature_names)
        corr = X_df.corrwith(fwd_ret).replace([np.inf, -np.inf], np.nan).dropna()
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
        for k, v in corr.head(15).items():
            leakage_scan.append({"feature": str(k), "corr_with_fwd_ret": float(v)})

    by_quarter = []
    tmp = pd.DataFrame({"y": y, "fwd_ret": fwd_ret.values}, index=aligned_df.index)
    tmp["period"] = tmp.index.to_period("Q").astype(str)
    for p, g in tmp.groupby("period"):
        n = len(g)
        if n == 0:
            continue
        by_quarter.append(
            {
                "period": p,
                "samples": int(n),
                "up_ratio": float((g["y"] == 1).mean()),
                "neutral_ratio": float((g["y"] == 0).mean()),
                "down_ratio": float((g["y"] == -1).mean()),
                "mean_fwd_ret": float(g["fwd_ret"].mean()),
                "std_fwd_ret": float(g["fwd_ret"].std(ddof=0)),
            }
        )

    report = {
        "data_file": args.data_file,
        "samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "class_distribution": class_dist,
        "class_balance": {
            "up_ratio": float((y == 1).mean()),
            "neutral_ratio": float((y == 0).mean()),
            "down_ratio": float((y == -1).mean()),
        },
        "fwd_return": {
            "mean": float(fwd_ret.mean()),
            "std": float(fwd_ret.std(ddof=0)),
            "p01": float(np.quantile(fwd_ret.values, 0.01)),
            "p99": float(np.quantile(fwd_ret.values, 0.99)),
        },
        "top_feature_corr_with_fwd_ret": leakage_scan,
        "quarterly_label_profile": by_quarter,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved={out_path}")
    print(f"Samples={report['samples']}")
    print(f"ClassDist={report['class_distribution']}")


if __name__ == "__main__":
    main()
