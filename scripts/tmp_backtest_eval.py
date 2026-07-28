import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo_trading.ml.dynamic_moe_v2_enhanced import load_moe_v2_enhanced
from algo_trading.ml.enhanced_multi_timeframe import prepare_enhanced_multi_timeframe_data_for_training


def calc_metrics(y, sig, ret_next, mask):
    y_m = y[mask]
    sig_m = sig[mask]
    ret_m = ret_next[mask]

    valid = (sig_m != 0) & (~np.isnan(ret_m))
    sret = ret_m[valid] * sig_m[valid]

    wins = sret[sret > 0]
    losses = sret[sret < 0]

    # Cost-adjusted scenario: round-trip fee + slippage (e.g., 0.12% + 0.08% = 0.20%)
    cost_per_trade = 0.002
    sret_net = sret - cost_per_trade
    wins_net = sret_net[sret_net > 0]
    losses_net = sret_net[sret_net < 0]

    return {
        "rows": int(mask.sum()),
        "signals": int((sig_m != 0).sum()),
        "coverage": float((sig_m != 0).mean()) if len(sig_m) else 0.0,
        "trade_accuracy": float((np.sign(y_m[valid]) == np.sign(sig_m[valid])).mean()) if valid.any() else 0.0,
        "winrate": float(len(wins) / max(1, len(wins) + len(losses))),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else 0.0,
        "expectancy": float(sret.mean()) if len(sret) else 0.0,
        "profit_factor_net": float(wins_net.sum() / abs(losses_net.sum())) if len(losses_net) and abs(losses_net.sum()) > 0 else 0.0,
        "expectancy_net": float(sret_net.mean()) if len(sret_net) else 0.0,
    }


def main():
    df = pd.read_csv("data/okx_1h.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    model = load_moe_v2_enhanced("models/dynamic_moe_v2_enhanced_final.pkl")
    artifact = joblib.load("models/dynamic_moe_v2_enhanced_final_artifact.pkl")
    threshold = float(artifact.get("threshold", 0.62))

    X, y, _, aligned = prepare_enhanced_multi_timeframe_data_for_training(
        df,
        return_dataframe=True,
        label_mode="ternary",
        move_threshold=0.001,
        neutral_quantile=0.35,
    )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
    X = X[valid]
    y = y[valid].astype(int)
    aligned = aligned.iloc[valid].copy()

    probs = model.predict_proba(X)
    classes = np.array(getattr(model, "classes_", [-1, 0, 1]))
    pred_cls = classes[np.argmax(probs, axis=1)]
    conf = np.max(probs, axis=1)
    sig = np.where(conf >= threshold, pred_cls, 0)

    ret_next = pd.to_numeric(aligned["close"], errors="coerce").pct_change().shift(-1).values
    idx = aligned.index
    n = len(idx)

    periods = {
        "ALL": np.ones(n, dtype=bool),
        "LAST_30pct": np.arange(n) >= int(n * 0.7),
        "LAST_10pct": np.arange(n) >= int(n * 0.9),
        "Y2024": (idx >= pd.Timestamp("2024-01-01", tz="UTC")) & (idx < pd.Timestamp("2025-01-01", tz="UTC")),
        "Y2025plus": idx >= pd.Timestamp("2025-01-01", tz="UTC"),
    }

    print(f"samples={n} threshold={threshold}")
    print("period,rows,signals,coverage,trade_accuracy,winrate,profit_factor,expectancy,profit_factor_net,expectancy_net")
    for name, mask in periods.items():
        m = calc_metrics(y, sig, ret_next, mask)
        print(
            f"{name},{m['rows']},{m['signals']},{m['coverage']:.6f},{m['trade_accuracy']:.6f},"
            f"{m['winrate']:.6f},{m['profit_factor']:.6f},{m['expectancy']:.8f},"
            f"{m['profit_factor_net']:.6f},{m['expectancy_net']:.8f}"
        )


if __name__ == "__main__":
    main()
