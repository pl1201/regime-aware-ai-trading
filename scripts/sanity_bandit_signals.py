"""
Quick sanity check for Bandit strategy signal generation.

Purpose:
- Ensure feature alignment fixes work (feature_names ordering)
- Ensure multiclass (-1/0/1) proba thresholding works
- Print ASCII-only diagnostics (safe for Windows terminals)
"""

from __future__ import annotations

import os
import sys

# ASCII-only marker to ensure output capture works
print("sanity_script_start", flush=True)

# Ensure project root is on sys.path when running as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from algo_trading.data_loader.loader import load_data
from algo_trading.strategies.ml.regime_ensemble_strategy import RegimeEnsembleBanditStrategy


def main() -> int:
    model_paths = {
        "xgb": "models/regime_bandit_xgb.pkl",
        "lgb": "models/regime_bandit_lgb.pkl",
        "cat": "models/regime_bandit_cat.pkl",
        "et": "models/regime_bandit_et_advanced.pkl",
        "hgb": "models/regime_bandit_hgb_advanced.pkl",
        "sgd": "models/regime_bandit_sgd.pkl",
    }

    df = load_data(
        source="binance",
        symbol="BTCUSDT",
        interval="1h",
        start="2025-01-01",
        end="2025-02-01",
        market="spot",
        add_features=True,
    )

    print("n_rows", len(df))
    if df.empty:
        print("empty dataframe - check data loader connectivity")
        return 2

    strat = RegimeEnsembleBanditStrategy(
        model_paths=model_paths,
        proba_threshold=0.50,
        allowed_regimes=["trending", "ranging", "volatile", "calm"],
        bandit_type="ucb",
        reward_mode="direction",
    )

    res = strat.generate_signals(df)
    sig = res.signals
    meta = res.meta or {}

    print("current_regime", meta.get("current_regime"))
    print("n_bars_meta", meta.get("n_bars"))
    print("signal_count_meta", meta.get("signal_count"))
    print("signal_count_series", int((sig != 0).sum()))

    counts = meta.get("bandit_counts") or {}
    total_sel = sum(counts.values()) if isinstance(counts, dict) else None
    print("bandit_total_selections", total_sel)
    print("selected_model", meta.get("selected_model"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


