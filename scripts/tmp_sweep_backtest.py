import itertools
import re
import subprocess
from pathlib import Path

PY = r"d:/Bot_Trading/.venv/Scripts/python.exe"
SCRIPT = r"d:/Bot_Trading/production/run_event_backtest_moe_current.py"
BASE_ARGS = [
    "--data-file", "okx_15m.csv",
    "--freq", "15min",
    "--model-file", "dynamic_moe_v2_enhanced_m15_ict_fibo_prod_sync_v1.pkl",
    "--artifact-file", "dynamic_moe_v2_enhanced_m15_ict_fibo_prod_sync_v1_artifact.pkl",
    "--commission", "0.0005",
    "--slippage-bps", "3",
    "--latency-bps", "1",
    "--threshold", "0.46",
]

# Focused grid to reduce signal flips and improve RR.
GRID = {
    "reversal_confirm_bars": [4, 6],
    "exit_confirm_bars": [2, 3],
    "min_hold_bars": [4, 8, 12],
    "sl_atr_k": [2.5, 3.0],
    "tp_atr_k": [3.0, 4.0],
}

KEYS_TO_PARSE = {
    "ProfitFactor": float,
    "WinRate": float,
    "TotalReturn": float,
    "MaxDrawdown": float,
    "TotalTrades": float,
    "Passed": lambda x: str(x).strip().lower() == "true",
}


def parse_output(out: str):
    parsed = {}
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k in KEYS_TO_PARSE:
            try:
                parsed[k] = KEYS_TO_PARSE[k](v)
            except Exception:
                pass
        if k == "SignalDiagnostics":
            m = re.search(r"coverage=([0-9.]+)", v)
            if m:
                parsed["coverage"] = float(m.group(1))
    return parsed


def score_row(row):
    pf = row.get("ProfitFactor", 0.0)
    wr = row.get("WinRate", 0.0)
    cov = row.get("coverage", 0.0)
    dd = abs(row.get("MaxDrawdown", -1.0))
    trades = row.get("TotalTrades", 0.0)
    # Weighted score prioritizing PF and drawdown.
    return pf * 2.0 + wr * 0.8 + cov * 0.6 - dd * 1.0 + min(trades / 3000.0, 1.0) * 0.2


def run_one(params):
    cmd = [PY, SCRIPT] + BASE_ARGS
    cmd += ["--reversal-confirm-bars", str(params["reversal_confirm_bars"])]
    cmd += ["--exit-confirm-bars", str(params["exit_confirm_bars"])]
    cmd += ["--min-hold-bars", str(params["min_hold_bars"])]
    cmd += ["--sl-atr-k", str(params["sl_atr_k"])]
    cmd += ["--tp-atr-k", str(params["tp_atr_k"])]

    proc = subprocess.run(cmd, cwd=r"d:/Bot_Trading", capture_output=True, text=True)
    merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
    metrics = parse_output(merged)
    metrics["returncode"] = proc.returncode
    metrics.update(params)
    metrics["score"] = score_row(metrics) if proc.returncode == 0 else -999.0
    return metrics


def main():
    all_params = []
    for rc, ec, mh, sl, tp in itertools.product(
        GRID["reversal_confirm_bars"],
        GRID["exit_confirm_bars"],
        GRID["min_hold_bars"],
        GRID["sl_atr_k"],
        GRID["tp_atr_k"],
    ):
        all_params.append(
            {
                "reversal_confirm_bars": rc,
                "exit_confirm_bars": ec,
                "min_hold_bars": mh,
                "sl_atr_k": sl,
                "tp_atr_k": tp,
            }
        )

    rows = []
    for i, p in enumerate(all_params, start=1):
        print(f"[{i}/{len(all_params)}] {p}")
        rows.append(run_one(p))

    rows = [r for r in rows if r.get("returncode", 1) == 0]
    rows.sort(key=lambda x: x.get("score", -999.0), reverse=True)

    top = rows[:10]
    print("\nTOP CONFIGS")
    for r in top:
        print(
            " | ".join(
                [
                    f"score={r.get('score', 0):.4f}",
                    f"PF={r.get('ProfitFactor', 0):.4f}",
                    f"WR={r.get('WinRate', 0):.4f}",
                    f"Ret={r.get('TotalReturn', 0):.4f}",
                    f"DD={r.get('MaxDrawdown', 0):.4f}",
                    f"Cov={r.get('coverage', 0):.4f}",
                    f"Trades={int(r.get('TotalTrades', 0))}",
                    f"rc={r['reversal_confirm_bars']}",
                    f"ec={r['exit_confirm_bars']}",
                    f"mh={r['min_hold_bars']}",
                    f"sl={r['sl_atr_k']}",
                    f"tp={r['tp_atr_k']}",
                ]
            )
        )


if __name__ == "__main__":
    main()
