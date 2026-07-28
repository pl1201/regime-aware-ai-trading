import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    q_path = ROOT / "results" / "event_backtest_walkforward_quarterly_15min.csv"
    y_path = ROOT / "results" / "event_backtest_walkforward_yearly_15min.csv"
    s_path = ROOT / "results" / "event_backtest_walkforward_summary_15min.json"

    q = pd.read_csv(q_path)
    y = pd.read_csv(y_path)
    with s_path.open("r", encoding="utf-8") as f:
        s = json.load(f)

    report = {
        "summary": s,
        "trades": {
            "total_trades": int(q["TotalTrades"].fillna(0).sum()),
            "mean_trades_per_quarter": float(q["TotalTrades"].fillna(0).mean()),
            "median_trades_per_quarter": float(q["TotalTrades"].fillna(0).median()),
            "mean_signal_rate": float(q["signal_rate"].fillna(0).mean()),
        },
        "best_quarters": q.sort_values("TotalReturn", ascending=False).head(5).to_dict(orient="records"),
        "worst_quarters": q.sort_values("TotalReturn", ascending=True).head(5).to_dict(orient="records"),
        "yearly": y.to_dict(orient="records"),
    }

    out_json = ROOT / "results" / "event_backtest_walkforward_report_15min.json"
    out_md = ROOT / "results" / "event_backtest_walkforward_report_15min.md"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    lines = []
    lines.append("# Walk-Forward Report M15")
    lines.append("")
    lines.append("## Summary")
    for k, v in s.items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Trade Activity")
    for k, v in report["trades"].items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Best 5 Quarters")
    for r in report["best_quarters"]:
        lines.append(
            f"- {r['period']}: return={float(r['TotalReturn']):.6f}, sharpe={float(r['Sharpe']):.4f}, trades={int(r['TotalTrades'])}"
        )

    lines.append("")
    lines.append("## Worst 5 Quarters")
    for r in report["worst_quarters"]:
        lines.append(
            f"- {r['period']}: return={float(r['TotalReturn']):.6f}, sharpe={float(r['Sharpe']):.4f}, trades={int(r['TotalTrades'])}"
        )

    lines.append("")
    lines.append("## Yearly")
    for r in report["yearly"]:
        lines.append(
            f"- {r['year']}: mean_return={float(r['mean_return']):.6f}, mean_sharpe={float(r['mean_sharpe']):.4f}, total_trades={int(r['total_trades'])}"
        )

    with out_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved={out_json}")
    print(f"Saved={out_md}")
    print(f"TotalTrades={report['trades']['total_trades']}")
    print(f"MeanTradesPerQuarter={report['trades']['mean_trades_per_quarter']}")


if __name__ == "__main__":
    main()
