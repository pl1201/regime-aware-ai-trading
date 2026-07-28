"""
Smoke test cho production.

Muc tieu:
- Verify import chain cac module critical
- Verify config loading
- Verify dry-run action (safe) cua start_trading_bot

Khong dat lenh trading that.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def check_imports() -> list[str]:
    failed = []
    modules = [
        "algo_trading.live.universal_bot",
        "algo_trading.live.okx_client",
        "algo_trading.filters.signal_quality_filter",
        "algo_trading.ml.dynamic_moe_v3_hmm_mtf",
        "algo_trading.risk.dynamic_risk_manager",
    ]

    for module_name in modules:
        try:
            __import__(module_name)
            print(f"[OK] import {module_name}")
        except Exception as exc:
            failed.append(f"import {module_name}: {exc}")
            print(f"[FAIL] import {module_name}: {exc}")

    return failed


def check_dry_run(project_root: Path) -> list[str]:
    failed = []
    cmd = [sys.executable, "start_trading_bot.py", "dry-run"]
    print(f"[INFO] run: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    print(result.stdout or "")
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        failed.append(f"dry-run return code={result.returncode}")

    return failed


def main() -> int:
    # Keep Windows terminals with legacy code pages from masking actual
    # smoke-test failures when child processes emit Unicode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")

    project_root = Path(__file__).parent

    failures = []
    failures.extend(check_imports())
    failures.extend(check_dry_run(project_root))

    if failures:
        print("\n[SUMMARY] Smoke test FAILED")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\n[SUMMARY] Smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
