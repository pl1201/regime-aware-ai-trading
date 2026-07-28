# Contributing

## Development setup

Create a Python 3.11 virtual environment, install `requirements.txt`, and copy
`.env.example` to `.env`. Keep paper/demo mode enabled.

## Quality checks

Run before opening a pull request:

```bash
python tests/run_quick_tests.py
pytest
cd production
python smoke_test_production.py
```

## Pull requests

- Keep changes focused and explain their quantitative rationale.
- Add tests for changes to signals, risk limits, metrics, or exchange behavior.
- Prevent look-ahead bias: features at time \(t\) may only use information
  available at or before \(t\).
- Report transaction-cost and slippage assumptions with backtest results.
- Never commit secrets, private market data, trained model binaries, or logs.

Backtest improvements should include out-of-sample or walk-forward evidence,
not only in-sample metrics.
