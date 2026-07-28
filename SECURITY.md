# Security Policy

## Trading credentials

Never commit exchange credentials, Telegram tokens, private keys, account
identifiers, or webhook secrets. Store them in a local `.env` file and use
exchange-scoped API keys with:

- withdrawal permission disabled;
- IP allowlisting where available;
- the minimum required trading permissions;
- demo/paper trading enabled during development.

If a credential has ever been committed or shared, revoke it immediately.
Deleting a file from the latest commit does not remove it from Git history.

## Live-trading safeguards

Before enabling live mode:

1. Run the complete test suite and the production smoke test.
2. Verify symbol, leverage, contract size, margin mode, and position mode.
3. Configure maximum position, per-trade risk, daily-loss, and drawdown limits.
4. Confirm order precision and minimum notional against the exchange.
5. Start with the smallest practical size and actively monitor the first run.

## Reporting a vulnerability

Please report security issues privately through GitHub's security advisory
feature. Do not include active credentials or sensitive account data in an
issue, discussion, log, or screenshot.
