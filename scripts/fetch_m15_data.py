import argparse
from pathlib import Path
import sys
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BINANCE_REST = {
    "spot": "https://api.binance.com/api/v3/klines",
    "futures": "https://fapi.binance.com/fapi/v1/klines",
}


def fetch_binance_klines(symbol: str, interval: str, start_ms: int | None, end_ms: int | None, market: str) -> pd.DataFrame:
    url = BINANCE_REST.get(market, BINANCE_REST["spot"])
    params = {"symbol": symbol.upper(), "interval": interval, "limit": 1000}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    if df.empty:
        return df
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open_time", "open", "high", "low", "close", "volume"]].dropna()


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch M15 OHLCV data and save CSV.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol, e.g. BTCUSDT.")
    parser.add_argument("--market", default="spot", choices=["spot", "futures"], help="Binance market type.")
    parser.add_argument("--start", default="2022-01-01", help="UTC start date, e.g. 2022-01-01.")
    parser.add_argument("--end", default=None, help="UTC end date, e.g. 2026-04-01.")
    parser.add_argument("--interval", default="15m", help="Kline interval, default 15m.")
    parser.add_argument("--out", default="okx_15m.csv", help="Output filename under data/.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = ROOT / "data" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000) if args.start else None
    end_ms = int(pd.Timestamp(args.end, tz="UTC").timestamp() * 1000) if args.end else None

    frames = []
    current = start_ms
    while True:
        batch = fetch_binance_klines(args.symbol, args.interval, current, end_ms, args.market)
        if batch.empty:
            break
        frames.append(batch)
        current = int(batch["open_time"].iloc[-1].timestamp() * 1000) + 1
        if end_ms is not None and current >= end_ms:
            break
        if len(batch) < 1000:
            break
        time.sleep(0.2)

    if not frames:
        raise RuntimeError("No data fetched from Binance API.")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    df = df.rename(columns={"open_time": "timestamp"})

    df.to_csv(out_path, index=False)
    print(f"Saved={out_path}")
    print(f"Rows={len(df)}")
    if len(df) > 0:
        print(f"From={df['timestamp'].iloc[0]}")
        print(f"To={df['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()
