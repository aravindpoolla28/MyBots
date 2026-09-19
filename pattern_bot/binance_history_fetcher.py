from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def interval_to_milliseconds(interval: str) -> int:
    unit = interval[-1].lower()
    value = int(interval[:-1])
    mapping = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }
    if unit not in mapping:
        raise ValueError(f"Unsupported Binance interval '{interval}'. Use values like 1m, 5m, 15m, 1h, 4h, 1d.")
    return value * mapping[unit]


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[list]:
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    response = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "code" in payload:
        raise RuntimeError(f"Binance error: {payload}")
    return payload


def build_dataframe_from_klines(raw_rows: list[list]) -> pd.DataFrame:
    if not raw_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    rows = []
    for row in raw_rows:
        open_ts_ms, open_price, high_price, low_price, close_price, volume, *_ = row
        rows.append(
            {
                "timestamp": pd.to_datetime(open_ts_ms, unit="ms", utc=True),
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": float(volume),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_history_csv(symbol: str, interval: str, days: int, output_path: str | Path) -> pd.DataFrame:
    interval_ms = interval_to_milliseconds(interval)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 60 * 60 * 1000)
    all_rows: list[list] = []
    current_start_ms = start_ms

    while current_start_ms < end_ms:
        page = fetch_klines(symbol, interval, current_start_ms, end_ms, limit=1000)
        if not page:
            break
        all_rows.extend(page)
        last_open_ts = page[-1][0]
        if last_open_ts <= current_start_ms:
            break
        current_start_ms = last_open_ts + interval_ms

    frame = build_dataframe_from_klines(all_rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Download fresh OHLCV data from Binance into a dedicated cache CSV for session pattern matching.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance trading pair to fetch.")
    parser.add_argument("--interval", default="1h", help="Binance candle interval such as 1h, 30m, 15m, 4h.")
    parser.add_argument("--days", type=int, default=1825, help="How many days of historical data to fetch (default is 5 years).")
    parser.add_argument("--output", default="binance_1h_history.csv", help="CSV file path for the output cache.")
    args = parser.parse_args()

    frame = fetch_history_csv(args.symbol, args.interval, args.days, args.output)
    print(f"Fetched {len(frame)} rows from {args.symbol} at {args.interval} interval")
    print(f"Saved to {args.output}")
    if not frame.empty:
        print(frame.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
