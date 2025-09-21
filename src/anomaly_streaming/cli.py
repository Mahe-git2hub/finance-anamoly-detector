"""Command line interface for the streaming anomaly detector."""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import Iterable

import pandas as pd

from .config import PipelineConfig, default_symbols
from .pipeline import RealTimeAnomalyPipeline


def format_results(results) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["symbol", "timestamp", "price", "combined", "anomaly"])
    rows = []
    for result in results:
        rows.append(
            {
                "symbol": result.symbol,
                "timestamp": result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "price": round(result.price, 2),
                "isolation": None if result.isolation_score is None else round(result.isolation_score, 3),
                "lstm": None if result.lstm_score is None else round(result.lstm_score, 3),
                "spc": None if result.spc_score is None else round(result.spc_score, 3),
                "combined": None if result.combined_score is None else round(result.combined_score, 3),
                "anomaly": "⚠️" if result.is_anomaly else "✅",
            }
        )
    return pd.DataFrame(rows)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time market anomaly detector")
    parser.add_argument("--symbols", nargs="+", default=default_symbols(), help="Symbols to monitor in SYMBOL:EXCHANGE format")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between fetches")
    parser.add_argument("--iterations", type=int, default=0, help="Number of iterations (0 for infinite)")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = PipelineConfig(symbols=list(args.symbols), fetch_interval_seconds=args.interval)
    pipeline = RealTimeAnomalyPipeline(config)
    iteration = 0
    try:
        while args.iterations == 0 or iteration < args.iterations:
            start = time.time()
            results = pipeline.fetch_and_update()
            table = format_results(results)
            if not table.empty:
                print("\n" + table.to_string(index=False))
            else:
                print(f"[{datetime.now():%H:%M:%S}] Waiting for enough data...")
            iteration += 1
            elapsed = time.time() - start
            sleep_for = max(0.0, args.interval - elapsed)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
