"""Utilities for streaming intraday price data from Yahoo Finance."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterable, List

import pandas as pd
import pytz
import yfinance as yf


@dataclass
class MarketBar:
    """A single market datapoint."""

    symbol: str
    timestamp: pd.Timestamp
    close: float
    volume: float


class YahooFinanceStreamer:
    """Polls Yahoo Finance for near real-time intraday market bars.

    Yahoo Finance exposes 1-minute delayed quotes for Indian equities that are
    sufficient for live anomaly detection demos. The streamer keeps track of the
    last timestamp seen for every symbol and only emits new bars.
    """

    def __init__(
        self,
        tickers: Iterable[str],
        interval: str = "1m",
        lookback_period: str = "1d",
        poll_interval: int = 60,
        timezone: str = "Asia/Kolkata",
    ) -> None:
        self._tickers: List[str] = list(tickers)
        if not self._tickers:
            raise ValueError("At least one ticker must be provided")
        self.interval = interval
        self.lookback_period = lookback_period
        self.poll_interval = poll_interval
        self.tz = pytz.timezone(timezone)
        self._last_timestamps: Dict[str, pd.Timestamp] = {}

    def _convert_timezone(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if index.tz is None:
            index = index.tz_localize("UTC")
        return index.tz_convert(self.tz)

    def _download(self, ticker: str) -> pd.DataFrame:
        data = yf.download(
            tickers=ticker,
            period=self.lookback_period,
            interval=self.interval,
            progress=False,
            auto_adjust=False,
        )
        if data.empty:
            return data
        data = data.rename(columns={"Close": "close", "Volume": "volume"})[
            ["close", "volume"]
        ]
        data.index = self._convert_timezone(data.index)
        data["symbol"] = ticker
        data = data.reset_index().rename(columns={"index": "timestamp"})
        return data

    def fetch(self) -> pd.DataFrame:
        """Fetch the most recent bars for every configured ticker."""

        frames: List[pd.DataFrame] = []
        for ticker in self._tickers:
            raw = self._download(ticker)
            if raw.empty:
                continue
            last_timestamp = self._last_timestamps.get(ticker)
            if last_timestamp is not None:
                raw = raw[raw["timestamp"] > last_timestamp]
            if raw.empty:
                continue
            self._last_timestamps[ticker] = raw["timestamp"].max()
            frames.append(raw)
        if not frames:
            return pd.DataFrame(columns=["timestamp", "symbol", "close", "volume"])
        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values(["symbol", "timestamp"], inplace=True)
        return combined

    async def stream(self) -> AsyncIterator[pd.DataFrame]:
        """Asynchronously yield new market bars as they become available."""

        while True:
            data = await asyncio.to_thread(self.fetch)
            if not data.empty:
                yield data
            await asyncio.sleep(self.poll_interval)


__all__ = ["YahooFinanceStreamer", "MarketBar"]
