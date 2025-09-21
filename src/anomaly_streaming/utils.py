"""Utility helpers for the anomaly streaming pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List

import pandas as pd


@dataclass(slots=True)
class Quote:
    """Represents a single market quote from a live data source."""

    symbol: str
    price: float
    currency: str
    timestamp: datetime
    source: str

    @classmethod
    def create(
        cls, symbol: str, price: float, currency: str = "INR", source: str = "google_finance"
    ) -> "Quote":
        return cls(symbol=symbol, price=price, currency=currency, timestamp=datetime.now(timezone.utc), source=source)


@dataclass(slots=True)
class AnomalyResult:
    """Container for a single anomaly detection output."""

    symbol: str
    timestamp: datetime
    price: float
    isolation_score: float | None
    lstm_score: float | None
    spc_score: float | None
    combined_score: float | None
    is_anomaly: bool
    details: dict


COLORS = {
    True: "#d7191c",
    False: "#2c7bb6",
}


def keep_last_rows(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Restrict the dataframe to the last ``limit`` rows per symbol."""

    if limit <= 0 or frame.empty:
        return frame
    return (
        frame.sort_values("timestamp")
        .groupby("symbol", group_keys=False)
        .apply(lambda part: part.iloc[-limit:], include_groups=False)
        .reset_index(drop=True)
    )


def ensure_list(value: str | Iterable[str]) -> List[str]:
    if isinstance(value, str):
        return [value]
    return list(value)
