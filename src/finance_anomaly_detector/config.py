"""Configuration defaults for the finance anomaly detection system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class StreamConfig:
    """Configuration for the real-time data streamer."""

    tickers: List[str] = field(
        default_factory=lambda: ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    )
    interval: str = "1m"
    lookback_period: str = "1d"
    poll_interval: int = 60


@dataclass(frozen=True)
class IsolationForestConfig:
    window_size: int = 240
    contamination: float = 0.02
    min_train_size: int = 120
    random_state: int = 7


@dataclass(frozen=True)
class LSTMAutoencoderConfig:
    sequence_length: int = 20
    encoding_dim: int = 16
    learning_rate: float = 1e-3
    epochs: int = 10
    retrain_interval: int = 60
    min_train_size: int = 120


@dataclass(frozen=True)
class SPCConfig:
    window_size: int = 30
    sigma_threshold: float = 3.0


@dataclass(frozen=True)
class DashboardConfig:
    refresh_seconds: int = 30


__all__ = [
    "StreamConfig",
    "IsolationForestConfig",
    "LSTMAutoencoderConfig",
    "SPCConfig",
    "DashboardConfig",
]
