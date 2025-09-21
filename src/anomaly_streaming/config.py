"""Default configuration for the anomaly streaming system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

def default_symbols() -> List[str]:
    """Symbols tracked by default (symbol:exchange format)."""
    return [
        "RELIANCE:NSE",
        "TCS:NSE",
        "HDFCBANK:NSE",
        "INFY:NSE",
        "SBIN:NSE",
    ]


@dataclass(slots=True)
class DetectorConfig:
    """Configuration for the streaming detectors."""

    window: int = 120
    feature_window: int = 20
    isolation_contamination: float = 0.05
    isolation_retrain: int = 15
    lstm_sequence: int = 30
    lstm_retrain: int = 30
    lstm_epochs: int = 25
    lstm_hidden_size: int = 32
    spc_sigma: float = 3.0


@dataclass(slots=True)
class PipelineConfig:
    """Top level configuration for the pipeline."""

    symbols: List[str] = field(default_factory=default_symbols)
    fetch_interval_seconds: int = 30
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    history_limit: int = 720
