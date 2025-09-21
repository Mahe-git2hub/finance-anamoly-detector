"""Base classes for anomaly detectors."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class StreamingDetector(ABC):
    """Abstract interface for detectors operating on streaming features."""

    def __init__(self) -> None:
        self._ready = False

    @abstractmethod
    def update(self, features: pd.DataFrame) -> None:
        """Consume the latest feature frame to update the model."""

    @abstractmethod
    def score(self, latest: pd.Series) -> float | None:
        """Return an anomaly score for the latest row."""

    def is_ready(self) -> bool:
        return self._ready

    def _mark_ready(self) -> None:
        self._ready = True
