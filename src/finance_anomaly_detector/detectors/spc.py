"""Statistical process control based anomaly detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class SPCDetector:
    """Applies a rolling z-score based control chart."""

    window_size: int = 30
    sigma_threshold: float = 3.0

    buffers: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)

    def update(self, features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return pd.DataFrame()

        required = {"timestamp", "symbol", "return_1"}
        if not required.issubset(features.columns):
            raise ValueError("Features must include timestamp, symbol and return_1 columns")

        results: List[pd.DataFrame] = []

        for symbol, group in features.groupby("symbol"):
            buffer = self.buffers.get(symbol)
            if buffer is None:
                buffer = group.copy()
            else:
                buffer = pd.concat([buffer, group], ignore_index=True)
            buffer = buffer.tail(self.window_size * 10)
            self.buffers[symbol] = buffer

            stats = buffer["return_1"].rolling(self.window_size)
            mean = stats.mean().iloc[-len(group) :].reset_index(drop=True)
            std = stats.std(ddof=0).replace(0, np.nan).iloc[-len(group) :].reset_index(drop=True)

            recent = group.reset_index(drop=True)
            z_scores = (recent["return_1"].reset_index(drop=True) - mean) / std
            z_scores = z_scores.fillna(0.0)
            anomalies = z_scores.abs() > self.sigma_threshold

            result = recent[["timestamp", "symbol"]].copy()
            result["spc_zscore"] = z_scores
            result["spc_anomaly"] = anomalies
            result["spc_ucl"] = self.sigma_threshold
            results.append(result)

        if not results:
            return pd.DataFrame(columns=["timestamp", "symbol", "spc_zscore", "spc_anomaly", "spc_ucl"])

        return pd.concat(results, ignore_index=True)


__all__ = ["SPCDetector"]
