"""Isolation Forest based anomaly detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestDetector:
    """Fits an Isolation Forest on rolling windows of engineered features."""

    window_size: int = 240
    contamination: float = 0.02
    min_train_size: int = 120
    random_state: int = 7

    models: Dict[str, IsolationForest] = field(default_factory=dict, init=False)
    buffers: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    thresholds: Dict[str, float] = field(default_factory=dict, init=False)

    def update(self, features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return pd.DataFrame()

        required = {"timestamp", "symbol"}
        if not required.issubset(features.columns):
            raise ValueError("Feature set must contain timestamp and symbol columns")

        results: List[pd.DataFrame] = []
        feature_cols = [c for c in features.columns if c not in {"timestamp", "symbol"}]

        for symbol, group in features.groupby("symbol"):
            buffer = self.buffers.get(symbol)
            if buffer is None:
                buffer = group.copy()
            else:
                buffer = pd.concat([buffer, group], ignore_index=True)
            buffer = buffer.tail(self.window_size)
            self.buffers[symbol] = buffer

            if len(buffer) < self.min_train_size:
                continue

            model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
            )
            model.fit(buffer[feature_cols])
            self.models[symbol] = model

            buffer_scores = model.score_samples(buffer[feature_cols])
            threshold = np.quantile(buffer_scores, self.contamination)
            self.thresholds[symbol] = threshold

            scores = model.score_samples(group[feature_cols])
            result = group[["timestamp", "symbol"]].copy()
            result["iforest_score"] = scores
            result["iforest_threshold"] = threshold
            result["iforest_anomaly"] = scores < threshold
            results.append(result)

        if not results:
            return pd.DataFrame(columns=[
                "timestamp",
                "symbol",
                "iforest_score",
                "iforest_threshold",
                "iforest_anomaly",
            ])

        return pd.concat(results, ignore_index=True)


__all__ = ["IsolationForestDetector"]
