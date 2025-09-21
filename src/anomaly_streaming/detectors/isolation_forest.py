"""Isolation Forest detector."""
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .base import StreamingDetector


class IsolationForestDetector(StreamingDetector):
    """Isolation Forest tuned for streaming updates."""

    def __init__(
        self,
        feature_columns: Sequence[str],
        contamination: float = 0.05,
        retrain_interval: int = 15,
        max_history: int = 500,
    ) -> None:
        super().__init__()
        self._feature_columns = list(feature_columns)
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            max_samples="auto",
            random_state=42,
            warm_start=False,
        )
        self._retrain_interval = retrain_interval
        self._max_history = max_history
        self._fit_counter = 0
        self._score_history: deque[float] = deque(maxlen=max_history)
        self._trained = False

    def update(self, features: pd.DataFrame) -> None:
        if features.empty:
            return
        feature_matrix = features[self._feature_columns].to_numpy(dtype=float)
        if feature_matrix.shape[0] < len(self._feature_columns) * 2:
            return
        if not self._trained or self._fit_counter >= self._retrain_interval:
            self._model.fit(feature_matrix)
            self._trained = True
            self._fit_counter = 0
            self._mark_ready()
        else:
            self._fit_counter += 1

    def score(self, latest: pd.Series) -> float | None:
        if not self._trained:
            return None
        vector = latest[self._feature_columns].to_numpy(dtype=float).reshape(1, -1)
        raw = float(self._model.decision_function(vector)[0])
        score = max(0.0, -raw)
        self._score_history.append(score)
        if len(self._score_history) < 5:
            return score
        min_score = min(self._score_history)
        max_score = max(self._score_history)
        if max_score == min_score:
            return score
        normalised = (score - min_score) / (max_score - min_score)
        return float(np.clip(normalised, 0.0, 1.0))
