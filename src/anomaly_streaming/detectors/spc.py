"""Statistical process control style detector."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import StreamingDetector


class SPCDetector(StreamingDetector):
    def __init__(self, sigma_level: float = 3.0) -> None:
        super().__init__()
        self._sigma_level = sigma_level
        self._latest_row: Optional[pd.Series] = None

    def update(self, features: pd.DataFrame) -> None:
        if features.empty:
            return
        self._latest_row = features.iloc[-1]
        self._mark_ready()

    def score(self, latest: pd.Series) -> float | None:
        row = latest if not latest.empty else self._latest_row
        if row is None:
            return None
        price_z = float(abs(row.get("zscore", 0.0)))
        ewma_z = float(abs(row.get("ewma_zscore", 0.0)))
        vol = float(abs(row.get("volatility", 0.0)))
        score = max(price_z / self._sigma_level, ewma_z / self._sigma_level, vol)
        return float(np.clip(score, 0.0, 5.0))

    def is_out_of_control(self, latest: pd.Series) -> bool:
        row = latest if not latest.empty else self._latest_row
        if row is None:
            return False
        return abs(row.get("zscore", 0.0)) >= self._sigma_level or abs(row.get("ewma_zscore", 0.0)) >= self._sigma_level
