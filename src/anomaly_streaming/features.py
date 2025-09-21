"""Feature engineering for streaming price series."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return",
    "log_return",
    "rolling_mean",
    "rolling_std",
    "volatility",
    "rolling_mad",
    "ewma",
    "ewm_std",
    "zscore",
    "ewma_zscore",
    "momentum",
]


@dataclass(slots=True)
class FeatureBuilder:
    """Create derived features used by the detectors."""

    window: int = 20

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        df = frame.sort_values("timestamp").copy()
        df["return"] = df["price"].pct_change()
        df["log_return"] = np.log(df["price"]).diff()
        df["rolling_mean"] = df["price"].rolling(self.window).mean()
        df["rolling_std"] = df["price"].rolling(self.window).std()
        df["volatility"] = df["return"].rolling(self.window).std()
        df["rolling_mad"] = df["price"].rolling(self.window).apply(
            lambda values: np.mean(np.abs(values - np.mean(values))) if len(values) else np.nan,
            raw=True,
        )
        df["ewma"] = df["price"].ewm(span=self.window, adjust=False).mean()
        df["ewm_std"] = df["return"].ewm(span=self.window, adjust=False).std()
        df["zscore"] = (df["price"] - df["rolling_mean"]) / df["rolling_std"]
        df["ewma_zscore"] = (df["price"] - df["ewma"]) / df["ewm_std"]
        df["momentum"] = df["price"].diff(self.window)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df


def combine_feature_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, axis=0).sort_values("timestamp").reset_index(drop=True)
