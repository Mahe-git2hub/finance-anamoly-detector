"""Feature engineering helpers for anomaly detection models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class FeatureEngineer:
    """Constructs rolling statistical features for market bars."""

    return_windows: Iterable[int] = (1, 5, 15)
    volatility_window: int = 20
    volume_window: int = 20

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data
        if not {"timestamp", "symbol", "close", "volume"}.issubset(data.columns):
            raise ValueError("Input data must include timestamp, symbol, close and volume columns")

        data = data.copy()
        data.sort_values(["symbol", "timestamp"], inplace=True)
        grouped = data.groupby("symbol", group_keys=False)

        features: List[pd.DataFrame] = []
        for symbol, frame in grouped:
            frame = frame.set_index("timestamp")
            frame = self._add_return_features(frame)
            frame = self._add_volatility(frame)
            frame = self._add_volume_features(frame)
            frame["symbol"] = symbol
            features.append(frame.reset_index())

        return pd.concat(features, ignore_index=True)

    def _add_return_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        for window in self.return_windows:
            frame[f"return_{window}"] = frame["close"].pct_change(window)
        frame.dropna(inplace=True)
        return frame

    def _add_volatility(self, frame: pd.DataFrame) -> pd.DataFrame:
        returns = frame["return_1"].fillna(0.0)
        frame["volatility"] = returns.rolling(self.volatility_window).std().fillna(0.0)
        return frame

    def _add_volume_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        vol_roll = frame["volume"].rolling(self.volume_window)
        volume_mean = vol_roll.mean().bfill().ffill()
        frame["volume_mean"] = volume_mean
        volume_std = vol_roll.std().replace(0, np.nan)
        zscore = (frame["volume"] - volume_mean) / volume_std
        frame["volume_zscore"] = zscore.fillna(0.0)
        return frame


__all__ = ["FeatureEngineer"]
