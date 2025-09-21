"""Pipeline orchestration for the anomaly detection system."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .data_stream import YahooFinanceStreamer
from .feature_engineering import FeatureEngineer


@dataclass
class AnomalyOrchestrator:
    """Coordinates streaming, feature engineering and anomaly detection."""

    streamer: YahooFinanceStreamer
    detectors: Dict[str, object]
    feature_engineer: FeatureEngineer = field(default_factory=FeatureEngineer)

    raw_history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    feature_history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    detection_history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    def run_step(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Poll the streamer once and run the detectors on the new data."""

        raw = self.streamer.fetch()
        if raw.empty:
            return None

        self.raw_history = pd.concat([self.raw_history, raw], ignore_index=True)
        self.raw_history.drop_duplicates(subset=["symbol", "timestamp"], keep="last", inplace=True)
        self.raw_history.sort_values(["symbol", "timestamp"], inplace=True)

        features = self.feature_engineer.transform(raw)
        if features.empty:
            return {"raw": raw, "features": features, "detections": pd.DataFrame()}

        combined_results: List[pd.DataFrame] = []
        for name, detector in self.detectors.items():
            result = detector.update(features)
            if result is None or result.empty:
                continue
            combined_results.append(result)

        merged = features.copy()
        if combined_results:
            merged = reduce(
                lambda left, right: pd.merge(
                    left, right, on=["timestamp", "symbol"], how="left"
                ),
                combined_results,
                merged,
            )
            merged.sort_values(["symbol", "timestamp"], inplace=True)

        self.feature_history = pd.concat([self.feature_history, merged], ignore_index=True)
        self.feature_history.drop_duplicates(subset=["symbol", "timestamp"], keep="last", inplace=True)
        self.feature_history.sort_values(["symbol", "timestamp"], inplace=True)

        anomaly_columns = [
            col
            for col in merged.columns
            if col.endswith("_anomaly")
        ]
        if anomaly_columns:
            merged["any_anomaly"] = merged[anomaly_columns].fillna(False).any(axis=1)
        else:
            merged["any_anomaly"] = False

        self.detection_history = pd.concat([self.detection_history, merged], ignore_index=True)
        self.detection_history.drop_duplicates(subset=["symbol", "timestamp"], keep="last", inplace=True)
        self.detection_history.sort_values(["symbol", "timestamp"], inplace=True)

        return {"raw": raw, "features": features, "detections": merged}


__all__ = ["AnomalyOrchestrator"]
