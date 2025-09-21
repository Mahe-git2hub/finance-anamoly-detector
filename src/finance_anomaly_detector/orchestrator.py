"""Pipeline orchestration for the anomaly detection system."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .config import (
    IsolationForestConfig,
    LSTMAutoencoderConfig,
    SPCConfig,
    StreamConfig,
)
from .data_stream import YahooFinanceStreamer
from .detectors.isolation_forest import IsolationForestDetector
from .detectors.lstm_autoencoder import LSTMAutoencoderDetector
from .detectors.spc import SPCDetector
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

        # Compute features with full history so rolling windows have sufficient
        # context, then isolate the rows that correspond to the newly streamed
        # batch. This ensures feature engineering succeeds even when the
        # streamer delivers a single bar per poll.
        engineered = self.feature_engineer.transform(self.raw_history)

        if engineered.empty:
            return {"raw": raw, "features": engineered, "detections": pd.DataFrame()}

        batch_keys = raw[["symbol", "timestamp"]].drop_duplicates()
        features = engineered.merge(batch_keys, on=["symbol", "timestamp"], how="inner")

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


def create_orchestrator(
    *,
    tickers: Sequence[str] | None = None,
    stream_config: StreamConfig | None = None,
    iforest_config: IsolationForestConfig | None = None,
    lstm_config: LSTMAutoencoderConfig | None = None,
    spc_config: SPCConfig | None = None,
) -> AnomalyOrchestrator:
    """Build an :class:`AnomalyOrchestrator` using configuration objects.

    The helper centralises the construction logic used by the dashboard and the
    API so both entry points remain in sync. Callers may override any
    configuration dataclass; unspecified values fall back to the library
    defaults.
    """

    stream_config = stream_config or StreamConfig()
    iforest_config = iforest_config or IsolationForestConfig()
    lstm_config = lstm_config or LSTMAutoencoderConfig()
    spc_config = spc_config or SPCConfig()

    target_tickers = list(tickers) if tickers is not None else list(stream_config.tickers)
    if not target_tickers:
        raise ValueError("At least one ticker must be provided to build an orchestrator")

    streamer = YahooFinanceStreamer(
        tickers=target_tickers,
        interval=stream_config.interval,
        lookback_period=stream_config.lookback_period,
        poll_interval=stream_config.poll_interval,
    )

    detectors: Dict[str, object] = {
        "Isolation Forest": IsolationForestDetector(
            window_size=iforest_config.window_size,
            contamination=iforest_config.contamination,
            min_train_size=iforest_config.min_train_size,
            random_state=iforest_config.random_state,
        ),
        "LSTM Autoencoder": LSTMAutoencoderDetector(
            sequence_length=lstm_config.sequence_length,
            encoding_dim=lstm_config.encoding_dim,
            learning_rate=lstm_config.learning_rate,
            epochs=lstm_config.epochs,
            retrain_interval=lstm_config.retrain_interval,
            min_train_size=lstm_config.min_train_size,
        ),
        "SPC": SPCDetector(
            window_size=spc_config.window_size,
            sigma_threshold=spc_config.sigma_threshold,
        ),
    }

    return AnomalyOrchestrator(streamer=streamer, detectors=detectors)


__all__ = ["AnomalyOrchestrator", "create_orchestrator"]
