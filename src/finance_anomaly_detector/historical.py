"""Historical anomaly detection utilities."""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

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


@dataclass(frozen=True)
class PeriodDefinition:
    """Defines how to query Yahoo Finance for a given human friendly period."""

    label: str
    yfinance_period: str
    interval: str


@dataclass
class HistoricalDetectionResult:
    """Container for historical anomaly detection results."""

    tickers: List[str]
    period: str
    detections: pd.DataFrame
    anomalies: pd.DataFrame


class HistoricalAnomalyDetector:
    """Detect anomalies over configurable historical periods."""

    #: Canonical period definitions mapped to the period label returned in the API.
    _PERIODS: Mapping[str, PeriodDefinition] = {
        "24h": PeriodDefinition(label="24h", yfinance_period="1d", interval="5m"),
        "2d": PeriodDefinition(label="2d", yfinance_period="2d", interval="5m"),
        "1w": PeriodDefinition(label="1w", yfinance_period="1wk", interval="15m"),
        "2w": PeriodDefinition(label="2w", yfinance_period="2wk", interval="30m"),
        "1mo": PeriodDefinition(label="1mo", yfinance_period="1mo", interval="1h"),
        "1q": PeriodDefinition(label="1q", yfinance_period="3mo", interval="1d"),
        "1y": PeriodDefinition(label="1y", yfinance_period="1y", interval="1d"),
    }

    #: Accept user facing aliases, normalising to one of the canonical keys.
    _ALIASES: Mapping[str, str] = {
        "24h": "24h",
        "1d": "24h",
        "24hr": "24h",
        "24hrs": "24h",
        "1day": "24h",
        "2d": "2d",
        "2day": "2d",
        "2days": "2d",
        "48h": "2d",
        "1w": "1w",
        "1wk": "1w",
        "7d": "1w",
        "1week": "1w",
        "2w": "2w",
        "2wk": "2w",
        "14d": "2w",
        "2weeks": "2w",
        "1mo": "1mo",
        "1month": "1mo",
        "30d": "1mo",
        "1mth": "1mo",
        "1q": "1q",
        "1quarter": "1q",
        "3mo": "1q",
        "1y": "1y",
        "1yr": "1y",
        "1year": "1y",
        "12mo": "1y",
    }

    def __init__(
        self,
        *,
        stream_config: StreamConfig | None = None,
        iforest_config: IsolationForestConfig | None = None,
        lstm_config: LSTMAutoencoderConfig | None = None,
        spc_config: SPCConfig | None = None,
        feature_engineer: FeatureEngineer | None = None,
        enable_lstm: bool = False,
    ) -> None:
        self.stream_config = stream_config or StreamConfig()
        self.iforest_config = iforest_config or IsolationForestConfig()
        self.lstm_config = lstm_config or LSTMAutoencoderConfig()
        self.spc_config = spc_config or SPCConfig()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.enable_lstm = enable_lstm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def supported_periods(self) -> List[str]:
        """Return the canonical period labels supported by the detector."""

        return list(self._PERIODS.keys())

    def detect(
        self,
        tickers: Iterable[str] | None = None,
        period: str = "24h",
    ) -> HistoricalDetectionResult:
        """Run anomaly detection over the requested historical window."""

        symbols = self._resolve_tickers(tickers)
        period_definition = self._resolve_period(period)
        raw_history = self._download_history(symbols, period_definition)

        if raw_history.empty:
            empty = pd.DataFrame(columns=["timestamp", "symbol", "close", "volume"])
            return HistoricalDetectionResult(symbols, period_definition.label, empty, empty)

        features = self.feature_engineer.transform(raw_history)
        if features.empty:
            empty = pd.DataFrame(columns=["timestamp", "symbol", "close", "volume"])
            return HistoricalDetectionResult(symbols, period_definition.label, empty, empty)

        merged = self._run_detectors(features)
        anomalies = merged[merged.get("any_anomaly", False)].copy()
        return HistoricalDetectionResult(symbols, period_definition.label, merged, anomalies)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_tickers(self, tickers: Iterable[str] | None) -> List[str]:
        if tickers is None:
            return list(self.stream_config.tickers)
        symbols = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
        if not symbols:
            raise ValueError("At least one ticker symbol must be provided")
        return symbols

    def _resolve_period(self, period: str) -> PeriodDefinition:
        key = period.lower().replace(" ", "")
        canonical = self._ALIASES.get(key)
        if canonical is None:
            raise ValueError(
                f"Unsupported period '{period}'. Allowed values: {', '.join(self._PERIODS)}"
            )
        return self._PERIODS[canonical]

    def _download_history(
        self, tickers: Sequence[str], definition: PeriodDefinition
    ) -> pd.DataFrame:
        streamer = YahooFinanceStreamer(
            tickers=tickers,
            interval=definition.interval,
            lookback_period=definition.yfinance_period,
            poll_interval=self.stream_config.poll_interval,
        )
        return streamer.fetch()

    def _build_detectors(self) -> Dict[str, object]:
        detectors: MutableMapping[str, object] = {
            "Isolation Forest": IsolationForestDetector(
                window_size=self.iforest_config.window_size,
                contamination=self.iforest_config.contamination,
                min_train_size=self.iforest_config.min_train_size,
                random_state=self.iforest_config.random_state,
            ),
            "SPC": SPCDetector(
                window_size=self.spc_config.window_size,
                sigma_threshold=self.spc_config.sigma_threshold,
            ),
        }
        if self.enable_lstm:
            detectors["LSTM Autoencoder"] = LSTMAutoencoderDetector(
                sequence_length=self.lstm_config.sequence_length,
                encoding_dim=self.lstm_config.encoding_dim,
                learning_rate=self.lstm_config.learning_rate,
                epochs=self.lstm_config.epochs,
                retrain_interval=self.lstm_config.retrain_interval,
                min_train_size=self.lstm_config.min_train_size,
            )
        return dict(detectors)

    def _run_detectors(self, features: pd.DataFrame) -> pd.DataFrame:
        detectors = self._build_detectors()
        base_columns = [
            "timestamp",
            "symbol",
            "close",
            "volume",
            "return_1",
            "return_5",
            "return_15",
            "volatility",
            "volume_zscore",
        ]
        existing = [col for col in base_columns if col in features.columns]
        merged = features[existing].copy()

        results = []
        for detector in detectors.values():
            output = detector.update(features)
            if output is None or output.empty:
                continue
            results.append(output)

        if results:
            merged = reduce(
                lambda left, right: pd.merge(
                    left,
                    right,
                    on=["timestamp", "symbol"],
                    how="left",
                ),
                results,
                merged,
            )

        merged.sort_values(["symbol", "timestamp"], inplace=True)
        anomaly_columns = [col for col in merged.columns if col.endswith("_anomaly")]
        if anomaly_columns:
            merged["any_anomaly"] = merged[anomaly_columns].fillna(False).any(axis=1)
        else:
            merged["any_anomaly"] = False
        merged.reset_index(drop=True, inplace=True)
        return merged


__all__ = [
    "HistoricalAnomalyDetector",
    "HistoricalDetectionResult",
    "PeriodDefinition",
]
