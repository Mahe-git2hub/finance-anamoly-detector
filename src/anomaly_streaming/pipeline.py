"""End-to-end pipeline orchestrating ingestion and detectors."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data.google_finance import GoogleFinanceClient
from .detectors.isolation_forest import IsolationForestDetector
from .detectors.lstm_autoencoder import LSTMAutoencoderDetector
from .detectors.spc import SPCDetector
from .features import FeatureBuilder
from .utils import AnomalyResult, Quote, keep_last_rows


@dataclass(slots=True)
class DetectorBundle:
    isolation: IsolationForestDetector
    lstm: LSTMAutoencoderDetector
    spc: SPCDetector


class RealTimeAnomalyPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.client = GoogleFinanceClient()
        self.feature_builder = FeatureBuilder(window=self.config.detector.feature_window)
        self.history = pd.DataFrame(columns=["timestamp", "symbol", "price", "currency", "source"])
        self._detectors: Dict[str, DetectorBundle] = {}
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._anomaly_log = pd.DataFrame(columns=["timestamp", "symbol", "price", "combined", "is_anomaly"])

    def _get_detectors(self, symbol: str) -> DetectorBundle:
        if symbol not in self._detectors:
            feature_cols = [
                "return",
                "log_return",
                "rolling_std",
                "volatility",
                "zscore",
                "ewma_zscore",
                "momentum",
            ]
            bundle = DetectorBundle(
                isolation=IsolationForestDetector(
                    feature_columns=feature_cols,
                    contamination=self.config.detector.isolation_contamination,
                    retrain_interval=self.config.detector.isolation_retrain,
                ),
                lstm=LSTMAutoencoderDetector(
                    feature_columns=["return", "log_return", "volatility"],
                    sequence_length=self.config.detector.lstm_sequence,
                    hidden_size=self.config.detector.lstm_hidden_size,
                    epochs=self.config.detector.lstm_epochs,
                    retrain_interval=self.config.detector.lstm_retrain,
                ),
                spc=SPCDetector(self.config.detector.spc_sigma),
            )
            self._detectors[symbol] = bundle
        return self._detectors[symbol]

    def _append_history(self, quotes: Sequence[Quote]) -> None:
        if not quotes:
            return
        quote_fields = [field.name for field in fields(Quote)]
        frame = pd.DataFrame(
            [{name: getattr(quote, name) for name in quote_fields} for quote in quotes]
        )
        self.history = pd.concat([self.history, frame], ignore_index=True)
        self.history = keep_last_rows(self.history, self.config.history_limit)

    def _compute_features(self, symbol: str) -> pd.DataFrame:
        symbol_history = self.history[self.history["symbol"] == symbol]
        if symbol_history.empty:
            return pd.DataFrame()
        features = self.feature_builder.transform(symbol_history)
        self._feature_cache[symbol] = features
        return features

    def fetch_and_update(self, symbols: Iterable[str] | None = None) -> List[AnomalyResult]:
        target_symbols = list(symbols or self.config.symbols)
        quotes = self.client.fetch_quotes(target_symbols)
        self._append_history(quotes)
        results: List[AnomalyResult] = []
        for symbol in target_symbols:
            features = self._compute_features(symbol)
            if features.empty:
                continue
            latest = features.iloc[-1]
            detectors = self._get_detectors(symbol)
            detectors.isolation.update(features)
            iso_score = detectors.isolation.score(latest)
            detectors.lstm.update(features)
            lstm_score = detectors.lstm.score(latest)
            detectors.spc.update(features)
            spc_score = detectors.spc.score(latest)
            scores = [score for score in [iso_score, lstm_score, spc_score] if score is not None]
            combined = float(np.mean(scores)) if scores else None
            is_anomaly = bool(combined is not None and combined >= 0.65)
            if detectors.spc.is_out_of_control(latest):
                is_anomaly = True
            result = AnomalyResult(
                symbol=symbol,
                timestamp=latest["timestamp"],
                price=float(latest["price"]),
                isolation_score=iso_score,
                lstm_score=lstm_score,
                spc_score=spc_score,
                combined_score=combined,
                is_anomaly=is_anomaly,
                details={
                    "iso_ready": detectors.isolation.is_ready(),
                    "lstm_ready": detectors.lstm.is_ready(),
                    "spc_ready": detectors.spc.is_ready(),
                },
            )
            results.append(result)
            self._anomaly_log = pd.concat(
                [
                    self._anomaly_log,
                    pd.DataFrame(
                        {
                            "timestamp": [result.timestamp],
                            "symbol": [result.symbol],
                            "price": [result.price],
                            "combined": [result.combined_score],
                            "is_anomaly": [result.is_anomaly],
                        }
                    ),
                ],
                ignore_index=True,
            )
            self._anomaly_log = keep_last_rows(self._anomaly_log, self.config.history_limit)
        return results

    def get_history(self, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        if symbols is None:
            return self.history.copy()
        mask = self.history["symbol"].isin(list(symbols))
        return self.history[mask].copy()

    def get_features(self, symbol: str) -> pd.DataFrame:
        return self._feature_cache.get(symbol, pd.DataFrame()).copy()

    def get_anomaly_log(self, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        if symbols is None:
            return self._anomaly_log.copy()
        mask = self._anomaly_log["symbol"].isin(list(symbols))
        return self._anomaly_log[mask].copy()

    def close(self) -> None:
        self.client.close()
