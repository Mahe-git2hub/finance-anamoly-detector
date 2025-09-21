"""LSTM autoencoder based anomaly detection using TensorFlow/Keras."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf


def _build_autoencoder(
    sequence_length: int,
    input_dim: int,
    encoding_dim: int,
    learning_rate: float,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(sequence_length, input_dim))
    encoded = tf.keras.layers.LSTM(encoding_dim, return_sequences=False)(inputs)
    repeated = tf.keras.layers.RepeatVector(sequence_length)(encoded)
    decoded = tf.keras.layers.LSTM(encoding_dim, return_sequences=True)(repeated)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(input_dim)
    )(decoded)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


@dataclass
class LSTMAutoencoderDetector:
    """Maintains an LSTM autoencoder per symbol."""

    sequence_length: int = 20
    encoding_dim: int = 16
    learning_rate: float = 1e-3
    epochs: int = 10
    retrain_interval: int = 60
    min_train_size: int = 120

    buffers: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    models: Dict[str, tf.keras.Model] = field(default_factory=dict, init=False)
    thresholds: Dict[str, float] = field(default_factory=dict, init=False)
    rows_since_train: Dict[str, int] = field(default_factory=dict, init=False)

    def update(self, features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return pd.DataFrame()

        required = {"timestamp", "symbol"}
        if not required.issubset(features.columns):
            raise ValueError("Features must include timestamp and symbol columns")

        feature_cols = [c for c in features.columns if c not in {"timestamp", "symbol"}]
        results: List[pd.DataFrame] = []

        for symbol, group in features.groupby("symbol"):
            buffer = self.buffers.get(symbol)
            if buffer is None:
                buffer = group.copy()
            else:
                buffer = pd.concat([buffer, group], ignore_index=True)
            max_size = max(
                self.min_train_size + self.retrain_interval + self.sequence_length,
                len(group),
                500,
            )
            buffer = buffer.tail(max_size).reset_index(drop=True)
            self.buffers[symbol] = buffer

            self.rows_since_train[symbol] = self.rows_since_train.get(symbol, 0) + len(group)

            if len(buffer) >= self.min_train_size:
                trained = self._maybe_train(symbol, buffer[feature_cols].to_numpy(np.float32))
                if trained:
                    self.rows_since_train[symbol] = 0

            if symbol not in self.models or len(buffer) < self.sequence_length:
                continue

            threshold = self.thresholds.get(symbol, np.inf)
            new_rows = buffer.tail(len(group)).reset_index(drop=True)
            start_idx = len(buffer) - len(new_rows)
            result_rows: List[dict] = []
            for i, row in new_rows.iterrows():
                end_idx = start_idx + i + 1
                if end_idx < self.sequence_length:
                    continue
                sequence = buffer.iloc[end_idx - self.sequence_length : end_idx][
                    feature_cols
                ].to_numpy(np.float32)
                error = self._score_sequence(symbol, sequence)
                result_rows.append(
                    {
                        "timestamp": row["timestamp"],
                        "symbol": symbol,
                        "lstm_error": error,
                        "lstm_threshold": threshold,
                        "lstm_anomaly": error > threshold,
                    }
                )
            if result_rows:
                results.append(pd.DataFrame(result_rows))

        if not results:
            return pd.DataFrame(
                columns=["timestamp", "symbol", "lstm_error", "lstm_threshold", "lstm_anomaly"]
            )
        return pd.concat(results, ignore_index=True)

    def _maybe_train(self, symbol: str, feature_array: np.ndarray) -> bool:
        should_train = symbol not in self.models
        if not should_train and self.rows_since_train.get(symbol, 0) >= self.retrain_interval:
            should_train = True

        if not should_train:
            return False

        sequences = self._build_sequences(feature_array)
        if sequences.size == 0:
            return False

        model = _build_autoencoder(
            sequence_length=self.sequence_length,
            input_dim=feature_array.shape[1],
            encoding_dim=self.encoding_dim,
            learning_rate=self.learning_rate,
        )

        batch_size = min(32, len(sequences))
        model.fit(
            sequences,
            sequences,
            epochs=self.epochs,
            batch_size=batch_size,
            verbose=0,
        )

        reconstruction = model.predict(sequences, verbose=0)
        errors = np.mean(np.square(reconstruction - sequences), axis=(1, 2))
        threshold = float(errors.mean() + 2 * errors.std())

        self.models[symbol] = model
        self.thresholds[symbol] = threshold
        return True

    def _build_sequences(self, values: np.ndarray) -> np.ndarray:
        if len(values) < self.sequence_length:
            return np.empty((0, self.sequence_length, values.shape[1]), dtype=np.float32)
        sequences = [
            values[i : i + self.sequence_length]
            for i in range(0, len(values) - self.sequence_length + 1)
        ]
        return np.stack(sequences)

    def _score_sequence(self, symbol: str, sequence: np.ndarray) -> float:
        model = self.models[symbol]
        sequence = np.expand_dims(sequence, axis=0)
        reconstruction = model.predict(sequence, verbose=0)
        error = np.mean(np.square(reconstruction - sequence))
        return float(error)


__all__ = ["LSTMAutoencoderDetector"]
