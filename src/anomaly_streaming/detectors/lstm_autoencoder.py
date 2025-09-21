"""LSTM autoencoder for capturing temporal anomalies."""
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import pandas as pd

from .base import StreamingDetector

try:  # pragma: no cover - exercised in environments with torch installed
    import torch
    from torch import nn
except Exception:  # pragma: no cover - gracefully handle missing torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if torch is not None:

    class LSTMAutoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_size: int) -> None:
            super().__init__()
            self.encoder = nn.LSTM(input_dim, hidden_size, batch_first=True)
            self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.output_layer = nn.Linear(hidden_size, input_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            _, (hidden, _) = self.encoder(x)
            # repeat hidden state across the sequence length
            repeated = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)
            decoded, _ = self.decoder(repeated)
            return self.output_layer(decoded)


class LSTMAutoencoderDetector(StreamingDetector):
    def __init__(
        self,
        feature_columns: Sequence[str],
        sequence_length: int = 30,
        hidden_size: int = 32,
        epochs: int = 20,
        retrain_interval: int = 30,
        learning_rate: float = 5e-3,
        max_error_history: int = 400,
    ) -> None:
        super().__init__()
        self._feature_columns = list(feature_columns)
        self._sequence_length = sequence_length
        self._hidden_size = hidden_size
        self._epochs = epochs
        self._retrain_interval = retrain_interval
        self._learning_rate = learning_rate
        self._error_history: deque[float] = deque(maxlen=max_error_history)
        self._model: LSTMAutoencoder | None = None
        self._steps_since_fit = 0
        self._trained = False
        self._latest_features: pd.DataFrame | None = None
        if torch is None:
            self._available = False
            self._device = None
        else:
            self._available = True
            self._device = torch.device("cpu")

    def update(self, features: pd.DataFrame) -> None:
        if not self._available or features.empty or len(features) < self._sequence_length:
            return
        self._latest_features = features.copy()
        if not self._trained or self._steps_since_fit >= self._retrain_interval:
            sequences = self._build_sequences(features)
            if sequences is None:
                return
            self._train(sequences)
            self._steps_since_fit = 0
            self._mark_ready()
        else:
            self._steps_since_fit += 1

    def _build_sequences(self, features: pd.DataFrame) -> torch.Tensor | None:
        if torch is None:
            return None
        matrix = features[self._feature_columns].to_numpy(dtype=np.float32)
        if matrix.shape[0] < self._sequence_length:
            return None
        sequences = []
        for idx in range(matrix.shape[0] - self._sequence_length + 1):
            window = matrix[idx : idx + self._sequence_length]
            sequences.append(window)
        if len(sequences) < 4:
            return None
        return torch.tensor(np.stack(sequences), dtype=torch.float32, device=self._device)

    def _train(self, sequences: torch.Tensor) -> None:
        if torch is None:
            return
        if self._model is None or sequences.shape[-1] != len(self._feature_columns):
            self._model = LSTMAutoencoder(len(self._feature_columns), self._hidden_size).to(self._device)
        assert self._model is not None
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        loss_fn = nn.MSELoss()
        self._model.train()
        for _ in range(self._epochs):
            permutation = torch.randperm(sequences.size(0))
            for start in range(0, sequences.size(0), 32):
                indices = permutation[start : start + 32]
                batch = sequences[indices]
                optimizer.zero_grad()
                output = self._model(batch)
                loss = loss_fn(output, batch)
                loss.backward()
                optimizer.step()
        self._trained = True

    def score(self, latest: pd.Series) -> float | None:
        if not self._available or not self._trained or self._model is None or torch is None:
            return None
        assert self._latest_features is not None
        values = self._latest_features[self._feature_columns].to_numpy(dtype=np.float32)
        if values.shape[0] < self._sequence_length:
            return None
        sequence = torch.tensor(values[-self._sequence_length :], dtype=torch.float32, device=self._device).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            reconstructed = self._model(sequence)
        loss = torch.mean((sequence - reconstructed) ** 2).item()
        self._error_history.append(loss)
        if len(self._error_history) < 5:
            return loss
        mean_error = float(np.mean(self._error_history))
        std_error = float(np.std(self._error_history)) or 1e-6
        z = (loss - mean_error) / std_error
        normalised = 1 / (1 + np.exp(-z))
        return float(np.clip(normalised, 0.0, 1.0))
