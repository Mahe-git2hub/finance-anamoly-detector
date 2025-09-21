import numpy as np
import pandas as pd
import tensorflow as tf

from finance_anomaly_detector.detectors.lstm_autoencoder import LSTMAutoencoderDetector


def test_lstm_autoencoder_detector_returns_scores():
    tf.random.set_seed(42)
    np.random.seed(42)

    timestamps = pd.date_range("2024-01-01", periods=60, freq="min")
    signal = np.sin(np.linspace(0, 6 * np.pi, len(timestamps)))
    noise = np.random.normal(scale=0.01, size=len(timestamps))

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["TEST"] * len(timestamps),
            "close": 100 + signal + noise,
            "volume": 1000 + 10 * np.cos(np.linspace(0, 6 * np.pi, len(timestamps))),
            "return_1": np.gradient(signal + noise),
            "return_5": np.gradient(signal + noise, 5),
            "volatility": np.abs(np.gradient(signal + noise, 2)),
            "volume_mean": 1000,
            "volume_zscore": np.gradient(signal + noise),
        }
    )

    detector = LSTMAutoencoderDetector(
        sequence_length=5,
        encoding_dim=4,
        epochs=2,
        retrain_interval=5,
        min_train_size=20,
    )
    results = detector.update(data)

    assert not results.empty
    assert {"timestamp", "symbol", "lstm_error", "lstm_anomaly"}.issubset(results.columns)
    assert (results["lstm_error"] >= 0).all()
