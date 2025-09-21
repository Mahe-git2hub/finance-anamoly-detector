import numpy as np
import pandas as pd

from finance_anomaly_detector.detectors.isolation_forest import IsolationForestDetector


def test_isolation_forest_detector_flags_scores():
    timestamps = pd.date_range("2024-01-01", periods=40, freq="min")
    base = np.sin(np.linspace(0, 4 * np.pi, 40))
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["TEST"] * len(timestamps),
            "close": 100 + base,
            "volume": 1000 + 20 * base,
            "return_1": np.gradient(base),
            "return_5": np.gradient(base, 5),
            "volatility": np.abs(np.gradient(base, 2)),
            "volume_mean": 1000,
            "volume_zscore": np.gradient(base),
        }
    )

    detector = IsolationForestDetector(window_size=30, min_train_size=20, contamination=0.1)
    results = detector.update(data)

    assert not results.empty
    assert {"timestamp", "symbol", "iforest_score", "iforest_anomaly"}.issubset(results.columns)
