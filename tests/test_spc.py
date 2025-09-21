import numpy as np
import pandas as pd

from finance_anomaly_detector.detectors.spc import SPCDetector


def test_spc_detector_computes_zscores():
    timestamps = pd.date_range("2024-01-01", periods=50, freq="min")
    returns = np.random.normal(0, 0.01, size=len(timestamps))
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["TEST"] * len(timestamps),
            "return_1": returns,
        }
    )

    detector = SPCDetector(window_size=10, sigma_threshold=2.5)
    results = detector.update(data)

    assert not results.empty
    assert {"timestamp", "symbol", "spc_zscore", "spc_anomaly"}.issubset(results.columns)
    assert (results["spc_zscore"].abs() >= 0).all()
