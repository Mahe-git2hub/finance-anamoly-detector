from __future__ import annotations

from typing import Sequence

import pandas as pd
import pytest

from finance_anomaly_detector.config import (
    IsolationForestConfig,
    SPCConfig,
    StreamConfig,
)
from finance_anomaly_detector.historical import (
    HistoricalAnomalyDetector,
    PeriodDefinition,
)


class DummyHistoricalDetector(HistoricalAnomalyDetector):
    """Test double overriding network access."""

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(
            stream_config=StreamConfig(tickers=["TEST.NS"]),
            iforest_config=IsolationForestConfig(
                window_size=20,
                contamination=0.2,
                min_train_size=10,
            ),
            spc_config=SPCConfig(window_size=5, sigma_threshold=1.0),
            enable_lstm=False,
        )
        self._data = data

    def _download_history(
        self, tickers: Sequence[str], definition: PeriodDefinition
    ) -> pd.DataFrame:
        return self._data.copy()


@pytest.fixture
def synthetic_history() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=60,
        freq="H",
        tz="Asia/Kolkata",
    )
    prices = pd.Series(100.0, index=timestamps).cumsum() / 10
    prices.iloc[-5:] += [0.5, 1.0, 1.5, 2.0, 15.0]
    volume = pd.Series(1_000.0, index=timestamps)
    volume.iloc[-1] = 10_000.0
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["TEST.NS"] * len(timestamps),
            "close": prices.to_numpy(),
            "volume": volume.to_numpy(),
        }
    )
    return frame


def test_detect_returns_anomalies_for_alias_period(synthetic_history: pd.DataFrame) -> None:
    detector = DummyHistoricalDetector(synthetic_history)
    result = detector.detect(["TEST.NS"], period="1 month")
    assert result.period == "1mo"
    assert result.tickers == ["TEST.NS"]
    assert "any_anomaly" in result.detections.columns
    assert not result.detections.empty


def test_invalid_period_raises() -> None:
    detector = HistoricalAnomalyDetector(enable_lstm=False)
    with pytest.raises(ValueError):
        detector.detect(period="5years")
