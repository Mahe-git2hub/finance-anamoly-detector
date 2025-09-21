"""Tests for the HTML visualisation layer exposed by the API."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

import finance_anomaly_detector.api as api
from finance_anomaly_detector.historical import HistoricalDetectionResult


class _StubHistoricalDetector:
    """Minimal stub to provide deterministic API responses."""

    def __init__(self) -> None:
        self.stream_config = SimpleNamespace(tickers=["AAA.NS", "BBB.NS"])

    def detect(self, tickers, period: str = "24h") -> HistoricalDetectionResult:  # pragma: no cover - exercised in tests
        timestamps = pd.to_datetime(
            ["2024-01-01 09:15", "2024-01-01 09:20", "2024-01-01 09:25"], utc=True
        )
        detections = pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": [tickers[0]] * len(timestamps),
                "close": [100.0, 101.5, 105.2],
                "volume": [1_000, 1_050, 1_400],
                "any_anomaly": [False, False, True],
                "return_1": [0.0, 0.015, 0.036],
            }
        )
        anomalies = detections[detections["any_anomaly"]].copy()
        return HistoricalDetectionResult(list(tickers), period, detections, anomalies)

    def supported_periods(self):  # pragma: no cover - trivial
        return ["24h", "2d", "1w"]


def test_visualize_flag_returns_html_dashboard(monkeypatch):
    """When visualize=true the endpoint should return an HTML document."""

    stub = _StubHistoricalDetector()
    monkeypatch.setattr(api, "_historical_detector", stub)

    client = TestClient(api.app)
    response = client.get("/anomalies", params={"visualize": "true"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Historical anomaly overview" in response.text
    assert "No anomalies" not in response.text  # anomalies exist in the stub


def test_visualize_flag_handles_empty_results(monkeypatch):
    """Ensure a pleasant message is rendered when no data is returned."""

    class EmptyStub(_StubHistoricalDetector):
        def detect(self, tickers, period: str = "24h") -> HistoricalDetectionResult:  # pragma: no cover - exercised in test
            empty = pd.DataFrame(columns=["timestamp", "symbol", "close", "volume", "any_anomaly"])
            return HistoricalDetectionResult(list(tickers), period, empty, empty)

    stub = EmptyStub()
    monkeypatch.setattr(api, "_historical_detector", stub)

    client = TestClient(api.app)
    response = client.get("/anomalies", params={"visualize": "true"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "No market data" in response.text

