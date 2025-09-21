"""FastAPI application exposing anomaly detection endpoints."""
from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from .config import (
    IsolationForestConfig,
    LSTMAutoencoderConfig,
    SPCConfig,
    StreamConfig,
)
from .historical import HistoricalAnomalyDetector
from .orchestrator import AnomalyOrchestrator, create_orchestrator

app = FastAPI(title="Finance Anomaly Detector", version="0.1.0")

_stream_config = StreamConfig()
_iforest_config = IsolationForestConfig()
_lstm_config = LSTMAutoencoderConfig()
_spc_config = SPCConfig()

_historical_detector = HistoricalAnomalyDetector(
    stream_config=_stream_config,
    iforest_config=_iforest_config,
    lstm_config=_lstm_config,
    spc_config=_spc_config,
    enable_lstm=False,
)

_live_state: Dict[str, object] = {"tickers": None, "orchestrator": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tickers(raw: str | None, default: Sequence[str]) -> List[str]:
    if raw is None:
        return list(default)
    symbols = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not symbols:
        raise ValueError("At least one ticker symbol must be supplied")
    return symbols


def _frame_to_records(frame: pd.DataFrame) -> List[dict]:
    if frame.empty:
        return []
    serialisable = frame.copy()
    for column in serialisable.columns:
        if pd.api.types.is_datetime64_any_dtype(serialisable[column]):
            serialisable[column] = serialisable[column].apply(
                lambda value: value.isoformat() if pd.notna(value) else None
            )
    serialisable = serialisable.where(pd.notnull(serialisable), None)
    return serialisable.to_dict(orient="records")


def _get_live_orchestrator(tickers: Sequence[str]) -> AnomalyOrchestrator:
    cached_tickers = _live_state.get("tickers")
    orchestrator = _live_state.get("orchestrator")
    if orchestrator is None or cached_tickers != tuple(tickers):
        orchestrator = create_orchestrator(
            tickers=tickers,
            stream_config=_stream_config,
            iforest_config=_iforest_config,
            lstm_config=_lstm_config,
            spc_config=_spc_config,
        )
        _live_state["orchestrator"] = orchestrator
        _live_state["tickers"] = tuple(tickers)
    return orchestrator  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", summary="Detect anomalies for the last 24 hours by default")
@app.get("/anomalies", summary="Detect anomalies across historical windows")
def historical_anomalies(
    period: str = Query(
        "24h",
        description="Lookback period, e.g. 24h, 2d, 1w, 2w, 1mo, 1q, 1y",
    ),
    tickers: str | None = Query(
        None, description="Comma separated ticker symbols (defaults to config)"
    ),
) -> dict:
    """Return anomaly detections for the requested historical window."""

    try:
        symbols = _parse_tickers(tickers, _historical_detector.stream_config.tickers)
        result = _historical_detector.detect(symbols, period=period)
    except ValueError as exc:  # invalid tickers or period
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "tickers": result.tickers,
        "period": result.period,
        "available_periods": _historical_detector.supported_periods(),
        "row_count": int(len(result.detections)),
        "detections": _frame_to_records(result.detections),
        "anomalies": _frame_to_records(result.anomalies),
    }


@app.get("/live", summary="Stream the most recent anomaly detections")
def live_anomalies(
    tickers: str | None = Query(
        None, description="Comma separated ticker symbols (defaults to config)"
    ),
) -> dict:
    """Expose the previous streaming implementation under the /live endpoint."""

    try:
        symbols = _parse_tickers(tickers, _stream_config.tickers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    orchestrator = _get_live_orchestrator(symbols)
    result = orchestrator.run_step()
    if result is None:
        return {
            "tickers": symbols,
            "message": "No fresh data available yet",
            "detections": [],
            "anomalies": [],
        }

    detections = result.get("detections", pd.DataFrame())
    if not isinstance(detections, pd.DataFrame):
        detections = pd.DataFrame()

    if not detections.empty and "any_anomaly" in detections.columns:
        anomalies = detections[detections["any_anomaly"]].copy()
    else:
        anomalies = pd.DataFrame()

    return {
        "tickers": symbols,
        "raw": _frame_to_records(result.get("raw", pd.DataFrame())),
        "features": _frame_to_records(result.get("features", pd.DataFrame())),
        "detections": _frame_to_records(detections),
        "anomalies": _frame_to_records(anomalies),
    }


__all__ = ["app"]
