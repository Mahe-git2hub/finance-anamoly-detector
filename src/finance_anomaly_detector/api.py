"""FastAPI application exposing anomaly detection endpoints."""
from __future__ import annotations

from itertools import cycle
from typing import Dict, List, Sequence

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from plotly.colors import qualitative

from .config import (
    IsolationForestConfig,
    LSTMAutoencoderConfig,
    SPCConfig,
    StreamConfig,
)
from .historical import HistoricalAnomalyDetector, HistoricalDetectionResult
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


def _format_frame_for_display(frame: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Prepare a DataFrame for human friendly HTML rendering."""

    if frame.empty:
        return frame

    formatted = frame.copy()
    if limit is not None:
        formatted = formatted.head(limit)

    for column in formatted.columns:
        series = formatted[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            formatted[column] = series.dt.strftime("%Y-%m-%d %H:%M")
        elif pd.api.types.is_bool_dtype(series):
            formatted[column] = series.map({True: "Yes", False: "No"})
        elif pd.api.types.is_float_dtype(series):
            formatted[column] = series.round(3)

    return formatted.fillna("")


def _build_price_figure(result: HistoricalDetectionResult) -> go.Figure:
    """Create a Plotly figure showing close prices with anomaly markers."""

    detections = result.detections.copy()
    detections = detections.sort_values(["symbol", "timestamp"])

    palette_source = qualitative.Plotly
    if not palette_source:
        palette_source = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    palette = cycle(palette_source)
    fig = go.Figure()

    for symbol, group in detections.groupby("symbol"):
        colour = next(palette)
        fig.add_trace(
            go.Scatter(
                x=group["timestamp"],
                y=group["close"],
                mode="lines",
                name=f"{symbol} close",
                line=dict(color=colour, width=2),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Close: %{y:.2f}<extra>{symbol}</extra>",
            )
        )

        anomalies = result.anomalies[result.anomalies["symbol"] == symbol]
        if not anomalies.empty:
            anomalies = anomalies.sort_values("timestamp")
            fig.add_trace(
                go.Scatter(
                    x=anomalies["timestamp"],
                    y=anomalies["close"],
                    mode="markers",
                    name=f"{symbol} anomaly",
                    marker=dict(color="#EF553B", size=10, symbol="diamond", line=dict(color="white", width=1)),
                    hovertemplate=(
                        "%{x|%Y-%m-%d %H:%M}<br>Close: %{y:.2f}<br>"
                        + "<extra>Anomaly</extra>"
                    ),
                    showlegend=True,
                )
            )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Close price")
    fig.update_xaxes(title_text="Timestamp")
    return fig


def _build_symbol_summary(result: HistoricalDetectionResult) -> pd.DataFrame:
    if result.detections.empty:
        return pd.DataFrame(columns=["Symbol", "Data points", "Anomalies", "Anomaly %"])

    detections = result.detections.copy()
    if "any_anomaly" not in detections.columns:
        detections["any_anomaly"] = False
    detections["any_anomaly"] = detections["any_anomaly"].fillna(False).astype(bool)

    summary = (
        detections.groupby("symbol", as_index=False)
        .agg(
            data_points=("timestamp", "size"),
            anomalies=("any_anomaly", "sum"),
        )
        .rename(columns={"symbol": "Symbol", "data_points": "Data points", "anomalies": "Anomalies"})
    )
    summary["Anomalies"] = summary["Anomalies"].astype(int)
    summary["Anomaly %"] = (
        (summary["Anomalies"] / summary["Data points"].replace(0, pd.NA)) * 100
    ).round(2).fillna(0.0)
    return summary


def _render_historical_visualisation(
    result: HistoricalDetectionResult, available_periods: Sequence[str]
) -> str:
    """Render a lightweight HTML dashboard for historical detections."""

    title = "Historical anomaly overview"
    subtitle = (
        f"Analysing {', '.join(result.tickers)} over the past {result.period}. "
        f"Switch periods with: {', '.join(available_periods)}."
    )

    if result.detections.empty:
        return f"""<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>{title}</title>
    <style>
      body {{ font-family: 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 2rem; background: #f7f9fc; color: #1f2933; }}
      .card {{ background: white; padding: 2rem; border-radius: 18px; box-shadow: 0 15px 35px rgba(15, 23, 42, 0.08); max-width: 960px; margin: 2rem auto; text-align: center; }}
      h1 {{ margin-top: 0; font-size: 2.2rem; }}
      p {{ margin: 0.5rem 0; font-size: 1.05rem; }}
    </style>
  </head>
  <body>
    <section class=\"card\">
      <h1>{title}</h1>
      <p>{subtitle}</p>
      <p><strong>No market data</strong> was returned for the selected configuration. Try a different period or ticker.</p>
    </section>
  </body>
</html>"""

    figure = _build_price_figure(result)
    figure_html = pio.to_html(figure, include_plotlyjs="cdn", full_html=False)

    summary = _build_symbol_summary(result)
    summary_html = _format_frame_for_display(summary).to_html(
        index=False,
        classes="table summary",
        border=0,
        justify="center",
    )

    anomalies = result.anomalies.copy()
    preferred_columns = [
        "timestamp",
        "symbol",
        "close",
        "volume",
        "return_1",
        "return_5",
        "any_anomaly",
    ]
    ordered_columns = [col for col in preferred_columns if col in anomalies.columns]
    for col in anomalies.columns:
        if col not in ordered_columns:
            ordered_columns.append(col)
    if ordered_columns:
        anomalies = anomalies[ordered_columns]
    anomalies = anomalies.sort_values("timestamp", ascending=False)
    anomalies_html = (
        _format_frame_for_display(anomalies, limit=200).to_html(
            index=False,
            classes="table anomalies",
            border=0,
            justify="center",
        )
        if not anomalies.empty
        else "<p class=\"empty\">No anomalies detected in this window 🎉</p>"
    )

    return f"""<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>{title}</title>
    <style>
      :root {{ color-scheme: light dark; }}
      body {{ font-family: 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 2rem; background: #f2f5fa; color: #0f172a; }}
      h1 {{ font-size: 2.4rem; margin: 0; }}
      h2 {{ font-size: 1.4rem; margin-top: 2.5rem; }}
      p.subtitle {{ font-size: 1.05rem; color: #475569; margin-top: 0.75rem; max-width: 960px; }}
      main {{ max-width: 1100px; margin: 0 auto; }}
      section.card {{ background: white; padding: 2rem; border-radius: 20px; box-shadow: 0 25px 55px rgba(15, 23, 42, 0.12); margin-top: 2rem; }}
      .chart {{ margin-top: 1.5rem; }}
      table.table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.95rem; }}
      table.table th, table.table td {{ padding: 0.6rem 0.75rem; text-align: center; border-bottom: 1px solid #e2e8f0; }}
      table.table tr:nth-child(even) {{ background: #f8fafc; }}
      table.table thead {{ background: #0f172a; color: white; }}
      table.table.summary thead {{ background: #1d4ed8; }}
      table.table.anomalies thead {{ background: #be123c; }}
      p.empty {{ font-style: italic; color: #64748b; margin-top: 1rem; }}
      @media (max-width: 768px) {{ body {{ padding: 1.2rem; }} section.card {{ padding: 1.4rem; }} }}
    </style>
  </head>
  <body>
    <main>
      <section class=\"card\">
        <h1>{title}</h1>
        <p class=\"subtitle\">{subtitle}</p>
        <div class=\"chart\">{figure_html}</div>
        <h2>Symbol summary</h2>
        {summary_html}
        <h2>Anomaly log</h2>
        {anomalies_html}
      </section>
    </main>
  </body>
</html>"""


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
    visualize: bool = Query(
        False,
        description="Return an interactive Plotly dashboard instead of JSON",
    ),
) -> dict:
    """Return anomaly detections for the requested historical window."""

    try:
        symbols = _parse_tickers(tickers, _historical_detector.stream_config.tickers)
        result = _historical_detector.detect(symbols, period=period)
    except ValueError as exc:  # invalid tickers or period
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if visualize:
        html = _render_historical_visualisation(
            result, available_periods=_historical_detector.supported_periods()
        )
        return HTMLResponse(content=html)

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
