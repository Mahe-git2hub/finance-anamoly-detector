"""Dash application exposing the real-time anomaly detector."""
from __future__ import annotations

from typing import List

import dash
from dash import Dash, Input, Output, dcc, html
from dash.dash_table import DataTable
import pandas as pd
import plotly.graph_objects as go

from .config import PipelineConfig
from .pipeline import RealTimeAnomalyPipeline


CONFIG = PipelineConfig()
PIPELINE = RealTimeAnomalyPipeline(CONFIG)


def build_price_figure(history: pd.DataFrame, anomalies: pd.DataFrame, symbols: List[str]) -> go.Figure:
    fig = go.Figure()
    for symbol in symbols:
        symbol_history = history[history["symbol"] == symbol]
        if symbol_history.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=symbol_history["timestamp"],
                y=symbol_history["price"],
                mode="lines+markers",
                name=f"{symbol} price",
            )
        )
        anomaly_rows = anomalies[(anomalies["symbol"] == symbol) & (anomalies["is_anomaly"])]
        if not anomaly_rows.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_rows["timestamp"],
                    y=anomaly_rows["price"],
                    mode="markers",
                    marker=dict(color="#d7191c", size=12, symbol="x"),
                    name=f"{symbol} anomalies",
                )
            )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Timestamp",
        yaxis_title="Price",
        legend_title="Legend",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def results_to_table(rows) -> List[dict]:
    data = []
    for result in rows:
        data.append(
            {
                "symbol": result.symbol,
                "timestamp": result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "price": round(result.price, 2),
                "isolation": None if result.isolation_score is None else round(result.isolation_score, 3),
                "lstm": None if result.lstm_score is None else round(result.lstm_score, 3),
                "spc": None if result.spc_score is None else round(result.spc_score, 3),
                "combined": None if result.combined_score is None else round(result.combined_score, 3),
                "anomaly": "Yes" if result.is_anomaly else "No",
            }
        )
    return data


app = Dash(__name__)
app.title = "Indian Markets - Real-time Anomaly Detection"
app.layout = html.Div(
    [
        html.H2("Indian Markets - Real-time Anomaly Detection"),
        html.P("Live anomaly detection using Isolation Forest, LSTM autoencoder and SPC"),
        html.Div(
            [
                html.Label("Symbols"),
                dcc.Dropdown(
                    id="symbol-selector",
                    options=[{"label": symbol, "value": symbol} for symbol in CONFIG.symbols],
                    value=CONFIG.symbols,
                    multi=True,
                ),
            ],
            className="controls",
        ),
        dcc.Graph(id="price-graph"),
        DataTable(
            id="anomaly-table",
            columns=[
                {"name": "Symbol", "id": "symbol"},
                {"name": "Timestamp", "id": "timestamp"},
                {"name": "Price", "id": "price"},
                {"name": "Isolation", "id": "isolation"},
                {"name": "LSTM", "id": "lstm"},
                {"name": "SPC", "id": "spc"},
                {"name": "Combined", "id": "combined"},
                {"name": "Anomaly", "id": "anomaly"},
            ],
            data=[],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "0.5rem"},
        ),
        dcc.Interval(id="refresh-timer", interval=CONFIG.fetch_interval_seconds * 1000, n_intervals=0),
    ],
    style={"backgroundColor": "#111", "color": "#fff", "minHeight": "100vh", "padding": "1rem"},
)


@app.callback(
    Output("price-graph", "figure"),
    Output("anomaly-table", "data"),
    Output("anomaly-table", "style_data_conditional"),
    Input("refresh-timer", "n_intervals"),
    Input("symbol-selector", "value"),
)
def update_dashboard(_: int, selected_symbols: List[str]):  # type: ignore[override]
    symbols = selected_symbols or CONFIG.symbols
    results = PIPELINE.fetch_and_update(symbols)
    history = PIPELINE.get_history(symbols)
    anomalies = PIPELINE.get_anomaly_log(symbols)
    figure = build_price_figure(history, anomalies, symbols)
    table_data = results_to_table(results)
    styles = [
        {
            "if": {"filter_query": '{anomaly} = "Yes"'},
            "backgroundColor": "#660708",
            "color": "#fff",
        }
    ]
    return figure, table_data, styles


def main() -> None:
    app.run_server(host="0.0.0.0", port=8050, debug=False)


if __name__ == "__main__":
    main()
