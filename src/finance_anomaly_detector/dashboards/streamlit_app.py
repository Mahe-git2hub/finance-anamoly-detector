"""Streamlit dashboard for the real-time anomaly detector."""
from __future__ import annotations

import time
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure the package is importable when running the file directly via Streamlit.
# This supports "src/" layout without requiring an editable install during dev.
import sys
from pathlib import Path

if __package__ in (None, ""):
    pkg_root = Path(__file__).resolve().parents[2]  # points to the "src" folder
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

from finance_anomaly_detector.config import (
    DashboardConfig,
    IsolationForestConfig,
    LSTMAutoencoderConfig,
    SPCConfig,
    StreamConfig,
)
from finance_anomaly_detector.orchestrator import AnomalyOrchestrator, create_orchestrator


def _render_price_panel(data: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    st.subheader("Live Prices")
    if data.empty:
        st.info("Waiting for the first batch of streaming data...")
        return
    fig = px.line(
        data,
        x="timestamp",
        y="close",
        color="symbol",
        title="Intraday price evolution",
    )
    fig.update_layout(height=400)
    if not anomalies.empty:
        fig.add_scatter(
            x=anomalies["timestamp"],
            y=anomalies["close"],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Anomaly",
        )
    st.plotly_chart(fig, use_container_width=True)


def _render_detector_summary(detections: pd.DataFrame) -> None:
    st.subheader("Detector scores")
    if detections.empty:
        st.info("Detectors are warming up – need more history to make decisions.")
        return

    latest = (
        detections.sort_values("timestamp")
        .groupby("symbol", as_index=False)
        .tail(1)
        .sort_values("symbol")
    )
    cols = st.columns(len(latest) or 1)
    for col, (_, row) in zip(cols, latest.iterrows()):
        col.metric(
            label=f"{row['symbol']} anomalies",
            value="Yes" if row.get("any_anomaly", False) else "No",
        )

    detector_cols = [
        col
        for col in detections.columns
        if col.endswith("_score") or col.endswith("_error") or col.endswith("_zscore")
    ]
    if detector_cols:
        display_cols = ["timestamp", "symbol"] + detector_cols + ["any_anomaly"]
        st.dataframe(
            detections[display_cols].sort_values("timestamp", ascending=False).head(50),
            use_container_width=True,
        )

    anomalies = detections[detections["any_anomaly"]]
    if not anomalies.empty:
        st.warning("⚠️ Real-time anomaly detected!", icon="⚠️")
        st.dataframe(anomalies.tail(20), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Market Anomaly Monitor", layout="wide")
    st.title("📈 Real-time Market Anomaly Detection – India")
    st.write(
        "Monitor NSE listed equities and surface unusual price and volume behaviour using "
        "Isolation Forests, LSTM autoencoders and statistical process control charts."
    )

    stream_config = StreamConfig()
    iforest_config = IsolationForestConfig()
    lstm_config = LSTMAutoencoderConfig()
    spc_config = SPCConfig()
    dashboard_config = DashboardConfig()

    default_tickers = ",".join(stream_config.tickers)
    ticker_text = st.sidebar.text_input(
        "Tracked tickers (comma separated)", value=default_tickers
    )
    tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
    if not tickers:
        st.error("Please configure at least one NSE ticker symbol to monitor.")
        st.stop()

    auto_refresh = st.sidebar.toggle("Auto refresh", value=True)
    refresh_seconds = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=15,
        max_value=180,
        value=dashboard_config.refresh_seconds,
        step=5,
    )

    if (
        "orchestrator" not in st.session_state
        or st.session_state.get("orchestrator_tickers") != tuple(tickers)
    ):
        st.session_state["orchestrator"] = create_orchestrator(
            tickers=tickers,
            stream_config=stream_config,
            iforest_config=iforest_config,
            lstm_config=lstm_config,
            spc_config=spc_config,
        )
        st.session_state["orchestrator_tickers"] = tuple(tickers)

    orchestrator: AnomalyOrchestrator = st.session_state["orchestrator"]

    result = orchestrator.run_step()
    if result is None:
        st.info("No fresh data available yet. Yahoo Finance updates 1-minute bars shortly after publication.")
    else:
        detections = result["detections"]
        anomaly_rows = detections[detections.get("any_anomaly", False)] if not detections.empty else pd.DataFrame()
        with st.container():
            _render_price_panel(orchestrator.raw_history, anomaly_rows)
        with st.container():
            _render_detector_summary(detections if not detections.empty else pd.DataFrame())

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Data Source: [Yahoo Finance](https://finance.yahoo.com) – 1 minute delayed NSE market data."
    )
    st.sidebar.markdown("Built with ❤️ using Streamlit by Mahesh G, scikit-learn, TensorFlow and pandas.")

    if auto_refresh:
        time.sleep(refresh_seconds)
        if hasattr(st, 'rerun'):
            st.rerun()
        else:
            st.experimental_rerun()


if __name__ == "__main__":
    main()

