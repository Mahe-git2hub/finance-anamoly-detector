"""Streamlit dashboard for the real-time anomaly detector."""
from __future__ import annotations

import time
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from ..config import (
    DashboardConfig,
    IsolationForestConfig,
    LSTMAutoencoderConfig,
    SPCConfig,
    StreamConfig,
)
from ..data_stream import YahooFinanceStreamer
from ..detectors.isolation_forest import IsolationForestDetector
from ..detectors.lstm_autoencoder import LSTMAutoencoderDetector
from ..detectors.spc import SPCDetector
from ..orchestrator import AnomalyOrchestrator


def _build_orchestrator(
    tickers: List[str],
    stream_config: StreamConfig,
    iforest_config: IsolationForestConfig,
    lstm_config: LSTMAutoencoderConfig,
    spc_config: SPCConfig,
) -> AnomalyOrchestrator:
    streamer = YahooFinanceStreamer(
        tickers=tickers,
        interval=stream_config.interval,
        lookback_period=stream_config.lookback_period,
        poll_interval=stream_config.poll_interval,
    )
    detectors: Dict[str, object] = {
        "Isolation Forest": IsolationForestDetector(
            window_size=iforest_config.window_size,
            contamination=iforest_config.contamination,
            min_train_size=iforest_config.min_train_size,
            random_state=iforest_config.random_state,
        ),
        "LSTM Autoencoder": LSTMAutoencoderDetector(
            sequence_length=lstm_config.sequence_length,
            encoding_dim=lstm_config.encoding_dim,
            learning_rate=lstm_config.learning_rate,
            epochs=lstm_config.epochs,
            retrain_interval=lstm_config.retrain_interval,
            min_train_size=lstm_config.min_train_size,
        ),
        "SPC": SPCDetector(
            window_size=spc_config.window_size,
            sigma_threshold=spc_config.sigma_threshold,
        ),
    }
    return AnomalyOrchestrator(streamer=streamer, detectors=detectors)


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
        st.session_state["orchestrator"] = _build_orchestrator(
            tickers, stream_config, iforest_config, lstm_config, spc_config
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
    st.sidebar.markdown("Built with ❤️ using Streamlit, scikit-learn, TensorFlow and pandas.")

    if auto_refresh:
        time.sleep(refresh_seconds)
        st.experimental_rerun()


if __name__ == "__main__":
    main()
