# Real-time Market Anomaly Detection for Indian Equities

This project demonstrates how to build a **streaming market monitoring system** for Indian equities that detects unusual trading behaviour in near real time. Intraday price and volume data are streamed from the publicly available [Yahoo Finance](https://finance.yahoo.com) API and analysed with multiple anomaly detection techniques. It can also continuously ingests quotes from the publicly available [Google Finance](https://www.google.com/finance/) pages for National Stock Exchange (NSE) listings. The pipeline retains context across polls so rolling features work whether the streamer emits a single new bar or a larger backfill, allowing both incremental and bulk updates to be processed reliably.

- **Isolation Forests** capture multivariate deviations across engineered statistical features for robust outlier detection on engineered features.
- **LSTM autoencoders** (TensorFlow/Keras) model temporal structure and highlight sequences that cannot be reconstructed well. Alternate Pytorch implementation also available - (PyTorch) to learn temporal behaviour and flag unusual sequences.
- **Statistical process control (SPC)** charts apply rolling z-score thresholds to short-term returns.

An interactive Streamlit dashboard ties the components together and visualises anomalies as they surface.

## Repository structure

**Tensorflow project idea**

```
├── README.md
├── requirements.txt
├── src/
│   └── finance_anomaly_detector/
│       ├── config.py
│       ├── data_stream.py
│       ├── feature_engineering.py
│       ├── orchestrator.py
│       ├── dashboards/
│       │   └── streamlit_app.py
│       └── detectors/
│           ├── isolation_forest.py
│           ├── lstm_autoencoder.py
│           └── spc.py
└── tests/
    ├── test_feature_engineering.py
    ├── test_isolation_forest.py
    ├── test_lstm_autoencoder.py
    └── test_spc.py
```


The system powers both a command line monitor and an interactive Plotly Dash dashboard that surfaces anomalies as they happen.

**Pytorch project structure**

```
├── pyproject.toml            # Packaging & dependency metadata
├── src/anomaly_streaming/
│   ├── cli.py                # CLI entry point
│   ├── config.py             # Config dataclasses
│   ├── dash_app.py           # Plotly Dash dashboard
│   ├── data/google_finance.py# Live data ingestion
│   ├── detectors/            # Isolation Forest, LSTM, SPC detectors
│   ├── features.py           # Feature engineering utilities
│   └── pipeline.py           # Streaming orchestration
└── tests/                    # Pytest-based regression tests
```

## Getting started

1. **Install dependencies** (preferably inside a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. **Install PyTorch CPU build** (once per environment):

   ```bash
   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu
   ```


3. **Run automated tests**:

   ```bash
   pytest
   ```

**Tensorflow specific instructions**

4. **Launch the real-time dashboard**:

   ```bash
   streamlit run src/finance_anomaly_detector/dashboards/streamlit_app.py
   ```

   The dashboard polls Yahoo Finance every minute for a configurable set of NSE-listed tickers (default: `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`). Toggle *Auto refresh* in the sidebar to keep the plots and detector tables up to date.

## How it works

1. **Streaming layer** – [`YahooFinanceStreamer`](src/finance_anomaly_detector/data_stream.py) polls Yahoo Finance for the latest one-minute bars, converts them to `Asia/Kolkata` timezone and yields only the new observations.

2. **Feature engineering** – [`FeatureEngineer`](src/finance_anomaly_detector/feature_engineering.py) creates rolling percentage returns, volatility estimates and volume z-scores used across all detectors.

3. **Anomaly detectors**:
   - [`IsolationForestDetector`](src/finance_anomaly_detector/detectors/isolation_forest.py) fits rolling isolation forests per symbol to flag low-density events.
   - [`LSTMAutoencoderDetector`](src/finance_anomaly_detector/detectors/lstm_autoencoder.py) trains lightweight TensorFlow/Keras LSTM autoencoders and scores the reconstruction error of recent sequences.
   - [`SPCDetector`](src/finance_anomaly_detector/detectors/spc.py) applies statistical process control limits to short-horizon returns to catch abrupt shocks.

4. **Orchestration** – [`AnomalyOrchestrator`](src/finance_anomaly_detector/orchestrator.py) stitches the pipeline together, preserves historical context and exposes detection results to other components. It feeds detectors with the cumulative history so both one-bar streaming updates and multi-row backfills share the same feature engineering context.

5. **Dashboard** – [`streamlit_app.py`](src/finance_anomaly_detector/dashboards/streamlit_app.py) provides live charts, detector scoreboards and anomaly alerts.

## Extending the system

- Add or remove detectors by updating the dictionary passed to `AnomalyOrchestrator` in the Streamlit app.
- Swap out the data source by implementing a new streamer class with the same interface as `YahooFinanceStreamer`.
- Integrate downstream alerting (Slack, email, etc.) by consuming the orchestrator's `detection_history` DataFrame.

## Notes

- Yahoo Finance intraday data is delayed by approximately one minute and subject to rate limits. The system retries on the next refresh cycle if no new data is available.
- The LSTM autoencoder keeps models small to run comfortably on CPU-only environments. Increase sequence length or model size for richer reconstructions.

Enjoy exploring live anomaly detection across Indian markets!

**Pytorch specific instructions**
4. **Start the real-time CLI monitor** (press `Ctrl+C` to stop):

   ```bash
   python -m anomaly_streaming.cli --interval 60 --symbols RELIANCE:NSE TCS:NSE
   ```

5. **Launch the interactive dashboard** (opens a Plotly Dash server on port 8050):

   ```bash
   python -m anomaly_streaming.dash_app
   ```

   The dashboard refreshes at the configured cadence, plots the latest prices and highlights anomalies from each detector ensemble.

## Data source

Quotes are scraped from the public Google Finance instrument pages (e.g. `https://www.google.com/finance/quote/RELIANCE:NSE`). Only the lightweight HTML response is used; no authentication or rate-limited APIs are required.

## Extending

- Add more tickers by editing `PipelineConfig` or passing them via CLI/ dashboard selector.
- Tune detection sensitivity through `DetectorConfig` in `config.py` (window sizes, contamination, sequence length, sigma limits, etc.).
- Swap the data ingestor to another Indian market source by implementing a new client that returns the `Quote` dataclass.

## Testing & observability

The `tests/` folder provides regression coverage for the Google Finance parser, feature engineering and history management utilities. The pipeline itself logs anomaly decisions via an internal dataframe (`RealTimeAnomalyPipeline.get_anomaly_log`) which is also exposed in the dashboard.

---

> **Note**: Live market data availability depends on Google Finance's/Yahoo Finance uptime and any corporate firewalls/proxy restrictions. When the service is unreachable, the pipeline keeps running and logs fetch errors without crashing.
