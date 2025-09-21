# Real-time Indian Market Anomaly Detection

This project delivers a full real-time anomaly detection stack for Indian equity markets. It continuously ingests quotes from the publicly available [Google Finance](https://www.google.com/finance/) pages for National Stock Exchange (NSE) listings, engineers streaming features and applies a rich ensemble of detectors:

- **Isolation Forests** for robust outlier detection on engineered features.
- **LSTM autoencoders** (PyTorch) to learn temporal behaviour and flag unusual sequences.
- **Statistical process control (SPC)** style rules (Shewhart & EWMA z-scores) for immediate control-limit breaches.

The system powers both a command line monitor and an interactive Plotly Dash dashboard that surfaces anomalies as they happen.

## Project structure

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

1. **Install PyTorch CPU build** (once per environment):

   ```bash
   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu
   ```

2. **Install project dependencies** (create a virtual environment first if desired):

   ```bash
   pip install -e .[dev]
   ```

3. **Run automated tests**:

   ```bash
   pytest
   ```

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

> **Note**: Live market data availability depends on Google Finance's uptime and any corporate firewalls/proxy restrictions. When the service is unreachable, the pipeline keeps running and logs fetch errors without crashing.
