from __future__ import annotations

from pathlib import Path

import pandas as pd

from anomaly_streaming.data.google_finance import parse_quote_from_html
from anomaly_streaming.features import FeatureBuilder, FEATURE_COLUMNS
from anomaly_streaming.utils import keep_last_rows


DATA_DIR = Path(__file__).parent / "data"


def test_parse_quote_from_html() -> None:
    html = (DATA_DIR / "google_reliance.html").read_text(encoding="utf-8")
    quote = parse_quote_from_html("RELIANCE:NSE", html)
    assert quote.symbol == "RELIANCE:NSE"
    assert quote.price > 0
    assert quote.currency == "INR"


def test_feature_builder_generates_expected_columns() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="min")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["TEST"] * len(dates),
            "price": pd.Series(range(1, len(dates) + 1), dtype=float) + 1000,
            "currency": ["INR"] * len(dates),
            "source": ["test"] * len(dates),
        }
    )
    builder = FeatureBuilder(window=10)
    features = builder.transform(df)
    for column in FEATURE_COLUMNS:
        assert column in features.columns
    assert not features.empty
    assert features["zscore"].abs().max() > 0


def test_keep_last_rows_limits_history() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="min"),
            "symbol": ["A"] * 10,
            "price": range(10),
        }
    )
    trimmed = keep_last_rows(df, limit=5)
    assert len(trimmed) == 5
    assert trimmed.iloc[0]["timestamp"] == df.iloc[5]["timestamp"]
