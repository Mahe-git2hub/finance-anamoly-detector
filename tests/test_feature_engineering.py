import pandas as pd

from finance_anomaly_detector.feature_engineering import FeatureEngineer


def test_feature_engineer_creates_expected_columns():
    timestamps = pd.date_range("2024-01-01", periods=10, freq="min")
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["TEST"] * len(timestamps),
            "close": [100 + i for i in range(len(timestamps))],
            "volume": [1000 + 10 * i for i in range(len(timestamps))],
        }
    )

    engineer = FeatureEngineer(return_windows=(1, 2), volatility_window=3, volume_window=3)
    features = engineer.transform(data)

    assert not features.empty
    expected_columns = {
        "timestamp",
        "symbol",
        "close",
        "volume",
        "return_1",
        "return_2",
        "volatility",
        "volume_mean",
        "volume_zscore",
    }
    assert expected_columns.issubset(set(features.columns))
    assert features[["return_1", "return_2", "volatility", "volume_zscore"]].isna().sum().sum() == 0
