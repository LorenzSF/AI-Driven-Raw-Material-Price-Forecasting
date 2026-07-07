import pandas as pd

from forecasting.features import engineer_features


def test_engineer_features_generates_expected_columns():
    idx = pd.date_range("2020-01-31", periods=15, freq="ME")
    panel = pd.DataFrame(
        {
            "target_price": range(15),
            "feat_a": range(10, 25),
            "feat_b": range(20, 35),
        },
        index=idx,
    )

    engineered, engineered_panel = engineer_features(panel, target_col="target_price")

    assert "feat_a" in engineered.columns
    assert "feat_a__mom" in engineered.columns
    assert "feat_a__yoy" in engineered.columns
    assert "target_price" in engineered_panel.columns
