import pandas as pd

from forecasting.evaluation import compute_metrics


def test_compute_metrics_contains_expected_fields():
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = pd.Series([1.1, 1.9, 3.2, 3.8])

    metrics = compute_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"RMSE", "MAE", "MAPE_pct", "R2", "DirectionalAcc"}
