from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100.0
    )
    r2 = float(r2_score(y_true, y_pred))

    yt_diff = y_true.diff()
    yp_diff = y_pred.diff()
    mask = yt_diff.notna() & yp_diff.notna()
    directional_acc = (
        float((np.sign(yt_diff[mask]) == np.sign(yp_diff[mask])).mean())
        if mask.any()
        else np.nan
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_pct": mape,
        "R2": r2,
        "DirectionalAcc": directional_acc,
    }
