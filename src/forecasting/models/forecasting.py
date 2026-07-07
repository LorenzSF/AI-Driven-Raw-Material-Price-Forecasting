from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.model_selection import TimeSeriesSplit

from forecasting.config import Config


def make_supervised(
    panel: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    df = panel.copy()
    y = df[target_col].shift(-horizon)
    X = df[feature_cols]
    df_xy = pd.concat([X, y.rename("y_future")], axis=1).dropna()
    return df_xy[feature_cols], df_xy["y_future"]


def rolling_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], RegressorMixin],
    cfg: Config,
) -> tuple[pd.DataFrame, RegressorMixin]:
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    preds = []
    fold_models: list[RegressorMixin] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < cfg.min_train_months:
            continue

        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

        model = model_factory()
        model.fit(Xtr, ytr)
        yhat = pd.Series(model.predict(Xte), index=Xte.index)

        fold_models.append(model)
        preds.append(pd.DataFrame({"fold": fold, "y_true": yte, "y_pred": yhat}))

    if preds:
        pred_df = pd.concat(preds).sort_index()
    else:
        pred_df = pd.DataFrame(columns=["fold", "y_true", "y_pred"])

    final_model = fold_models[-1] if fold_models else model_factory()
    final_model.fit(X, y)
    return pred_df, final_model
