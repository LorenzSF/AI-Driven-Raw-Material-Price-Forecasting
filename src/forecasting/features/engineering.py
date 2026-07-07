from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def engineer_features(panel: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = panel.copy()
    feature_cols = [c for c in df.columns if c != target_col]

    levels = df[feature_cols].copy()
    mom = levels.pct_change(1).add_suffix("__mom")
    yoy = levels.pct_change(12).add_suffix("__yoy")
    roll3 = levels.rolling(3).mean().add_suffix("__roll3")

    engineered = pd.concat([levels, mom, yoy, roll3], axis=1)
    engineered = engineered.dropna(axis=1, how="all")

    engineered_panel = pd.concat([df[[target_col]], engineered], axis=1)
    return engineered, engineered_panel


def univariate_r2(y: pd.Series, x: pd.Series) -> float:
    mask = y.notna() & x.notna()
    if int(mask.sum()) < 12:
        return np.nan
    lr = LinearRegression()
    lr.fit(x[mask].values.reshape(-1, 1), y[mask].values)
    yhat = lr.predict(x[mask].values.reshape(-1, 1))
    return r2_score(y[mask].values, yhat)


def compute_best_lags(
    panel: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    lags: list[int],
) -> pd.DataFrame:
    y = panel[target_col]
    rows: list[dict[str, object]] = []

    for f in feature_cols:
        best = {"feature": f, "best_lag": None, "best_corr": np.nan, "best_r2": np.nan}
        best_key = (-np.inf, -np.inf)

        x0 = panel[f]
        for lag in lags:
            x = x0.shift(lag)
            corr = y.corr(x)
            r2 = univariate_r2(y, x)

            key = (
                abs(corr) if pd.notna(corr) else -np.inf,
                r2 if pd.notna(r2) else -np.inf,
            )
            if key > best_key:
                best_key = key
                best = {"feature": f, "best_lag": lag, "best_corr": corr, "best_r2": r2}

        rows.append(best)

    return pd.DataFrame(rows).sort_values(["best_r2", "best_corr"], ascending=False)


def compute_missingness(raw_panel: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    return raw_panel[feature_cols].isna().mean().rename("missing_rate")


def decide_keep_drop(
    a: str,
    b: str,
    miss_a: float,
    miss_b: float,
    r2_a: float,
    r2_b: float,
) -> tuple[str, str]:
    if pd.notna(miss_a) and pd.notna(miss_b) and miss_a != miss_b:
        return (a, b) if miss_a < miss_b else (b, a)
    if pd.notna(r2_a) and pd.notna(r2_b) and r2_a != r2_b:
        return (a, b) if r2_a > r2_b else (b, a)
    return (a, b) if a < b else (b, a)


def correlation_filter(
    engineered_panel: pd.DataFrame,
    raw_panel_for_missingness: pd.DataFrame,
    target_col: str,
    lag_table: pd.DataFrame,
    corr_threshold: float,
) -> tuple[list[str], pd.DataFrame]:
    features = [c for c in engineered_panel.columns if c != target_col]
    X = engineered_panel[features]

    raw_base_cols = [c for c in raw_panel_for_missingness.columns if c != target_col]
    miss = compute_missingness(raw_panel_for_missingness, raw_base_cols)

    lag_r2 = lag_table.set_index("feature")["best_r2"]

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drops: set[str] = set()
    decisions: list[Dict[str, object]] = []

    for col in upper.columns:
        high = upper.index[upper[col] > corr_threshold].tolist()
        for row in high:
            if row in drops or col in drops:
                continue

            base_row = row.split("__")[0]
            base_col = col.split("__")[0]

            miss_row = miss.get(base_row, np.nan)
            miss_col = miss.get(base_col, np.nan)

            r2_row = lag_r2.get(base_row, np.nan)
            r2_col = lag_r2.get(base_col, np.nan)

            keep, drop = decide_keep_drop(
                row, col, miss_row, miss_col, r2_row, r2_col
            )

            drops.add(drop)
            decisions.append(
                {
                    "feature_a": row,
                    "feature_b": col,
                    "abs_corr": float(upper.loc[row, col]),
                    "keep": keep,
                    "drop": drop,
                    "missing_a": float(miss_row) if pd.notna(miss_row) else np.nan,
                    "missing_b": float(miss_col) if pd.notna(miss_col) else np.nan,
                    "lag_r2_a": float(r2_row) if pd.notna(r2_row) else np.nan,
                    "lag_r2_b": float(r2_col) if pd.notna(r2_col) else np.nan,
                }
            )

    selected = [f for f in features if f not in drops]
    return selected, pd.DataFrame(decisions)
