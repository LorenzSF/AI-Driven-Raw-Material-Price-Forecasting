"""lithium_forecasting_pipeline.py - monthly forecasting + causal screening.

Load indicators/target parquet -> monthly align (2010-2025) -> impute+engineer ->
lag scoring (corr+R2) -> corr filter (|corr|>0.9) -> DoubleML ATE screening ->
sensitivity curves -> forecast 1/3/6m (price level) -> SHAP -> trade-off report.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass(frozen=True)
class Config:
    data_dir: str = r"C:\Users\Usuario\Desktop\Bekaert\Data source"

    indicators_file: str = "master_data.parquet"
    target_file: str = "target_price.parquet"

    indicators_time_col: Optional[str] = None
    target_time_col: Optional[str] = None
    target_value_col: Optional[str] = None

    start_date: str = "2010-01-01"
    end_date: str = "2025-12-31"
    monthly_rule: str = "M"

    treat_zero_as_missing_for_indicators: bool = True

    impute_ffill: bool = True
    impute_time_interp: bool = True

    lag_months_min: int = 0
    lag_months_max: int = 12

    corr_threshold: float = 0.90

    forecast_horizons: Tuple[int, int, int] = (1, 3, 6)

    n_splits: int = 6
    min_train_months: int = 60

    max_indicators_to_plot: int = 24

    output_dir: str = "outputs"

    random_state: int = 42


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def guess_time_column(df: pd.DataFrame) -> str:
    candidates = [
        c
        for c in df.columns
        if any(k in c.lower() for k in ["date", "time", "month", "period"])
    ]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    return df.columns[0]


def coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def to_monthly(df: pd.DataFrame, how: str = "last") -> pd.DataFrame:
    if how == "last":
        return df.resample("M").last()
    if how == "mean":
        return df.resample("M").mean()
    raise ValueError(f"Unsupported monthly aggregation: {how}")


def date_slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    return df.loc[(df.index >= s) & (df.index <= e)].copy()


def normalize_numeric_strings_to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"[^\d\.,\-\+]", "", regex=True)

    def _coerce_one(x: str) -> float:
        if x in ("", "nan", "None"):
            return np.nan
        x = x.strip()
        last_comma = x.rfind(",")
        last_dot = x.rfind(".")
        if last_comma == -1 and last_dot == -1:
            try:
                return float(x)
            except Exception:
                return np.nan
        dec_sep = "," if last_comma > last_dot else "."
        thou_sep = "." if dec_sep == "," else ","
        x2 = x.replace(thou_sep, "")
        if dec_sep == ",":
            x2 = x2.replace(",", ".")
        try:
            return float(x2)
        except Exception:
            return np.nan

    return s.apply(_coerce_one)


def load_and_standardize_indicators(
    path: str, time_col: Optional[str] = None
) -> pd.DataFrame:
    df = load_parquet(path).copy()

    if time_col is None:
        time_col = guess_time_column(df)

    df[time_col] = coerce_datetime(df[time_col])
    df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="last")]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    obj_cols = [c for c in df.columns if c not in numeric_cols]
    for c in obj_cols:
        coerced = normalize_numeric_strings_to_float(df[c])
        if coerced.notna().sum() > 0:
            df[c] = coerced
            numeric_cols.append(c)

    numeric_cols = sorted(set(numeric_cols))
    return df[numeric_cols]


def load_and_standardize_target(
    path: str,
    time_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    df = load_parquet(path).copy()

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        if value_col is None:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) == 1:
                value_col = numeric_cols[0]
            elif len(numeric_cols) > 1:
                nn = df[numeric_cols].notna().sum().sort_values(ascending=False)
                value_col = nn.index[0]
            else:
                best_col, best_nonnull = None, 0
                for c in df.columns:
                    coerced = normalize_numeric_strings_to_float(df[c])
                    nonnull = int(coerced.notna().sum())
                    if nonnull > best_nonnull:
                        best_col, best_nonnull = c, nonnull
                if best_col is None or best_nonnull == 0:
                    raise ValueError(
                        "Target has DatetimeIndex but no numeric value column could be inferred. "
                        "Set Config.target_value_col explicitly."
                    )
                df[best_col] = normalize_numeric_strings_to_float(df[best_col])
                value_col = best_col

        out = df[[value_col]].copy()
        out = out.rename(columns={value_col: "target_price"})
        out["target_price"] = pd.to_numeric(out["target_price"], errors="coerce")
        out = out.dropna(subset=["target_price"])
        return out, "target_price"

    if time_col is None:
        time_col = guess_time_column(df)

    df[time_col] = coerce_datetime(df[time_col])
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    if value_col is None:
        numeric_cols = [
            c
            for c in df.columns
            if c != time_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(numeric_cols) == 1:
            value_col = numeric_cols[0]
        elif len(numeric_cols) > 1:
            nn = df[numeric_cols].notna().sum().sort_values(ascending=False)
            value_col = nn.index[0]
        else:
            candidate_cols = [c for c in df.columns if c != time_col]
            best_col, best_nonnull = None, 0
            for c in candidate_cols:
                coerced = normalize_numeric_strings_to_float(df[c])
                nonnull = int(coerced.notna().sum())
                if nonnull > best_nonnull:
                    best_col, best_nonnull = c, nonnull
            if best_col is None or best_nonnull == 0:
                raise ValueError(
                    "Could not infer target value column. Set Config.target_value_col explicitly."
                )
            df[best_col] = normalize_numeric_strings_to_float(df[best_col])
            value_col = best_col

    out = df[[time_col, value_col]].copy()
    out = out.rename(columns={time_col: "date", value_col: "target_price"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").set_index("date")

    if not out.index.is_unique:
        out = out[~out.index.duplicated(keep="last")]

    out["target_price"] = pd.to_numeric(out["target_price"], errors="coerce")
    out = out.dropna(subset=["target_price"])
    out = out[["target_price"]]
    return out, "target_price"


def build_monthly_panel(df_ind: pd.DataFrame, df_tgt: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df_ind_m = to_monthly(df_ind, how="last")
    df_tgt_m = to_monthly(df_tgt, how="last")

    panel = df_ind_m.join(df_tgt_m, how="outer")
    panel = date_slice(panel, cfg.start_date, cfg.end_date)

    full_idx = pd.date_range(panel.index.min(), panel.index.max(), freq=cfg.monthly_rule)
    panel = panel.reindex(full_idx)
    return panel


def treat_zeros_as_missing(df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in out.columns if c not in exclude_cols]
    out[cols] = out[cols].replace(0, np.nan)
    return out


def impute_panel(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    if cfg.impute_ffill:
        out = out.ffill()
    if cfg.impute_time_interp:
        out = out.interpolate(method="time", limit_direction="both")
    return out


def engineer_features(panel: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    feature_cols: List[str],
    lags: List[int],
) -> pd.DataFrame:
    y = panel[target_col]
    rows: List[Dict[str, object]] = []

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


def compute_missingness(raw_panel: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    return raw_panel[feature_cols].isna().mean().rename("missing_rate")


def decide_keep_drop(
    a: str,
    b: str,
    miss_a: float,
    miss_b: float,
    r2_a: float,
    r2_b: float,
) -> Tuple[str, str]:
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
) -> Tuple[List[str], pd.DataFrame]:
    features = [c for c in engineered_panel.columns if c != target_col]
    X = engineered_panel[features]

    raw_base_cols = [c for c in raw_panel_for_missingness.columns if c != target_col]
    miss = compute_missingness(raw_panel_for_missingness, raw_base_cols)

    lag_r2 = lag_table.set_index("feature")["best_r2"]

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drops: set[str] = set()
    decisions: List[Dict[str, object]] = []

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


def run_doubleml_screening(
    panel: pd.DataFrame,
    target_col: str,
    selected_features: List[str],
    base_feature_to_best_lag: Dict[str, int],
    cfg: Config,
) -> pd.DataFrame:
    try:
        from doubleml import DoubleMLData, DoubleMLPLR
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    except Exception as e:
        print("DoubleML is not available. Install with: pip install doubleml")
        print(f"Reason: {e}")
        return pd.DataFrame()

    learner_specs = {
        "linear_enet": ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=cfg.random_state
        ),
        "rf": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            random_state=cfg.random_state,
            n_jobs=-1,
        ),
        "hgb": HistGradientBoostingRegressor(random_state=cfg.random_state),
    }

    base_cols = sorted({c.split("__")[0] for c in selected_features})
    base_cols = [c for c in base_cols if c in panel.columns and c != target_col]

    y = panel[target_col].copy()
    results: List[Dict[str, object]] = []

    for t in base_cols:
        t_lag = int(base_feature_to_best_lag.get(t, 0))
        T = panel[t].shift(t_lag).rename("T")

        ctrl = [c for c in base_cols if c != t]
        X = pd.concat(
            [
                panel[c]
                .shift(int(base_feature_to_best_lag.get(c, 0)))
                .rename(c)
                for c in ctrl
            ],
            axis=1,
        )

        dml_df = pd.concat([y.rename("Y"), T, X], axis=1).dropna()
        if len(dml_df) < max(cfg.min_train_months, 120):
            continue

        dml_data = DoubleMLData(dml_df, y_col="Y", d_cols="T", x_cols=ctrl)

        for spec_name, ml in learner_specs.items():
            try:
                plr = DoubleMLPLR(dml_data, ml_l=ml, ml_m=ml, n_folds=5)
                plr.fit()
                summ = plr.summary

                results.append(
                    {
                        "treatment": t,
                        "treatment_lag": t_lag,
                        "spec": spec_name,
                        "ate": float(summ["coef"].iloc[0]),
                        "std_err": float(summ["std err"].iloc[0]),
                        "p_value": float(summ["P>|t|"].iloc[0]),
                        "ci_low": float(summ["2.5 %"].iloc[0]),
                        "ci_high": float(summ["97.5 %"].iloc[0]),
                        "n_obs": int(len(dml_df)),
                    }
                )
            except Exception:
                continue

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out = out.sort_values(["p_value", "ate"], ascending=[True, False]).reset_index(
        drop=True
    )
    return out


def plot_partial_dependence_curves(
    model: RegressorMixin,
    X: pd.DataFrame,
    features: List[str],
    out_path: str,
    title: str,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(model, X, features=features, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_supervised(
    panel: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    horizon: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = panel.copy()
    y = df[target_col].shift(-horizon)
    X = df[feature_cols]
    df_xy = pd.concat([X, y.rename("y_future")], axis=1).dropna()
    return df_xy[feature_cols], df_xy["y_future"]


def rolling_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    cfg: Config,
) -> Tuple[pd.DataFrame, RegressorMixin]:
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    preds = []
    fold_models = []

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


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
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


def run_shap(
    final_model: RegressorMixin,
    X: pd.DataFrame,
    out_dir: str,
    tag: str,
) -> pd.DataFrame:
    try:
        import shap
    except Exception as e:
        print("SHAP not available. Install with: pip install shap")
        print(f"Reason: {e}")
        return pd.DataFrame()

    ensure_dir(out_dir)

    explainer = shap.Explainer(final_model, X)
    shap_values = explainer(X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"shap_summary_{tag}.png"), dpi=150)
    plt.close()

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    imp.to_csv(os.path.join(out_dir, f"shap_importance_{tag}.csv"), index=False)
    return imp


def plot_indicator_panel(
    panel: pd.DataFrame,
    cols: List[str],
    out_path: str,
    title: str,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(12, 6))
    panel[cols].plot(ax=ax, legend=False)
    ax.set_title(title)
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_target_with_splits(
    target: pd.Series,
    split_points: List[pd.Timestamp],
    out_path: str,
    title: str,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(12, 6))
    target.plot(ax=ax)
    for sp in split_points:
        ax.axvline(sp, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(cfg: Config) -> None:
    ensure_dir(cfg.output_dir)

    ind_path = os.path.join(cfg.data_dir, cfg.indicators_file)
    tgt_path = os.path.join(cfg.data_dir, cfg.target_file)

    df_ind = load_and_standardize_indicators(ind_path, time_col=cfg.indicators_time_col)
    df_tgt, target_col = load_and_standardize_target(
        tgt_path,
        time_col=cfg.target_time_col,
        value_col=cfg.target_value_col,
    )

    if df_tgt[target_col].notna().sum() < 24:
        raise ValueError(
            "Target series has too few numeric observations after parsing. "
            "Check target_value_col or file content."
        )

    raw_panel = build_monthly_panel(df_ind, df_tgt, cfg)

    if cfg.treat_zero_as_missing_for_indicators:
        raw_panel = treat_zeros_as_missing(raw_panel, exclude_cols=[target_col])

    ind_cols = [c for c in raw_panel.columns if c != target_col]
    plot_cols = ind_cols[: min(cfg.max_indicators_to_plot, len(ind_cols))]
    if plot_cols:
        plot_indicator_panel(
            raw_panel,
            plot_cols,
            os.path.join(cfg.output_dir, "01_raw_plots", "raw_indicators_subset.png"),
            title="Raw indicators (subset, unscaled)",
        )

    imputed_panel = impute_panel(raw_panel, cfg)

    engineered_features, engineered_panel = engineer_features(
        imputed_panel, target_col=target_col
    )

    eng_cols = list(engineered_features.columns)
    plot_eng_cols = eng_cols[: min(cfg.max_indicators_to_plot, len(eng_cols))]
    if plot_eng_cols:
        plot_indicator_panel(
            engineered_panel,
            plot_eng_cols,
            os.path.join(
                cfg.output_dir,
                "02_engineered_plots",
                "engineered_indicators_subset.png",
            ),
            title="Engineered indicators (subset, unscaled)",
        )

    base_ind_cols = [c for c in imputed_panel.columns if c != target_col]
    lags = list(range(cfg.lag_months_min, cfg.lag_months_max + 1))
    lag_table = compute_best_lags(imputed_panel, target_col, base_ind_cols, lags)

    ensure_dir(os.path.join(cfg.output_dir, "03_lag_analysis"))
    lag_table.to_csv(os.path.join(cfg.output_dir, "03_lag_analysis", "lag_scores.csv"), index=False)

    base_to_lag = lag_table.set_index("feature")["best_lag"].fillna(0).astype(int).to_dict()

    selected_features, dropped_pairs = correlation_filter(
        engineered_panel=engineered_panel,
        raw_panel_for_missingness=raw_panel,
        target_col=target_col,
        lag_table=lag_table,
        corr_threshold=cfg.corr_threshold,
    )

    ensure_dir(os.path.join(cfg.output_dir, "04_correlation_filter"))
    pd.Series(selected_features, name="feature").to_csv(
        os.path.join(cfg.output_dir, "04_correlation_filter", "selected_features.csv"),
        index=False,
    )
    dropped_pairs.to_csv(
        os.path.join(cfg.output_dir, "04_correlation_filter", "dropped_pairs.csv"),
        index=False,
    )

    dml_results = run_doubleml_screening(
        panel=imputed_panel,
        target_col=target_col,
        selected_features=selected_features,
        base_feature_to_best_lag=base_to_lag,
        cfg=cfg,
    )
    ensure_dir(os.path.join(cfg.output_dir, "05_doubleml"))
    if not dml_results.empty:
        dml_results.to_csv(
            os.path.join(cfg.output_dir, "05_doubleml", "doubleml_results.csv"),
            index=False,
        )

    if not dml_results.empty:
        shortlisted_base = dml_results["treatment"].drop_duplicates().head(25).tolist()
    else:
        shortlisted_base = lag_table["feature"].head(25).tolist()

    shortlisted_features = [
        c for c in selected_features if c.split("__")[0] in set(shortlisted_base)
    ]
    if not shortlisted_features:
        shortlisted_features = selected_features

    def model_linear():
        return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=cfg.random_state)

    def model_xgb():
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=cfg.random_state,
                n_jobs=-1,
            )
        except Exception:
            return model_linear()

    model_variants = {
        "linear_baseline": (model_linear, selected_features),
        "tree_no_causal": (model_xgb, selected_features),
        "tree_causal_shortlist": (model_xgb, shortlisted_features),
    }

    ensure_dir(os.path.join(cfg.output_dir, "06_forecasts"))

    comparison_rows = []
    final_models: Dict[int, Tuple[RegressorMixin, pd.DataFrame, str]] = {}

    for h in cfg.forecast_horizons:
        horizon_variant_rows = []

        for variant_name, (factory, feat_cols) in model_variants.items():
            X, y = make_supervised(engineered_panel, target_col, feat_cols, horizon=h)
            pred_df, final_model = rolling_backtest(X, y, factory, cfg)

            if pred_df.empty:
                continue

            metrics = compute_metrics(pred_df["y_true"], pred_df["y_pred"])
            row = {
                "horizon_months": h,
                "variant": variant_name,
                "n_features": len(feat_cols),
                **metrics,
            }
            horizon_variant_rows.append(row)
            comparison_rows.append(row)

            pred_out = pred_df.copy()
            pred_out["horizon"] = h
            pred_out["variant"] = variant_name
            pred_out.to_csv(
                os.path.join(
                    cfg.output_dir,
                    "06_forecasts",
                    f"predictions_h{h}_{variant_name}.csv",
                )
            )

        if horizon_variant_rows:
            best = sorted(horizon_variant_rows, key=lambda r: r["RMSE"])[0]
            best_variant = best["variant"]
            best_factory, best_feats = model_variants[best_variant]

            X_best, y_best = make_supervised(engineered_panel, target_col, best_feats, horizon=h)
            _, best_model = rolling_backtest(X_best, y_best, best_factory, cfg)
            final_models[h] = (best_model, X_best, best_variant)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(os.path.join(cfg.output_dir, "model_comparison.csv"), index=False)

    X_tmp, _ = make_supervised(engineered_panel, target_col, selected_features, horizon=1)
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    split_points = []
    for _, test_idx in tscv.split(X_tmp):
        split_points.append(X_tmp.index[test_idx[0]])

    plot_target_with_splits(
        target=engineered_panel[target_col].dropna(),
        split_points=split_points,
        out_path=os.path.join(
            cfg.output_dir,
            "06_forecasts",
            "target_train_test_splits.png",
        ),
        title="Target price with rolling-origin split boundaries",
    )

    if cfg.forecast_horizons and cfg.forecast_horizons[0] in final_models:
        h_for_curves = cfg.forecast_horizons[0]
        best_model, X_best, best_variant = final_models[h_for_curves]
        top_features_for_curves = list(X_best.columns[: min(6, X_best.shape[1])])

        plot_partial_dependence_curves(
            model=best_model,
            X=X_best,
            features=top_features_for_curves,
            out_path=os.path.join(
                cfg.output_dir,
                "05_doubleml",
                "causal_curves",
                f"pdp_sensitivity_h{h_for_curves}.png",
            ),
            title=(
                "Sensitivity curves (Partial Dependence) "
                f"- horizon {h_for_curves} - best variant: {best_variant}"
            ),
        )

    ensure_dir(os.path.join(cfg.output_dir, "07_shap"))
    for h, (m, Xh, variant) in final_models.items():
        run_shap(m, Xh, os.path.join(cfg.output_dir, "07_shap"), tag=f"h{h}_{variant}")

    report_path = os.path.join(cfg.output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Forecasting + Causality Screening Report\n\n")
        f.write("## Data window and frequency\n")
        f.write(f"- Monthly (month-end)\n- {cfg.start_date} to {cfg.end_date}\n\n")

        f.write("## Inputs\n")
        f.write(f"- Indicators: {ind_path}\n")
        f.write(f"- Target: {tgt_path}\n\n")

        f.write("## Lag analysis\n")
        f.write("- Best-lag table saved: `03_lag_analysis/lag_scores.csv`\n\n")

        f.write("## Correlation filtering\n")
        f.write(f"- Threshold: |corr| > {cfg.corr_threshold}\n")
        f.write("- Selected features: `04_correlation_filter/selected_features.csv`\n")
        f.write("- Dropped pairs: `04_correlation_filter/dropped_pairs.csv`\n\n")

        f.write("## DoubleML screening (Option A, ATE)\n")
        if dml_results.empty:
            f.write(
                "- DoubleML results not produced (package missing or insufficient data after alignment).\n\n"
            )
        else:
            f.write("- DoubleML results: `05_doubleml/doubleml_results.csv`\n")
            top = dml_results.head(10)[["treatment", "spec", "ate", "p_value"]]
            f.write("- Top 10 by significance:\n\n")
            f.write(top.to_markdown(index=False))
            f.write("\n\n")

        f.write("## Forecast performance and trade-offs\n")
        f.write("- Model comparison: `model_comparison.csv`\n\n")
        if not comparison.empty:
            f.write(comparison.sort_values(["horizon_months", "RMSE"]).to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write(
                "- No forecast results produced. Check that engineered features and target overlap sufficiently.\n\n"
            )

        f.write("## Plots\n")
        f.write("- Raw indicators subset: `01_raw_plots/raw_indicators_subset.png`\n")
        f.write(
            "- Engineered indicators subset: `02_engineered_plots/engineered_indicators_subset.png`\n"
        )
        f.write("- Target splits: `06_forecasts/target_train_test_splits.png`\n")
        f.write("- Sensitivity curves: `05_doubleml/causal_curves/pdp_sensitivity_h1.png`\n")
        f.write("- SHAP: `07_shap/`\n\n")

    print(f"Done. Outputs saved to: {os.path.abspath(cfg.output_dir)}")


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
