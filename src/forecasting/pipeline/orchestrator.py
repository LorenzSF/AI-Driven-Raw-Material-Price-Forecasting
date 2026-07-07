from __future__ import annotations

import os
import warnings

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

from forecasting.causal import run_doubleml_screening
from forecasting.config import Config
from forecasting.data import (
    build_monthly_panel,
    ensure_dir,
    impute_panel,
    load_and_standardize_indicators,
    load_and_standardize_target,
    treat_zeros_as_missing,
    validate_input_contract,
    write_run_metadata,
)
from forecasting.evaluation import compute_metrics, write_markdown_report
from forecasting.features import compute_best_lags, correlation_filter, engineer_features
from forecasting.models import make_supervised, rolling_backtest
from forecasting.visualization import (
    plot_indicator_panel,
    plot_partial_dependence_curves,
    plot_target_with_splits,
    run_shap,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def run_pipeline(cfg: Config) -> str:
    ensure_dir(cfg.output_dir)

    ind_path, tgt_path = validate_input_contract(cfg)

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

    def model_linear() -> RegressorMixin:
        return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=cfg.random_state)

    def model_xgb() -> RegressorMixin:
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
    final_models: dict[int, tuple[RegressorMixin, pd.DataFrame, str]] = {}

    for h in cfg.forecast_horizons:
        horizon_variant_rows = []

        for variant_name, (factory, feat_cols) in model_variants.items():
            X, y = make_supervised(engineered_panel, target_col, feat_cols, horizon=h)
            pred_df, _ = rolling_backtest(X, y, factory, cfg)

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

    write_markdown_report(
        cfg=cfg,
        report_path=os.path.join(cfg.output_dir, "report.md"),
        ind_path=ind_path,
        tgt_path=tgt_path,
        dml_results=dml_results,
        comparison=comparison,
    )

    write_run_metadata(
        cfg,
        cfg.output_dir,
        inputs={"indicators": ind_path, "target": tgt_path},
        artifacts={
            "model_comparison": os.path.join(cfg.output_dir, "model_comparison.csv"),
            "report": os.path.join(cfg.output_dir, "report.md"),
        },
    )

    return cfg.output_dir
