from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.linear_model import ElasticNetCV

from forecasting.config import Config


def run_doubleml_screening(
    panel: pd.DataFrame,
    target_col: str,
    selected_features: list[str],
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
    results: list[dict[str, object]] = []

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
