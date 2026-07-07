from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecasting.config import Config


def write_markdown_report(
    cfg: Config,
    report_path: str,
    ind_path: str,
    tgt_path: str,
    dml_results: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Forecasting + Causality Screening Report\n\n")
        f.write("## Run Metadata\n")
        f.write(f"- Run ID: `{cfg.effective_run_id}`\n")
        f.write(f"- Output directory: `{cfg.output_dir}`\n\n")

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
