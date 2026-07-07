from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.inspection import PartialDependenceDisplay

from forecasting.data import ensure_dir


def plot_partial_dependence_curves(
    model: RegressorMixin,
    X: pd.DataFrame,
    features: list[str],
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
    cols: list[str],
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
    split_points: list[pd.Timestamp],
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
