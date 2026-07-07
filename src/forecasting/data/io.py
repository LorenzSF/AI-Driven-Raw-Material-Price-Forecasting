from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from forecasting.config import Config


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


def validate_input_contract(cfg: Config) -> tuple[str, str]:
    ind_path = os.path.join(cfg.data_dir, cfg.indicators_file)
    tgt_path = os.path.join(cfg.data_dir, cfg.target_file)

    missing = [p for p in [ind_path, tgt_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required input files: " + ", ".join(missing)
        )
    return ind_path, tgt_path


def write_run_metadata(
    cfg: Config,
    output_dir: str,
    inputs: dict[str, str],
    artifacts: Optional[dict[str, str]] = None,
) -> None:
    ensure_dir(output_dir)
    metadata = {
        "run_id": cfg.effective_run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "data_dir": cfg.data_dir,
            "indicators_file": cfg.indicators_file,
            "target_file": cfg.target_file,
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "forecast_horizons": list(cfg.forecast_horizons),
            "corr_threshold": cfg.corr_threshold,
            "n_splits": cfg.n_splits,
            "min_train_months": cfg.min_train_months,
            "output_root": cfg.output_root,
        },
        "inputs": inputs,
        "artifacts": artifacts or {},
    }
    metadata_path = Path(output_dir) / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


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
) -> tuple[pd.DataFrame, str]:
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
                        "Set target_value_col explicitly."
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
                    "Could not infer target value column. Set target_value_col explicitly."
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


def treat_zeros_as_missing(df: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
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
