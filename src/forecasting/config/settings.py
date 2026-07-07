from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class Config:
    data_dir: str = "data"

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

    output_root: str = "outputs/runs"
    run_id: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    random_state: int = 42

    @property
    def effective_run_id(self) -> str:
        return self.run_id

    @property
    def output_dir(self) -> str:
        return str(Path(self.output_root) / self.effective_run_id)


def _as_tuple_int(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def load_config(config_path: str | None = None) -> Config:
    resolved = Path(config_path or os.getenv("FORECASTING_CONFIG_PATH", "config/default.toml"))
    data: dict[str, object] = {}

    if resolved.exists():
        with resolved.open("rb") as f:
            data = tomllib.load(f)

    env_overrides = {
        "data_dir": os.getenv("FORECASTING_DATA_DIR"),
        "output_root": os.getenv("FORECASTING_OUTPUT_ROOT"),
        "run_id": os.getenv("FORECASTING_RUN_ID"),
    }
    for key, value in env_overrides.items():
        if value not in (None, ""):
            data[key] = value

    if "forecast_horizons" in data and data["forecast_horizons"] is not None:
        data["forecast_horizons"] = _as_tuple_int(data["forecast_horizons"])  # type: ignore[arg-type]

    if "run_id" not in data or not data["run_id"]:
        data["run_id"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    return Config(**data)
