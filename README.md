# AI-Driven-Raw-Material-Price-Forecasting

Package-based monthly forecasting and causal screening pipeline for raw-material prices.

## Project structure

```text
/home/runner/work/AI-Driven-Raw-Material-Price-Forecasting/AI-Driven-Raw-Material-Price-Forecasting/
├── config/
│   └── default.toml
├── src/forecasting/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── causal/
│   ├── models/
│   ├── evaluation/
│   ├── visualization/
│   └── pipeline/
├── tests/
│   ├── config/
│   ├── data/
│   ├── features/
│   └── evaluation/
├── .github/workflows/ci.yml
├── pyproject.toml
└── lithium_forecasting_pipeline.py
```

## Setup

```bash
python -m pip install --upgrade pip
pip install -e .[dev]
```

## Configuration

Default runtime config lives at:

- `/home/runner/work/AI-Driven-Raw-Material-Price-Forecasting/AI-Driven-Raw-Material-Price-Forecasting/config/default.toml`

Environment overrides:

- `FORECASTING_CONFIG_PATH`
- `FORECASTING_DATA_DIR`
- `FORECASTING_OUTPUT_ROOT`
- `FORECASTING_RUN_ID`

## Run commands

Compatibility command (unchanged):

```bash
python lithium_forecasting_pipeline.py
```

Direct package entrypoint:

```bash
python -m forecasting.pipeline --config config/default.toml
```

## Reproducibility and artifacts

Each run writes versioned outputs under `outputs/runs/<run_id>/`, including:

- run metadata (`run_metadata.json`)
- comparison metrics (`model_comparison.csv`)
- report (`report.md`)
- stage-specific artifacts (`01_raw_plots` ... `07_shap`)

## Development workflow

1. Edit module in `src/forecasting/<layer>/`
2. Run local checks
3. Add/update tests in `tests/`
4. Open PR (CI runs lint + type checks + tests)

## Testing and quality checks

```bash
ruff check .
mypy src
pytest -q
```
