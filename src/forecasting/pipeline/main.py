from __future__ import annotations

import argparse
import os

from forecasting.config import load_config

from .orchestrator import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the raw material forecasting pipeline")
    parser.add_argument(
        "--config",
        default=os.getenv("FORECASTING_CONFIG_PATH", "config/default.toml"),
        help="Path to TOML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = run_pipeline(cfg)
    print(f"Done. Outputs saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
