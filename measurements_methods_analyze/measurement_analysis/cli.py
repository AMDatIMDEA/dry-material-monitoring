"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .analysis import analyze_experiment
from .config import load_config
from .export import export_tables
from .plots import create_all_plots


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare human, YOLO, and D405 volumes against mass-based "
            "reference volumes."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        result = analyze_experiment(config)
        config.output_directory.mkdir(parents=True, exist_ok=True)
        tables = export_tables(result, config)
        plots = create_all_plots(result, config)
        print(f"Experiment: {config.experiment_directory}")
        for row in result.summary:
            print(f"{row['method']}: {row['valid_measurement_count']} valid comparison(s)")
        print(f"Skipped comparisons: {len(result.excluded)}")
        print(f"Outputs: {config.output_directory}")
        print(f"Created {len(tables) + len(plots)} files.")
        return 0
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
