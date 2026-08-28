"""CLI for offline processing of Run_Experiment capture groups."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .config import load_config
from .errors import YoloFoundationError
from .synchronized import process_synchronized_captures
from .roi_policy import ROI_MODES, apply_roi_mode, choose_roi_mode


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process saved Run_Experiment C920 groups offline. This command never "
            "opens a camera and never compares YOLO with depth estimates."
        )
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Run_Experiment session directory or one measurements/<id> directory.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        help=(
            "Explicit model profile. It must match a recorded material alias; "
            "for a legacy blank material it is the deliberate recovery choice."
        ),
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help=(
            "Re-run a completed group and preserve the prior result as an explicit "
            "processing revision."
        ),
    )
    parser.add_argument(
        "--start-measurement-index",
        type=int,
        help=(
            "For a session target, process only groups at or after this index. "
            "Useful for resuming a long forced reprocessing run."
        ),
    )
    parser.add_argument(
        "--roi-mode",
        choices=ROI_MODES,
        help=(
            "Use the frozen configured tube ROI or deliberately evaluate the whole "
            "image. When omitted, the terminal asks."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        print(
            "STAGE DISCOVERY - locating capture manifests and verifying each saved group.",
            flush=True,
        )

        def report_progress(position, total, group) -> None:
            print(
                f"STAGE OFFLINE_GROUP {position}/{total} - {group.measurement_id} "
                f"({len(group.source_paths)} images). Segmentation can take several minutes.",
                flush=True,
            )

        project = load_config(args.config)
        project = apply_roi_mode(
            project,
            choose_roi_mode(
                project,
                args.roi_mode,
                stdin_is_tty=sys.stdin.isatty(),
            ),
        )
        report = process_synchronized_captures(
            project,
            args.target,
            profile_name=args.profile,
            force_reprocess=args.force_reprocess,
            start_measurement_index=args.start_measurement_index,
            software_repository=Path(__file__).resolve().parents[2],
            progress_callback=report_progress,
        )
        print(
            json.dumps(
                {
                    "target": str(report.target),
                    "session_directory": str(report.session_directory),
                    "method": "yolo",
                    "acquisition_performed": False,
                    "method_comparison_performed": False,
                    "groups": [
                        {
                            **asdict(group),
                            "artifact_directory": str(group.artifact_directory),
                        }
                        for group in report.groups
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    except (YoloFoundationError, OSError, ValueError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
