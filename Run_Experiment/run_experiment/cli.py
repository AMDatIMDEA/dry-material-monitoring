"""Operator CLI and lightweight C920/status window loop."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Callable

from material_level_yolo.camera import OpenCVCameraPreview, PreviewEvent
from material_volume.config import load_config as load_depth_config
from material_volume.models import CalibrationData

from .config import load_config, validate_config
from .core import SynchronizedExperiment
from .depth import PreparedDepthController
from .errors import ExperimentError
from .models import ExperimentInputs, ExperimentState, MeasurementReferenceInputs


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open and warm C920/D405. Each operator key press or left click requests "
            "fresh per-measurement references before confirmed synchronized capture. "
            "YOLO is never loaded."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--experiment-id")
    parser.add_argument("--purpose")
    parser.add_argument("--material-name")
    parser.add_argument("--total-capacity-ml", type=float)
    parser.add_argument("--bulk-density-g-per-ml", type=float)
    parser.add_argument("--total-possible-weight-g", type=float)
    parser.add_argument(
        "--reference-material-weight-g",
        type=float,
        help="One-measurement weight; valid only with --once or --hardware-smoke-test.",
    )
    parser.add_argument(
        "--manual-material-level-mm",
        type=float,
        help="One-measurement manual level; valid only with --once or --hardware-smoke-test.",
    )
    parser.add_argument("--operator-notes", help="General experiment/session notes.")
    parser.add_argument(
        "--measurement-notes",
        help="One-measurement notes; valid only with --once or --hardware-smoke-test.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--depth-config", type=Path)
    parser.add_argument("--depth-calibration", type=Path)
    parser.add_argument("--c920-device-index", type=int)
    parser.add_argument(
        "--c920-device-name-contains",
        help="On Windows, select the unique DirectShow camera whose name contains this text.",
    )
    parser.add_argument("--c920-backend", choices=("auto", "dshow", "msmf", "v4l2"))
    parser.add_argument("--c920-width", type=int)
    parser.add_argument("--c920-height", type=int)
    parser.add_argument("--c920-fps", type=float)
    parser.add_argument("--c920-warmup-frames", type=int)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--span-seconds", type=float)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Trigger once immediately after ARMED; useful for scripted laboratory operation.",
    )
    parser.add_argument(
        "--hardware-smoke-test",
        action="store_true",
        help="Open both real cameras, acquire once, and exit; never used by automated tests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment: SynchronizedExperiment | None = None
    try:
        config = validate_config(_apply_overrides(load_config(args.config), args))
        depth_config = load_depth_config(config.depth_config_path)
        calibration = CalibrationData.load(config.depth_calibration_path)
        inputs = _inputs(args, depth_config.tube.capacity_ml)
        _validate_reference_cli_mode(args)
        depth = PreparedDepthController(depth_config, calibration)
        c920 = OpenCVCameraPreview()

        def report_state(state: ExperimentState) -> None:
            print(f"STATE {state.value}")

        experiment = SynchronizedExperiment(
            config,
            inputs,
            c920=c920,
            depth=depth,
            state_callback=report_state,
            software_repository=REPOSITORY_ROOT,
        )
        summary = experiment.confirmation_summary()
        print("Operator confirmation summary:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if not args.yes:
            if not sys.stdin.isatty():
                raise ValueError("Confirmation requires --yes when stdin is noninteractive.")
            if input("Open and warm both cameras? [y/N]: ").strip().casefold() not in {"y", "yes"}:
                print("Cancelled before camera initialization.")
                return 0
        experiment.initialize()
        if args.once or args.hardware_smoke_test:
            if args.interactive:
                outcome = _guided_measurement(experiment)
                if outcome is not None:
                    _print_outcome(outcome)
            else:
                index = experiment.next_measurement_index()
                references = _command_reference_inputs(args)
                _print_reference_summary(index, references)
                if not args.yes and not _confirm_capture(index):
                    print(
                        f"Measurement {index} cancelled before allocation; "
                        "no camera acquisition was performed."
                    )
                    return 0
                _print_outcome(experiment.trigger(references))
            return 0
        while experiment.state is ExperimentState.ARMED:
            event = experiment.preview_once()
            if event is PreviewEvent.CAPTURE:
                outcome = _guided_measurement(experiment)
                if outcome is not None:
                    _print_outcome(outcome)
            elif event in {PreviewEvent.CANCEL, PreviewEvent.CLOSED}:
                break
        return 0
    except (ExperimentError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Graceful stop requested.", file=sys.stderr)
        return 130
    finally:
        if experiment is not None:
            experiment.request_stop()


def _apply_overrides(config, args):
    camera = replace(
        config.c920,
        device_index=(
            config.c920.device_index if args.c920_device_index is None else args.c920_device_index
        ),
        device_name_contains=(
            config.c920.device_name_contains
            if args.c920_device_name_contains is None
            else args.c920_device_name_contains
        ),
        backend=config.c920.backend if args.c920_backend is None else args.c920_backend,
        width=config.c920.width if args.c920_width is None else args.c920_width,
        height=config.c920.height if args.c920_height is None else args.c920_height,
        fps=config.c920.fps if args.c920_fps is None else args.c920_fps,
        warmup_frames=(
            config.c920.warmup_frames
            if args.c920_warmup_frames is None
            else args.c920_warmup_frames
        ),
    )
    acquisition = replace(
        config.acquisition,
        frame_count=(
            config.acquisition.frame_count if args.frame_count is None else args.frame_count
        ),
        span_seconds=(
            config.acquisition.span_seconds if args.span_seconds is None else args.span_seconds
        ),
    )
    return replace(
        config,
        depth_config_path=(
            config.depth_config_path
            if args.depth_config is None
            else args.depth_config.expanduser().resolve()
        ),
        depth_calibration_path=(
            config.depth_calibration_path
            if args.depth_calibration is None
            else args.depth_calibration.expanduser().resolve()
        ),
        c920=camera,
        acquisition=acquisition,
        output_root=(
            config.output_root
            if args.output_root is None
            else args.output_root.expanduser().resolve()
        ),
    )


def _inputs(args, default_capacity: float) -> ExperimentInputs:
    interactive = bool(args.interactive)
    experiment_id = _required(args.experiment_id, "Experiment ID", interactive)
    purpose = _required(args.purpose, "Purpose", interactive)
    capacity = args.total_capacity_ml
    if capacity is None:
        if interactive:
            # Display enough significant digits that re-entering the shown frozen
            # geometry value passes the same capacity-consistency validation.
            entered = input(f"Total capacity mL [{default_capacity:.15g}]: ").strip()
            capacity = default_capacity if not entered else float(entered)
        else:
            raise ValueError("--total-capacity-ml is required.")
    density = args.bulk_density_g_per_ml
    total_weight = args.total_possible_weight_g
    if interactive:
        density, total_weight = _prompt_mass_reference_choice(density, total_weight)
    material = args.material_name
    notes = args.operator_notes
    if interactive:
        material = (
            material
            or input(
                "Material name (recommended for automatic offline YOLO profile): "
            ).strip()
            or None
        )
        notes = notes or input("General experiment notes (optional): ").strip() or None
    return ExperimentInputs(
        experiment_id=experiment_id,
        purpose=purpose,
        material_name=material,
        total_capacity_ml=capacity,
        bulk_density_g_per_ml=density,
        total_possible_weight_g=total_weight,
        reference_material_weight_g=None,
        manual_material_level_mm=None,
        operator_notes=notes,
        output_root=args.output_root,
    )


def _prompt_mass_reference_choice(
    density: float | None,
    total_weight: float | None,
    *,
    input_func: Callable[[str], str] | None = None,
    print_func: Callable[[str], None] = print,
) -> tuple[float | None, float | None]:
    """Keep current mass metadata or replace it with one explicit input basis."""

    ask = input if input_func is None else input_func
    print_func("Mass reference currently configured:")
    print_func(
        "  Bulk density: "
        + ("unavailable" if density is None else f"{density:g} g/mL")
    )
    print_func(
        "  Total possible material mass: "
        + ("unavailable" if total_weight is None else f"{total_weight:g} g")
    )
    if not _prompt_yes_no(
        "Do you want to change the density/total mass? [y/N]: ",
        ask,
        print_func,
        default=False,
    ):
        return density, total_weight

    while True:
        choice = ask("Use bulk [d]ensity or total full-tube [m]ass? [d/m]: ").strip().casefold()
        if choice in {"d", "density"}:
            value = _prompt_positive("Bulk density [g/mL]: ", ask, print_func)
            print_func(
                "Bulk density selected. Total possible material mass will remain blank."
            )
            return value, None
        if choice in {"m", "mass", "total", "total_mass"}:
            value = _prompt_positive(
                "Total possible material mass for a full tube [g]: ", ask, print_func
            )
            print_func(
                "Total mass selected. Bulk density will remain blank and no "
                "mass-to-volume reference will be calculated."
            )
            return None, value
        print_func("Enter d for bulk density or m for total full-tube material mass.")


def _prompt_yes_no(
    prompt: str,
    ask: Callable[[str], str],
    print_func: Callable[[str], None],
    *,
    default: bool,
) -> bool:
    while True:
        entered = ask(prompt).strip().casefold()
        if not entered:
            return default
        if entered in {"y", "yes"}:
            return True
        if entered in {"n", "no"}:
            return False
        print_func("Enter y for yes or n for no.")


def _prompt_positive(
    prompt: str,
    ask: Callable[[str], str],
    print_func: Callable[[str], None],
) -> float:
    while True:
        entered = ask(prompt).strip()
        try:
            value = float(entered)
        except ValueError:
            print_func("Enter a finite number greater than zero.")
            continue
        if not math.isfinite(value) or value <= 0.0:
            print_func("Enter a finite number greater than zero.")
            continue
        return value


def _validate_reference_cli_mode(args: argparse.Namespace) -> None:
    has_one_shot_values = any(
        value is not None
        for value in (
            args.reference_material_weight_g,
            args.manual_material_level_mm,
            args.measurement_notes,
        )
    )
    if has_one_shot_values and args.interactive:
        raise ValueError(
            "Do not combine --interactive with command-line measurement reference "
            "values; enter them at that measurement's guided prompt."
        )
    if has_one_shot_values and not (args.once or args.hardware_smoke_test):
        raise ValueError(
            "--reference-material-weight-g, --manual-material-level-mm, and "
            "--measurement-notes are per-measurement values. Use them only with "
            "--once/--hardware-smoke-test; the continuous guided workflow prompts "
            "for new values after every trigger request."
        )


def _command_reference_inputs(args: argparse.Namespace) -> MeasurementReferenceInputs:
    return MeasurementReferenceInputs(
        reference_material_weight_g=args.reference_material_weight_g,
        manual_material_level_mm=args.manual_material_level_mm,
        measurement_notes=args.measurement_notes,
    )


def _guided_measurement(
    experiment: SynchronizedExperiment,
    *,
    input_func: Callable[[str], str] | None = None,
    print_func: Callable[[str], None] = print,
):
    """Collect and confirm one measurement's references before allocation/capture."""
    ask = input if input_func is None else input_func
    index = experiment.next_measurement_index()
    print_func("")
    print_func(f"Measurement {index} reference inputs")
    weight = _prompt_optional_nonnegative(
        "Real material left weight [g] (blank = unavailable): ",
        ask,
        print_func,
    )
    manual = _prompt_optional_nonnegative(
        "Manual human material level [mm] (blank = unavailable): ",
        ask,
        print_func,
        maximum=float(experiment.depth.config.tube.usable_height_mm),
    )
    notes = ask("Measurement notes (optional): ").strip() or None
    references = MeasurementReferenceInputs(
        reference_material_weight_g=weight,
        manual_material_level_mm=manual,
        measurement_notes=notes,
    )
    _print_reference_summary(index, references, print_func=print_func)
    if _confirm_capture(index, input_func=ask, print_func=print_func):
        return experiment.trigger(references)
    print_func(f"Measurement {index} cancelled before allocation; system remains ARMED.")
    return None


def _prompt_optional_nonnegative(
    prompt: str,
    ask: Callable[[str], str],
    print_func: Callable[[str], None],
    *,
    maximum: float | None = None,
) -> float | None:
    while True:
        entered = ask(prompt).strip()
        if not entered:
            return None
        try:
            value = float(entered)
        except ValueError:
            print_func("Enter a finite nonnegative number, or leave it blank.")
            continue
        if not math.isfinite(value) or value < 0.0:
            print_func("Enter a finite nonnegative number, or leave it blank.")
            continue
        if maximum is not None and value > maximum:
            print_func(f"Value cannot exceed the usable tube height ({maximum:g} mm).")
            continue
        return value


def _print_reference_summary(
    measurement_index: int,
    references: MeasurementReferenceInputs,
    *,
    print_func: Callable[[str], None] = print,
) -> None:
    weight = (
        "unavailable"
        if references.reference_material_weight_g is None
        else f"{references.reference_material_weight_g:g} g"
    )
    manual = (
        "unavailable"
        if references.manual_material_level_mm is None
        else f"{references.manual_material_level_mm:g} mm"
    )
    notes = references.measurement_notes or "none"
    print_func(f"Reference data for measurement {measurement_index}:")
    print_func(f"  Weight:       {weight}")
    print_func(f"  Manual level: {manual}")
    print_func(f"  Notes:        {notes}")


def _confirm_capture(
    measurement_index: int,
    *,
    input_func: Callable[[str], str] | None = None,
    print_func: Callable[[str], None] = print,
) -> bool:
    ask = input if input_func is None else input_func
    while True:
        entered = ask("Press Enter to capture, or type c to cancel: ").strip().casefold()
        if not entered:
            return True
        if entered in {"c", "cancel"}:
            return False
        print_func(
            f"Enter captures measurement {measurement_index}; c cancels the request."
        )


def _required(value: str | None, label: str, interactive: bool) -> str:
    if value is not None and value.strip():
        return value.strip()
    if interactive:
        entered = input(f"{label}: ").strip()
        if entered:
            return entered
    raise ValueError(f"{label} is required.")


def _print_outcome(outcome) -> None:
    print(
        json.dumps(
            {
                "measurement_id": outcome.measurement_id,
                "measurement_index": outcome.measurement_index,
                "state": outcome.final_state.value,
                "c920_frames": len(outcome.c920.frames),
                "depth_status": outcome.depth_record.status,
                "yolo_status": outcome.yolo_record.status,
                "overlap_duration_seconds": outcome.overlap_duration_seconds,
                "capture_manifest": str(outcome.capture_manifest),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
