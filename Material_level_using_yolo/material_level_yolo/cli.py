"""Command line interface for profile validation and operator acquisition workflows."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

from .camera_discovery import CameraDiscoveryError, list_directshow_camera_names
from .config import load_config
from .errors import YoloFoundationError
from .inference import UltralyticsInferenceAdapter
from .logging_utils import configure_logging
from .operator import OperatorInputs, discover_images, run_operator_workflow
from .roi_policy import ROI_MODES, apply_roi_mode, choose_roi_mode


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an offline YOLO profile or run folder/manual/timed still "
            "acquisition. Preview frames are never sent to inference."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List Windows DirectShow camera indexes and friendly names, then exit.",
    )
    parser.add_argument(
        "--profile",
        help=(
            "Optional model-profile assertion. Operator workflows select the profile "
            "from --material-name; configuration-only commands use default_profile."
        ),
    )
    parser.add_argument(
        "--camera-name-contains",
        help="Override camera.device_name_contains for this run (Windows DirectShow).",
    )
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "dshow", "msmf", "v4l2"),
        help="Override the selected profile's camera backend for this run.",
    )
    parser.add_argument(
        "--validate-model",
        action="store_true",
        help="Load weights and strictly validate task/model.names without prediction.",
    )
    parser.add_argument(
        "--roi-mode",
        choices=ROI_MODES,
        help=(
            "Use the frozen configured tube ROI or deliberately evaluate the whole "
            "image. When omitted for an operator workflow, the terminal asks."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for missing operator fields; otherwise missing required values are errors.",
    )
    parser.add_argument(
        "--acquisition-mode",
        choices=("manual_camera", "timed_camera", "folder"),
    )
    parser.add_argument("--input", type=Path, help="Saved image/folder for folder mode only.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--experiment-id")
    parser.add_argument("--purpose")
    parser.add_argument("--total-capacity-ml", type=float)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--measurement-index", type=int)
    parser.add_argument("--measurement-id")
    parser.add_argument("--trigger-time-utc")
    parser.add_argument("--material-name")
    parser.add_argument("--bulk-density-g-per-ml", type=float)
    parser.add_argument("--total-possible-weight-g", type=float)
    parser.add_argument(
        "--remaining-weight-g",
        type=float,
        help="Independent remaining-material weight; never inferred from material/profile.",
    )
    parser.add_argument("--manual-material-level", type=float)
    parser.add_argument("--manual-level-unit", choices=("mm", "cm"))
    parser.add_argument("--usable-internal-height-mm", type=float)
    parser.add_argument("--notes")
    parser.add_argument("--capture-count", type=int)
    parser.add_argument("--capture-interval-seconds", type=float)
    parser.add_argument("--capture-duration-seconds", type=float)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Accept the printed confirmation noninteractively.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.list_cameras:
            names = list_directshow_camera_names()
            print(
                json.dumps(
                    [
                        {"directshow_index": index, "friendly_name": name}
                        for index, name in enumerate(names)
                    ],
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        config = load_config(args.config)
        wants_workflow = args.interactive or args.acquisition_mode is not None or args.input is not None
        if wants_workflow:
            inputs = _operator_inputs(args)
            config = apply_roi_mode(
                config,
                choose_roi_mode(
                    config,
                    args.roi_mode,
                    stdin_is_tty=sys.stdin.isatty(),
                ),
            )
            profile = (
                config.select_profile_for_material(inputs.material_name, inputs.profile_name)
                if inputs.material_name is not None
                else config.select_profile(inputs.profile_name)
            )
            config, profile = _camera_overrides(config, profile, args)
            logger = configure_logging(profile.output.log_level)

            def confirm(summary: Mapping[str, Any]) -> bool:
                print("Operator confirmation summary:")
                print(json.dumps(dict(summary), indent=2, ensure_ascii=False))
                if args.yes:
                    return True
                if not sys.stdin.isatty():
                    raise ValueError("Confirmation requires --yes when stdin is noninteractive.")
                return input("Proceed with acquisition? [y/N]: ").strip().casefold() in {"y", "yes"}

            result = run_operator_workflow(config, inputs, confirm=confirm)
            if result.cancelled:
                print(
                    json.dumps(
                        {
                            "cancelled": True,
                            "reason": result.cancellation_reason,
                            "captured_still_count": result.captured_still_count,
                            "measurement_written": False,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                return 0
            assert result.outcome is not None
            outcome = result.outcome
            logger.info(
                "Processed %d deliberate still(s) for %s: %s",
                result.captured_still_count,
                outcome.record.measurement_id,
                outcome.record.status,
            )
            print(
                json.dumps(
                    {
                        "experiment_id": outcome.record.experiment_id,
                        "measurement_id": outcome.record.measurement_id,
                        "method": "yolo",
                        "acquisition_mode": outcome.record.acquisition_mode,
                        "status": outcome.record.status,
                        "valid": outcome.group.valid,
                        "roi_mode": profile.tube_roi.usage_mode,
                        "level_calibration_modes": sorted(
                            {
                                image.level_calibration_mode
                                for image in outcome.group.images
                            }
                        ),
                        "level_calibration_warning": (
                            None
                            if profile.tube_roi.usage_mode == "configured"
                            else (
                                "No calibrated full/empty vertical limits were used; "
                                "percentages are normalized to detected semantic extent."
                            )
                        ),
                        "estimated_material_percent": outcome.record.estimated_material_percent,
                        "captured_or_supplied_still_count": result.captured_still_count,
                        "accepted_image_count": outcome.group.accepted_count,
                        "rejected_image_count": outcome.group.rejected_count,
                        "rejection_reasons": outcome.group.rejection_reasons,
                        "session_directory": str(outcome.session_directory),
                        "result": str(outcome.result_path),
                        "workbook": str(outcome.session_directory / "yolo_measurements.xlsx"),
                        "csv": str(outcome.session_directory / "yolo_measurements.csv"),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        profile = config.select_profile(args.profile)
        config, profile = _camera_overrides(config, profile, args)
        logger = configure_logging(profile.output.log_level)
        summary: dict[str, object] = {
            "config": str(config.config_path),
            "profile": profile.name,
            "weights": str(profile.weights_path),
            "weights_present": profile.weights_path.is_file(),
            "expected_task": profile.expected_task,
            "estimation_mode": profile.estimation_mode,
            "device": profile.inference.device,
            "continuous_real_time_inference": False,
            "saved_still_level_estimation": True,
            "camera_device_name_contains": profile.camera.device_name_contains,
            "camera_backend": profile.camera.backend,
        }
        if args.validate_model:
            adapter = UltralyticsInferenceAdapter(profile)
            mapping = adapter.load()
            summary["model_validated"] = True
            summary["weights_sha256"] = adapter.weights_sha256
            summary["semantic_mapping"] = {
                "material": {"id": mapping.material_id, "name": mapping.material_name},
                "empty": {"id": mapping.empty_id, "name": mapping.empty_name},
            }
        else:
            summary["model_validated"] = False
        logger.info("Validated configuration for profile %s", profile.name)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except (CameraDiscoveryError, YoloFoundationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _operator_inputs(args: argparse.Namespace) -> OperatorInputs:
    interactive = bool(args.interactive)
    profile_name = args.profile
    mode = args.acquisition_mode
    if mode is None and args.input is not None:
        mode = "folder"
    mode = _required_value("acquisition mode", mode, interactive)
    experiment_id = _required_value("experiment id", args.experiment_id, interactive)
    purpose = _required_value("experiment purpose", args.purpose, interactive)
    capacity = args.total_capacity_ml
    if capacity is None and interactive:
        capacity = _prompt_float("Total capacity (mL)", required=True)
    if capacity is None:
        raise ValueError("--total-capacity-ml is required for an operator workflow.")
    folder_path = args.input
    if mode == "folder" and folder_path is None and interactive:
        folder_path = Path(_prompt("Saved image or folder path"))
    output_root = args.output_root
    if interactive and output_root is None:
        entered = _prompt("Absolute output directory (blank uses profile default)", "")
        output_root = None if not entered else Path(entered)
    density = args.bulk_density_g_per_ml
    total_weight = args.total_possible_weight_g
    remaining = args.remaining_weight_g
    manual = args.manual_material_level
    unit = args.manual_level_unit
    usable = args.usable_internal_height_mm
    material = args.material_name
    notes = args.notes
    if interactive:
        material = material or _prompt("Material name (selects YOLO model profile)")
        density, total_weight = _prompt_mass_reference_choice(density, total_weight)
        remaining = remaining if remaining is not None else _prompt_float("Independently measured remaining weight g (optional)")
        if manual is None:
            manual = _prompt_float("Manual material level (optional)")
        if manual is not None and unit is None:
            unit = _prompt("Manual level unit (mm/cm)")
        if manual is not None and usable is None:
            usable = _prompt_float("Usable internal height (mm)", required=True)
        notes = notes or _prompt("Operator notes (optional)", "") or None
    return OperatorInputs(
        profile_name=profile_name,
        experiment_id=str(experiment_id),
        purpose=str(purpose),
        total_capacity_ml=capacity,
        acquisition_mode=str(mode),
        output_root=output_root,
        folder_path=folder_path,
        recursive=args.recursive,
        bulk_density_g_per_ml=density,
        total_possible_weight_g=total_weight,
        remaining_weight_g=remaining,
        manual_material_level=manual,
        manual_material_level_unit=unit,
        usable_internal_height_mm=usable,
        material_name=material,
        operator_notes=notes,
        measurement_index=args.measurement_index,
        measurement_id=args.measurement_id,
        trigger_time_utc=args.trigger_time_utc,
        timed_capture_count=args.capture_count,
        capture_interval_seconds=args.capture_interval_seconds,
        capture_duration_seconds=args.capture_duration_seconds,
    )


def _prompt_mass_reference_choice(
    density: float | None,
    total_weight: float | None,
) -> tuple[float | None, float | None]:
    """Keep current mass metadata or replace it with one explicit input basis."""

    print("Mass reference currently configured:")
    print(
        "  Bulk density: "
        + ("unavailable" if density is None else f"{density:g} g/mL")
    )
    print(
        "  Total possible material mass: "
        + ("unavailable" if total_weight is None else f"{total_weight:g} g")
    )
    if not _prompt_yes_no("Do you want to change the density/total mass? [y/N]: "):
        return density, total_weight
    while True:
        choice = _prompt("Use bulk [d]ensity or total full-tube [m]ass? [d/m]").casefold()
        if choice in {"d", "density"}:
            value = _prompt_positive_float("Bulk density (g/mL)")
            print("Bulk density selected. Total possible material mass will remain blank.")
            return value, None
        if choice in {"m", "mass", "total", "total_mass"}:
            value = _prompt_positive_float(
                "Total possible material mass for a full tube (g)"
            )
            print(
                "Total mass selected. Bulk density will remain blank and no "
                "mass-to-volume reference will be calculated."
            )
            return None, value
        print("Enter d for bulk density or m for total full-tube material mass.")


def _prompt_yes_no(prompt: str) -> bool:
    while True:
        entered = input(prompt).strip().casefold()
        if not entered or entered in {"n", "no"}:
            return False
        if entered in {"y", "yes"}:
            return True
        print("Enter y for yes or n for no.")


def _prompt_positive_float(label: str) -> float:
    while True:
        value = input(f"{label}: ").strip()
        try:
            parsed = float(value)
        except ValueError:
            print("Enter a finite number greater than zero.")
            continue
        if not math.isfinite(parsed) or parsed <= 0.0:
            print("Enter a finite number greater than zero.")
            continue
        return parsed


def _camera_overrides(config, profile, args):
    if args.camera_name_contains is None and args.camera_backend is None:
        return config, profile
    requested_name = profile.camera.device_name_contains
    if args.camera_name_contains is not None:
        requested_name = args.camera_name_contains.strip()
        if not requested_name:
            raise ValueError("--camera-name-contains cannot be blank.")
    profile = replace(
        profile,
        camera=replace(
            profile.camera,
            device_name_contains=requested_name,
            backend=(
                profile.camera.backend
                if args.camera_backend is None
                else args.camera_backend
            ),
        ),
    )
    profiles = dict(config.profiles)
    profiles[profile.name] = profile
    return replace(config, profiles=profiles), profile


def _required_value(label: str, value: Any, interactive: bool) -> Any:
    if value is not None and (not isinstance(value, str) or value.strip()):
        return value
    if interactive:
        return _prompt(label)
    option = "--" + label.replace(" ", "-")
    raise ValueError(f"{option} is required for an operator workflow.")


def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    if value:
        return value
    if default is not None:
        return default
    raise ValueError(f"{label} is required.")


def _prompt_float(label: str, *, required: bool = False) -> float | None:
    value = input(f"{label}: ").strip()
    if not value and not required:
        return None
    if not value:
        raise ValueError(f"{label} is required.")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be numeric.") from exc


if __name__ == "__main__":
    raise SystemExit(main())
