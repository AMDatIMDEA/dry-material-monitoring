"""Measure material and empty volume once or at a configured interval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

# Running this file by path puts only ``3d_camera`` on sys.path. Add the
# repository root so the shared experiment_records package is importable without
# requiring an editable install; the legacy material_volume imports remain local.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from material_volume.capture import RealSenseDepthSource
from material_volume.config import AppConfig, load_config
from material_volume.errors import MaterialMeasurementError, MeasurementQualityError
from material_volume.export import save_measurement
from material_volume.models import CalibrationData, VolumeEstimate
from material_volume.pipeline import MaterialVolumePipeline, RefillWarningLatch
from material_volume.session import (
    DepthMeasurementRequest,
    PreparedDepthMethod,
    StorageProfile,
)
from material_volume.synthetic import SyntheticDepthSource


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 3D material surface map and estimate filled/empty tube volume."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", choices=("camera", "bag", "synthetic"), default="camera")
    parser.add_argument("--bag", type=Path, help="RealSense .bag file when --source bag is used.")
    parser.add_argument("--calibration", type=Path, help="Override the calibration .npz path.")
    parser.add_argument(
        "--fill-percent",
        type=float,
        help="Synthetic-mode fill percentage; defaults to synthetic.fill_percent in YAML.",
    )
    parser.add_argument("--continuous", action="store_true", help="Repeat at the configured interval.")
    parser.add_argument(
        "--max-measurements",
        type=int,
        default=0,
        help="Stop continuous mode after N measurements; 0 means no limit.",
    )
    parser.add_argument("--interval", type=float, help="Override update interval in seconds.")
    parser.add_argument("--no-export", action="store_true", help="Print results without saving files.")
    session = parser.add_argument_group("optional research session")
    session.add_argument("--experiment-id", help="Portable session identifier; enables session mode.")
    session.add_argument("--purpose", help="Session research purpose.")
    session.add_argument("--operator-notes", help="Optional notes stored in the common row.")
    session.add_argument("--material-name", help="Operator-provided material name.")
    session.add_argument("--total-capacity-ml", type=float, help="Capacity; must match frozen tube geometry.")
    session.add_argument("--bulk-density-g-per-ml", type=float, help="Explicit bulk density; never inferred.")
    session.add_argument("--total-possible-weight-g", type=float)
    session.add_argument("--reference-material-weight-g", type=float)
    session.add_argument("--manual-material-level-mm", type=float)
    session.add_argument(
        "--output-root",
        type=Path,
        help="Absolute session output root; defaults to research.directory in YAML.",
    )
    session.add_argument(
        "--storage-profile",
        choices=tuple(profile.value for profile in StorageProfile),
        help="Session evidence profile; defaults to research.storage_profile in YAML.",
    )
    session.add_argument("--measurement-index", type=int, help="Externally allocated index.")
    session.add_argument("--measurement-id", help="Externally allocated shared measurement ID.")
    session.add_argument("--trigger-time-utc", help="Timezone-aware ISO 8601 trigger time.")
    session.add_argument("--capture-start-utc", help="Externally observed capture start time.")
    session.add_argument("--capture-end-utc", help="Externally observed capture end time.")
    session.add_argument("--processing-time-utc", help="Externally observed result time.")
    return parser.parse_args(argv)


def build_source(config: AppConfig, args: argparse.Namespace):
    if args.source == "synthetic":
        return SyntheticDepthSource(config, fill_percent=args.fill_percent)
    if args.source == "bag":
        if args.bag is None:
            raise ValueError("--bag is required when --source bag is selected.")
        return RealSenseDepthSource(config, bag_path=args.bag)
    return RealSenseDepthSource(config)


def load_or_create_calibration(config: AppConfig, args: argparse.Namespace) -> CalibrationData:
    if args.source == "synthetic":
        with SyntheticDepthSource(config, fill_percent=0.0) as empty_source:
            empty_burst = empty_source.capture_burst(config.fusion.burst_frames)
        return MaterialVolumePipeline.calibrate(empty_burst, config)
    path = args.calibration.expanduser().resolve() if args.calibration else config.calibration_path
    return CalibrationData.load(path)


def print_result(estimate: VolumeEstimate, folder: Path | None) -> None:
    result = estimate.result
    print("\nMaterial volume measurement")
    print(f"  Status:              {result.warning_state}")
    print(f"  Fill:                {result.fill_percent:.2f} %")
    print(f"  Mean material level: {result.mean_level_mm:.2f} mm")
    print(f"  Material volume:     {result.material_volume_ml:.2f} mL")
    print(f"  Empty volume:        {result.empty_volume_ml:.2f} mL")
    print(f"  Tube capacity:       {result.capacity_ml:.2f} mL")
    print(f"  Quality score:       {result.quality.score:.3f}")
    print(f"  Surface coverage:    {result.quality.surface_coverage:.1%}")
    print(
        "  Heuristic variability: +/-"
        f"{result.quality.uncertainty_percent:.2f} percentage points (not GUM uncertainty)"
    )
    if result.quality.reasons:
        print("  Decision withheld:   " + "; ".join(result.quality.reasons))
    if result.warning_state == "REFILL_WARNING_ACTIVE":
        print("  *** REFILL WARNING: material is below the configured threshold. ***")
    elif result.refill_required:
        print("  Low level detected; waiting for configured repeat confirmation.")
    if folder is not None:
        print(f"  Output:              {folder}")


def _session_request(args: argparse.Namespace, config: AppConfig) -> DepthMeasurementRequest:
    calibration_path = None
    if args.source != "synthetic":
        calibration_path = (
            args.calibration.expanduser().resolve()
            if args.calibration
            else config.calibration_path
        )
    acquisition_mode = {
        "camera": "standalone_camera",
        "bag": "bag",
        "synthetic": "synthetic",
    }[args.source]
    return DepthMeasurementRequest(
        experiment_id=args.experiment_id,
        purpose=args.purpose,
        operator_notes=args.operator_notes,
        material_name=args.material_name,
        total_capacity_ml=args.total_capacity_ml,
        bulk_density_g_per_ml=args.bulk_density_g_per_ml,
        total_possible_weight_g=args.total_possible_weight_g,
        reference_material_weight_g=args.reference_material_weight_g,
        manual_material_level_mm=args.manual_material_level_mm,
        output_root=args.output_root,
        storage_profile=args.storage_profile,
        measurement_index=args.measurement_index,
        measurement_id=args.measurement_id,
        trigger_time_utc=args.trigger_time_utc,
        capture_start_utc=args.capture_start_utc,
        capture_end_utc=args.capture_end_utc,
        processing_time_utc=args.processing_time_utc,
        acquisition_mode=acquisition_mode,
        calibration_path=calibration_path,
        software_repository=REPOSITORY_ROOT,
    )


def _validate_session_args(args: argparse.Namespace) -> None:
    session_only_values = (
        args.purpose,
        args.operator_notes,
        args.material_name,
        args.total_capacity_ml,
        args.bulk_density_g_per_ml,
        args.total_possible_weight_g,
        args.reference_material_weight_g,
        args.manual_material_level_mm,
        args.output_root,
        args.storage_profile,
        args.measurement_index,
        args.measurement_id,
        args.trigger_time_utc,
        args.capture_start_utc,
        args.capture_end_utc,
        args.processing_time_utc,
    )
    if args.experiment_id is None and any(value is not None for value in session_only_values):
        raise ValueError("Session metadata options require --experiment-id.")
    if args.experiment_id is not None and args.no_export:
        raise ValueError("--no-export cannot be combined with --experiment-id.")
    external_identity_or_time = any(
        value is not None
        for value in (
            args.measurement_index,
            args.measurement_id,
            args.trigger_time_utc,
            args.capture_start_utc,
            args.capture_end_utc,
            args.processing_time_utc,
        )
    )
    if args.continuous and external_identity_or_time:
        raise ValueError(
            "Externally supplied identity/timestamps are accepted only for one measurement; "
            "omit --continuous or let continuous session mode allocate them automatically."
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_session_args(args)
        config = load_config(args.config)
        for warning in config.validate():
            print(f"SETUP WARNING: {warning}")
        interval = config.runtime.update_interval_seconds if args.interval is None else args.interval
        if interval <= 0.0:
            raise ValueError("Measurement interval must be positive.")
        calibration = load_or_create_calibration(config, args)
        pipeline = MaterialVolumePipeline(config, calibration)
        latch = RefillWarningLatch(config)
        measurement_count = 0

        with build_source(config, args) as source:
            prepared = (
                PreparedDepthMethod(config, calibration, source, latch=latch)
                if args.experiment_id is not None
                else None
            )
            while True:
                started = time.monotonic()
                try:
                    session_failed = False
                    if prepared is not None:
                        outcome = prepared.run_one(_session_request(args, config))
                        if outcome.estimate is None:
                            print(
                                "INVALID MEASUREMENT - no refill decision was issued; "
                                f"evidence: {outcome.artifact_directory}",
                                file=sys.stderr,
                            )
                            if (
                                not args.continuous
                                or not config.runtime.continue_after_invalid_measurement
                            ):
                                return 3
                            session_failed = True
                        else:
                            estimate = outcome.estimate
                            folder = outcome.artifact_directory
                    else:
                        burst = source.capture_burst(config.fusion.burst_frames)
                        estimate = pipeline.measure(burst)
                        latch.update(estimate)
                        folder = None if args.no_export else save_measurement(estimate, config)
                    if not session_failed:
                        print_result(estimate, folder)
                        measurement_count += 1
                except MeasurementQualityError as exc:
                    print(
                        f"INVALID MEASUREMENT - no refill decision was issued: {exc}",
                        file=sys.stderr,
                    )
                    if not args.continuous or not config.runtime.continue_after_invalid_measurement:
                        return 3

                if not args.continuous:
                    break
                if args.max_measurements > 0 and measurement_count >= args.max_measurements:
                    break
                remaining = interval - (time.monotonic() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
        return 0
    except (MaterialMeasurementError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Measurement stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
