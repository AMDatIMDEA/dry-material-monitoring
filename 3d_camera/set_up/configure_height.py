"""Interactive D405 camera-to-rim placement optimizer."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from enum import Enum
from pathlib import Path
import sys
from typing import TextIO

# Direct execution from ``3d_camera/set_up`` must still see the independent
# sibling ``material_volume`` package.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
for import_root in (REPOSITORY_ROOT, PACKAGE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from material_volume.config import AppConfig, load_config
from material_volume.errors import MaterialMeasurementError
from set_up.precision_test import PrecisionTestResult, run_flat_target_precision_test
from set_up.setup_config import save_setup_to_config
from set_up.setup_optimizer import (
    CameraDiscoveryError,
    MountingRange,
    OptimizationResult,
    ProfileDiscovery,
    Recommendation,
    SafetyAllowances,
    SetupEvaluation,
    SetupStatus,
    SetupValidationError,
    TubeDimensions,
    best_valid_at_exact_distance,
    discover_connected_d405_profiles,
    evaluate_exact_distance,
    make_fallback_profiles,
    nearest_valid_alternatives,
    optimize_setups,
)


DEFAULT_CONFIG = PACKAGE_ROOT / "config.yaml"
InputFunction = Callable[[str], str]


class UserCancelled(RuntimeError):
    """The user ended interactive input without requesting a save."""


class OutputMode(str, Enum):
    ABSTRACT = "ABSTRACT"
    LONG_REVIEW = "LONG REVIEW"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize D405 camera-to-rim distance and depth profile for a circular tube."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run the hardware-free 58/55/115 mm example without prompts or saving.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("abstract", "long"),
        help="Skip the output-mode prompt (primarily for automated review/example runs).",
    )
    parser.add_argument(
        "--precision-test",
        action="store_true",
        help="After setup selection, test repeatability on a connected flat matte target.",
    )
    parser.add_argument(
        "--precision-frames",
        type=int,
        default=60,
        help="Frames for --precision-test (default: 60; minimum: 10).",
    )
    parser.add_argument(
        "--known-distance-mm",
        type=float,
        help=(
            "Known optical-origin-to-target distance for an optional absolute-error "
            "result during --precision-test."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show an unexpected developer traceback instead of a short error.",
    )
    return parser.parse_args()


def collect_interactive_inputs(
    input_fn: InputFunction = input,
    output: TextIO = sys.stdout,
) -> tuple[TubeDimensions, SafetyAllowances, bool, MountingRange]:
    print("D405 CAMERA PLACEMENT OPTIMIZER", file=output)
    print(
        "Distances are measured from the camera depth optical origin, not the housing.",
        file=output,
    )
    outer = _prompt_float(
        "Tube outer diameter in mm",
        input_fn=input_fn,
        output=output,
        minimum=0.0,
        minimum_inclusive=False,
    )
    inner = _prompt_float(
        "Tube inner diameter in mm",
        input_fn=input_fn,
        output=output,
        minimum=0.0,
        maximum=outer,
        minimum_inclusive=False,
    )
    usable_height = _prompt_float(
        "Tube usable internal height in mm",
        input_fn=input_fn,
        output=output,
        minimum=0.0,
        minimum_inclusive=False,
    )
    maximum_fill = _prompt_float(
        "Maximum filling height in mm",
        default=usable_height,
        input_fn=input_fn,
        output=output,
        minimum=0.0,
        maximum=usable_height,
    )
    free_margin_percent = _prompt_float(
        "Required free image margin around each side of the tube in %",
        default=10.0,
        input_fn=input_fn,
        output=output,
        minimum=0.0,
        maximum=99.999,
    )
    centring_error = _prompt_float(
        "Expected centring error in mm",
        default=3.0,
        input_fn=input_fn,
        output=output,
        minimum=0.0,
    )
    maximum_tilt = _prompt_float(
        "Maximum camera tilt in degrees",
        default=2.0,
        input_fn=input_fn,
        output=output,
        minimum=0.0,
        maximum=89.999,
    )
    connected = _prompt_yes_no(
        "Is a D405 currently connected?",
        default=False,
        input_fn=input_fn,
        output=output,
    )
    mechanical_min = _prompt_optional_float(
        "Minimum mechanically available camera-to-rim distance in mm (optional)",
        input_fn=input_fn,
        output=output,
        minimum=0.0,
        minimum_inclusive=False,
    )
    mechanical_max = _prompt_optional_float(
        "Maximum mechanically available camera-to-rim distance in mm (optional)",
        input_fn=input_fn,
        output=output,
        minimum=mechanical_min if mechanical_min is not None else 0.0,
        minimum_inclusive=mechanical_min is not None,
    )
    try:
        tube = TubeDimensions(
            outer_diameter_mm=outer,
            inner_diameter_mm=inner,
            usable_height_mm=usable_height,
            maximum_fill_height_mm=maximum_fill,
        )
        safety = SafetyAllowances(
            free_margin_fraction=free_margin_percent / 100.0,
            centring_error_mm=centring_error,
            maximum_tilt_degrees=maximum_tilt,
        )
        mounting = MountingRange(
            minimum_camera_to_rim_mm=mechanical_min,
            maximum_camera_to_rim_mm=mechanical_max,
        )
    except SetupValidationError as exc:
        # Cross-field checks are repeated interactively instead of exposing a traceback.
        print(f"Invalid setup: {exc}", file=output)
        raise UserCancelled(
            "Please rerun the optimizer and correct the conflicting dimensions."
        ) from exc
    return tube, safety, connected, mounting


def run_interactive(
    config: AppConfig,
    args: argparse.Namespace,
    *,
    input_fn: InputFunction = input,
    output: TextIO = sys.stdout,
) -> int:
    tube, safety, camera_expected, mounting = collect_interactive_inputs(
        input_fn=input_fn,
        output=output,
    )
    discovery = _obtain_profiles(
        camera_expected,
        maximum_reliable_depth_mm=config.filters.max_distance_mm,
        output=output,
    )
    result = optimize_setups(
        tube=tube,
        profiles=discovery.profiles,
        safety=safety,
        configured_minimum_depth_mm=config.filters.min_distance_mm,
        configured_maximum_depth_mm=config.filters.max_distance_mm,
        mounting_range=mounting,
        distance_step_mm=1.0,
    )
    output_mode = (
        OutputMode.ABSTRACT
        if getattr(args, "output_mode", None) == "abstract"
        else (
            OutputMode.LONG_REVIEW
            if getattr(args, "output_mode", None) == "long"
            else _prompt_output_mode(input_fn=input_fn, output=output)
        )
    )
    if not result.valid:
        if output_mode is OutputMode.LONG_REVIEW:
            _print_profile_source(discovery, output=output)
            _print_no_solution(
                result=result,
                tube=tube,
                safety=safety,
                discovery=discovery,
                config=config,
                mounting=mounting,
                output=output,
            )
        else:
            _print_abstract_no_solution(result, discovery=discovery, output=output)
        return 1

    selected = result.best_compromise
    if selected is None:
        print("No best-compromise recommendation could be selected.", file=output)
        return 1
    selected_range = _mounting_range_for_evaluation(result, selected)
    if output_mode is OutputMode.LONG_REVIEW:
        _print_profile_source(discovery, output=output)
        print("\nD405 SETUP RECOMMENDATIONS", file=output)
        seen_setups: dict[tuple[int, int, int, float], str] = {}
        for recommendation in result.recommendations:
            duplicate_of = seen_setups.get(recommendation.evaluation.setup_key)
            _print_recommendation(
                recommendation,
                discovery=discovery,
                duplicate_of=duplicate_of,
                output=output,
            )
            seen_setups.setdefault(
                recommendation.evaluation.setup_key, recommendation.category
            )
    else:
        _print_abstract(
            selected,
            discovery=discovery,
            mounting_range=selected_range,
            heading="D405 SETUP – ABSTRACT",
            output=output,
        )
    suitable = _prompt_yes_no(
        "Is the recommended camera-to-rim distance mechanically suitable?",
        default=True,
        input_fn=input_fn,
        output=output,
    )
    if not suitable:
        preferred = _prompt_float(
            "Preferred camera-to-rim distance in mm",
            input_fn=input_fn,
            output=output,
            minimum=0.0,
            minimum_inclusive=False,
        )
        exact = evaluate_exact_distance(
            tube=tube,
            profiles=discovery.profiles,
            camera_to_rim_distance_mm=preferred,
            safety=safety,
            configured_minimum_depth_mm=config.filters.min_distance_mm,
            configured_maximum_depth_mm=config.filters.max_distance_mm,
            mounting_range=mounting,
        )
        selected = best_valid_at_exact_distance(exact)
        if output_mode is OutputMode.LONG_REVIEW:
            print(
                f"\nEXACT-DISTANCE EVALUATION: camera_to_rim_mm = {preferred:.1f}",
                file=output,
            )
            for evaluation in exact:
                _print_evaluation(
                    evaluation,
                    heading=f"PROFILE {evaluation.profile.name}",
                    discovery=discovery,
                    output=output,
                )
        if selected is None:
            alternatives = nearest_valid_alternatives(
                tube=tube,
                profiles=discovery.profiles,
                requested_camera_to_rim_mm=preferred,
                safety=safety,
                configured_minimum_depth_mm=config.filters.min_distance_mm,
                configured_maximum_depth_mm=config.filters.max_distance_mm,
                mounting_range=mounting,
            )
            if output_mode is OutputMode.LONG_REVIEW:
                print(
                    "\nThe requested distance is PHYSICALLY_INVALID for the complete fill range.",
                    file=output,
                )
                if alternatives:
                    print("Nearest physically usable alternatives:", file=output)
                    for alternative in alternatives:
                        print(
                            f"  {alternative.camera_to_rim_mm:.1f} mm with "
                            f"{alternative.profile.name} "
                            f"(status {alternative.status.value})",
                            file=output,
                        )
                else:
                    print(
                        "No physically usable alternative exists inside the stated mechanical range.",
                        file=output,
                    )
            else:
                print(
                    "\nStatus: PHYSICALLY_INVALID\n"
                    "Warning: requested distance cannot measure the complete range.\n"
                    + (
                        f"Next action: try {alternatives[0].camera_to_rim_mm:.1f} mm "
                        f"with {alternatives[0].profile.name}."
                        if alternatives
                        else "Next action: widen the mechanical mounting range."
                    ),
                    file=output,
                )
            return 1
        selected_range = _mounting_range_for_evaluation(result, selected)
        if output_mode is OutputMode.ABSTRACT:
            _print_abstract(
                selected,
                discovery=discovery,
                mounting_range=selected_range,
                heading="EXACT-DISTANCE RESULT – ABSTRACT",
                output=output,
            )
        else:
            print(
                f"\nBest physically usable profile at exactly {preferred:.1f} mm: "
                f"{selected.profile.name} ({selected.status.value})",
                file=output,
            )

    if output_mode is OutputMode.LONG_REVIEW:
        print("\nFINAL SETUP SUMMARY", file=output)
        _print_evaluation(
            selected,
            heading="ACCEPTED GEOMETRIC SETUP",
            discovery=discovery,
            output=output,
        )
        _print_accuracy_interpretation(output=output)

    if args.precision_test:
        if discovery.approximate:
            print(
                "\nPRECISION TEST NOT RUN: a connected D405 with actual profiles is required.",
                file=output,
            )
        else:
            _read_input(
                "Place a flat matte target perpendicular to the camera in the central "
                "image area, then press Enter to start the precision test: ",
                input_fn,
            )
            try:
                precision = run_flat_target_precision_test(
                    selected.profile,
                    serial_number=discovery.serial_number,
                    frame_count=args.precision_frames,
                    known_reference_distance_mm=args.known_distance_mm,
                )
            except RuntimeError as exc:
                print(f"PRECISION TEST ERROR: {exc}", file=output)
                print(
                    "The accepted geometry is unchanged; no repeatability value was recorded.",
                    file=output,
                )
            else:
                _print_precision_result(precision, output=output)

    save = _prompt_yes_no(
        f"Save this setup to {config.config_path}?",
        default=False,
        input_fn=input_fn,
        output=output,
    )
    if not save:
        print("Setup accepted for review; configuration was not changed.", file=output)
        return 0
    saved = save_setup_to_config(
        config.config_path,
        evaluation=selected,
        discovery=discovery,
    )
    print(f"Configuration saved: {saved.config_path}", file=output)
    print(f"Timestamped backup:   {saved.backup_path}", file=output)
    print(
        "IMPORTANT: repeat empty-tube calibration before measuring because the "
        "geometry/profile changed.",
        file=output,
    )
    return 0


def run_example(
    config: AppConfig,
    output: TextIO = sys.stdout,
    output_mode: OutputMode = OutputMode.ABSTRACT,
) -> int:
    """Hardware-free representative example required by the project documentation."""
    tube = TubeDimensions(58.0, 55.0, 115.0, 115.0)
    safety = SafetyAllowances()
    discovery = make_fallback_profiles(config.filters.max_distance_mm)
    result = optimize_setups(
        tube=tube,
        profiles=discovery.profiles,
        safety=safety,
        configured_minimum_depth_mm=config.filters.min_distance_mm,
        configured_maximum_depth_mm=config.filters.max_distance_mm,
    )
    if not result.valid:
        if output_mode is OutputMode.LONG_REVIEW:
            _print_profile_source(discovery, output=output)
            _print_no_solution(
                result=result,
                tube=tube,
                safety=safety,
                discovery=discovery,
                config=config,
                mounting=MountingRange(),
                output=output,
            )
        else:
            _print_abstract_no_solution(result, discovery=discovery, output=output)
        return 1
    if output_mode is OutputMode.LONG_REVIEW:
        print("HARDWARE-FREE D405 OPTIMIZER EXAMPLE", file=output)
        _print_profile_source(discovery, output=output)
        for recommendation in result.recommendations:
            _print_recommendation(
                recommendation,
                discovery=discovery,
                duplicate_of=None,
                output=output,
            )
        print(
            "\nExample only: values are approximate and config.yaml was not changed.",
            file=output,
        )
    else:
        selected = result.best_compromise
        if selected is None:
            return 1
        _print_abstract(
            selected,
            discovery=discovery,
            mounting_range=_mounting_range_for_evaluation(result, selected),
            heading="HARDWARE-FREE D405 EXAMPLE – ABSTRACT",
            output=output,
        )
    return 0


def _obtain_profiles(
    camera_expected: bool,
    *,
    maximum_reliable_depth_mm: float,
    output: TextIO,
) -> ProfileDiscovery:
    if camera_expected:
        try:
            return discover_connected_d405_profiles(maximum_reliable_depth_mm)
        except CameraDiscoveryError as exc:
            print(f"\nCAMERA QUERY WARNING: {exc}", file=output)
            print(
                "Continuing with the documented offline D405 table. Results are approximate.",
                file=output,
            )
            fallback = make_fallback_profiles(maximum_reliable_depth_mm)
            return ProfileDiscovery(
                profiles=fallback.profiles,
                source=f"{fallback.source} after camera query failure",
                approximate=True,
                notes=(str(exc),) + fallback.notes,
            )
    return make_fallback_profiles(maximum_reliable_depth_mm)


def _mounting_range_for_evaluation(
    result: OptimizationResult,
    evaluation: SetupEvaluation,
) -> tuple[float, float]:
    candidates = [
        candidate
        for candidate in result.valid_candidates
        if (
            candidate.profile.width,
            candidate.profile.height,
            candidate.profile.fps,
        )
        == (
            evaluation.profile.width,
            evaluation.profile.height,
            evaluation.profile.fps,
        )
        and (candidate.valid if evaluation.valid else candidate.usable)
    ]
    if not candidates:
        return evaluation.camera_to_rim_mm, evaluation.camera_to_rim_mm
    return (
        min(candidate.camera_to_rim_mm for candidate in candidates),
        max(candidate.camera_to_rim_mm for candidate in candidates),
    )


def _print_abstract(
    evaluation: SetupEvaluation,
    *,
    discovery: ProfileDiscovery,
    mounting_range: tuple[float, float],
    heading: str,
    output: TextIO,
) -> None:
    fill_min = evaluation.measurable_fill_min_mm
    fill_max = evaluation.measurable_fill_max_mm
    maximum_fill = max(evaluation.tube.maximum_fill_height_mm, 1e-9)
    if fill_min is None or fill_max is None:
        fill_text = "none"
    else:
        fill_text = (
            f"{fill_min:.1f}-{fill_max:.1f} mm "
            f"({100.0 * fill_min / maximum_fill:.0f}-"
            f"{100.0 * fill_max / maximum_fill:.0f}%)"
        )
    warning = _essential_warning(evaluation, discovery)
    next_action = (
        "Increase distance or relax an allowance only after a real-camera trial."
        if evaluation.status is SetupStatus.CONDITIONALLY_USABLE
        else "Mount rigidly, query the real D405, then repeat empty-tube calibration."
    )
    print(f"\n{heading}", file=output)
    print(f"Camera to rim:       {evaluation.camera_to_rim_mm:.1f} mm", file=output)
    print(f"Camera to bottom:    {evaluation.camera_to_bottom_mm:.1f} mm", file=output)
    print(f"Profile:             {evaluation.profile.name} FPS", file=output)
    print(
        f"Valid mounting range:{mounting_range[0]:8.1f}-{mounting_range[1]:.1f} mm",
        file=output,
    )
    print(f"Measurable fill:     {fill_text}", file=output)
    print(
        f"Projected outer tube:{evaluation.nearest.outer_pixels_x:8.1f} x "
        f"{evaluation.nearest.outer_pixels_y:.1f} px at nearest surface",
        file=output,
    )
    print(
        f"Approx. geometry:   {evaluation.nearest.sampling_x_mm_per_pixel:.4f} x "
        f"{evaluation.nearest.sampling_y_mm_per_pixel:.4f} mm/pixel",
        file=output,
    )
    print(f"Final status:        {evaluation.status.value}", file=output)
    print(f"Essential warning:   {warning}", file=output)
    print(f"Next action:         {next_action}", file=output)


def _print_abstract_no_solution(
    result: OptimizationResult,
    *,
    discovery: ProfileDiscovery,
    output: TextIO,
) -> None:
    print("\nD405 SETUP – ABSTRACT", file=output)
    print("Best camera-to-rim:  none", file=output)
    print("Valid mounting range:none", file=output)
    print("Measurable fill:     none", file=output)
    print("Final status:        PHYSICALLY_INVALID", file=output)
    print(
        "Essential warning:   No profile/distance combination covers the complete tube.",
        file=output,
    )
    print(
        "Next action:         Review tube size, depth range, or mechanical mounting limits.",
        file=output,
    )
    if discovery.approximate:
        print("Camera data:         offline fallback (approximate)", file=output)


def _essential_warning(
    evaluation: SetupEvaluation, discovery: ProfileDiscovery
) -> str:
    if evaluation.allowance_reasons:
        return evaluation.allowance_reasons[0]
    if discovery.approximate:
        return "Offline nominal intrinsics; confirm with the connected D405."
    if evaluation.warnings:
        return evaluation.warnings[0]
    return "Geometry is feasible, but real material and known fill levels still require validation."


def _format_measurable_range(
    minimum_mm: float | None,
    maximum_mm: float | None,
    requested_maximum_mm: float,
) -> str:
    if minimum_mm is None or maximum_mm is None:
        return "none"
    denominator = max(requested_maximum_mm, 1e-9)
    return (
        f"{minimum_mm:.1f}-{maximum_mm:.1f} mm "
        f"({100.0 * minimum_mm / denominator:.0f}-"
        f"{100.0 * maximum_mm / denominator:.0f}%)"
    )


def _print_profile_source(discovery: ProfileDiscovery, *, output: TextIO) -> None:
    print("\nCAMERA INFORMATION", file=output)
    print(f"  Source:                         {discovery.source}", file=output)
    print(
        f"  Values approximate:             {'YES' if discovery.approximate else 'NO'}",
        file=output,
    )
    if discovery.device_model:
        print(f"  Connected model:                {discovery.device_model}", file=output)
    if discovery.serial_number:
        print(f"  Serial number:                  {discovery.serial_number}", file=output)
    print(f"  Evaluated profiles:             {len(discovery.profiles)}", file=output)
    for profile in discovery.profiles:
        print(
            f"    {profile.name}: H-FOV {profile.horizontal_fov_degrees:.2f} deg, "
            f"V-FOV {profile.vertical_fov_degrees:.2f} deg, "
            f"Min-Z {profile.minimum_usable_depth_mm:.1f} mm",
            file=output,
        )
    for note in discovery.notes:
        print(f"  Note: {note}", file=output)


def _print_recommendation(
    recommendation: Recommendation,
    *,
    discovery: ProfileDiscovery,
    duplicate_of: str | None,
    output: TextIO,
) -> None:
    heading = recommendation.category
    if duplicate_of:
        heading += f" (same physical setup as {duplicate_of})"
    _print_evaluation(
        recommendation.evaluation,
        heading=heading,
        discovery=discovery,
        output=output,
    )
    print(
        f"  Valid mounting-distance range for profile: "
        f"{recommendation.mounting_minimum_mm:.1f}-"
        f"{recommendation.mounting_maximum_mm:.1f} mm",
        file=output,
    )


def _print_evaluation(
    evaluation: SetupEvaluation,
    *,
    heading: str,
    discovery: ProfileDiscovery,
    output: TextIO,
) -> None:
    rim = evaluation.rim
    nearest = evaluation.nearest
    bottom = evaluation.bottom
    print(f"\n{heading}", file=output)
    print("", file=output)
    print("Tube:", file=output)
    print(
        f"  Outer diameter:                         {evaluation.tube.outer_diameter_mm:9.2f} mm",
        file=output,
    )
    print(
        f"  Inner diameter:                         {evaluation.tube.inner_diameter_mm:9.2f} mm",
        file=output,
    )
    print(
        f"  Usable internal height:                 {evaluation.tube.usable_height_mm:9.2f} mm",
        file=output,
    )
    print(
        f"  Maximum filling height:                 {evaluation.tube.maximum_fill_height_mm:9.2f} mm",
        file=output,
    )
    print("", file=output)
    print("Depth profile:", file=output)
    print(
        f"  Resolution and rate:                    {evaluation.profile.width} x "
        f"{evaluation.profile.height} @ {evaluation.profile.fps} FPS",
        file=output,
    )
    print(
        f"  Intrinsics source:                      {evaluation.profile.intrinsics_source}",
        file=output,
    )
    print(
        f"  Horizontal / vertical FOV:              "
        f"{evaluation.profile.horizontal_fov_degrees:.2f} / "
        f"{evaluation.profile.vertical_fov_degrees:.2f} deg",
        file=output,
    )
    print(
        f"  Stereo baseline used for overlap:        "
        f"{evaluation.profile.stereo_baseline_mm:9.2f} mm",
        file=output,
    )
    print(
        f"  Camera data approximate:                {'YES' if discovery.approximate else 'NO'}",
        file=output,
    )
    print("", file=output)
    print("Explicit distance definitions:", file=output)
    print(
        f"  camera_to_rim_mm:                       {evaluation.camera_to_rim_mm:9.2f} mm",
        file=output,
    )
    print(
        f"  camera_to_bottom_mm:                    {evaluation.camera_to_bottom_mm:9.2f} mm",
        file=output,
    )
    print(
        f"  nearest_surface_distance_mm:            "
        f"{evaluation.nearest_surface_distance_mm:9.2f} mm",
        file=output,
    )
    print(
        f"  farthest_surface_distance_mm:           "
        f"{evaluation.farthest_surface_distance_mm:9.2f} mm",
        file=output,
    )
    print("", file=output)
    print("Tube rim / stereo-overlap check:", file=output)
    print(
        f"  Centred image availability (H / V):     "
        f"{rim.centred_width_mm:9.2f} / {rim.centred_height_mm:.2f} mm",
        file=output,
    )
    print(
        f"  Conservative stereo-overlap width:      "
        f"{rim.stereo_overlap_width_mm:9.2f} mm",
        file=output,
    )
    print(
        f"  Nominal tube margins (H / V / stereo):  "
        f"{rim.nominal_horizontal_margin_mm:9.2f} / "
        f"{rim.nominal_vertical_margin_mm:.2f} / "
        f"{rim.nominal_stereo_overlap_margin_mm:.2f} mm",
        file=output,
    )
    print(
        f"  Bottom rim/wall clearance (nominal / conservative): "
        f"{bottom.nominal_rim_clearance_mm:.2f} / "
        f"{bottom.conservative_rim_clearance_mm:.2f} mm",
        file=output,
    )
    print("", file=output)
    print("Coverage at nearest allowed material surface:", file=output)
    print(
        f"  Visible full image area:                 {nearest.visible_width_mm:9.2f} x "
        f"{nearest.visible_height_mm:.2f} mm",
        file=output,
    )
    print(
        f"  Principal-point-centred availability:   {nearest.centred_width_mm:9.2f} x "
        f"{nearest.centred_height_mm:.2f} mm",
        file=output,
    )
    print(
        f"  Conservative stereo-overlap width:      "
        f"{nearest.stereo_overlap_width_mm:9.2f} mm",
        file=output,
    )
    print(
        f"  Required diameter with allowances:      "
        f"{nearest.required_diameter_with_allowance_mm:9.2f} mm",
        file=output,
    )
    print(
        f"  Horizontal / vertical margin:           "
        f"{nearest.horizontal_remaining_margin_mm:9.2f} / "
        f"{nearest.vertical_remaining_margin_mm:.2f} mm",
        file=output,
    )
    print("", file=output)
    print("Coverage at empty tube bottom:", file=output)
    print(
        f"  Visible full image area:                 {bottom.visible_width_mm:9.2f} x "
        f"{bottom.visible_height_mm:.2f} mm",
        file=output,
    )
    print(
        f"  Principal-point-centred availability:   {bottom.centred_width_mm:9.2f} x "
        f"{bottom.centred_height_mm:.2f} mm",
        file=output,
    )
    print(
        f"  Conservative stereo-overlap width:      "
        f"{bottom.stereo_overlap_width_mm:9.2f} mm",
        file=output,
    )
    print(
        f"  Required diameter with allowances:      "
        f"{bottom.required_diameter_with_allowance_mm:9.2f} mm",
        file=output,
    )
    print(
        f"  Horizontal / vertical margin:           "
        f"{bottom.horizontal_remaining_margin_mm:9.2f} / "
        f"{bottom.vertical_remaining_margin_mm:.2f} mm",
        file=output,
    )
    print(
        f"  Minimum H / V margin over all heights:  "
        f"{evaluation.minimum_horizontal_remaining_margin_mm:9.2f} / "
        f"{evaluation.minimum_vertical_remaining_margin_mm:.2f} mm",
        file=output,
    )
    print(
        f"  Maximum supported outer diameter:       "
        f"{evaluation.maximum_supported_tube_diameter_mm:9.2f} mm",
        file=output,
    )
    print("", file=output)
    print("Projected tube diameter:", file=output)
    print(
        f"  Outer at nearest (X / Y):                {nearest.outer_pixels_x:9.1f} / "
        f"{nearest.outer_pixels_y:.1f} px",
        file=output,
    )
    print(
        f"  Inner at nearest (X / Y):                {nearest.inner_pixels_x:9.1f} / "
        f"{nearest.inner_pixels_y:.1f} px",
        file=output,
    )
    print(
        f"  Outer at bottom (X / Y):                 {bottom.outer_pixels_x:9.1f} / "
        f"{bottom.outer_pixels_y:.1f} px",
        file=output,
    )
    print(
        f"  Inner at bottom (X / Y):                 {bottom.inner_pixels_x:9.1f} / "
        f"{bottom.inner_pixels_y:.1f} px",
        file=output,
    )
    print("", file=output)
    print("lateral_sampling_mm_per_pixel (geometric, not precision):", file=output)
    print(
        f"  At nearest surface (X / Y):              "
        f"{nearest.sampling_x_mm_per_pixel:9.4f} / "
        f"{nearest.sampling_y_mm_per_pixel:.4f} mm/pixel",
        file=output,
    )
    print(
        f"  At tube bottom (X / Y):                  "
        f"{bottom.sampling_x_mm_per_pixel:9.4f} / "
        f"{bottom.sampling_y_mm_per_pixel:.4f} mm/pixel",
        file=output,
    )
    print("", file=output)
    print("Depth range:", file=output)
    print(
        f"  Effective minimum / maximum:             "
        f"{evaluation.effective_minimum_depth_mm:9.2f} / "
        f"{evaluation.effective_maximum_depth_mm:.2f} mm",
        file=output,
    )
    print(
        f"  Nearest / farthest tilt-envelope point:  "
        f"{evaluation.nearest_tilt_envelope_distance_mm:9.2f} / "
        f"{evaluation.farthest_tilt_envelope_distance_mm:.2f} mm",
        file=output,
    )
    print(
        f"  Near / far depth margin:                 "
        f"{evaluation.near_depth_margin_mm:9.2f} / "
        f"{evaluation.far_depth_margin_mm:.2f} mm",
        file=output,
    )
    print(
        "  Physical measurable material heights:   "
        + _format_measurable_range(
            evaluation.physical_measurable_fill_min_mm,
            evaluation.physical_measurable_fill_max_mm,
            evaluation.tube.maximum_fill_height_mm,
        ),
        file=output,
    )
    print(
        "  Safety-compliant material heights:       "
        + _format_measurable_range(
            evaluation.safe_measurable_fill_min_mm,
            evaluation.safe_measurable_fill_max_mm,
            evaluation.tube.maximum_fill_height_mm,
        ),
        file=output,
    )
    print(
        f"  Expected depth quality (estimate):       {evaluation.estimated_depth_quality}",
        file=output,
    )
    print(
        "  estimated_depth_uncertainty_mm:          Not available from geometry alone.",
        file=output,
    )
    print(
        "  empirical_repeatability_mm:              Not available until tested with the real camera and target.",
        file=output,
    )
    print("", file=output)
    print(
        f"Complete geometric/physical status:        {evaluation.status.value}",
        file=output,
    )
    if evaluation.physical_reasons:
        print("Physical constraints that fail:", file=output)
        for reason in evaluation.physical_reasons:
            print(f"  - {reason}", file=output)
    if evaluation.allowance_reasons:
        print("Conservative allowances that fail:", file=output)
        for reason in evaluation.allowance_reasons:
            print(f"  - {reason}", file=output)
    if evaluation.warnings:
        print("Warnings:", file=output)
        for warning in evaluation.warnings:
            print(f"  - {warning}", file=output)


def _print_accuracy_interpretation(*, output: TextIO) -> None:
    print("\nMEASUREMENT INTERPRETATION", file=output)
    print(
        "  mm/pixel is lateral geometric sampling, not depth precision or absolute accuracy.",
        file=output,
    )
    print(
        "  No documented project coefficient maps this geometry to D405 depth uncertainty.",
        file=output,
    )
    print(
        "  Empirical repeatability requires repeated frames from the installed camera/target.",
        file=output,
    )
    print(
        "  Actual accuracy also depends on material texture, transparency, reflections, "
        "lighting, surface angle, stereo matching, calibration, camera temperature, "
        "and missing depth pixels.",
        file=output,
    )


def _print_precision_result(
    result: PrecisionTestResult, *, output: TextIO
) -> None:
    print("\nEMPIRICAL FLAT-TARGET PRECISION TEST", file=output)
    print(f"  Frames:                             {result.frame_count}", file=output)
    print(
        f"  Central ROI fraction:                {result.roi_fraction:.0%}",
        file=output,
    )
    print(
        f"  Median measured distance:            {result.median_distance_mm:.4f} mm",
        file=output,
    )
    print(
        f"  Frame-median repeatability (robust):  {result.frame_median_repeatability_mm:.4f} mm",
        file=output,
    )
    print(
        f"  Median pixel robust standard dev.:    {result.median_pixel_robust_std_mm:.4f} mm",
        file=output,
    )
    print(
        f"  Best-fit plane residual RMS:          {result.plane_residual_rms_mm:.4f} mm",
        file=output,
    )
    print(
        f"  Valid-depth coverage:                 {result.valid_depth_coverage:.1%}",
        file=output,
    )
    if result.known_reference_distance_mm is None:
        print(
            "  Absolute accuracy:                    NOT REPORTED (no known reference distance).",
            file=output,
        )
    else:
        print(
            f"  Known reference distance:             "
            f"{result.known_reference_distance_mm:.4f} mm",
            file=output,
        )
        print(
            f"  Absolute error for this test:          {result.absolute_error_mm:+.4f} mm",
            file=output,
        )


def _print_no_solution(
    *,
    result: OptimizationResult,
    tube: TubeDimensions,
    safety: SafetyAllowances,
    discovery: ProfileDiscovery,
    config: AppConfig,
    mounting: MountingRange,
    output: TextIO,
) -> None:
    print("\nNO VALID D405 SETUP EXISTS FOR THE STATED CONSTRAINTS", file=output)
    print(
        f"  Candidate combinations evaluated: {result.evaluated_count}",
        file=output,
    )
    print(
        f"  Camera-to-rim search: {result.search_minimum_camera_to_rim_mm:.1f} to "
        f"{result.search_maximum_camera_to_rim_mm:.1f} mm",
        file=output,
    )
    unrestricted = optimize_setups(
        tube=tube,
        profiles=discovery.profiles,
        safety=safety,
        configured_minimum_depth_mm=config.filters.min_distance_mm,
        configured_maximum_depth_mm=config.filters.max_distance_mm,
        mounting_range=None,
    )
    has_mechanical_limit = (
        mounting.minimum_camera_to_rim_mm is not None
        or mounting.maximum_camera_to_rim_mm is not None
    )
    if has_mechanical_limit and unrestricted.valid:
        print(
            "  Cause: the mechanical mounting range excludes otherwise valid geometry.",
            file=output,
        )
        nearest = sorted(
            unrestricted.valid_candidates,
            key=lambda candidate: (
                0.0
                if mounting.contains(candidate.camera_to_rim_mm)
                else min(
                    abs(
                        candidate.camera_to_rim_mm
                        - (mounting.minimum_camera_to_rim_mm or candidate.camera_to_rim_mm)
                    ),
                    abs(
                        candidate.camera_to_rim_mm
                        - (mounting.maximum_camera_to_rim_mm or candidate.camera_to_rim_mm)
                    ),
                )
            ),
        )[:3]
        print("  Nearest physical candidates outside that range:", file=output)
        for candidate in nearest:
            print(
                f"    {candidate.camera_to_rim_mm:.1f} mm, {candidate.profile.name}",
                file=output,
            )
        return

    print("  Profile diagnostics:", file=output)
    for profile in discovery.profiles:
        lower = max(
            1.0,
            profile.minimum_usable_depth_mm
            - tube.usable_height_mm
            + tube.maximum_fill_height_mm,
            config.filters.min_distance_mm
            - tube.usable_height_mm
            + tube.maximum_fill_height_mm,
        )
        upper = min(
            profile.maximum_reliable_depth_mm,
            config.filters.max_distance_mm,
        ) - tube.usable_height_mm
        probe_distance = max(1.0, (lower + max(lower, upper)) / 2.0)
        evaluation = evaluate_exact_distance(
            tube,
            (profile,),
            probe_distance,
            safety,
            configured_minimum_depth_mm=config.filters.min_distance_mm,
            configured_maximum_depth_mm=config.filters.max_distance_mm,
        )[0]
        print(
            f"    {profile.name}: depth-compatible camera-to-rim interval "
            f"{lower:.1f} to {upper:.1f} mm; maximum supported diameter near probe "
            f"{evaluation.maximum_supported_tube_diameter_mm:.1f} mm.",
            file=output,
        )
        for reason in evaluation.reasons:
            print(f"      - {reason}", file=output)


def _prompt_float(
    prompt: str,
    *,
    input_fn: InputFunction,
    output: TextIO,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    while True:
        suffix = f" [{default:g}]" if default is not None else ""
        raw = _read_input(f"{prompt}{suffix}: ", input_fn).strip()
        if not raw and default is not None:
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.", file=output)
            continue
        if value != value or value in (float("inf"), float("-inf")):
            print("Please enter a finite number.", file=output)
            continue
        if minimum is not None:
            invalid_min = value < minimum if minimum_inclusive else value <= minimum
            if invalid_min:
                relation = "at least" if minimum_inclusive else "greater than"
                print(f"Value must be {relation} {minimum:g}.", file=output)
                continue
        if maximum is not None and value > maximum:
            print(f"Value must not exceed {maximum:g}.", file=output)
            continue
        return value


def _prompt_optional_float(
    prompt: str,
    *,
    input_fn: InputFunction,
    output: TextIO,
    minimum: float | None = None,
    minimum_inclusive: bool = True,
) -> float | None:
    while True:
        raw = _read_input(f"{prompt} [none]: ", input_fn).strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number or leave the answer blank.", file=output)
            continue
        if value != value or value in (float("inf"), float("-inf")):
            print("Please enter a finite number.", file=output)
            continue
        if minimum is not None:
            invalid_min = value < minimum if minimum_inclusive else value <= minimum
            if invalid_min:
                relation = "at least" if minimum_inclusive else "greater than"
                print(f"Value must be {relation} {minimum:g}.", file=output)
                continue
        return value


def _prompt_yes_no(
    prompt: str,
    *,
    default: bool,
    input_fn: InputFunction,
    output: TextIO,
) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = _read_input(f"{prompt} {suffix} ", input_fn).strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer Y or N.", file=output)


def _prompt_output_mode(
    *,
    input_fn: InputFunction,
    output: TextIO,
) -> OutputMode:
    print("\nChoose output mode:", file=output)
    print("1. ABSTRACT – practical recommendation only", file=output)
    print("2. LONG REVIEW – complete technical evaluation", file=output)
    while True:
        raw = _read_input("Selection [1]: ", input_fn).strip()
        if raw in {"", "1"}:
            return OutputMode.ABSTRACT
        if raw == "2":
            return OutputMode.LONG_REVIEW
        print("Please enter 1 or 2.", file=output)


def _read_input(prompt: str, input_fn: InputFunction) -> str:
    try:
        return input_fn(prompt)
    except EOFError as exc:
        raise UserCancelled("Input ended; configuration was not changed.") from exc


def main() -> int:
    args = parse_args()
    try:
        if args.precision_frames < 10:
            raise SetupValidationError("--precision-frames must be at least 10.")
        if args.known_distance_mm is not None and args.known_distance_mm <= 0.0:
            raise SetupValidationError("--known-distance-mm must be greater than zero.")
        config = load_config(args.config)
        if args.example:
            example_mode = (
                OutputMode.LONG_REVIEW
                if args.output_mode == "long"
                else OutputMode.ABSTRACT
            )
            return run_example(config, output_mode=example_mode)
        return run_interactive(config, args)
    except UserCancelled as exc:
        print(f"Cancelled: {exc}", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        print("\nCancelled; configuration was not changed.", file=sys.stderr)
        return 130
    except (
        MaterialMeasurementError,
        SetupValidationError,
        CameraDiscoveryError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if args.debug:
            raise
        print(
            f"UNEXPECTED ERROR: {exc}. Rerun with --debug for a developer traceback.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
