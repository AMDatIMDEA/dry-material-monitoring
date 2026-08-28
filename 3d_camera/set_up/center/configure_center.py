"""Interactively configure the D405 measurement-circle centre."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys
from typing import TextIO

# Direct execution from ``3d_camera/set_up/center`` must see sibling packages.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
for import_root in (REPOSITORY_ROOT, PACKAGE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from material_volume.config import AppConfig, load_config
from material_volume.errors import MaterialMeasurementError
from set_up.center.center_config import CenterSelection, save_center_config
from set_up.center.preview import PreviewCancelled, select_center


DEFAULT_CONFIG = PACKAGE_ROOT / "config.yaml"
InputFunction = Callable[[str], str]


class UserCancelled(RuntimeError):
    """Terminal input ended or the user explicitly cancelled."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the physical measurement-circle centre in the native D405 depth stream."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show a developer traceback for unexpected errors.",
    )
    return parser.parse_args()


def run_interactive(
    config: AppConfig,
    *,
    input_fn: InputFunction = input,
    output: TextIO = sys.stdout,
) -> CenterSelection | None:
    """Run all prompts and return the final unsaved selection."""
    previous_diameter = (
        config.detection_roi.diameter_mm
        if config.detection_roi.diameter_mm is not None
        else config.tube.inner_diameter_mm
    )
    previous_margin = config.detection_roi.free_image_margin_percent

    print(
        "The detection circle is the physical inner measurement ROI; enter its diameter in mm.",
        file=output,
    )
    print(
        "The preview uses the native processed depth frame used by calibration and measurement.",
        file=output,
    )
    print(
        "Centre profile: 1280x720@30. Note: the D405 approximate minimum-Z at this "
        "resolution is 100 mm, so nearer surfaces may not return valid depth.",
        file=output,
    )

    while True:
        diameter = _prompt_float(
            "Detection-circle diameter in mm",
            default=previous_diameter,
            minimum=0.0,
            maximum=None,
            minimum_inclusive=False,
            input_fn=input_fn,
            output=output,
        )
        margin = _prompt_float(
            "Free image margin on each side (% of circle diameter)",
            default=previous_margin,
            minimum=0.0,
            maximum=100.0,
            input_fn=input_fn,
            output=output,
        )
        print(
            "Opening the D405 preview. Left-click the desired centre, press S to switch "
            "between depth, infrared, and normal-camera views, then press Enter to accept.",
            file=output,
        )
        preview = select_center(config, diameter, margin)
        selection = CenterSelection(
            center_x_px=preview.center_x_px,
            center_y_px=preview.center_y_px,
            diameter_mm=diameter,
            free_image_margin_percent=margin,
            frame_width_px=preview.frame_width_px,
            frame_height_px=preview.frame_height_px,
            fps=preview.fps,
            depth_width=preview.depth_width,
            depth_height=preview.depth_height,
            camera_serial_number=preview.camera_serial_number,
        )
        print(
            f"Selected depth-frame centre ({selection.center_x_px:g}, "
            f"{selection.center_y_px:g}) at {selection.frame_width_px}x"
            f"{selection.frame_height_px}.",
            file=output,
        )
        if not _prompt_yes_no(
            "Do you want to repeat the process?",
            default=False,
            input_fn=input_fn,
            output=output,
        ):
            return selection
        previous_diameter = diameter
        previous_margin = margin
        print("Repeating without saving the previous selection.\n", file=output)


def confirm_and_save(
    config: AppConfig,
    selection: CenterSelection,
    *,
    input_fn: InputFunction = input,
    output: TextIO = sys.stdout,
) -> bool:
    print("\nConfiguration to save:", file=output)
    print(
        f"  Centre: ({selection.center_x_px:g}, {selection.center_y_px:g}) px "
        f"in processed depth {selection.frame_width_px}x{selection.frame_height_px}",
        file=output,
    )
    print(f"  Detection diameter: {selection.diameter_mm:g} mm", file=output)
    print(f"  Free image margin: {selection.free_image_margin_percent:g}%", file=output)
    print(
        f"  Camera depth stream: {selection.depth_width or selection.frame_width_px}x"
        f"{selection.depth_height or selection.frame_height_px}@{selection.fps} z16",
        file=output,
    )
    if not _prompt_yes_no(
        "Save these values to config.yaml?",
        default=False,
        input_fn=input_fn,
        output=output,
    ):
        print("Cancelled; configuration was not changed.", file=output)
        return False

    result = save_center_config(config.config_path, selection)
    print(f"Configuration saved: {result.config_path}", file=output)
    print(f"Timestamped backup: {result.backup_path}", file=output)
    print(
        "Run the empty-tube calibration again before measuring; the ROI geometry has changed.",
        file=output,
    )
    return True


def _prompt_float(
    prompt: str,
    *,
    default: float,
    minimum: float | None,
    maximum: float | None,
    input_fn: InputFunction,
    output: TextIO,
    minimum_inclusive: bool = True,
) -> float:
    while True:
        raw = _read_input(f"{prompt} [{default:g}]: ", input_fn).strip()
        if raw.lower() in {"q", "quit", "c", "cancel"}:
            raise UserCancelled("Configuration was not changed.")
        if not raw:
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number, or type 'cancel'.", file=output)
            continue
        if value != value or value in (float("inf"), float("-inf")):
            print("Please enter a finite number.", file=output)
            continue
        if minimum is not None:
            invalid = value < minimum if minimum_inclusive else value <= minimum
            if invalid:
                relation = "at least" if minimum_inclusive else "greater than"
                print(f"Value must be {relation} {minimum:g}.", file=output)
                continue
        if maximum is not None and value > maximum:
            print(f"Value must not exceed {maximum:g}.", file=output)
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
        if raw in {"q", "quit", "c", "cancel"}:
            raise UserCancelled("Configuration was not changed.")
        print("Please answer Y or N, or type 'cancel'.", file=output)


def _read_input(prompt: str, input_fn: InputFunction) -> str:
    try:
        return input_fn(prompt)
    except EOFError as exc:
        raise UserCancelled("Input ended; configuration was not changed.") from exc


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        selection = run_interactive(config)
        if selection is None:
            return 0
        return 0 if confirm_and_save(config, selection) else 0
    except (UserCancelled, PreviewCancelled) as exc:
        print(f"Cancelled: {exc}", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        print("\nCancelled; configuration was not changed.", file=sys.stderr)
        return 130
    except (MaterialMeasurementError, OSError, ValueError) as exc:
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
