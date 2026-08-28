"""Create a virtual reference or capture an automatic empty-tube calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

# Direct execution puts only ``3d_camera`` on sys.path. Add the repository root
# so material_volume can import the shared experiment_records package without
# requiring an editable install first.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from material_volume.capture import RealSenseDepthSource
from material_volume.config import AppConfig, load_config
from material_volume.errors import MaterialMeasurementError
from material_volume.calibration import create_virtual_bottom_calibration
from material_volume.pipeline import MaterialVolumePipeline
from material_volume.synthetic import SyntheticDepthSource


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a configured virtual bottom or calibrate against an empty-tube reference."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", choices=("camera", "bag", "synthetic"), default="camera")
    parser.add_argument("--bag", type=Path, help="RealSense .bag file when --source bag is used.")
    parser.add_argument("--output", type=Path, help="Override calibration output path.")
    return parser.parse_args()


def choose_calibration_mode(input_fn=input) -> int:
    print("Choose calibration mode:")
    print("1 - Create a virtual surface using the current configuration")
    print("2 - Run the automatic calibration")
    while True:
        selection = input_fn("Selection [2]: ").strip()
        if selection in ("", "2"):
            return 2
        if selection == "1":
            return 1
        print("Please enter 1 or 2.")


def build_source(config: AppConfig, source: str, bag: Path | None):
    if source == "synthetic":
        return SyntheticDepthSource(config, fill_percent=0.0)
    if source == "bag":
        if bag is None:
            raise ValueError("--bag is required when --source bag is selected.")
        return RealSenseDepthSource(config, bag_path=bag)
    return RealSenseDepthSource(config)


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        for warning in config.validate():
            print(f"SETUP WARNING: {warning}")
        output = args.output.expanduser().resolve() if args.output else config.calibration_path
        mode = choose_calibration_mode()
        if mode == 2 and args.source != "synthetic":
            print(
                "Calibration requirement: empty the tube and place a flat, matte reference "
                "at the true inner bottom. Do not move the camera or tube after calibration."
            )
        with build_source(config, args.source, args.bag) as source:
            if mode == 1:
                print(
                    "Creating a camera-aligned virtual bottom from config.yaml. "
                    "One frame is captured only to read the active camera intrinsics."
                )
                burst = source.capture_burst(1)
                calibration = create_virtual_bottom_calibration(burst.intrinsics, config)
            else:
                burst = source.capture_burst(config.fusion.burst_frames)
                calibration = MaterialVolumePipeline.calibrate(burst, config)

        if output.exists():
            backup = output.with_name(f"{output.stem}.previous{output.suffix}")
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output, backup)
            print(f"Previous calibration backed up to: {backup}")
        calibration.save(output)
        print(f"Calibration saved: {output}")
        print(f"Calibration method: {calibration.calibration_method}")
        if calibration.calibration_method == "automatic_plane_fit":
            print(f"Bottom-plane RMS: {calibration.plane_rms_mm:.3f} mm")
            print(f"Bottom-reference coverage: {calibration.plane_coverage:.1%}")
        else:
            print("Bottom-plane RMS: not applicable (configured virtual surface)")
            print("Bottom-reference coverage: not applicable (no surface fit)")
        print(f"Calculated camera-to-rim distance: {calibration.measured_rim_distance_mm:.2f} mm")
        return 0
    except (MaterialMeasurementError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Calibration cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
