"""Validate geometry and D405 range before connecting the camera."""

from __future__ import annotations

import argparse
from math import radians, tan
from pathlib import Path
import sys

# Running this file by path puts only ``3d_camera`` on sys.path. Add the
# repository root so the shared experiment_records package is importable even
# before an editable install, matching run_measurement.py's legacy behavior.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from material_volume.config import approximate_d405_min_z_mm, load_config
from material_volume.errors import MaterialMeasurementError


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check tube geometry and D405 profile constraints.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        warnings = config.validate()
        tube = config.tube
        camera = config.camera
        min_z = approximate_d405_min_z_mm(camera.depth_width, camera.depth_height)
        rim_view_width = 2.0 * camera.distance_to_rim_mm * tan(radians(87.0 / 2.0))
        print("D405 tube-volume setup")
        print(f"  Inner diameter used:       {tube.inner_diameter_mm:.2f} mm")
        print(f"  Usable internal height:    {tube.usable_height_mm:.2f} mm")
        print(f"  Calculated capacity:       {tube.capacity_ml:.2f} mL")
        print(f"  Camera-to-rim distance:    {camera.distance_to_rim_mm:.2f} mm")
        print(f"  Camera-to-bottom distance: {camera.distance_to_rim_mm + tube.usable_height_mm:.2f} mm")
        print(f"  Depth profile:             {camera.depth_width}x{camera.depth_height}@{camera.fps}")
        print(f"  Approximate profile min-Z: {min_z if min_z is not None else 'unknown'} mm")
        print(f"  Nominal rim view width:    {rim_view_width:.1f} mm")
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  - {warning}")
            return 1
        print("\nConfiguration is internally consistent.")
        return 0
    except (MaterialMeasurementError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
