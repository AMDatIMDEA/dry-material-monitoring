"""Path-executable tube ROI configuration command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
for candidate in (REPOSITORY_ROOT, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from material_level_yolo.roi_config import PixelRoi, freeze_tube_roi, select_pixel_roi
from material_level_yolo.errors import YoloFoundationError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and freeze the useful internal tube rectangle shared by all YOLO profiles."
        )
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--axis", choices=("top_to_bottom", "bottom_to_top"))
    parser.add_argument("--left-px", type=int)
    parser.add_argument("--top-px", type=int)
    parser.add_argument("--right-px", type=int)
    parser.add_argument("--bottom-px", type=int)
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        supplied = (args.left_px, args.top_px, args.right_px, args.bottom_px)
        if any(value is not None for value in supplied):
            if not all(value is not None for value in supplied):
                raise ValueError("Supply all four pixel bounds or none of them.")
            bounds = PixelRoi(*supplied)
        else:
            bounds = select_pixel_roi(args.image)
        summary = {
            "image": str(args.image.expanduser().resolve()),
            "config": str(args.config.expanduser().resolve()),
            "pixel_roi": {
                "left": bounds.left,
                "top": bounds.top,
                "right": bounds.right,
                "bottom": bounds.bottom,
            },
            "axis": args.axis,
            "applies_to": "all model profiles",
        }
        print("Tube ROI confirmation:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if not args.yes:
            if input("Freeze this ROI? [y/N]: ").strip().casefold() not in {"y", "yes"}:
                print("Cancelled; config was not changed.")
                return 0
        reference = freeze_tube_roi(
            args.config,
            args.image,
            bounds,
            axis=args.axis,
        )
        print(json.dumps({"frozen": True, "reference": str(reference)}, indent=2))
        return 0
    except (YoloFoundationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
