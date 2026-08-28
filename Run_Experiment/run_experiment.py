"""Path-executable entry point for synchronized acquisition."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
for candidate in (
    REPOSITORY_ROOT,
    PROJECT_ROOT,
    REPOSITORY_ROOT / "3d_camera",
    REPOSITORY_ROOT / "Material_level_using_yolo",
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from run_experiment.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
