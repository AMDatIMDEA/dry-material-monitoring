"""Repository entry point for offline YOLO validation and operator workflows."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
for candidate in (REPOSITORY_ROOT, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from material_level_yolo.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
