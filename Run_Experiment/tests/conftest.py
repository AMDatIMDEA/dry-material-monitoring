from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for candidate in (
    REPOSITORY_ROOT,
    PROJECT_ROOT,
    REPOSITORY_ROOT / "3d_camera",
    REPOSITORY_ROOT / "Material_level_using_yolo",
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
