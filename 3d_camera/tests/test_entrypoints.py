from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "relative_script, expected_help",
    [
        (Path("3d_camera/validate_setup.py"), "D405 profile constraints"),
        (Path("3d_camera/calibrate_empty.py"), "--output OUTPUT"),
        (
            Path("3d_camera/set_up/center/configure_center.py"),
            "measurement-circle centre",
        ),
        (Path("3d_camera/set_up/configure_height.py"), "--output-mode"),
    ],
)
def test_path_entrypoint_finds_shared_package(
    tmp_path: Path,
    relative_script: Path,
    expected_help: str,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(repository / relative_script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert expected_help in completed.stdout
