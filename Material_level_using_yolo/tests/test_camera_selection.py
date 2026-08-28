from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from material_level_yolo.camera import resolve_camera_selection
from material_level_yolo.config import load_config
from material_level_yolo.errors import AcquisitionError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def camera_settings():
    return load_config(PROJECT_ROOT / "config.yaml").select_profile("powder").camera


def test_friendly_name_selects_c920_regardless_of_numeric_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("material_level_yolo.camera.platform.system", lambda: "Windows")
    selection = resolve_camera_selection(
        camera_settings(),
        directshow_names=(
            "LG Camera",
            "Intel(R) RealSense(TM) Depth Camera 405  Depth",
            "HD Pro Webcam C920",
        ),
    )
    assert selection.device_index == 2
    assert selection.backend == "dshow"
    assert selection.friendly_name == "HD Pro Webcam C920"


def test_friendly_name_missing_or_ambiguous_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("material_level_yolo.camera.platform.system", lambda: "Windows")
    with pytest.raises(AcquisitionError, match="Available cameras"):
        resolve_camera_selection(
            camera_settings(),
            directshow_names=("LG Camera",),
        )
    with pytest.raises(AcquisitionError, match="ambiguous"):
        resolve_camera_selection(
            camera_settings(),
            directshow_names=("C920 left", "C920 right"),
        )


def test_null_friendly_name_preserves_legacy_numeric_selection() -> None:
    settings = replace(
        camera_settings(),
        device_name_contains=None,
        device_index=4,
        backend="msmf",
    )
    selection = resolve_camera_selection(settings)
    assert selection.device_index == 4
    assert selection.backend == "msmf"
    assert selection.friendly_name is None
