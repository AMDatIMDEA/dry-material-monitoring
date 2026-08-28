from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from experiment_records import sha256_file
from material_level_yolo.config import load_config
from material_level_yolo.errors import ConfigurationError
from material_level_yolo.image_io import write_image
from material_level_yolo.roi_config import PixelRoi, freeze_tube_roi


FROZEN_AT = datetime(2026, 8, 17, 10, 0, tzinfo=timezone.utc)


def test_freeze_shared_roi_from_unicode_image(
    write_config,
    base_config_data: dict,
    tmp_path: Path,
) -> None:
    config_path = write_config(base_config_data, folder_name="Étude ROI avec espaces")
    image_path = write_image(
        tmp_path / "Images étalon" / "tube intérieur.png",
        np.zeros((200, 100, 3), dtype=np.uint8),
    )
    source_hash = sha256_file(image_path)

    reference = freeze_tube_roi(
        config_path,
        image_path,
        PixelRoi(left=20, top=10, right=80, bottom=190),
        now_utc=lambda: FROZEN_AT,
    )

    assert sha256_file(image_path) == source_hash
    assert reference == config_path.parent / "calibration" / "tube_roi.json"
    payload = json.loads(reference.read_text(encoding="utf-8"))
    assert payload["pixel_roi"] == {
        "left": 20,
        "top": 10,
        "right": 80,
        "bottom": 190,
    }
    assert payload["normalized_roi"] == {
        "left": 0.2,
        "top": 0.05,
        "right": 0.8,
        "bottom": 0.95,
    }
    project = load_config(config_path)
    for profile in project.profiles.values():
        profile.require_frozen_roi()
        assert profile.tube_roi.frozen
        assert profile.tube_roi.calibration_reference == reference
        assert profile.tube_roi.left == pytest.approx(0.2)
        assert profile.tube_roi.right == pytest.approx(0.8)


def test_whole_image_placeholder_cannot_be_frozen(
    write_config,
    base_config_data: dict,
    tmp_path: Path,
) -> None:
    config_path = write_config(base_config_data)
    original_config = config_path.read_bytes()
    image_path = write_image(
        tmp_path / "tube.png",
        np.zeros((100, 80, 3), dtype=np.uint8),
    )

    with pytest.raises(ConfigurationError, match="unsafe placeholder"):
        freeze_tube_roi(config_path, image_path, PixelRoi(0, 0, 80, 100))

    assert config_path.read_bytes() == original_config
    assert not (config_path.parent / "calibration" / "tube_roi.json").exists()


def test_invalid_or_partial_bounds_do_not_modify_config(
    write_config,
    base_config_data: dict,
    tmp_path: Path,
) -> None:
    config_path = write_config(base_config_data)
    original_config = config_path.read_bytes()
    image_path = write_image(
        tmp_path / "tube.png",
        np.zeros((100, 80, 3), dtype=np.uint8),
    )

    with pytest.raises(ConfigurationError, match="inside"):
        freeze_tube_roi(config_path, image_path, PixelRoi(-1, 0, 60, 90))

    assert config_path.read_bytes() == original_config
