from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def base_config_data() -> dict:
    return deepcopy(yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8")))


@pytest.fixture
def write_config(tmp_path: Path, base_config_data: dict):
    def write(data: dict | None = None, *, folder_name: str = "configuration") -> Path:
        folder = tmp_path / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "config.yaml"
        path.write_text(
            yaml.safe_dump(base_config_data if data is None else data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    return write


@pytest.fixture
def config_with_fake_weights(write_config, base_config_data: dict):
    base_config_data["model_profiles"]["powder"]["weights"] = (
        "model_weights/powder_empty.pt"
    )
    base_config_data["model_profiles"]["polymer"]["weights"] = (
        "model_weights/polymer_empty.pt"
    )
    base_config_data["defaults"]["tube_roi"].update(
        left=0.1,
        top=0.05,
        right=0.9,
        bottom=0.95,
        frozen=True,
        calibration_reference="calibration/tube_roi.json",
    )
    path = write_config(base_config_data, folder_name="Étude avec espaces")
    weights = path.parent / "model_weights"
    weights.mkdir()
    (weights / "powder_empty.pt").write_bytes(b"mock powder weights\x00")
    (weights / "polymer_empty.pt").write_bytes(b"mock polymer weights\x00")
    calibration = path.parent / "calibration"
    calibration.mkdir()
    (calibration / "tube_roi.json").write_text("{}\n", encoding="utf-8")
    return path
