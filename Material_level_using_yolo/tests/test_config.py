from __future__ import annotations

from pathlib import Path

import pytest

from material_level_yolo.config import load_config
from material_level_yolo.errors import ConfigurationError, ProfileNotFoundError, WeightsError


def test_supplied_config_has_two_named_auto_device_profiles() -> None:
    project = Path(__file__).resolve().parents[1]
    config = load_config(project / "config.yaml")
    assert config.default_profile == "powder"
    assert set(config.profiles) == {"powder", "polymer"}
    assert config.select_profile().material_role.class_name == "Powder"
    assert config.select_profile("polymer").material_role.class_name == "polymer"
    assert config.select_profile_for_material("POWDER").name == "powder"
    assert config.select_profile_for_material("white powder").name == "powder"
    assert config.select_profile_for_material(" polymer ").name == "polymer"
    assert config.select_profile_for_material("black polymer").name == "polymer"
    assert config.select_profile_for_material(None, "powder").name == "powder"
    assert all(profile.inference.device == "auto" for profile in config.profiles.values())
    assert config.select_profile("powder").aggregation.manual_minimum_valid_images == 1
    assert config.select_profile("powder").camera.capture_duration_seconds is None
    assert config.select_profile("powder").camera.preview_wait_ms == 10
    assert config.select_profile("powder").camera.device_name_contains == "C920"
    assert config.select_profile("powder").weights_path.is_absolute()


def test_missing_recorded_material_requires_explicit_profile() -> None:
    project = Path(__file__).resolve().parents[1]
    config = load_config(project / "config.yaml")
    with pytest.raises(ConfigurationError, match="recorded material_name or an explicit"):
        config.select_profile_for_material(None)


def test_unknown_profile_fails_with_available_names(write_config) -> None:
    config = load_config(write_config())
    with pytest.raises(ProfileNotFoundError, match="polymer, powder"):
        config.select_profile("pellets")


def test_recorded_material_aliases_are_unique_across_profiles(
    write_config, base_config_data: dict
) -> None:
    base_config_data["model_profiles"]["polymer"]["recorded_material_names"] = ["Powder"]
    with pytest.raises(ConfigurationError, match="shared by profiles"):
        load_config(write_config(base_config_data))


def test_relative_paths_resolve_against_unicode_yaml_not_cwd(
    write_config,
    base_config_data: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_config_data["model_profiles"]["powder"]["weights"] = (
        "model_weights/powder_empty.pt"
    )
    path = write_config(base_config_data, folder_name="Étude matériau avec espaces")
    monkeypatch.chdir(tmp_path)
    profile = load_config(path).select_profile("powder")
    assert profile.weights_path == (path.parent / "model_weights" / "powder_empty.pt").resolve()
    assert profile.output.root == (path.parent / "research_records").resolve()


def test_absolute_weights_calibration_and_output_paths_are_preserved(
    write_config,
    base_config_data: dict,
    tmp_path: Path,
) -> None:
    absolute_root = (tmp_path / "Résultats absolus").resolve()
    absolute_weights = (tmp_path / "Modèles" / "powder.pt").resolve()
    absolute_calibration = (tmp_path / "Étalonnage" / "tube.yaml").resolve()
    powder = base_config_data["model_profiles"]["powder"]
    powder["weights"] = str(absolute_weights)
    powder["overrides"] = {
        "tube_roi": {"calibration_reference": str(absolute_calibration)},
        "output": {"root": str(absolute_root)},
    }
    profile = load_config(write_config(base_config_data)).select_profile("powder")
    assert profile.weights_path == absolute_weights
    assert profile.tube_roi.calibration_reference == absolute_calibration
    assert profile.output.root == absolute_root


def test_missing_weights_fail_only_when_model_use_is_requested(
    write_config,
    base_config_data: dict,
) -> None:
    base_config_data["model_profiles"]["powder"]["weights"] = (
        "model_weights/powder_empty.pt"
    )
    profile = load_config(write_config(base_config_data)).select_profile("powder")
    assert not profile.weights_path.exists()
    with pytest.raises(WeightsError, match="were not found"):
        profile.require_weights()


def test_weights_directory_is_rejected(write_config, base_config_data: dict) -> None:
    base_config_data["model_profiles"]["powder"]["weights"] = (
        "model_weights/powder_empty.pt"
    )
    path = write_config(base_config_data)
    target = path.parent / "model_weights" / "powder_empty.pt"
    target.mkdir(parents=True)
    profile = load_config(path).select_profile("powder")
    with pytest.raises(WeightsError, match="are not a file"):
        profile.require_weights()


def test_new_operator_camera_settings_have_backward_compatible_defaults(
    write_config, base_config_data: dict
) -> None:
    defaults = base_config_data["defaults"]
    defaults["aggregation"].pop("manual_minimum_valid_images")
    for field in (
        "capture_duration_seconds",
        "preview_wait_ms",
        "mirror_preview",
        "close_on_capture",
        "freeze_after_capture_ms",
        "device_name_contains",
    ):
        defaults["camera"].pop(field)
    profile = load_config(write_config(base_config_data)).select_profile("powder")
    assert profile.aggregation.manual_minimum_valid_images == 1
    assert profile.camera.capture_duration_seconds is None
    assert profile.camera.preview_wait_ms == 10
    assert profile.camera.close_on_capture
    assert profile.camera.device_name_contains is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["defaults"]["inference"].update(device="gpu"), "must be 'auto'"),
        (lambda data: data["defaults"].update(estimation_mode="bbox_vertical_extent"), "requires"),
        (lambda data: data["defaults"]["tube_roi"].update(left=0.8, right=0.2), "left < right"),
        (lambda data: data["defaults"]["estimation"].update(row_smoothing_window=4), "must be odd"),
        (lambda data: data["defaults"]["estimation"].update(removed_interface_spread_threshold=0.2), "unknown"),
        (lambda data: data["model_profiles"]["powder"].update(typo=True), "unknown: typo"),
    ],
)
def test_invalid_configuration_fails_closed(
    write_config,
    base_config_data: dict,
    mutation,
    message: str,
) -> None:
    mutation(base_config_data)
    with pytest.raises(ConfigurationError, match=message):
        load_config(write_config(base_config_data))


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "cuda:0", "cuda:12"])
def test_supported_inference_devices_are_accepted(
    write_config, base_config_data: dict, device: str
) -> None:
    base_config_data["defaults"]["inference"]["device"] = device
    profile = load_config(write_config(base_config_data)).select_profile("powder")
    assert profile.inference.device == device
