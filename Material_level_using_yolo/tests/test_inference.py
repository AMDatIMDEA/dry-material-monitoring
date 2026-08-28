from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiment_records import sha256_file
from material_level_yolo.config import load_config
from material_level_yolo.errors import ModelCompatibilityError, WeightsError
from material_level_yolo.inference import (
    UltralyticsInferenceAdapter,
    resolve_inference_device,
)


class FakeModel:
    def __init__(self, *, names=None, task: str = "segment") -> None:
        self.names = {0: "Empty", 1: "Powder"} if names is None else names
        self.task = task
        self.predict_calls: list[dict] = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return ["opaque-result"]


@pytest.mark.parametrize(
    ("profile_name", "material_name"),
    [("powder", "Powder"), ("polymer", "polymer")],
)
def test_each_configured_model_profile_resolves_its_own_semantics(
    config_with_fake_weights: Path,
    profile_name: str,
    material_name: str,
) -> None:
    profile = load_config(config_with_fake_weights).select_profile(profile_name)
    fake = FakeModel(names={4: "Empty", 8: material_name})
    mapping = UltralyticsInferenceAdapter(
        profile, model_factory=lambda weights, task: fake
    ).load()
    assert mapping.material_id == 8
    assert mapping.empty_id == 4
    assert fake.predict_calls == []


def test_completion_gate_valid_mocked_profile_loads_before_inference(
    config_with_fake_weights: Path,
) -> None:
    profile = load_config(config_with_fake_weights).select_profile("powder")
    fake = FakeModel(names={7: "Powder", 3: "Empty"})
    factory_calls = []

    def factory(weights: Path, task: str):
        factory_calls.append((weights, task))
        return fake

    adapter = UltralyticsInferenceAdapter(profile, model_factory=factory)
    mapping = adapter.load()
    assert mapping.material_id == 7
    assert mapping.empty_id == 3
    assert fake.predict_calls == []
    assert factory_calls == [(profile.weights_path, "segment")]
    assert adapter.weights_sha256 == sha256_file(profile.weights_path)


@pytest.mark.parametrize(
    "names",
    [
        {0: "polymer", 1: "Empty"},
        {0: "Powder", 1: "powder", 2: "Empty"},
    ],
)
def test_completion_gate_wrong_or_ambiguous_mapping_never_reaches_predict(
    config_with_fake_weights: Path,
    names,
) -> None:
    profile = load_config(config_with_fake_weights).select_profile("powder")
    fake = FakeModel(names=names)
    adapter = UltralyticsInferenceAdapter(profile, model_factory=lambda weights, task: fake)
    with pytest.raises(ModelCompatibilityError):
        adapter.predict(np.zeros((12, 12, 3), dtype=np.uint8))
    assert fake.predict_calls == []
    assert adapter.loaded is False


def test_task_mismatch_never_reaches_predict(config_with_fake_weights: Path) -> None:
    profile = load_config(config_with_fake_weights).select_profile("powder")
    fake = FakeModel(task="detect")
    adapter = UltralyticsInferenceAdapter(profile, model_factory=lambda weights, task: fake)
    with pytest.raises(ModelCompatibilityError, match="expects Ultralytics task"):
        adapter.predict(np.zeros((12, 12, 3), dtype=np.uint8))
    assert fake.predict_calls == []


def test_valid_prediction_auto_falls_back_to_cpu_and_uses_configured_options(
    config_with_fake_weights: Path,
) -> None:
    profile = load_config(config_with_fake_weights).select_profile("powder")
    fake = FakeModel()
    adapter = UltralyticsInferenceAdapter(
        profile,
        model_factory=lambda weights, task: fake,
        cuda_available=lambda: False,
    )
    output = adapter.predict(np.zeros((12, 12, 3), dtype=np.uint8))
    assert output.raw_results == ("opaque-result",)
    assert output.model_task == "segment"
    assert output.inference_device == "cpu"
    assert len(fake.predict_calls) == 1
    call = fake.predict_calls[0]
    assert call["device"] == "cpu"
    assert call["classes"] == [1, 0]
    assert call["conf"] == profile.inference.confidence
    assert call["iou"] == profile.inference.iou
    assert call["max_det"] == profile.inference.max_detections


def test_auto_device_prefers_first_cuda_gpu() -> None:
    assert resolve_inference_device("auto", cuda_available=lambda: True) == "cuda:0"


def test_auto_device_survives_cuda_discovery_failure() -> None:
    def broken_discovery() -> bool:
        raise RuntimeError("driver mismatch")

    assert resolve_inference_device("auto", cuda_available=broken_discovery) == "cpu"


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("cpu", "cpu"), ("cuda", "cuda:0"), ("cuda:2", "cuda:2")],
)
def test_explicit_device_is_not_silently_changed(requested: str, expected: str) -> None:
    assert resolve_inference_device(requested, cuda_available=lambda: False) == expected


def test_missing_weights_prevent_model_factory_call(
    write_config,
    base_config_data: dict,
) -> None:
    base_config_data["model_profiles"]["powder"]["weights"] = (
        "model_weights/missing_powder_empty.pt"
    )
    profile = load_config(write_config(base_config_data)).select_profile("powder")
    calls = []
    adapter = UltralyticsInferenceAdapter(
        profile,
        model_factory=lambda weights, task: calls.append((weights, task)),
    )
    with pytest.raises(WeightsError, match="were not found"):
        adapter.load()
    assert calls == []


def test_corrupt_or_incorrect_weights_get_clear_load_error(config_with_fake_weights: Path) -> None:
    profile = load_config(config_with_fake_weights).select_profile("powder")

    def bad_factory(weights: Path, task: str):
        raise RuntimeError("invalid checkpoint header")

    with pytest.raises(WeightsError, match="invalid checkpoint header"):
        UltralyticsInferenceAdapter(profile, model_factory=bad_factory).load()
