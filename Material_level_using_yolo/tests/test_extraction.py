from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from material_level_yolo.config import load_config
from material_level_yolo.domain import InferenceOutput, SemanticClassMap
from material_level_yolo.estimation import estimate_image_level
from material_level_yolo.extraction import observation_from_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_MAP = SemanticClassMap(
    material_id=7,
    material_name="Powder",
    empty_id=3,
    empty_name="Empty",
    model_names={3: "Empty", 7: "Powder"},
)


class Values:
    def __init__(self, value) -> None:
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Boxes:
    def __init__(self, classes, confidences, xyxy=()) -> None:
        self.cls = Values(classes)
        self.conf = Values(confidences)
        self.xyxy = Values(xyxy)


class Masks:
    def __init__(self, masks) -> None:
        self.data = Values(masks)


class Result:
    def __init__(self, classes, confidences, *, masks=None, xyxy=()) -> None:
        self.boxes = Boxes(classes, confidences, xyxy)
        self.masks = None if masks is None else Masks(masks)


def output(result, *, task="segment") -> InferenceOutput:
    return InferenceOutput(
        raw_results=(result,),
        class_map=CLASS_MAP,
        model_task=task,
        weights_sha256="a" * 64,
    )


def configured_profile():
    profile = load_config(PROJECT_ROOT / "config.yaml").select_profile("powder")
    return replace(profile, tube_roi=replace(profile.tube_roi, frozen=True))


def test_segmentation_extraction_resizes_masks_and_estimates_interface() -> None:
    profile = configured_profile()
    material = np.zeros((50, 40), dtype=np.float32)
    empty = np.zeros_like(material)
    material[25:] = 1.0
    empty[:25] = 1.0
    observation = observation_from_inference(
        output(Result([7, 3], [0.92, 0.93], masks=[material, empty])),
        profile,
        (100, 80, 3),
    )
    estimate = estimate_image_level(observation, profile)
    assert observation.material_mask.shape == (100, 80)
    assert observation.material_instance_count == 1
    assert estimate.valid
    assert estimate.material_percent == pytest.approx(50.0)


def test_missing_segmentation_masks_are_explicitly_invalid() -> None:
    profile = configured_profile()
    observation = observation_from_inference(
        output(Result([7, 3], [0.92, 0.93], masks=None)),
        profile,
        (100, 80, 3),
    )
    estimate = estimate_image_level(observation, profile)
    assert not estimate.valid
    assert "missing_segmentation_masks" in estimate.rejection_reasons
    assert "missing_material_mask" in estimate.rejection_reasons
    assert "missing_empty_mask" in estimate.rejection_reasons


def test_explicit_bbox_vertical_extent_mode_uses_axis_boundary() -> None:
    segment_profile = configured_profile()
    profile = replace(
        segment_profile,
        expected_task="detect",
        estimation_mode="bbox_vertical_extent",
    )
    result = Result(
        [7, 3],
        [0.92, 0.93],
        xyxy=[[0, 50, 80, 100], [0, 0, 80, 50]],
    )
    observation = observation_from_inference(output(result, task="detect"), profile, (100, 80, 3))
    estimate = estimate_image_level(observation, profile)
    assert observation.estimation_mode == "bbox_vertical_extent"
    assert estimate.valid
    assert estimate.material_percent == pytest.approx(50.0)


def test_multiple_raw_results_are_ambiguous() -> None:
    profile = configured_profile()
    result = Result([], [], masks=[])
    inference = InferenceOutput(
        raw_results=(result, result),
        class_map=CLASS_MAP,
        model_task="segment",
        weights_sha256="a" * 64,
    )
    observation = observation_from_inference(inference, profile, (100, 80, 3))
    estimate = estimate_image_level(observation, profile)
    assert not estimate.valid
    assert "ambiguous_multiple_inference_results" in estimate.rejection_reasons


@pytest.mark.parametrize(
    ("box", "reason"),
    [
        ([0, 50, 0, 100], "invalid_material_bbox"),
        ([100, 50, 120, 100], "material_bbox_outside_image"),
    ],
)
def test_invalid_semantic_bbox_is_rejected_without_inventing_support(box, reason) -> None:
    segment_profile = configured_profile()
    profile = replace(
        segment_profile,
        expected_task="detect",
        estimation_mode="bbox_vertical_extent",
    )
    result = Result(
        [7, 3],
        [0.92, 0.93],
        xyxy=[box, [0, 0, 80, 50]],
    )
    observation = observation_from_inference(output(result, task="detect"), profile, (100, 80, 3))
    estimate = estimate_image_level(observation, profile)
    assert not estimate.valid
    assert reason in estimate.rejection_reasons
    assert not observation.material_mask.any()


def test_highest_confidence_instance_is_selected_and_duplicates_are_recorded() -> None:
    profile = configured_profile()
    wrong_material = np.zeros((100, 80), dtype=np.float32)
    wrong_material[80:] = 1.0
    chosen_material = np.zeros_like(wrong_material)
    chosen_material[50:] = 1.0
    empty = np.zeros_like(wrong_material)
    empty[:50] = 1.0
    observation = observation_from_inference(
        output(
            Result(
                [7, 7, 3],
                [0.60, 0.95, 0.90],
                masks=[wrong_material, chosen_material, empty],
            )
        ),
        profile,
        (100, 80, 3),
    )
    estimate = estimate_image_level(observation, profile)
    assert observation.material_instance_count == 2
    assert observation.material_confidences == pytest.approx((0.60, 0.95))
    assert estimate.valid, estimate.rejection_reasons
    assert estimate.material_percent == pytest.approx(50.0)
