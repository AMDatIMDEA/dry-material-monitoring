from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from material_level_yolo.config import load_config
from material_level_yolo.errors import ConfigurationError
from material_level_yolo.estimation import (
    SemanticObservation,
    aggregate_image_estimates,
    estimate_image_level,
)
from material_level_yolo.roi_policy import (
    apply_roi_mode,
    choose_roi_mode,
    configured_roi_available,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_explicit_whole_image_mode_runs_without_frozen_roi() -> None:
    project = load_config(PROJECT_ROOT / "config.yaml")
    assert not configured_roi_available(project)
    whole = apply_roi_mode(project, "whole_image")
    profile = whole.select_profile("powder")
    profile.require_frozen_roi()
    assert profile.tube_roi.usage_mode == "whole_image"
    assert profile.tube_roi.calibration_reference is None
    assert (
        profile.tube_roi.left,
        profile.tube_roi.top,
        profile.tube_roi.right,
        profile.tube_roi.bottom,
    ) == (0.0, 0.0, 1.0, 1.0)

    material = np.zeros((100, 80), dtype=bool)
    empty = np.zeros_like(material)
    material[50:] = True
    empty[:50] = True
    estimate = estimate_image_level(SemanticObservation(material, empty), profile)
    assert estimate.valid, estimate.rejection_reasons
    assert estimate.roi_mode == "whole_image"
    assert estimate.material_percent == pytest.approx(50.0)


def test_whole_image_mode_ignores_background_and_calibrated_roi_only_gates() -> None:
    project = apply_roi_mode(load_config(PROJECT_ROOT / "config.yaml"), "whole_image")
    profile = project.select_profile("powder")
    profile = replace(
        profile,
        estimation=replace(
            profile.estimation,
            row_occupancy_threshold=0.99,
            minimum_dominant_run_rows=1000,
            minimum_column_classified_fraction=1.0,
        ),
        aggregation=replace(
            profile.aggregation,
            maximum_unclassified_fraction=0.0,
        ),
    )
    material = np.zeros((200, 300), dtype=bool)
    empty = np.zeros_like(material)
    # The valid Powder/Empty tube segmentation occupies only 12% of the image;
    # all remaining pixels are laboratory background.
    empty[40:100, 120:180] = True
    material[100:160, 120:180] = True

    estimate = estimate_image_level(SemanticObservation(material, empty), profile)

    assert estimate.valid, estimate.rejection_reasons
    assert estimate.material_percent == pytest.approx(50.0)
    assert estimate.analysis_bounds_source == "detected_semantic_extent"
    assert estimate.level_calibration_mode == "detected_semantic_extent_uncalibrated"
    assert (estimate.roi.left, estimate.roi.top, estimate.roi.right, estimate.roi.bottom) == (
        120,
        40,
        180,
        160,
    )
    assert estimate.unclassified_fraction == 0.0
    assert "full_empty_vertical_limits_not_calibrated" in estimate.quality_warnings
    assert "insufficient_material_row_support" not in estimate.rejection_reasons
    assert "insufficient_empty_row_support" not in estimate.rejection_reasons


def test_whole_image_single_class_is_not_promoted_to_calibrated_endpoint() -> None:
    profile = apply_roi_mode(
        load_config(PROJECT_ROOT / "config.yaml"), "whole_image"
    ).select_profile("powder")
    material = np.zeros((200, 300), dtype=bool)
    material[80:160, 120:180] = True
    observation = SemanticObservation(
        material,
        np.zeros_like(material),
        empty_confidences=(),
        empty_instance_count=0,
        empty_present=False,
    )

    estimate = estimate_image_level(observation, profile)

    assert not estimate.valid
    assert estimate.material_percent is None
    assert estimate.candidate_material_percent is None
    assert estimate.level_calibration_mode == "unavailable_without_vertical_calibration"
    assert "vertical_level_calibration_unavailable_for_single_class" in estimate.rejection_reasons


def test_whole_image_mode_still_rejects_obviously_tiny_detection() -> None:
    profile = apply_roi_mode(
        load_config(PROJECT_ROOT / "config.yaml"), "whole_image"
    ).select_profile("powder")
    material = np.zeros((200, 300), dtype=bool)
    empty = np.zeros_like(material)
    empty[50, 50] = True
    material[50, 50] = True

    estimate = estimate_image_level(SemanticObservation(material, empty), profile)

    assert not estimate.valid
    assert "tiny_detection_lateral_extent" in estimate.rejection_reasons
    assert "tiny_detection_axial_extent" in estimate.rejection_reasons
    assert "tiny_detection_area_fraction" in estimate.rejection_reasons


def test_whole_image_six_frame_group_averages_background_heavy_masks() -> None:
    profile = apply_roi_mode(
        load_config(PROJECT_ROOT / "config.yaml"), "whole_image"
    ).select_profile("powder")
    estimates = []
    for percent in (47.0, 50.0, 53.0, 47.0, 50.0, 53.0):
        material = np.zeros((200, 300), dtype=bool)
        empty = np.zeros_like(material)
        top, bottom, left, right = 40, 160, 120, 180
        boundary = round(bottom - (bottom - top) * percent / 100.0)
        empty[top:boundary, left:right] = True
        material[boundary:bottom, left:right] = True
        estimates.append(
            estimate_image_level(SemanticObservation(material, empty), profile)
        )

    group = aggregate_image_estimates(estimates, profile)

    assert all(estimate.valid for estimate in estimates)
    assert group.valid, group.rejection_reasons
    assert group.majority_pattern == "material_and_empty"
    assert group.accepted_count == 6
    assert group.material_percent == pytest.approx(50.0)
    assert group.aggregation_method == "arithmetic_mean"


def test_prompt_allows_no_when_configured_roi_is_missing() -> None:
    project = load_config(PROJECT_ROOT / "config.yaml")
    messages: list[str] = []
    mode = choose_roi_mode(
        project,
        None,
        stdin_is_tty=True,
        input_fn=lambda _prompt: "n",
        output_fn=messages.append,
    )
    assert mode == "whole_image"
    assert any("WARNING" in message for message in messages)


def test_prompt_rejects_yes_when_configured_roi_is_missing() -> None:
    project = load_config(PROJECT_ROOT / "config.yaml")
    with pytest.raises(ConfigurationError, match="No frozen tube ROI is available"):
        choose_roi_mode(
            project,
            None,
            stdin_is_tty=True,
            input_fn=lambda _prompt: "y",
            output_fn=lambda _message: None,
        )


def test_noninteractive_mode_requires_explicit_cli_choice() -> None:
    project = load_config(PROJECT_ROOT / "config.yaml")
    with pytest.raises(ConfigurationError, match="--roi-mode is required"):
        choose_roi_mode(project, None, stdin_is_tty=False)
    assert choose_roi_mode(project, "whole_image", stdin_is_tty=False) == "whole_image"


def test_prompt_uses_frozen_configured_roi(config_with_fake_weights) -> None:
    project = load_config(config_with_fake_weights)
    assert configured_roi_available(project)
    mode = choose_roi_mode(
        project,
        None,
        stdin_is_tty=True,
        input_fn=lambda _prompt: "yes",
        output_fn=lambda _message: None,
    )
    configured = apply_roi_mode(project, mode)
    roi = configured.select_profile("powder").require_frozen_roi()
    assert roi.usage_mode == "configured"
    assert roi.calibration_reference is not None
