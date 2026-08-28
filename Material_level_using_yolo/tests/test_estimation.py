from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiment_records import derive_estimate
from material_level_yolo.config import load_config
from material_level_yolo.estimation import (
    SemanticObservation,
    aggregate_image_estimates,
    estimate_image_level,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def profile():
    loaded = load_config(PROJECT_ROOT / "config.yaml").select_profile("powder")
    return replace(loaded, tube_roi=replace(loaded.tube_roi, frozen=True))


def masks_for_percent(
    percent: float,
    *,
    height: int = 100,
    width: int = 80,
    slope_rows: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    base = float(round(height * (1.0 - percent / 100.0)))
    x = np.arange(width, dtype=np.float64)
    boundaries = base + slope_rows * (x - (width - 1) / 2.0) / max(width - 1, 1)
    y = np.arange(height, dtype=np.float64)[:, None]
    material = y >= boundaries[None, :]
    empty = ~material
    return material, empty


def observation_for_percent(percent: float, **kwargs) -> SemanticObservation:
    material, empty = masks_for_percent(percent, **kwargs)
    if percent == 0.0:
        return SemanticObservation(
            material,
            empty,
            material_confidences=(),
            material_instance_count=0,
            material_present=False,
        )
    if percent == 100.0:
        return SemanticObservation(
            material,
            empty,
            empty_confidences=(),
            empty_instance_count=0,
            empty_present=False,
        )
    return SemanticObservation(material, empty)


@pytest.mark.parametrize("percent", [0.0, 5.0, 50.0, 85.0, 100.0])
def test_known_flat_interfaces_produce_predictable_percentages(profile, percent: float) -> None:
    result = estimate_image_level(observation_for_percent(percent), profile)
    assert result.valid, result.rejection_reasons
    assert result.material_percent == pytest.approx(percent, abs=1e-12)
    assert result.material_coverage_fraction == pytest.approx(percent / 100.0)
    assert result.empty_coverage_fraction == pytest.approx(1.0 - percent / 100.0)
    assert result.overlap_fraction == 0.0
    assert result.unclassified_fraction == 0.0


def test_mask_area_is_diagnostic_while_sloped_interface_sets_level(profile) -> None:
    material, empty = masks_for_percent(50.0, slope_rows=10.0)
    result = estimate_image_level(SemanticObservation(material, empty), profile)
    assert result.valid, result.rejection_reasons
    assert result.material_percent == pytest.approx(50.0, abs=1.0)
    assert 0.05 < result.interface_spread_fraction < 0.12


def test_irregular_interface_spread_is_diagnostic_not_rejection(profile) -> None:
    material, empty = masks_for_percent(50.0, slope_rows=40.0)
    result = estimate_image_level(SemanticObservation(material, empty), profile)
    assert result.valid, result.rejection_reasons
    assert result.material_percent == pytest.approx(50.0, abs=1.0)
    assert result.interface_spread_fraction is not None
    assert result.interface_spread_fraction > 0.15
    assert "excessive_interface_spread" not in result.rejection_reasons


def test_sparse_mask_noise_does_not_move_robust_interface_materially(profile) -> None:
    material, empty = masks_for_percent(50.0)
    rng = np.random.default_rng(42)
    noisy = rng.random(material.shape) < 0.01
    material = material.copy()
    empty = empty.copy()
    material[noisy] = ~material[noisy]
    empty[noisy] = ~empty[noisy]
    result = estimate_image_level(SemanticObservation(material, empty), profile)
    assert result.valid, result.rejection_reasons
    assert result.material_percent == pytest.approx(50.0, abs=1.0)


def test_bottom_to_top_orientation_is_explicit(profile) -> None:
    material, empty = masks_for_percent(25.0)
    flipped_profile = replace(profile, tube_roi=replace(profile.tube_roi, axis="bottom_to_top"))
    result = estimate_image_level(
        SemanticObservation(material[::-1], empty[::-1]),
        flipped_profile,
    )
    assert result.valid, result.rejection_reasons
    assert result.material_percent == pytest.approx(25.0)


def test_roi_clipping_ignores_masks_outside_calibrated_tube(profile) -> None:
    height, width = 120, 100
    roi_profile = replace(
        profile,
        tube_roi=replace(profile.tube_roi, left=0.2, top=0.1, right=0.8, bottom=0.9),
    )
    material = np.ones((height, width), dtype=bool)
    empty = np.zeros((height, width), dtype=bool)
    top, bottom, left, right = 12, 108, 20, 80
    boundary = top + (bottom - top) // 2
    empty[top:boundary, left:right] = True
    material[top:boundary, left:right] = False
    result = estimate_image_level(SemanticObservation(material, empty), roi_profile)
    assert result.valid, result.rejection_reasons
    assert result.material_percent == pytest.approx(50.0)
    assert (result.roi.left, result.roi.top, result.roi.right, result.roi.bottom) == (
        left,
        top,
        right,
        bottom,
    )


def test_gaps_and_overlaps_remain_diagnostics_without_sole_rejection(profile) -> None:
    material, empty = masks_for_percent(50.0)
    gap_material, gap_empty = material.copy(), empty.copy()
    gap_material[35:65] = False
    gap_empty[35:65] = False
    gap = estimate_image_level(SemanticObservation(gap_material, gap_empty), profile)
    assert gap.valid, gap.rejection_reasons
    assert gap.unclassified_fraction == pytest.approx(0.30)
    assert "excessive_unclassified_fraction" not in gap.rejection_reasons

    overlap_material, overlap_empty = material.copy(), empty.copy()
    overlap_material[40:60] = True
    overlap_empty[40:60] = True
    overlap = estimate_image_level(
        SemanticObservation(overlap_material, overlap_empty), profile
    )
    assert overlap.valid, overlap.rejection_reasons
    assert overlap.overlap_fraction == pytest.approx(0.20)
    assert "excessive_material_empty_overlap" not in overlap.rejection_reasons


@pytest.mark.parametrize(
    ("observation_update", "reason"),
    [
        ({"material_confidences": (0.1,)}, "low_or_missing_material_confidence"),
        ({"empty_confidences": ()}, "low_or_missing_empty_confidence"),
    ],
)
def test_low_confidence_present_roles_are_invalid(
    profile, observation_update: dict, reason: str
) -> None:
    material, empty = masks_for_percent(50.0)
    observation = replace(SemanticObservation(material, empty), **observation_update)
    result = estimate_image_level(observation, profile)
    assert not result.valid
    assert result.material_percent is None
    assert reason in result.rejection_reasons


@pytest.mark.parametrize("percent", [0.0, 30.0, 70.0, 100.0])
def test_geometrically_sound_single_class_observations_are_valid(
    profile, percent: float
) -> None:
    material, empty = masks_for_percent(percent)
    if percent in {30.0, 70.0, 100.0}:
        observation = SemanticObservation(
            material,
            np.zeros_like(empty),
            empty_confidences=(),
            empty_instance_count=0,
            empty_present=False,
        )
    else:
        observation = SemanticObservation(
            np.zeros_like(material),
            empty,
            material_confidences=(),
            material_instance_count=0,
            material_present=False,
        )
    result = estimate_image_level(observation, profile)
    assert result.valid, result.rejection_reasons
    assert result.material_percent == pytest.approx(percent)


def test_multiple_instances_are_recorded_but_do_not_invalidate(profile) -> None:
    material, empty = masks_for_percent(50.0)
    result = estimate_image_level(
        SemanticObservation(
            material,
            empty,
            material_instance_count=2,
            empty_instance_count=3,
            material_confidences=(0.91, 0.72),
            empty_confidences=(0.88, 0.70, 0.65),
        ),
        profile,
    )
    assert result.valid, result.rejection_reasons
    assert result.material_instance_count == 2
    assert result.empty_instance_count == 3


def test_tiny_or_disconnected_single_class_regions_are_invalid(profile) -> None:
    tiny = np.zeros((100, 80), dtype=bool)
    tiny[48:52, 38:42] = True
    result = estimate_image_level(
        SemanticObservation(
            tiny,
            np.zeros_like(tiny),
            empty_confidences=(),
            empty_instance_count=0,
            empty_present=False,
        ),
        profile,
    )
    assert not result.valid
    assert "insufficient_material_lateral_support" in result.rejection_reasons
    assert "material_not_connected_to_tube_bottom" in result.rejection_reasons

    bottom_sliver = np.zeros((100, 80), dtype=bool)
    bottom_sliver[99:, :] = True
    sliver_result = estimate_image_level(
        SemanticObservation(
            bottom_sliver,
            np.zeros_like(bottom_sliver),
            empty_confidences=(),
            empty_instance_count=0,
            empty_present=False,
        ),
        profile,
    )
    assert not sliver_result.valid
    assert "tiny_material_axial_extent" in sliver_result.rejection_reasons


def test_multiple_or_reversed_interfaces_are_explicitly_invalid(profile) -> None:
    material = np.zeros((100, 80), dtype=bool)
    empty = np.zeros_like(material)
    empty[:20] = True
    material[20:40] = True
    empty[40:60] = True
    material[60:] = True
    result = estimate_image_level(SemanticObservation(material, empty), profile)
    assert not result.valid
    assert "multiple_interface_transitions" in result.rejection_reasons
    assert "implausible_material_empty_orientation" in result.rejection_reasons


def test_valid_estimate_conserves_percent_and_straight_tube_volume(profile) -> None:
    material, empty = masks_for_percent(35.0)
    result = estimate_image_level(SemanticObservation(material, empty), profile)
    values = derive_estimate(result.material_percent, 197.04)
    material_percent, material_volume, empty_percent, empty_volume = values
    assert material_percent + empty_percent == pytest.approx(100.0)
    assert material_volume + empty_volume == pytest.approx(197.04)
    assert material_volume == pytest.approx(197.04 * 0.35)


def test_six_image_group_uses_mean_of_retained_majority_pattern(profile) -> None:
    percentages = [48.0, 49.0, 50.0, 51.0]
    estimates = [
        estimate_image_level(SemanticObservation(*masks_for_percent(value)), profile)
        for value in percentages
    ]
    for value in (20.0, 80.0):
        material, empty = masks_for_percent(value)
        estimates.append(
            estimate_image_level(
                SemanticObservation(
                    material,
                    np.zeros_like(empty),
                    empty_confidences=(),
                    empty_instance_count=0,
                    empty_present=False,
                ),
                profile,
            )
        )
    group = aggregate_image_estimates(estimates, profile)
    assert group.valid
    assert group.majority_pattern == "material_and_empty"
    assert group.accepted_count == 4
    assert group.rejected_count == 2
    assert group.material_percent == pytest.approx(49.5)
    assert group.aggregation_method == "arithmetic_mean"
    assert group.retained_flags == (True, True, True, True, False, False)
    assert group.pattern_filter_reasons[-1] == (
        "class_pattern_disagrees_with_majority",
    )


def test_group_allows_one_retained_image_and_rejects_excessive_spread(profile) -> None:
    only = [estimate_image_level(observation_for_percent(49.0), profile)]
    one = aggregate_image_estimates(only, profile)
    assert one.valid
    assert one.accepted_count == 1
    assert one.minimum_valid_images_required == 1
    assert one.material_percent == pytest.approx(49.0)

    dispersed = [
        estimate_image_level(SemanticObservation(*masks_for_percent(value)), profile)
        for value in (20.0, 50.0, 80.0)
    ]
    spread = aggregate_image_estimates(dispersed, profile)
    assert not spread.valid
    assert spread.material_percent is None
    assert "excessive_group_spread" in spread.rejection_reasons


def test_tied_class_patterns_are_explicitly_invalid(profile) -> None:
    dual = estimate_image_level(observation_for_percent(50.0), profile)
    material, empty = masks_for_percent(50.0)
    material_only = estimate_image_level(
        SemanticObservation(
            material,
            np.zeros_like(empty),
            empty_confidences=(),
            empty_instance_count=0,
            empty_present=False,
        ),
        profile,
    )
    group = aggregate_image_estimates((dual, material_only), profile)
    assert not group.valid
    assert group.majority_pattern is None
    assert "ambiguous_majority_class_pattern" in group.rejection_reasons


def test_three_vs_three_tie_prefers_material_and_empty(profile) -> None:
    dual = [estimate_image_level(observation_for_percent(50.0), profile) for _ in range(3)]
    material, empty = masks_for_percent(50.0)
    material_only = [
        estimate_image_level(
            SemanticObservation(
                material,
                np.zeros_like(empty),
                empty_confidences=(),
                empty_instance_count=0,
                empty_present=False,
            ),
            profile,
        )
        for _ in range(3)
    ]
    group = aggregate_image_estimates((*dual, *material_only), profile)
    assert group.valid, group.rejection_reasons
    assert group.majority_pattern == "material_and_empty"
    assert group.accepted_count == 3
    assert group.rejected_count == 3
    assert group.material_percent == pytest.approx(50.0)
    assert group.retained_flags == (True, True, True, False, False, False)


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [("material_only", 100.0), ("empty_only", 0.0)],
)
def test_six_unanimous_single_class_images_use_semantic_endpoint(
    profile, pattern: str, expected: float
) -> None:
    material, empty = masks_for_percent(40.0)
    if pattern == "material_only":
        observation = SemanticObservation(
            material,
            np.zeros_like(empty),
            empty_confidences=(),
            empty_instance_count=0,
            empty_present=False,
        )
    else:
        observation = SemanticObservation(
            np.zeros_like(material),
            empty,
            material_confidences=(),
            material_instance_count=0,
            material_present=False,
        )
    estimates = tuple(estimate_image_level(observation, profile) for _ in range(6))
    assert estimates[0].material_percent != expected
    group = aggregate_image_estimates(estimates, profile)
    assert group.valid, group.rejection_reasons
    assert group.majority_pattern == pattern
    assert group.accepted_count == 6
    assert group.rejected_count == 0
    assert group.material_percent == expected
    assert group.accepted_material_percentages == (expected,) * 6


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [("material_only", 100.0), ("empty_only", 0.0)],
)
def test_six_unanimous_single_class_images_work_without_calibrated_roi(
    profile, pattern: str, expected: float
) -> None:
    whole_image = replace(
        profile,
        tube_roi=replace(
            profile.tube_roi,
            left=0.0,
            top=0.0,
            right=1.0,
            bottom=1.0,
            frozen=False,
            usage_mode="whole_image",
        ),
    )
    material, empty = masks_for_percent(40.0)
    observation = (
        SemanticObservation(
            material,
            np.zeros_like(empty),
            empty_confidences=(),
            empty_instance_count=0,
            empty_present=False,
        )
        if pattern == "material_only"
        else SemanticObservation(
            np.zeros_like(material),
            empty,
            material_confidences=(),
            material_instance_count=0,
            material_present=False,
        )
    )
    estimates = tuple(estimate_image_level(observation, whole_image) for _ in range(6))
    assert all(not item.valid for item in estimates)
    group = aggregate_image_estimates(estimates, whole_image)
    assert group.valid, group.rejection_reasons
    assert group.material_percent == expected
