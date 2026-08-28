from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tube_measure import AppConfig, MaskInstance, TubeAnalyzer


SHAPE = (130, 120)


def rectangle(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def instance(name: str, mask: np.ndarray, confidence: float = 0.95) -> MaskInstance:
    return MaskInstance(name, mask, confidence)


def configured(expected_height: float | None = 100.0) -> AppConfig:
    base = AppConfig()
    return replace(
        base,
        completeness=replace(
            base.completeness,
            expected_full_height_px=expected_height,
            min_reference_pairs=1,
        ),
    )


def test_multiple_tubes_are_paired_and_sorted_left_to_right() -> None:
    masks = [
        instance("Material", rectangle(55, 80, 75, 110)),
        instance("Empty", rectangle(10, 10, 30, 70)),
        instance("Polymer", rectangle(10, 70, 30, 110)),
        instance("Empty", rectangle(55, 10, 75, 80)),
    ]
    result = TubeAnalyzer(configured()).analyze(masks)

    assert [tube.tube_id for tube in result.measurements] == ["tube_001", "tube_002"]
    assert all(tube.valid for tube in result.measurements)
    assert result.measurements[0].percentage == pytest.approx(40.0)
    assert result.measurements[1].percentage == pytest.approx(30.0)


def test_close_neighbouring_tubes_do_not_cross_pair() -> None:
    masks = [
        instance("Empty", rectangle(10, 10, 24, 60)),
        instance("Material", rectangle(10, 60, 24, 110)),
        instance("Empty", rectangle(27, 10, 41, 75)),
        instance("Material", rectangle(27, 75, 41, 110)),
    ]
    result = TubeAnalyzer(configured()).analyze(masks)

    assert len(result.measurements) == 2
    assert [tube.percentage for tube in result.measurements] == pytest.approx([50.0, 35.0])
    assert [tube.bbox[0] for tube in result.measurements] == [10, 27]


def test_irregular_interface_and_small_gap_are_accepted() -> None:
    empty = np.zeros(SHAPE, dtype=bool)
    material = np.zeros(SHAPE, dtype=bool)
    for x in range(15, 40):
        surface = 60 + int(round(4 * np.sin(x / 3.0)))
        empty[10:surface, x] = True
        material[surface + 2 : 110, x] = True

    result = TubeAnalyzer(configured()).analyze(
        [instance("Empty", empty), instance("Material", material)]
    )
    tube = result.measurements[0]
    assert tube.valid
    assert tube.percentage == pytest.approx(50.0, abs=3.0)


def test_minor_overlap_is_split_without_double_counting() -> None:
    empty = rectangle(20, 10, 40, 62)
    material = rectangle(20, 58, 40, 110)
    result = TubeAnalyzer(configured()).analyze(
        [instance("Empty", empty), instance("Material", material)]
    )
    tube = result.measurements[0]

    assert tube.valid
    assert tube.percentage == pytest.approx(50.0)
    assert tube.empty_area_px + tube.material_area_px == pytest.approx(2000.0)


def test_partial_single_class_mask_is_invalid_not_zero_percent() -> None:
    result = TubeAnalyzer(configured()).analyze(
        [instance("Empty", rectangle(10, 10, 30, 70))]
    )
    tube = result.measurements[0]

    assert not tube.valid
    assert tube.percentage is None
    assert tube.rejection_reason == "single_class_mask_is_incomplete"


@pytest.mark.parametrize(
    ("class_name", "expected_percentage"),
    [("Empty", 0.0), ("Material", 100.0), ("Polymer", 100.0)],
)
def test_geometrically_complete_single_class_tubes_are_valid(
    class_name: str, expected_percentage: float
) -> None:
    result = TubeAnalyzer(configured()).analyze(
        [instance(class_name, rectangle(10, 10, 30, 110))]
    )
    tube = result.measurements[0]

    assert tube.valid
    assert tube.percentage == expected_percentage
    assert tube.source.startswith("single_")


def test_false_match_with_no_horizontal_overlap_is_rejected() -> None:
    result = TubeAnalyzer(configured()).analyze(
        [
            instance("Empty", rectangle(10, 10, 30, 60)),
            instance("Material", rectangle(50, 60, 70, 110)),
        ]
    )

    assert len(result.measurements) == 2
    assert all(not tube.valid for tube in result.measurements)
    assert all(tube.percentage is None for tube in result.measurements)


def test_pair_statistics_can_validate_a_full_single_class_tube() -> None:
    config = configured(expected_height=None)
    result = TubeAnalyzer(config).analyze(
        [
            instance("Empty", rectangle(10, 10, 30, 65)),
            instance("Material", rectangle(10, 65, 30, 110)),
            instance("Empty", rectangle(50, 10, 70, 110)),
        ]
    )

    assert result.reference_height_px == 100.0
    assert result.reference_width_px == 20.0
    assert len(result.measurements) == 2
    assert result.measurements[1].valid
    assert result.measurements[1].percentage == 0.0


def test_unknown_classes_are_ignored() -> None:
    result = TubeAnalyzer(configured()).analyze(
        [instance("Tube", rectangle(10, 10, 30, 110))]
    )
    assert result.measurements == []
    assert result.ignored_instances == 1

