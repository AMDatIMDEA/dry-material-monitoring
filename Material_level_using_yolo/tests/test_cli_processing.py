from __future__ import annotations

from pathlib import Path

import numpy as np

from material_level_yolo.cli import (
    _prompt_mass_reference_choice,
    discover_images,
    main,
    parse_args,
)
from material_level_yolo.image_io import write_image


def test_discover_images_is_deterministic_and_unicode_safe(tmp_path: Path) -> None:
    folder = tmp_path / "Images avec espaces et accents é"
    write_image(folder / "z.PNG", np.zeros((4, 4, 3), dtype=np.uint8))
    write_image(folder / "à.png", np.zeros((4, 4, 3), dtype=np.uint8))
    (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    assert [path.name for path in discover_images(folder)] == ["z.PNG", "à.png"]


def test_processing_cli_requires_common_record_inputs(capsys) -> None:
    result = main(["--input", "missing.png"])
    assert result == 2
    assert "--experiment-id is required" in capsys.readouterr().err


def test_operator_cli_exposes_all_scientific_and_schedule_inputs() -> None:
    args = parse_args(
        [
            "--profile", "polymer",
            "--acquisition-mode", "timed_camera",
            "--experiment-id", "study-001",
            "--purpose", "repeatability",
            "--total-capacity-ml", "250",
            "--bulk-density-g-per-ml", "0.5",
            "--total-possible-weight-g", "125",
            "--remaining-weight-g", "25",
            "--manual-material-level", "4",
            "--manual-level-unit", "cm",
            "--usable-internal-height-mm", "80",
            "--notes", "operator note",
            "--output-root", "C:\\Research records",
            "--capture-count", "6",
            "--capture-interval-seconds", "0.2",
            "--roi-mode", "whole_image",
            "--yes",
        ]
    )
    assert args.profile == "polymer"
    assert args.acquisition_mode == "timed_camera"
    assert args.purpose == "repeatability"
    assert args.remaining_weight_g == 25.0
    assert args.manual_level_unit == "cm"
    assert args.capture_count == 6
    assert args.roi_mode == "whole_image"
    assert args.yes


def test_interactive_mass_reference_can_keep_current_values(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    assert _prompt_mass_reference_choice(0.51, 100.0) == (0.51, 100.0)


def test_interactive_mass_reference_selects_one_basis(monkeypatch) -> None:
    answers = iter(("y", "m", "bad", "0", "98.4"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert _prompt_mass_reference_choice(0.51, None) == (None, 98.4)
