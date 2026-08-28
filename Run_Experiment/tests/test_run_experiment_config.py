from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from run_experiment.config import load_config
from run_experiment.cli import (
    _inputs,
    _prompt_mass_reference_choice,
    _validate_reference_cli_mode,
    parse_args,
)
from run_experiment.errors import ExperimentConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_supplied_config_resolves_paths_and_freezes_six_over_one_second() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    assert config.acquisition.frame_count == 6
    assert config.acquisition.span_seconds == 1.0
    assert config.acquisition.worker_count == 2
    assert config.c920.backend == "dshow"
    assert config.c920.device_name_contains == "C920"
    assert config.depth_config_path == (PROJECT_ROOT.parent / "3d_camera" / "config.yaml").resolve()
    assert config.output_root == (PROJECT_ROOT / "research_records").resolve()


def test_interactive_capacity_prompt_round_trips_frozen_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_capacity = 197.04069123315182
    prompts: list[str] = []
    answers = iter(("study-1", "purpose", "197.040691233152", "", "Powder", ""))

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", answer)
    inputs = _inputs(parse_args(["--interactive"]), default_capacity)

    assert inputs.total_capacity_ml == pytest.approx(default_capacity, abs=1e-12)
    assert inputs.material_name == "Powder"
    assert inputs.reference_material_weight_g is None
    assert inputs.manual_material_level_mm is None
    assert "197.040691233152" in prompts[2]


def test_mass_reference_prompt_keeps_existing_defaults() -> None:
    answers = iter(("",))
    result = _prompt_mass_reference_choice(
        0.52,
        102.46,
        input_func=lambda _prompt: next(answers),
        print_func=lambda _message: None,
    )
    assert result == (0.52, 102.46)


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (("y", "d", "0.52"), (0.52, None)),
        (("yes", "mass", "102.46"), (None, 102.46)),
    ],
)
def test_mass_reference_prompt_uses_one_explicit_basis(answers, expected) -> None:
    responses = iter(answers)
    result = _prompt_mass_reference_choice(
        None,
        None,
        input_func=lambda _prompt: next(responses),
        print_func=lambda _message: None,
    )
    assert result == expected


def test_mass_reference_prompt_retries_invalid_choice_and_value() -> None:
    responses = iter(("maybe", "y", "x", "m", "0", "nan", "95.5"))
    messages: list[str] = []
    result = _prompt_mass_reference_choice(
        None,
        None,
        input_func=lambda _prompt: next(responses),
        print_func=messages.append,
    )
    assert result == (None, 95.5)
    assert messages


def test_per_measurement_cli_values_are_rejected_for_repeating_session() -> None:
    repeating = parse_args(["--reference-material-weight-g", "54.7"])
    with pytest.raises(ValueError, match="per-measurement values"):
        _validate_reference_cli_mode(repeating)

    one_shot = parse_args(
        [
            "--once",
            "--reference-material-weight-g",
            "54.7",
            "--manual-material-level-mm",
            "41.5",
        ]
    )
    _validate_reference_cli_mode(one_shot)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda data: data["acquisition"].update(frame_count=1), ">= 2"),
        (lambda data: data["acquisition"].update(worker_count=3), "exactly 2"),
        (lambda data: data["acquisition"].update(span_seconds=0), "finite"),
        (lambda data: data["c920"].update(backend="hard-coded"), "must be one of"),
        (lambda data: data["c920"].update(capture_key="q"), "must differ"),
        (lambda data: data.update(unknown=True), "unknown"),
    ],
)
def test_invalid_config_fails_closed(tmp_path: Path, update, message: str) -> None:
    data = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    update(data)
    path = tmp_path / "bad config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ExperimentConfigurationError, match=message):
        load_config(path)
