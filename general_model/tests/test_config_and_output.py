from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tube_measure import MaskInstance, TubeAnalyzer, load_config
from tube_measure.output import write_results
from tube_measure.pipeline import resolve_inference_device


def test_example_configuration_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    config = load_config(path)
    assert "empty" in config.empty_names
    assert {"polymer", "material"} <= config.material_names
    assert config.inference.device == "auto"


def test_explicit_cpu_device_is_stable() -> None:
    assert resolve_inference_device("cpu") == "cpu"


def test_auto_device_prefers_cuda_then_falls_back_to_cpu() -> None:
    assert resolve_inference_device("auto", cuda_available=lambda: True) == "cuda:0"
    assert resolve_inference_device("auto", cuda_available=lambda: False) == "cpu"


def test_json_and_csv_include_required_measurement_fields(tmp_path: Path) -> None:
    mask = np.zeros((120, 40), dtype=bool)
    mask[10:110, 10:30] = True
    config_path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    config = load_config(config_path)
    # The example uses inferred height, so use a paired tube for this output test.
    empty = mask.copy()
    empty[60:110] = False
    material = mask.copy()
    material[10:60] = False
    analysis = TubeAnalyzer(config).analyze(
        [MaskInstance("Empty", empty), MaskInstance("Material", material)]
    )
    records = [{"source": "synthetic.png", "frame_index": None, "analysis": analysis.to_dict()}]
    write_results(records, tmp_path)

    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    tube = payload[0]["analysis"]["tubes"][0]
    assert {"tube_id", "percentage", "confidence", "valid", "rejection_reason"} <= tube.keys()
    with (tmp_path / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["tube_id"] == "tube_001"
