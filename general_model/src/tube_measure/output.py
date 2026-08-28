from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def write_results(records: List[Dict[str, Any]], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    fieldnames = [
        "source",
        "frame_index",
        "tube_id",
        "percentage",
        "confidence",
        "valid",
        "rejection_reason",
        "x1",
        "y1",
        "x2",
        "y2",
        "empty_area_px",
        "material_area_px",
        "pair_score",
        "measurement_source",
        "reference_height_px",
        "reference_width_px",
    ]
    with (output_directory / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for row in _csv_rows(record):
                writer.writerow(row)


def _csv_rows(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    analysis = record["analysis"]
    for tube in analysis["tubes"]:
        bbox = tube["bbox"]
        yield {
            "source": record["source"],
            "frame_index": record.get("frame_index"),
            "tube_id": tube["tube_id"],
            "percentage": tube["percentage"],
            "confidence": tube["confidence"],
            "valid": tube["valid"],
            "rejection_reason": tube["rejection_reason"],
            "x1": bbox["x1"],
            "y1": bbox["y1"],
            "x2": bbox["x2"],
            "y2": bbox["y2"],
            "empty_area_px": tube["empty_area_px"],
            "material_area_px": tube["material_area_px"],
            "pair_score": tube["pair_score"],
            "measurement_source": tube["source"],
            "reference_height_px": analysis["reference_height_px"],
            "reference_width_px": analysis["reference_width_px"],
        }

