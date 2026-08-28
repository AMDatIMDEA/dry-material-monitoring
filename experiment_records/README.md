# experiment_records

`experiment_records` is the one method-neutral implementation of the material
study's common `measurements` contract. It owns the 32-column order, typed record,
IDs, validation/formulas, SHA-256/Git provenance, file locking, and identical
Excel/CSV persistence. It contains no camera, inference, numerical measurement,
or method-comparison code.

Both `3d_camera` and `Material_level_using_yolo` import this package. They must
not copy `COMMON_COLUMNS` into method code.

Install the shared package from the repository root when an editable environment
is needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

## Minimal valid depth result

```python
from datetime import datetime, timezone
from pathlib import Path

from experiment_records import (
    CommonMeasurementRecord,
    MeasurementWorkbookStore,
    Method,
    make_measurement_id,
    prepare_record,
    with_estimate,
)

trigger = datetime.now(timezone.utc)
measurement_id = make_measurement_id(1, trigger)
record = CommonMeasurementRecord(
    experiment_id="study-001",
    measurement_id=measurement_id,
    measurement_index=1,
    method="depth",
    acquisition_mode="synthetic",
    trigger_time_utc=trigger,
    capture_start_utc=trigger,
    capture_end_utc=trigger,
    processing_time_utc=trigger,
    material_name="example material",
    total_capacity_ml=200.0,
    valid=True,
    status="complete_valid",
    artifact_directory=f"measurements/{measurement_id}/depth",
    source_artifact=f"measurements/{measurement_id}/depth/result.json",
    config_sha256="0" * 64,
    calibration_or_model_sha256="1" * 64,
)
record = prepare_record(with_estimate(record, 35.0))

store = MeasurementWorkbookStore(
    Path("study-001/depth_measurements.xlsx"),
    method=Method.DEPTH,
)
store.upsert(record)  # creates, appends, or idempotently replaces
assert store.verify_mirror()
```

Method implementations that need one atomic common-row plus diagnostic-sheet
update can use `upsert_with_diagnostic_rows(...)`. The caller supplies the
method-specific sheet name, exact columns and rows, and a replacement key such
as experiment/measurement ID. The store preserves unrelated sheets, verifies
the temporary workbook and common CSV mirror, then replaces both under the same
single-writer lock. Diagnostic columns remain method-owned; this shared package
owns only the common `measurements` contract.

Use the same store class with `method=Method.YOLO` and a
`yolo_measurements.xlsx` target for YOLO records. The store rejects rows labelled
for the other method. `validate_aligned_records(depth, yolo)` checks only frozen
shared capture/reference metadata; it deliberately does not compare estimates.

Use `prepare_record(..., usable_internal_height_mm=...)` when a manual material
level is supplied. Weight-derived volume is calculated only when both explicit
weight and bulk density exist. Material names never select or imply density.

## Persistence contract

- `create` refuses to overwrite either target.
- `append` rejects an existing `(experiment_id, measurement_id, method)` key.
- `upsert` creates or replaces that key idempotently.
- Missing values stay as blank Excel cells and empty CSV fields.
- A lock file serializes cross-process writers. After a crash, verify that no
  writer is active before manually removing a stale `.xlsx.lock` file.
- Temporary files are written and verified beside their destinations. XLSX is
  replaced first; if CSV replacement is interrupted, the next successful upsert
  reads the XLSX authority and regenerates both mirrors.
- Extra method-specific workbook sheets are preserved. Only the primary
  `measurements` sheet is owned by this package.

The numeric tolerances are software consistency checks from the data dictionary,
not scientific acceptance limits or measurement uncertainty. Percentage
conservation uses `1e-6` percentage points. Volume conservation uses
`max(1e-6 mL, total_capacity_ml * 1e-9)`.
