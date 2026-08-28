# Material-level study data dictionary

## Status and authority

This document freezes the stage-1 data design for the material-level study. It
expands the common Excel contract in `CODEX_PROMPTS_MATERIAL_LEVEL_STUDY.md`
without changing its column names, order, or meanings. The contract version
defined here is `1.0.0`.

This is a software data contract, not evidence that either measurement method
is scientifically or metrologically validated.

## Common workbook contract

Each method workbook has a primary sheet named exactly `measurements`. The sheet
contains exactly the 32 columns below, in the listed order. The depth and YOLO
workbooks use identical headers, types, units, null semantics, and validation
rules. Method-specific fields belong in additional sheets or structured
artifacts, never in the primary sheet.

Excel timestamps and identifiers are stored as text to prevent Excel from
dropping timezone or leading-zero information. Numeric values are stored as
numeric cells, booleans as boolean cells, and missing values as truly blank
cells. CSV mirrors use the identical header order, UTF-8 with BOM, `.` as the
decimal separator, lowercase `true`/`false`, and an empty field for a missing
value. The strings `null`, `None`, `NaN`, `inf`, `N/A`, and `0` are not missing
value substitutes.

“Conditional” in the Nullable column means that the status and cross-field
rules later in this document decide whether a value is required.

| # | Column | Logical/storage type | Unit | Nullable | Meaning and field validation |
|---:|---|---|---|---|---|
| 1 | `schema_version` | text | none | No | Version of this common record contract. Exactly `1.0.0` for this design. |
| 2 | `experiment_id` | text | none | No | Operator-selected experiment/session identifier. Must equal the session directory name and pass the identifier policy below. |
| 3 | `measurement_id` | text | none | No | Shared identifier for one trigger/capture group, such as `000001_taking_20260813T104512345Z`. Must pass the ID policy below. |
| 4 | `measurement_index` | integer | count | No | Monotonically increasing integer within the experiment. Must be at least 1 and equal the numeric prefix in `measurement_id`. Booleans and fractional values are invalid. |
| 5 | `method` | text enum | none | No | `depth` or `yolo`. |
| 6 | `acquisition_mode` | text enum | none | No | For example `synchronized`, `manual_camera`, `timed_camera`, or `folder`. The complete allowed set is defined below. |
| 7 | `trigger_time_utc` | UTC ISO 8601 text | UTC | No | Time the operator or caller requested the measurement. It supplies the timestamp component of `measurement_id`. |
| 8 | `capture_start_utc` | UTC ISO 8601 text | UTC | Conditional | Start of this method's acquisition. Required when any method acquisition began. |
| 9 | `capture_end_utc` | UTC ISO 8601 text | UTC | Conditional | End of this method's acquisition. Required when acquisition completed or ended partially; blank only if acquisition never began or no reliable source time exists for folder input. |
| 10 | `processing_time_utc` | UTC ISO 8601 text | UTC | Conditional | Time the method result was produced. Required for `complete_valid` and `complete_invalid`; blank while pending or when no result was produced. |
| 11 | `material_name` | text | none | Yes | Operator-provided material name. Blank means it was not supplied; software must not infer it from detections, file names, or density. A confirmed research session should supply it. |
| 12 | `total_capacity_ml` | finite number | mL | No | Usable tube capacity. Must be greater than zero and common to both methods for one synchronized measurement. |
| 13 | `bulk_density_g_per_ml` | finite number | g/mL | Yes | Bulk density used for any mass-to-volume calculation. When present, must be greater than zero. Never infer it from `material_name`. |
| 14 | `total_possible_weight_g` | finite number | g | Yes | Maximum material mass at the stated capacity, if known. When present, must be nonnegative. It is an explicit operator/reference input, not silently derived. |
| 15 | `reference_material_weight_g` | finite number | g | Yes | Independently measured material-left mass. When present, must be nonnegative. |
| 16 | `reference_volume_from_weight_ml` | finite number | mL | Conditional | `reference_material_weight_g / bulk_density_g_per_ml`, only when both inputs are valid. Must be blank if either input is blank. |
| 17 | `manual_material_level_mm` | finite number | mm | Yes | Independently measured human/reference fill height. When present, must be nonnegative and no greater than the documented usable internal height. |
| 18 | `manual_material_percent` | finite number | % | Conditional | Manual level as a percentage of usable internal height, if available. Must be in `[0, 100]` and follow the formula below when height and geometry are available. |
| 19 | `reference_volume_from_manual_ml` | finite number | mL | Conditional | Manual reference volume using the documented tube geometry. Must be blank unless the manual level and required geometry are available. |
| 20 | `estimated_material_percent` | finite number | % | Conditional | Result produced by the method named in `method`. Required only for an accepted result and constrained to `[0, 100]`. |
| 21 | `estimated_material_volume_ml` | finite number | mL | Conditional | Method material percentage multiplied by capacity. Required only for an accepted result and constrained to `[0, total_capacity_ml]`. |
| 22 | `estimated_empty_percent` | finite number | % | Conditional | Empty percentage produced by the method. Required only for an accepted result and constrained to `[0, 100]`. |
| 23 | `estimated_empty_volume_ml` | finite number | mL | Conditional | Method empty percentage multiplied by capacity. Required only for an accepted result and constrained to `[0, total_capacity_ml]`. |
| 24 | `valid` | boolean | none | Conditional | Whether the method accepted the measurement. `true` only for `complete_valid`, `false` for terminal rejection/failure, and blank for pending, partial, or cancelled work that has not reached a method decision. |
| 25 | `quality_score` | finite number | dimensionless | Yes | Method quality score when one is defined. When present, must be in `[0, 1]`. Scores are method-specific and are not directly comparable. |
| 26 | `status` | text enum | none | No | Machine-readable result/failure state. Allowed values and their required fields are defined below. |
| 27 | `notes` | text | none | Yes | Optional operator notes. Machine diagnostics and exception traces belong in method diagnostics/manifests, not in this field. |
| 28 | `artifact_directory` | path text | none | No | Path to the measurement evidence directory for this method. Use the path policy below. |
| 29 | `source_artifact` | path text | none | Conditional | Main source image, depth artifact, or capture-group manifest. Required once evidence exists; blank only when acquisition never produced an artifact. |
| 30 | `config_sha256` | lowercase hex text | none | Conditional | SHA-256 of the exact effective method-configuration snapshot. Required for a completed result; pending/failed records may leave it blank only when no effective method configuration exists. |
| 31 | `calibration_or_model_sha256` | lowercase hex text | none | Conditional | SHA-256 of the depth calibration artifact or YOLO weights. Required for a completed result. It is blank for a pending YOLO row until weights are selected and verified. |
| 32 | `software_commit` | text | none | Yes | Full Git commit, with `+dirty` when relevant. Blank only when Git metadata is unavailable; provenance capture must not make acquisition fail. |

The alignment key is `(experiment_id, measurement_id)`. A method result is
uniquely identified by `(experiment_id, measurement_id, method)`. Export is an
idempotent upsert on that three-field key; duplicate keys are invalid. Rows are
written in ascending `measurement_index` order.

## Controlled values

### Method

| Value | Meaning |
|---|---|
| `depth` | Existing Intel RealSense D405 depth-surface reconstruction method. |
| `yolo` | Offline still-image YOLO material-level method. |

### Acquisition mode

| Value | Method compatibility | Meaning |
|---|---|---|
| `standalone_camera` | depth | Standalone live D405 command. |
| `bag` | depth | D405 RealSense recording replay. |
| `synthetic` | depth | Hardware-free synthetic D405 source. |
| `synchronized` | depth, yolo | One Run_Experiment trigger shared by warmed D405 and C920 sources. Offline YOLO processing retains this mode; it does not become `folder`. |
| `manual_camera` | yolo | Operator deliberately captured one C920 still before offline inference. |
| `timed_camera` | yolo | Configured timed/count C920 still acquisition before offline inference. |
| `folder` | yolo | Existing still images supplied from a folder. |

Adding an acquisition mode requires a schema minor version and compatible
readers. Free-form acquisition modes are not allowed in version `1.0.0`.

### Status and null-state rules

| Status | `valid` | Estimate fields | Required timing/evidence semantics |
|---|---|---|---|
| `pending_offline_inference` | blank | all blank | A synchronized/manual/timed capture is durable but YOLO has not processed it. Capture start/end and source artifact are required. |
| `complete_valid` | `true` | all four required | Acquisition and processing timestamps, both hashes, and source artifact are required. All formula/conservation rules apply. |
| `complete_invalid` | `false` | all blank | Processing completed but quality/class/interface rules rejected the result. Diagnostics retain candidate values and rejection reasons if useful. Processing time and both hashes are required. |
| `partial_capture` | blank | all blank | Acquisition started but did not meet its planned evidence count. Preserve the partial capture and manifest; later processing may upsert this row to a completed status. |
| `acquisition_failed` | `false` | all blank | No usable acquisition was completed. Capture times and source artifact are populated only if they exist. |
| `processing_failed` | `false` | all blank | Acquisition exists, but no method result was produced. Capture times/source artifact are required; failure time and details live in the manifest. |
| `cancelled` | blank | all blank | Work was cancelled after a measurement was allocated. If cancellation occurred before allocation/acquisition, no common row is written. |

The D405's existing warning/refill states (`OK`, pending confirmation, hysteresis,
and refill-warning state) are depth diagnostics, not common `status` values.
Likewise, YOLO image rejection reasons are YOLO diagnostics. They must not expand
the common status vocabulary without a schema change.

## Formula and cross-field rules

All calculations use unrounded finite values. Rounding is display-only.

1. Weight-derived reference volume, when both explicit inputs exist:

   ```text
   reference_volume_from_weight_ml =
       reference_material_weight_g / bulk_density_g_per_ml
   ```

   If either input is blank, the derived field must be blank. A supplied derived
   value must match the formula within
   `max(1e-6 mL, abs(expected) * 1e-9)`.

2. For the documented straight, constant-cross-section tube:

   ```text
   manual_material_percent =
       100 * manual_material_level_mm / usable_internal_height_mm

   reference_volume_from_manual_ml =
       total_capacity_ml * manual_material_percent / 100
   ```

   `usable_internal_height_mm` and the geometry source are stored in the session
   manifest/effective configuration, because they are method setup rather than
   common row columns. If geometry is absent, both derived manual fields remain
   blank. A future non-uniform tube requires a new documented geometry formula
   and a schema change; arbitrary mask area is not a volume formula.

3. For `complete_valid` method results:

   ```text
   estimated_empty_percent = 100 - estimated_material_percent
   estimated_material_volume_ml =
       total_capacity_ml * estimated_material_percent / 100
   estimated_empty_volume_ml =
       total_capacity_ml - estimated_material_volume_ml
   ```

   The absolute percentage conservation tolerance is `1e-6` percentage points.
   The volume conservation/formula tolerance is
   `max(1e-6 mL, total_capacity_ml * 1e-9)`. These are software-consistency
   tolerances, not measurement uncertainty or scientific acceptance limits.

4. All numeric values must be finite. Masses, volumes, levels, capacity, and
   density cannot be negative; density and capacity must be strictly positive
   when present. Percentages are closed-range `[0, 100]` values.

5. `total_possible_weight_g` remains an explicit input. If it, density, and
   capacity are all present, compare it with
   `total_capacity_ml * bulk_density_g_per_ml`. A mismatch beyond the formula
   tolerance is a validation error unless the operator removes one conflicting
   value and documents the independent reference outside the common field.

6. For one `(experiment_id, measurement_id)`, shared operator/reference fields
   (`measurement_index`, `trigger_time_utc`, material, capacity, density, weight,
   manual references, and notes) must match in depth and YOLO rows. Method,
   capture/processing timestamps, results, quality, artifacts, and hashes may
   differ.

7. Missing or rejected scientific values remain blank. Failed or invalid methods
   must not publish candidate estimates as common results and must never use zero
   as a failure placeholder. Candidate/debug values may be retained in the
   method-specific diagnostic record with an explicit rejection reason.

## Identifier policy

### Experiment identifier

`experiment_id` is an operator-chosen portable slug:

- 1 to 64 ASCII characters;
- first character alphanumeric;
- remaining characters alphanumeric, `-`, `_`, or `.`;
- no whitespace, path separator, `..` path segment, trailing dot/space, or
  Windows reserved device name; and
- unique under the selected output root.

Human-readable Unicode belongs in purpose, material, and notes fields. Restricting
generated identifiers to a portable alphabet does not restrict the output root,
which may contain spaces and arbitrary valid Unicode.

### Measurement identifier

The exact version-1 form is:

```text
<index>_taking_<UTC timestamp>
000001_taking_20260813T104512345Z
```

- `<index>` is zero-padded to at least six digits and equals
  `measurement_index` (`000001`, `000002`, ...; it expands rather than wraps
  after `999999`).
- The timestamp is `YYYYMMDDTHHMMSSfffZ`, the shared trigger UTC instant
  truncated to milliseconds.
- The regular expression is
  `^[0-9]{6,}_taking_[0-9]{8}T[0-9]{9}Z$`.
- Allocation is serialized under the session lock. The next index is one more
  than the greatest durable index found in manifests or either workbook; an
  interrupted/reserved index is never silently reused.
- Retries and offline processing retain the original ID and index. They do not
  mint a new measurement merely because processing time or model/config hashes
  changed.

The timestamp aids sorting; uniqueness comes from the experiment plus serialized
index. The same ID names the measurement directory and is copied without change
to both method rows.

## UTC timestamp policy

- Canonical workbook/JSON representation is ISO 8601 text in UTC with a literal
  `Z`, for example `2026-08-13T10:45:12.345678Z`.
- APIs accept timezone-aware datetimes only. Naive datetimes are invalid. Inputs
  with another offset are converted to UTC before persistence.
- Internal timestamps retain microsecond precision when available. Measurement
  IDs use the same trigger instant truncated, not rounded, to milliseconds.
- For synchronized acquisition, one `trigger_time_utc` is captured once and used
  unchanged by both rows. Method capture times describe actual method activity.
- When both values exist, require
  `trigger_time_utc <= capture_start_utc <= capture_end_utc <= processing_time_utc`.
  For folder input whose historical capture times cannot be established without
  violating this rule, capture start/end remain blank and source-file metadata is
  kept separately; file modification time is never invented as capture time.
- Scheduling and elapsed-time calculations use a monotonic clock. The manifest
  records target and actual monotonic offsets for each C920 frame; wall-clock UTC
  is for traceability, not scheduling.
- System-clock discontinuities are recorded as a synchronization diagnostic and
  never corrected by fabricating timestamps.

## Canonical session and measurement layout

`<output_root>` may be the default `3d_camera/research_records` or any accessible
operator-selected absolute directory. The canonical layout is:

```text
<output_root>/
  <experiment_id>/
    session_manifest.json
    effective_config.yaml
    depth_measurements.xlsx
    depth_measurements.csv
    yolo_measurements.xlsx
    yolo_measurements.csv
    provenance/
      run_experiment_effective_config.yaml
      depth_effective_config.yaml
      yolo_effective_config.yaml
      calibration/<calibration file or immutable-reference metadata>
      environment.json
    measurements/
      <measurement_id>/
        capture_manifest.json
        c920/
          raw/
            frame_01_<filesystem-safe UTC>.png
            frame_02_<filesystem-safe UTC>.png
            ...
            frame_06_<filesystem-safe UTC>.png
        depth/
          artifact_manifest.json
          result.json
          height_map_mm.npy
          height_map.png
          material_surface_mm.ply
          color_snapshot.png
          raw_depth/                 # full_raw only
        yolo/
          artifact_manifest.json
          result.json
          image_details.csv
          overlays/
          masks/
```

Conditional/profile-controlled artifacts may be absent, but directory names and
ownership do not change. Standalone depth and standalone YOLO sessions use the
same layout; they create only applicable evidence. A synchronized session creates
both workbooks and reserves one pending YOLO row at capture time. Offline YOLO
processing idempotently replaces that row using the same result key.

`effective_config.yaml` is the human-facing combined session snapshot. Exact
method snapshots whose bytes are hashed live under `provenance/`. The session
manifest contains the experiment metadata, record schema version, storage
profile, method/config/calibration/model references and hashes, software state,
camera identity/profile, creation/update times, and measurement manifest index.
`capture_manifest.json` is the crash-recovery source of truth for trigger state,
planned/actual frame times, per-camera outcomes, source hashes, and partial
failure reasons.

Within a workbook, `artifact_directory` uses a session-relative POSIX-form path,
for example
`measurements/000001_taking_20260813T104512345Z/depth`. The preferred
`source_artifact` is likewise relative: the depth `result.json`, the YOLO source
image for a single-image run, or `capture_manifest.json` for a capture group.
An external folder input that is not copied may use a resolved absolute path.

## Unicode-safe path policy

- Runtime APIs use `pathlib.Path`; they do not construct paths with manual slash
  concatenation or assume the current working directory.
- Relative configuration paths resolve against the YAML file containing them,
  not against the process working directory.
- Output roots may contain spaces and non-ASCII characters, including a path such
  as `D:\Research records\Material level study`.
- Do not transliterate, ASCII-encode, case-fold, or otherwise rewrite an
  operator-supplied directory. Generated experiment/measurement/artifact names
  use the portable policies above.
- Text JSON/YAML is UTF-8 with `ensure_ascii=False` behavior and line endings
  normalized when creating hash-governed snapshots. CSV uses UTF-8 with BOM for
  reliable Windows Excel opening.
- OpenCV image I/O uses the repository's proven Unicode-safe pattern:
  `numpy.fromfile` plus `cv2.imdecode` for reads and `cv2.imencode` plus
  `ndarray.tofile` for writes. It must not rely on `cv2.imread`/`cv2.imwrite` for
  Unicode Windows paths.
- Validate that the resolved measurement artifact path remains beneath the
  resolved session root before creating, replacing, or cleaning files. Do not
  follow an ID containing traversal components.
- Workbook paths are never `file://` URIs. All created directories/files must be
  tested under a path containing both spaces and accented characters.

## Configuration and hash provenance

- An effective configuration is the fully resolved, validated configuration
  actually used, including defaults and selected named profile. Copy it before
  acquisition/processing as deterministic UTF-8 YAML with LF line endings.
- `config_sha256` is lowercase SHA-256 over the exact bytes of that saved method
  snapshot. Hashing the original, partially defaulted YAML is insufficient.
- `calibration_or_model_sha256` is lowercase SHA-256 over the calibration `.npz`
  bytes for depth or the model weights bytes for YOLO. Stream large files rather
  than loading them wholly into memory.
- A SHA-256 value is exactly 64 lowercase hexadecimal characters. A changed file
  produces a new hash and, for already completed YOLO output, a recorded
  processing revision rather than silent overwrite.
- Model weights are referenced and hashed but are not copied into the repository
  or evidence bundle. Research/full profiles copy the small depth calibration or
  retain an immutable reference plus hash when copying is prohibited.
- `software_commit` uses `<40 lowercase hex>` for a clean tree and
  `<40 lowercase hex>+dirty` for a dirty tree. It is blank when Git is unavailable;
  the environment manifest records that condition. Dirty provenance hashes do
  not authorize changing user files.
- Every retained artifact is listed in its artifact manifest with relative path,
  byte length, media/logical type, SHA-256, and creation UTC. The manifest itself
  is written last through a same-directory temporary file and atomic replacement.
- Hashes establish identity/integrity, not scientific correctness.

## Evidence profiles

The profile is selected per session and recorded in every measurement manifest.
Profiles control newly generated retention; they never delete an operator's
pre-existing input files. C920 stills captured for deferred synchronized YOLO are
retained in every profile because they are the unprocessed source evidence.

| Evidence | `compact` | `research` (recommended default) | `full_raw` |
|---|---|---|---|
| Common Excel/CSV rows, session/capture/artifact manifests | Keep | Keep | Keep |
| Effective config snapshots, hashes, software/environment metadata | Keep | Keep | Keep |
| D405 result JSON and compact height-map PNG | Keep | Keep | Keep |
| D405 numeric height-map NPY and PLY surface | Omit | Keep | Keep |
| D405 diagnostic color/infrared snapshot when enabled | Omit | Keep | Keep |
| D405 calibration | Hash plus immutable reference | Copy plus hash, or immutable reference plus reason | Copy plus hash, or immutable reference plus reason |
| Raw unfiltered D405 Z16 burst and per-frame timing/intrinsics | Omit | Omit; policy is explicit in manifest | Keep. A lossless per-frame bundle or retained bag is required to claim `full_raw`. |
| Captured C920 lossless source stills | Keep | Keep | Keep |
| External folder-mode source images | Hash and resolved reference; never modify | Byte-for-byte copy plus original-path/hash metadata | Byte-for-byte copy plus original-path/hash metadata |
| YOLO aggregate JSON and per-image diagnostic table | Keep | Keep | Keep |
| YOLO masks/annotated overlays | Aggregate/contact-sheet evidence only | Per-image masks and overlays | Per-image masks and overlays plus adapter raw outputs |
| Logs/exception details needed for recovery | Bounded summary | Keep relevant session logs | Keep full session logs subject to privacy policy |

`research` retains enough software evidence to audit a reported result, but does
not by itself make the result scientifically validated. `full_raw` is storage
intensive; the session must verify adequate disk space before arming. A capture
that cannot retain the profile's required evidence is marked partial/failed and
must not silently downgrade its profile.

## Workbook durability and ownership

- The two workbooks are method-specific stores, not a joined comparison table.
- `depth_measurements.xlsx/.csv` contains `method=depth` rows only;
  `yolo_measurements.xlsx/.csv` contains `method=yolo` rows only.
- One session writer owns workbook/CSV updates. Camera/inference workers return
  records to it and never write Excel directly. Cross-process updates use a
  session lock with stale-lock recovery metadata.
- Save a complete temporary workbook/CSV beside its destination, flush/close it,
  verify headers and row keys, then atomically replace the destination where the
  filesystem supports it. Never expose a half-written workbook as current.
- The capture/result manifests are the recovery source. On restart, regenerate or
  reconcile workbook/CSV rows idempotently from durable manifests. Excel is not
  the sole copy of acquisition state.
- After every write, workbook and CSV key sets and ordered common values must
  match. An interrupted second-file replacement is detected and repaired on next
  open.

## Schema-version rules

- Version format is semantic `MAJOR.MINOR.PATCH`, stored as text. This document
  defines `1.0.0`.
- All rows in a workbook and both workbooks in one session use the same schema
  version. Mixed versions are invalid.
- Increment **MAJOR** for removal, rename, reorder, changed type/unit/meaning,
  changed nullability, changed formula semantics, or any other incompatible
  primary-sheet change.
- Increment **MINOR** for a backward-compatible controlled-value addition or a
  new optional capability that leaves every versioned common column unchanged.
- Increment **PATCH** for documentation/validation clarifications that do not
  change accepted persisted values or layout.
- Readers reject an unknown major version. They may accept a newer minor/patch
  only when explicitly coded to do so; they never guess column mappings.
- Additional method-specific sheets and JSON fields may evolve independently,
  but each must carry its own schema version and cannot change the common sheet.
- The existing integer `schema_version` in `3d_camera/config.yaml` is a separate
  configuration schema and must not be confused with this record version.

## Method-specific diagnostics (outside the common sheet)

Depth diagnostics include surface coverage, median temporal valid fraction,
temporal MAD, maximum internal-hole radius, calibration-plane RMS, outliers
replaced, heuristic variability (never labelled GUM uncertainty), min/mean/max
level, refill/hysteresis state, active camera profile/serial/depth scale/filter
settings, and raw artifact metadata.

YOLO diagnostics include one row per still: source hash, semantic/model class
mapping, task/mode, confidence, material/empty coverage, overlap, unclassified
fraction, interface location/spread, ROI/calibration details, accepted/rejected
state and reasons, plus group accepted/rejected counts and robust spread.

Synchronization diagnostics include target/actual monotonic and UTC C920 frame
times, timing errors, D405 burst start/end, overlap duration, planned/actual frame
count, camera ownership/cleanup outcome, and partial/failure reasons.

No diagnostics sheet may calculate depth-versus-YOLO differences, errors,
agreement, correlations, rankings, or comparison plots. Method comparison is a
separate future project.
