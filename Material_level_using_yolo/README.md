# Offline YOLO material-level estimation

`Material_level_using_yolo` captures or accepts deliberate still images and
performs GPU-first inference only after acquisition, with automatic CPU
fallback when CUDA is unavailable. It estimates the material/empty
interface along a configured straight-tube axis, aggregates a capture group,
and writes the repository's shared measurement record plus YOLO diagnostics.
There is no YOLO inference on C920 preview frames, no continuous inference, no
depth-camera orchestration, and no method comparison.

The reusable `material_level_yolo.camera` module contains only camera settings,
preview events, warm-up, backend selection, and the OpenCV adapter. Importing it
does not import Ultralytics or the inference workflow, so a synchronized
acquisition process can safely reuse the C920 support without loading a model.

The implementation and its software tests are not scientific or metrological
validation. The ROI, model, camera geometry, and thresholds must be frozen and
validated for the actual apparatus before research conclusions are drawn.

## Install and place models

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

The default `device: auto` selects `cuda:0` when PyTorch reports CUDA as usable
and otherwise selects CPU. The resolved device is written to `result.json`.
Model weights are local research inputs and must not be committed. Portable
examples use `model_weights/powder_best.pt` and
`model_weights/polymer_best.pt` under this component; the ignored
`config.local.yaml` may retain other laboratory paths.

The source dataset's IDs 4 (`Powder`), 5 (`Empty`), and 6 (`polymer`) are not
trusted at runtime. The loader checks the trained model's `model.names` and
requires exactly one match for each configured semantic role. Optional explicit
IDs are additional assertions. A wrong task, missing or ambiguous class,
duplicate name, or ID/name mismatch fails before inference and output creation.

```powershell
# Validate configuration only
.\.venv\Scripts\python.exe .\Material_level_using_yolo\run_offline.py --profile powder

# Load weights and validate model task/classes without predicting
.\.venv\Scripts\python.exe .\Material_level_using_yolo\run_offline.py --profile powder --validate-model
```

## Capture first, run synchronized inference later

Acquisition and inference are deliberately different commands and processes.
For the geometrically controlled mode, freeze the useful internal tube rectangle
once from a representative image captured with the final camera mount. The
selector excludes the laboratory background and writes both the
normalized shared ROI in `config.yaml` and a hashed
`calibration/tube_roi.json` reference:

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\configure_roi.py `
  --image "C:\Study images\étalon tube.png"
```

Drag around the useful internal tube region, press Enter, inspect the printed
pixel bounds, and confirm. Escape cancels without changing the configuration.
For a scripted, already reviewed rectangle, supply all four bounds with
`--left-px`, `--top-px`, `--right-px`, and `--bottom-px`. The supplied
`0,0,1,1` placeholder is marked `frozen: false`. At runtime the terminal asks
`Use the tube ROI for this YOLO calculation? [y/n]`. Answer `y` to require the
frozen configured ROI, or `n` to run explicitly in `whole_image` mode without a
calibration file. In whole-image mode, mask-quality calculations use the tight
detected semantic extent, so unclassified laboratory background does not fail
row/column/coverage gates. For a valid Powder+Empty pair, percentage is
provisionally normalized between the detected semantic top and bottom. Those
limits are not calibrated full/empty tube limits, so the result and terminal
carry an explicit warning. A single-class whole-image detection is not silently
converted to calibrated 0% or 100%.
The selected mode and bounds are recorded in provenance and YOLO diagnostics.
For scripts/noninteractive stdin, supply `--roi-mode configured` or
`--roi-mode whole_image`. Re-run ROI configuration whenever the mount, framing,
resolution, or useful tube limits change.

First capture with both warmed cameras; `--material-name` must match one of the
case-insensitive `recorded_material_names` aliases configured for a model
profile:

```powershell
.\.venv\Scripts\python.exe .\Run_Experiment\run_experiment.py `
  --experiment-id polymer-sync-001 `
  --purpose "deferred synchronized material-level study" `
  --material-name Powder `
  --total-capacity-ml 211.527360357 `
  --output-root "C:\Research records\Material level étude" `
  --yes
```

After `Run_Experiment` reaches `ARMED`, each `c` key press or left click opens a
terminal reference step before acquisition. Enter that measurement's weight,
manual level, and notes (or leave any blank), review the summary, and press
Enter to capture. Those values are stored in both method rows and retained by
the offline command; a cancelled reference step creates no measurement.

After acquisition has closed—or from a separate PowerShell process—process all
pending capture manifests in the session:

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\process_synchronized.py `
  "C:\Research records\Material level étude\polymer-sync-001"
```

To process only one selected measurement, pass its directory:

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\process_synchronized.py `
  "C:\Research records\Material level étude\polymer-sync-001\measurements\000001_taking_20260814T100000000000Z"
```

Profile selection is normally automatic from the recorded material. An
optional `--profile powder` is an assertion and fails if that profile is not
configured for a recorded name. For an older session whose material was left
blank, an explicit `--profile powder` is accepted as the deliberate recovery
choice; it does not invent or backfill a material name in the common record.
Without an explicit profile, a blank material remains an error. Source paths are constrained to the session,
and every PNG is checked against its capture-manifest SHA-256 before the model
loads. The command then reuses the original experiment/measurement IDs,
timestamps, capacity, density/reference inputs, material, and notes.

The terminal reports each measurement group before processing it. A large
segmentation model processing six 1920x1080 images can take minutes, especially
on CPU; a
session command keeps the selected model loaded while it advances through all
pending groups. Interrupting is safe to retry because the processing manifest
journals incomplete work.

Completed groups are integrity-verified and skipped by default, without loading
YOLO. To deliberately reprocess with the current model/configuration:

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\process_synchronized.py `
  "C:\Research records\Material level étude\polymer-sync-001" `
  --force-reprocess
```

Forced processing preserves the prior common record and evidence beneath
`measurements/<measurement_id>/yolo_revisions/revision_NNNN/`, records old and
new configuration/model hashes, and only replaces the active YOLO row after a
successful run. `yolo_processing_manifest.json` journals validation,
processing, completion, and errors. A retry archives incomplete attempt
evidence and resumes the same ID. Missing or hash-mismatched source images fail
before inference and leave a pending row unchanged.

Every saved frame retains its diagnostic row and rejection reasons. Frames are
filtered by the majority detected-class pattern, and even one geometrically
valid retained frame may form a result; the retained count records the weaker
evidence. An exact 3-vs-3 tie selects the `material_and_empty` pattern when it
is one of the tied patterns; other ties remain invalid. A group with no valid
frame after filtering or excessive retained-frame spread becomes
`complete_invalid` with blank estimate fields. Offline bookkeeping may update shared
JSON manifests, but the depth workbook, depth CSV, depth estimates, and depth
evidence are never edited. No joined tables, differences, charts, or comparison
statistics are produced.

If a long forced session run is interrupted after some measurement groups have
completed, resume without reprocessing the earlier groups by supplying the
first unfinished index:

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\process_synchronized.py `
  "C:\Research records\Material level study\powder-sync-001" `
  --profile powder --roi-mode whole_image --force-reprocess `
  --start-measurement-index 18
```

On Windows, evidence-directory archival automatically retries short-lived
access-denied errors caused by Explorer previews, antivirus scanning, or file
indexing. If Windows continues to lock only the directory root, the workflow
copies the complete prior evidence into a temporary revision snapshot, verifies
its file list, sizes, and SHA-256 hashes, commits that snapshot, and then safely
reuses the existing directory root. Prior evidence is never cleared until its
revision copy has passed verification. If an individual file is itself locked,
close the program viewing that file and rerun from the same index.

## Operator startup and confirmation

### Select the C920 by camera name on Windows

OpenCV numeric camera index `0` is often the laptop's built-in camera. The
supplied profiles now set `camera.device_name_contains: C920`. At each camera
open, Windows DirectShow devices are enumerated again, the unique friendly name
containing `C920` is resolved to its current index, and that index is opened with
the configured backend. Moving the C920 to another USB port or changing camera
enumeration order therefore does not require editing `device_index`.

List what the application sees:

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\run_offline.py --list-cameras
```

Override the match for one run with `--camera-name-contains "C920"`. Set
`device_name_contains: null` only when deliberate legacy numeric selection is
needed. A missing or ambiguous friendly-name match fails clearly instead of
silently opening another camera.

This Windows installation defaults to `backend: dshow` because automatic
backend probing was observed to stall at camera open. DirectShow is also used
for friendly-name enumeration; `auto` and `msmf` remain configurable alternatives.

Start an operator workflow by supplying `--acquisition-mode`, or add
`--interactive` to prompt for missing fields. Before any camera is opened or
folder is processed, every value is validated and a JSON confirmation summary
is printed. Without `--yes`, the terminal asks for confirmation. Noninteractive
runs must use `--yes`. The ROI choice is independent of confirmation: when
`--roi-mode` is omitted an interactive terminal always asks; redirected or
automated commands must provide the explicit mode.

For camera modes, startup reports `STAGE MODEL_LOADING`, `MODEL_READY`,
`CAMERA_OPENING`, and `CAMERA_WARMUP`. Model task/class validation deliberately
occurs before the camera opens. A large segmentation model can take one or more
minutes to load; do not interpret an active `MODEL_LOADING` message as a
camera-selection failure. Use `--list-cameras` for an immediate camera-only
identity check and `--validate-model` to time model loading without opening a
camera.

The collected inputs are material name, experiment ID and purpose, capacity,
optional bulk density, optional total-possible weight, optional independently
measured remaining weight, optional manual level with `mm` or `cm`, operator
notes, acquisition mode, and an absolute output root. The material name is
collected once and selects the configured model profile automatically; an
optional `--profile` is only an assertion/recovery input. Manual level also needs
the usable internal height in millimetres so the shared contract can calculate
its reference percentage. Density is never selected from a material name or
profile.

Interactive startup first displays any density and total-mass values already
supplied on the command line and asks whether to keep them. If you choose to
change them, select exactly one input basis: `d` for bulk density in g/mL or
`m` for the total possible material mass of a full tube in g. The unselected
field is left blank. Neither value is inferred from the material name, and the
software does not silently derive one from the other. A total-mass-only choice
therefore cannot calculate the independent remaining-mass reference volume;
that formula requires an explicitly supplied bulk density.

### Folder mode

One image:

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\run_offline.py `
  --material-name Powder `
  --acquisition-mode folder `
  --input "C:\Study images\sample 01.png" `
  --experiment-id powder-run-001 `
  --purpose "single-image protocol check" `
  --total-capacity-ml 250 `
  --output-root "C:\Research records" `
  --yes
```

A folder, sorted deterministically by Unicode path:

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\run_offline.py `
  --material-name polymer `
  --acquisition-mode folder `
  --input "C:\Study images\capture group 001" `
  --experiment-id polymer-run-001 `
  --purpose "offline six-image repeatability" `
  --total-capacity-ml 250 `
  --remaining-weight-g 42.6 `
  --notes "operator supplied note" `
  --output-root "C:\Research records" `
  --yes
```

Folder discovery accepts `.bmp`, `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`, and
`.webp`, case-insensitively. It is nonrecursive unless `--recursive` is present,
and paths are sorted deterministically. Originals are never changed. Folder
mode accepts a geometrically valid one-image folder. The retained-frame count is
always written to YOLO diagnostics so evidence strength is not hidden.

### Manual C920 mode

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\run_offline.py `
  --material-name Powder `
  --acquisition-mode manual_camera `
  --experiment-id powder-manual-001 `
  --purpose "manual operator reading" `
  --total-capacity-ml 250 `
  --manual-material-level 42 `
  --manual-level-unit mm `
  --usable-internal-height-mm 100 `
  --output-root "C:\Research records"
```

After warm-up, press the configured capture key or left-click the preview. The
unmodified current frame is saved, acquisition closes or briefly freezes as
configured, and inference then starts on that one saved still. Quit, Escape, or
closing the window cancels without creating a measurement. Manual mode uses the
same one-retained-image rule and records a retained count of one when accepted.

### Timed C920 mode

Capture a fixed count:

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\run_offline.py `
  --material-name polymer `
  --acquisition-mode timed_camera `
  --experiment-id polymer-timed-001 `
  --purpose "six stills over one second" `
  --total-capacity-ml 250 `
  --capture-count 6 `
  --capture-interval-seconds 0.2 `
  --output-root "C:\Research records" `
  --yes
```

Or choose a start/end duration; captures occur at time zero and each interval
not exceeding the duration:

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\run_offline.py `
  --material-name polymer `
  --acquisition-mode timed_camera `
  --experiment-id polymer-duration-001 `
  --purpose "duration scheduled stills" `
  --total-capacity-ml 250 `
  --capture-duration-seconds 1.0 `
  --capture-interval-seconds 0.2 `
  --output-root "C:\Research records" `
  --yes
```

Do not pass count and duration together. Timed frames are encoded immediately
to a bounded temporary spool; no frame queue or inference worker runs during
preview. After the requested stills are captured, the camera closes and the
saved-image workflow processes them sequentially. Cancellation discards a
partial spool and writes no measurement. A per-image inference failure is
retained as an invalid diagnostic image rather than an invented estimate.

For custom programmatic integrations, the low-level saved-image API still
accepts `resume_pending_synchronized=True`. Normal operators should use
`process_synchronized.py`, which performs manifest discovery, source-integrity
checks, material/profile selection, journaling, and revision handling first.

For a guided terminal workflow:

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\run_offline.py --interactive
```

## Measurand and formula

The ROI endpoints are the calibrated full and empty limits along the configured
tube axis. After mask thresholding, the estimator computes material-only and
empty-only occupancy for every ROI row. It chooses the boundary that minimizes
material support on the empty side plus empty support on the material side.
Mask area is retained only as a diagnostic; it is never equated with volume.

When `roi_mode=whole_image`, this calibrated-ROI procedure is not claimed. The
estimator crops quality analysis to the detected semantic envelope, disables
ROI-only occupancy/column/unclassified gates, and uses a dual-class detected
vertical extent only as an explicitly uncalibrated provisional normalization.
Confidence, mask threshold, class validation, duplicate selection, interface
sanity, majority-pattern filtering, and averaging remain enabled. Interface
spread remains a saved diagnostic but never invalidates an image.

For interface fraction `q` from the full limit toward the empty limit:

```text
q = interface_distance_from_full / calibrated_full_to_empty_distance
estimated_material_percent = 100 * (1 - q)
estimated_empty_percent = 100 - estimated_material_percent
estimated_material_volume_ml = total_capacity_ml * estimated_material_percent / 100
estimated_empty_volume_ml = total_capacity_ml - estimated_material_volume_ml
```

Overlapping pixels are excluded from both semantic-only row occupancies.
Unclassified pixels stay unclassified. Neither is normalized away. Column-wise
interfaces provide the configured P90-P10 spread diagnostic. See
[`docs/METHOD.md`](docs/METHOD.md) for the full algorithm, assumptions, gates,
and limitations.

## Rejection and aggregation

An image may detect material only, Empty only, or both. For repeated detections
of one semantic role, the highest-confidence instance supplies the mask while
the full confidence list and instance count remain diagnostic. A single-class
mask is accepted only when it spans a reasonable lateral/axial fraction and
connects to the physically expected ROI end; a tiny blob is never converted to
0% or 100%. Low confidence, missing mask geometry, reversed/multiple
transitions, or inadequate row support can invalidate an image. Interface
spread remains an explicit irregular-boundary diagnostic but is not a
rejection rule. Overlap and unclassified fractions remain explicit diagnostics but do
not, by themselves, invalidate it.

All images are inferred first and classified as `material_only`, `empty_only`,
or `material_and_empty`. The majority pattern is retained; an exact 3-vs-3 tie
selects `material_and_empty` when present, while other ties remain invalid.
Disagreeing patterns receive `class_pattern_disagrees_with_majority`. Six
unanimous material-only frames use 100% and six unanimous Empty-only frames use
0%; otherwise the final percentage is the arithmetic mean of valid retained
percentages. One retained image is allowed. P90-P10 spread,
total/retained/rejected counts, patterns, individual
percentages, and all reasons remain in `yolo_image_details`, `image_details.csv`,
and `result.json`. Invalid common rows preserve estimate fields as blanks,
never zeros.

Segmentation is the normal mode. Detection weights are rejected unless the
profile explicitly pairs `expected_task: detect` with
`estimation_mode: bbox_vertical_extent`. In that limited mode, configured-role
box extents create tube-axis support; box area is not a volume proxy. The mode
is less geometrically specific and must be validated independently.

## Outputs

```text
<output_root>/<experiment_id>/
  session_manifest.json
  provenance/yolo_effective_config_<profile>.yaml
  yolo_measurements.xlsx
  yolo_measurements.csv
  measurements/<measurement_id>/yolo/
    result.json
    image_details.csv
    artifact_manifest.json
    source_images/...
    per_image/...
    overlays/...          # when configured
    masks/...             # when configured
```

The primary `measurements` sheet and CSV use the one shared column contract.
The workbook also contains `yolo_image_details`, one row per image, including
class pattern, majority pattern, retained flag, instance counts, individual
percentage, and filtering/rejection reasons. Source
images are copied byte-for-byte as evidence and originals are hash-checked
after inference; derived overlays and masks never overwrite them. Workbook/CSV
writes use a single-writer lock, temporary files, verification, and atomic
replacement where the platform permits it.

The normal overlay is intentionally compact: translucent material/Empty masks,
the frozen ROI, selected class names and confidences, the candidate interface,
and the final/candidate level when available. Long rejection messages and row
occupancy data stay in the separate JSON/CSV diagnostics rather than obscuring
the specimen image.

For synchronized sessions, lossless C920 sources are retained by every storage
profile and are never deleted by offline processing. For standalone folder
processing, this release copies source images byte-for-byte into the evidence
bundle for every profile. Per-image masks and overlays are governed by the
explicit `save_masks` and `save_overlays` settings; the recommended `research`
configuration enables both. The selected profile and both flags are frozen in
the effective configuration, so the manifest remains honest about what was
retained.

The current inference adapter does not serialize raw Ultralytics result
objects, so YOLO `full_raw` has no additional adapter-raw payload beyond the
lossless sources, masks, overlays, diagnostics, and hashes. Treat that as a
known retention limitation. D405 raw Z16 retention is controlled separately by
the depth profile and occurs only for `full_raw`.

## Configuration and future materials

`config.yaml` has one `defaults` section for the shared Ultralytics task,
inference/NMS settings, frozen ROI and axis, estimation/aggregation gates, C920
acquisition, and output policy. Each material profile normally contains only
recorded material aliases, its weights, and the material/Empty semantic names.
Use a small `overrides` mapping only when a trained model genuinely needs a
different common setting.
Camera configuration includes friendly-name match, numeric fallback index,
backend, requested width/height/FPS,
warm-up frames, count, interval or duration, capture/quit keys, window name,
preview polling, mirror behavior, and close/freeze behavior. `backend: dshow`
or `backend: msmf` is supported on Windows. The backend remains configuration,
not an unconditional platform hard-code. Relative paths resolve against the YAML,
not the process working directory. Absolute paths and paths with spaces or
accents are supported.

To add a material, copy one small profile and change its name,
`recorded_material_names`, weights, and semantic role configuration. Aliases
are exact after trimming and case-folding, and cannot be shared by two profiles.
No material-name branch is needed in Python.
The configured ROI is shared by all profiles because it describes the frozen
apparatus, not the material. The ROI tool rejects a whole-image selection and
profile-level ROI overrides. Deliberate `whole_image` runtime mode remains a
separate explicit choice and is recorded rather than presented as calibration.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q .\Material_level_using_yolo\tests
```

Tests use synthetic masks, temporary images, mock cameras, and mock adapters.
They prove preview frames are not inferred and require no camera, trained
weights, Torch execution, GUI, or network access.

Before research acquisition, complete the repository
[research checklist](../docs/material_level_study/RESEARCH_CHECKLIST.md). The
[manual camera smoke-test checklist](../docs/material_level_study/MANUAL_SMOKE_TEST.md)
is optional and is not a metrological validation.
