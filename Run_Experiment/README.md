# Synchronized C920 and D405 laboratory acquisition

`Run_Experiment` keeps one C920 and one D405 open and warmed, then creates one
shared measurement per operator key press or left mouse click. It captures six
lossless C920 stills across a configured one-second span while the D405 acquires
one normal configured depth burst. Depth processing starts only after both
capture workers finish. YOLO is neither imported nor run by this project.

This is software synchronization, not hardware triggering. The manifest records
the intended schedule, actual timestamps, errors, intervals, and overlap; it
does not claim exact synchronization or scientific validation.

## Install

From the repository root, using the supplied environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

The orchestrator imports only `material_level_yolo.camera`, whose module has no
Ultralytics, model, or inference imports. It reuses `PreparedDepthMethod` for the
existing D405 burst and numerical algorithm.

## Configuration

[`config.yaml`](config.yaml) resolves relative paths against its own directory.
It defines:

- D405 config and calibration paths;
- C920 Windows friendly-name match (`C920` by default), numeric fallback index,
  portable/backend selection, resolution, FPS, warm-up,
  trigger/quit keys, status window, polling, and preview mirroring;
- C920 frame count, acquisition span, exactly two bounded workers, and monotonic
  scheduler polling;
- default absolute-resolved output root and depth evidence profile.

The default targets are six frames over one second:

```text
0.0, 0.2, 0.4, 0.6, 0.8, 1.0 seconds
```

This Windows installation sets `c920.backend: dshow` explicitly to avoid slow
automatic backend probing. The setting remains configurable for another system.
The configured `c920.device_name_contains` is resolved at every open, so the
C920 may move between USB ports without relying on a fixed index. Use
`--c920-device-name-contains C920` as a CLI override.

## Operator workflow

The CLI validates and prints all metadata before opening either camera. Without
`--yes`, it asks for confirmation. It then reports:

```text
INITIALIZING -> ARMED -> CAPTURING -> PROCESSING -> SAVED -> ARMED
                                \-> PARTIAL -----/
                                \-> ERROR -------/
```

While `ARMED`, press the configured capture key or left-click the C920 window.
That action requests a measurement but does not yet allocate an ID or start
either camera acquisition. The terminal asks for a new real material-left
weight, manual human level, and optional notes for that measurement. Each may
be left blank. Review the short summary, then press Enter to capture or type
`c` to cancel. Cancellation leaves the system `ARMED` and creates no directory,
workbook row, measurement ID, or camera capture. Quit/Escape/window close
requests a graceful stop. A trigger received while busy is rejected before a
second ID or directory is allocated.

```text
Measurement 12 reference inputs
Real material left weight [g] (blank = unavailable): 54.7
Manual human material level [mm] (blank = unavailable): 41.5
Measurement notes (optional): slight slope visible
Reference data for measurement 12:
  Weight:       54.7 g
  Manual level: 41.5 mm
  Notes:        slight slope visible
Press Enter to capture, or type c to cancel:
```

Continuous guided example (the values here are session constants):

```powershell
.\.venv\Scripts\python.exe .\Run_Experiment\run_experiment.py `
  --experiment-id polymer-sync-001 `
  --purpose "C920-D405 synchronized repeatability" `
  --material-name "operator supplied polymer" `
  --total-capacity-ml 211.527360357 `
  --bulk-density-g-per-ml 0.52 `
  --total-possible-weight-g 109.994227386 `
  --operator-notes "mount and lighting frozen for laboratory run 1" `
  --output-root "C:\Research records\Material level étude" `
  --yes
```

Guided entry:

```powershell
.\.venv\Scripts\python.exe .\Run_Experiment\run_experiment.py --interactive
```

Interactive startup displays the currently supplied bulk density and total
possible full-tube material mass (or `unavailable`) and asks whether to keep
them. If you choose to change them, select exactly one input basis: `d` for
bulk density in g/mL or `m` for total possible material mass in g. The other
field is left blank; it is not inferred from the material name or silently
derived. Consequently, choosing total mass alone does not enable the
remaining-mass-to-volume reference formula, which requires an explicit bulk
density.

For scripted acquisition of exactly one measurement, the existing CLI values
are explicitly one-shot and require `--once` (or `--hardware-smoke-test`):

```powershell
.\.venv\Scripts\python.exe .\Run_Experiment\run_experiment.py `
  --once --experiment-id polymer-sync-001 `
  --purpose "scripted single measurement" `
  --material-name "operator supplied polymer" `
  --total-capacity-ml 211.527360357 `
  --bulk-density-g-per-ml 0.52 `
  --reference-material-weight-g 48.2 `
  --manual-material-level-mm 41.0 `
  --measurement-notes "slight slope visible" `
  --output-root "C:\Research records\Material level study" --yes
```

Programmatic callers pass a new `MeasurementReferenceInputs` object to each
`SynchronizedExperiment.trigger(...)` call. Former reference fields on
`ExperimentInputs` remain compatible for the first trigger only and are never
reused on subsequent triggers.

Relevant CLI overrides include:

```text
--depth-config --depth-calibration
--c920-device-index --c920-backend
--c920-width --c920-height --c920-fps --c920-warmup-frames
--frame-count --span-seconds --output-root
```

Capacity must match the frozen depth tube geometry. Density is never inferred
from the material name. A measurement's reference and manual inputs remain
blank unless the operator explicitly supplies them for that trigger. The prior
measurement's inputs never become defaults. Existing shared-record formulas
apply independently to every row:

```text
reference_volume_from_weight_ml = reference_material_weight_g / bulk_density_g_per_ml
manual_material_percent = 100 * manual_material_level_mm / usable_height_mm
reference_volume_from_manual_ml = total_capacity_ml * manual_material_percent / 100
```

Each derived field remains blank when any required input is unavailable.

## Persistence and recovery policy

One trigger produces:

```text
<output_root>/<experiment_id>/
  session_manifest.json
  depth_measurements.xlsx
  depth_measurements.csv
  yolo_measurements.xlsx
  yolo_measurements.csv
  measurements/<measurement_id>/
    capture_manifest.json
    c920/raw/
      c920_000001_<UTC>.png
      ... six lossless images ...
    depth/
      ... existing depth evidence ...
```

The depth reservation allocates the monotonic measurement index and stable ID
only after the operator confirms the per-measurement references and immediately
before acquisition. `capture_manifest.json` records those raw inputs and is
atomically written as `ALLOCATED`, updated during frame persistence, and finalized as `SAVED` or
`PARTIAL`. A crash therefore leaves durable frame paths/hashes and enough state
to inspect or resume without silently reusing the ID.

The synchronization section records:

- one UTC trigger and monotonic trigger value;
- acquisition barrier time and trigger-to-barrier delay;
- each C920 target UTC/monotonic time, actual UTC/monotonic time, timing error,
  dimensions, path, and SHA-256;
- D405 monotonic and UTC acquisition start/end;
- computed interval overlap, partial/failure status, and the explicit statement
  that exact hardware synchronization is not claimed.

Both workbooks use the identical shared 32-column `measurements` contract. The
depth row is completed normally. A complete C920 group gets a YOLO
`pending_offline_inference` row; a partial group gets `partial_capture`; zero
frames get `acquisition_failed`. Pending/partial estimate fields and `valid`
remain blank. No zero placeholders or comparison columns are written.
Both rows receive the same per-measurement weight, manual level, derived
reference values, and notes under the same measurement ID. Deferred YOLO
processing preserves them while replacing the pending YOLO result.

The saved C920 paths and pending row are the offline handoff. The YOLO API
supports an explicit `resume_pending_synchronized=True` request with the same
measurement ID/index and capture timestamps; it creates only the missing
`measurements/<id>/yolo` evidence and idempotently replaces the pending common
row after inference. A persisted partial group with at least one frame may use
the same explicit handoff, but no missing frame is synthesized: the retained
count records that weaker evidence, while a tied class-pattern majority or no
geometrically valid frame produces a clearly invalid result. A zero-frame
failure requires a new synchronized acquisition and a new measurement ID.
`Run_Experiment` itself never performs inference.

Process the resulting session later, after acquisition has exited or from a
separate PowerShell process:

```powershell
# Optional apparatus ROI setup (repeat after reframing)
.\.venv\Scripts\python.exe .\Material_level_using_yolo\configure_roi.py `
  --image ".\Run_Experiment\research_records\polymer-sync-001\measurements\<measurement-id>\c920\raw\<frame>.png"
```

Select only the useful internal tube rectangle. This setup is shared across
material profiles and does not run during synchronized acquisition. The later
offline command asks whether to use this configured ROI or deliberately process
the whole image. For automation, pass `--roi-mode configured` or
`--roi-mode whole_image`.

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\process_synchronized.py `
  "C:\Research records\Material level étude\polymer-sync-001"
```

The offline command verifies capture hashes, selects the configured profile
from the recorded material name, retains the shared ID, and updates only YOLO
records plus non-destructive manifest bookkeeping. See the YOLO README for
selected-measurement and forced-revision commands.

For a captured session whose material name was left blank, select the intended
model explicitly. This preserves the blank material field while processing all
pending measurements:

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\process_synchronized.py `
  ".\Run_Experiment\research_records\1" `
  --profile powder
```

For future sessions, enter a configured material alias such as `Powder` or
`white powder` so profile selection is automatic. At the frozen-capacity prompt,
press Enter to accept the calibrated value; the prompt displays enough digits
for the shown value to be safely re-entered.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q .\Run_Experiment\tests
```

Tests use fake prepared cameras and no GUI, weights, Ultralytics, or hardware.
They verify timing, shared IDs, workbooks, manifests, partial/failure paths,
cleanup, busy rejection, Unicode paths, and isolated imports with no YOLO
inference module.

## Optional hardware smoke test

This command is manual and is never collected by automated tests. It opens and
warms both real cameras, performs one confirmed synchronized acquisition, saves
the evidence, and exits:

```powershell
.\.venv\Scripts\python.exe .\Run_Experiment\run_experiment.py `
  --hardware-smoke-test `
  --experiment-id hardware-smoke-001 `
  --purpose "manual hardware smoke test" `
  --total-capacity-ml 211.527360357 `
  --output-root "C:\Research records\Hardware smoke" 
```

Verify the configured depth calibration, camera mounting, tube geometry, and
output path before approving the confirmation summary.

This is software-timed overlap, not hardware synchronization: operating-system
scheduling, camera buffering, exposure, and USB contention remain sources of
timing error and are recorded rather than hidden. Before research acquisition,
complete the repository [research
checklist](../docs/material_level_study/RESEARCH_CHECKLIST.md) and use the
optional [manual smoke-test
checklist](../docs/material_level_study/MANUAL_SMOKE_TEST.md) for connected
cameras. Passing either checklist does not establish metrological validity.
