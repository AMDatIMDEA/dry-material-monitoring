# Integrated architecture

## Scope

Dry-Material Monitoring is one research repository with several
method-specific boundaries. It acquires synchronized evidence, estimates
material volume by two independent computer-vision methods, stores both methods
under a common record contract, and produces descriptive comparison outputs.

The architecture preserves separation between acquisition, estimation, record
management, and analysis so that one method cannot silently alter another
method's measurements.

## Component boundaries

| Component | Owns | Must not own |
|---|---|---|
| `Run_Experiment` | Camera warm-up, operator trigger, shared identity, timing, capture manifests, C920 PNGs, D405 burst coordination. | YOLO inference, model loading, method comparison. |
| `3d_camera` | D405 capture, calibration, depth processing, visible-surface reconstruction, cylindrical volume integration, depth evidence and quality decisions. | YOLO semantics or comparison statistics. |
| `Material_level_using_yolo` | Saved-image model validation/inference, ROI/interface estimation, image aggregation, YOLO evidence and record completion. | Live synchronized inference, D405 numerical internals, comparison conclusions. |
| `experiment_records` | Common schema, measurement identity, validation, derived conservation fields, provenance, locking, atomic XLSX/CSV persistence. | Camera, model, estimator, or statistical logic. |
| `measurements_methods_analyze` | Read-only alignment of completed records, descriptive statistics, exclusions, tables, and plots. | Mutation of acquisition records or claims of validation. |

## Execution flow

1. The operator freezes the physical mount, geometry, calibration, camera
   settings, C920 ROI/axis, model weights, semantic roles, thresholds, and
   storage profile.
2. `Run_Experiment` opens and warms both cameras. A confirmed trigger allocates
   one measurement identity and records changing reference observations.
3. Two bounded workers capture the D405 burst and six C920 stills. The manifest
   records target and observed timing; this is software coordination, not
   hardware triggering.
4. Depth processing runs after both capture workers finish and writes the depth
   result/evidence through the common record contract.
5. After acquisition closes, the offline YOLO command verifies capture hashes,
   selects the profile from the recorded material, validates the model task and
   semantic names, and estimates the saved image group.
6. YOLO processing completes the pending YOLO row with the original identity.
   A forced rerun creates a preserved processing revision.
7. The analysis component opens the method tables read-only, checks shared
   fields, applies documented inclusion rules, and exports tables and plots.

## Invariants

- One confirmed trigger maps to one immutable `measurement_id` shared by the
  method records and evidence directories.
- A cancelled reference confirmation creates neither an identity nor evidence.
- Source data and effective configuration are hashed before or during use.
- Model task and `model.names` semantic roles must validate before inference.
- Invalid results keep estimate fields blank and retain rejection diagnostics.
- Material and empty percentages/volumes conserve their configured totals.
- Workbook and CSV views are written atomically and remain schema-compatible.
- Completed evidence is never silently overwritten; revisions are explicit.
- Software tests are not described as scientific or metrological validation.

## Configuration and privacy

Each component keeps its familiar `config.yaml` as a portable tracked example.
Machine-specific operational values use a sibling `config.local.yaml`, ignored
by Git. Paths are resolved relative to the selected YAML file unless the CLI
requires an explicit absolute output root.

Tracked files contain no personal username, OneDrive location, or camera serial
number. Model weights, calibrations, datasets, recordings, research sessions,
and generated outputs remain untracked and are distributed through controlled
research-data archives when approved.

## Inference device policy

YOLO inference happens only on saved stills. `device: auto` asks PyTorch whether
CUDA is usable, selects `cuda:0` when possible, and otherwise selects CPU. A
CUDA discovery error also falls back to CPU in auto mode. Explicit `cpu` or
`cuda:<index>` choices are preserved. The requested and resolved devices are
written into YOLO result provenance.

For reproducibility studies, the device should be frozen explicitly because
CPU and GPU kernels or dependency builds can produce small numerical
differences even when the algorithm is unchanged.

## Persistence and recovery

The session manifest and per-measurement capture/processing manifests are the
durable state for recovery. Evidence is written before a row is finalized.
Temporary workbook/CSV files are replaced atomically, guarded by session locks.
Interrupted offline processing can be retried; already completed groups are
integrity-checked and skipped unless an explicit revision is requested.

The `research` storage profile is the recommended scientific default. The
`full_raw` profile retains lossless D405 source bursts and can require much more
storage. No profile automatically deletes source C920 images.

## Scientific boundaries

The depth method estimates volume beneath the visible reconstructed surface; it
cannot observe hidden voids and is affected by depth coverage, material optical
properties, calibration, and mount geometry. The YOLO method estimates a
calibrated image-space interface; it depends on representative model training,
class semantics, ROI/axis, lighting, and camera geometry.

The analysis outputs describe the available validation dataset under documented
inclusion rules. Accuracy, bias, repeatability, agreement, uncertainty, and
generalization claims require a preregistered or otherwise defensible
experimental design and independent reference data.
