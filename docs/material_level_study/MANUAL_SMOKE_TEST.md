# Manual connected-camera smoke-test checklist

This procedure is optional, requires connected hardware, and is never an
automated/CI test. Passing it confirms basic operation only—not accuracy,
traceability, synchronization precision, or scientific validation.

## Before opening cameras

- [ ] Close RealSense Viewer and every application that may own the D405/C920.
- [ ] Verify the frozen mount, tube, lighting, D405 calibration file, C920 ROI,
  configured model aliases/weights, output root, and available disk space.
- [ ] Confirm the chosen material name exactly matches a configured
  `recorded_material_names` alias.
- [ ] Run the hardware-free suite and configuration checks first.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe .\3d_camera\validate_setup.py
.\.venv\Scripts\python.exe .\Material_level_using_yolo\run_offline.py --profile polymer --validate-model
```

## One manual synchronized trigger

```powershell
.\.venv\Scripts\python.exe .\Run_Experiment\run_experiment.py `
  --hardware-smoke-test `
  --experiment-id manual-smoke-001 `
  --purpose "manual connected-camera smoke test" `
  --material-name "operator supplied polymer" `
  --total-capacity-ml 211.527360357 `
  --reference-material-weight-g 48.2 `
  --manual-material-level-mm 41.0 `
  --measurement-notes "manual smoke reference" `
  --output-root "C:\Research records\Manual smoke test"
```

- [ ] Review the confirmation before cameras open.
- [ ] Confirm both devices warm successfully and the state reaches `ARMED`.
- [ ] Confirm reference values are displayed for the one-shot measurement before
  acquisition; in the continuous guided command, verify every trigger request
  prompts for fresh values and allows blanks.
- [ ] Confirm one trigger is accepted and retriggering while busy is rejected.
- [ ] Confirm six lossless C920 files span the configured interval, one D405
  burst is recorded, and the state is `SAVED` or an auditable `PARTIAL`.
- [ ] Verify the capture manifest contains target/actual times, frame hashes,
  depth interval, overlap, the entered per-measurement references, cleanup
  outcome, and no exact hardware-sync claim.
- [ ] Close acquisition and confirm both cameras can be reopened by their vendor
  tools, demonstrating cleanup.

## Deferred inference

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\process_synchronized.py `
  "C:\Research records\Manual smoke test\manual-smoke-001"
```

- [ ] Confirm all source hashes verify before inference.
- [ ] Confirm the YOLO row retains the capture measurement ID and changes from
  pending/partial to `complete_valid` or `complete_invalid`.
- [ ] Confirm invalid results retain blank common estimates and per-frame reasons.
- [ ] Confirm depth and YOLO primary headers match, diagnostics remain separate,
  and depth workbook/CSV bytes were not modified by offline processing.
- [ ] Run the same offline command again and confirm it reports an existing
  verified result without loading the model or adding a duplicate row.

Archive or delete smoke-test runtime data according to the laboratory policy;
do not treat it as research evidence unless it was collected under the approved
study protocol.
