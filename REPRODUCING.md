# Reproducing the software workflow

This guide reproduces the software execution and evidence structure. It does
not, by itself, reproduce a scientific result: released weights, calibration,
an immutable dataset, hardware metadata, and the paper's experiment protocol
must accompany the archived software release.

## 1. Record the release

Use a tagged repository release or archive DOI. Record the Git commit, Python
version, operating system, camera firmware, RealSense SDK version, model
weight SHA-256 values, and configuration SHA-256 values. The measurement
workflow already writes provenance and artifact hashes into each session.

## 2. Create the environment

From the repository root on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r pylock.toml
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
```

`pylock.toml` is the exact Windows/Python 3.11 dependency resolution for the
`v0.1.2` beta research release. `pyproject.toml` and `requirements.txt` retain bounded
compatibility ranges for maintainers and other supported installations.

For CUDA execution, create a separate environment and install the PyTorch build
appropriate for the laboratory's driver/CUDA combination before installing the
repository from `pyproject.toml`; the default-index lock may otherwise replace
that build. The tracked YOLO
configuration uses `device: auto`: it selects `cuda:0` only when PyTorch
reports CUDA as usable, and otherwise selects CPU. For strict replication,
set `device: cpu` or a specific `cuda:<index>` and report it.

## 3. Restore non-Git research assets

Copy the archived model weights to the paths selected in
`Material_level_using_yolo/config.local.yaml`. Restore the D405 calibration and
input research session without modifying their archived bytes. Verify every
published SHA-256 checksum before processing.

Research data, weights, calibration files, and generated results are ignored
by design. Publish approved immutable assets in a research-data archive and
link its DOI from the release notes and paper rather than placing large or
sensitive artifacts in Git.

## 4. Prepare local configurations

Copy each tracked configuration to `config.local.yaml` beside it, then enter
only laboratory-specific values in the local copy. These files are ignored.
Never edit a tracked example to contain a username, serial number, or private
storage path.

## 5. Verify the implementation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe .\3d_camera\run_measurement.py `
  --config .\3d_camera\config.yaml `
  --source synthetic --fill-percent 35 --no-export
```

Hardware validation is separate from unit testing. Complete
`docs/material_level_study/MANUAL_SMOKE_TEST.md`, then the preregistered or
paper-specific validation plan before collecting reportable data.

## 6. Preserve the research record

Do not overwrite a completed session. Archive the effective configurations,
weights, calibration, raw inputs required by the selected storage profile,
tabular outputs, plots, environment report, release identifier, and checksums.
State known limitations and excluded measurements in the paper.
