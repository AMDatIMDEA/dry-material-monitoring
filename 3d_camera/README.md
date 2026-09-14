# D405 Material-Level and Bulk-Volume Measurement

This component of Dry-Material Monitoring uses the Intel RealSense
D405 depth stream to reconstruct the visible top surface of material inside a
known cylindrical tube, integrate the occupied bulk volume, calculate the
remaining empty volume, and raise a configurable refill warning.

> **Restricted component:** [`LICENSE`](LICENSE) requires prior written
> permission to use, copy, modify, or distribute this directory. The root MIT
> license does not override these terms. See the repository
> [`LICENSES.md`](../LICENSES.md) for the complete boundary.

No custom dataset or trained AI model is required for this fixed, known tube geometry. The depth measurement itself is the useful signal; RGB is saved only as a diagnostic snapshot.

## Research status

The numerical pipeline has hardware-free tests and explicit quality gates, but it
is not yet a validated metrological method. Read [the algorithm definition](docs/ALGORITHM.md)
and [experimental validation plan](docs/VALIDATION.md) before citing measurement
accuracy or uncertainty. Repository-wide citation metadata is maintained in
[`../CITATION.cff`](../CITATION.cff); add the release and paper DOIs before
publication.

## Interactive camera-placement optimizer

Run the placement optimizer from the repository root:

```powershell
.\.venv\Scripts\python.exe .\3d_camera\set_up\configure_height.py
```

It asks for the outer and inner tube diameters, usable internal height, maximum
fill height, image margin, expected centring error, maximum tilt, camera
availability, and optional mechanical mounting limits. It searches the
profile/distance combinations at 1 mm camera-to-rim increments and reports up
to three objectives: `HIGHEST_SPATIAL_SAMPLING`, `SAFEST_SETUP`, and
`BEST_COMPROMISE`. The optimizer checks the complete 0-100% requested filling
range, profile Min-Z, configured far depth, horizontal/vertical field,
principal-point asymmetry, stereo overlap, tube-rim/wall visibility,
projected pixels, and the requested margin/centring/tilt envelope.

Before results are displayed, choose:

```text
Choose output mode:
1. ABSTRACT – practical recommendation only
2. LONG REVIEW – complete technical evaluation
Selection [1]:
```

`ABSTRACT` is the default and contains only the practical recommendation in
about ten lines. `LONG REVIEW` shows every retained profile, calculation,
assumption, constraint, and warning.

The complete implementation model and assumptions are documented in
[`set_up/result.md`](set_up/result.md).

Results use three statuses:

- `VALID`: nominal physics and every requested safety allowance pass.
- `CONDITIONALLY_USABLE`: the real tube and nominal depth range fit, but a
  conservative requested margin, centring tolerance, or tilt envelope fails.
- `PHYSICALLY_INVALID`: nominal FOV/depth, projected pixels, stereo overlap,
  rim/wall visibility, or the mechanical range prevents complete measurement.

The optimizer uses these exact distance definitions:

- `camera_to_rim_mm`: distance from the D405 depth optical origin to the tube
  opening/rim plane. The legacy `camera.distance_to_rim_mm` configuration key
  already has this meaning.
- `camera_to_bottom_mm`: `camera_to_rim_mm + usable_height_mm`.
- `nearest_surface_distance_mm`: distance to the maximum allowed material
  surface. It equals `camera_to_rim_mm` when maximum fill equals usable height.
- `farthest_surface_distance_mm`: distance to the empty inner bottom; normally
  equal to `camera_to_bottom_mm`.

When a D405 is connected, the tool queries its Z16 depth profiles and uses the
actual `fx`, `fy`, `cx`, and `cy` intrinsics. It reports the device model and
serial number and accounts for an asymmetric principal point. Offline mode
uses the documented D401/D405 modes and profile Min-Z values with a nominal
87 x 58 degree FOV. Every offline coverage and sampling result is marked
approximate; it is not a substitute for connected-camera validation.

The optimizer never edits `config.yaml` merely by evaluating a tube. After the
user accepts a valid result, it asks separately whether to save. A confirmed
save creates a timestamped backup, updates only the relevant existing
tube/camera keys (including wall thickness derived from the entered outer and
inner diameters), adds reproducibility metadata under `setup_optimizer`, and
warns that empty-tube calibration must be repeated.

For a non-interactive hardware-free example:

```powershell
.\.venv\Scripts\python.exe .\3d_camera\set_up\configure_height.py --example
.\.venv\Scripts\python.exe .\3d_camera\set_up\configure_height.py --example --output-mode long
```

An optional installed-camera repeatability check is available with
`--precision-test`. Place a flat matte target in the central image area. The
test reports robust temporal repeatability, median pixel robust standard
deviation, plane residual RMS, and valid-depth coverage. It reports absolute
error only when `--known-distance-mm` supplies a traceable reference distance.

### Sampling, repeatability, and accuracy are different

- `lateral_sampling_mm_per_pixel` is pinhole geometry at a stated distance. It
  describes X/Y sample spacing, not depth precision.
- `estimated_depth_uncertainty_mm` is reported as "Not available from geometry
  alone" because this project contains no documented D405 uncertainty
  coefficient model.
- `empirical_repeatability_mm` comes from repeated observations of the real
  installed camera and target. Repeatability still is not absolute accuracy.
- Absolute accuracy requires known reference distances/fill levels and a
  controlled validation procedure.

Real performance also depends on material texture, transparency, reflections,
lighting, surface angle, stereo matching, calibration, camera temperature, and
missing depth pixels. The optimizer establishes geometric feasibility; the
complete system must still be validated at known fill levels.

## Important setup facts

The configured dimensions give the nominal capacity:

\[
V = \pi r^2 H
\]

The volume algorithm always uses the explicit `inner_diameter_mm`; outer
diameter and wall thickness are checked for consistency but do not override
that measured value. Verify all dimensions with calipers.
`usable_height_mm` must be the **internal fillable height**, not necessarily the
external tube height.

Do not select a profile from Min-Z alone. The real tube must fit vertically,
horizontally, and inside the common stereo field; the open rim must not occlude
the measurable inner surface; and the empty bottom must stay within
`filters.max_distance_mm`. Run `set_up/configure_height.py` for the actual tube
instead of treating a previously used distance/profile as universal.

## How the algorithm works

1. **Empty-tube calibration**
   - Capture a static burst of the empty tube.
   - Fuse frames with a temporal median and median-absolute-deviation (MAD) outlier rejection.
   - Fit a robust 3D plane to a flat matte reference at the true inner bottom.
   - Use this plane, the configured tube center, and the known internal height to define a tube coordinate frame: X/Y across the bottom and Z upward toward the camera.

2. **Depth acquisition**
   - Read the actual depth scale and intrinsics from the D405 SDK; nothing is hard-coded.
   - Preserve the native depth resolution.
   - Apply configurable RealSense threshold, disparity-domain spatial, and temporal filters.
   - Do not use aggressive SDK hole filling by default because it can invent wrong depths.
   - Capture 45 frames (about 1.5 seconds at 30 FPS), reject zeros/NaN/infinity, and fuse the burst.

3. **Transparent-wall rejection**
   - Reconstruct all valid pixels into camera XYZ coordinates.
   - Transform points into the calibrated tube frame.
   - Keep only physically possible points inside the inner cylinder.
   - Exclude a configurable 2 mm annulus next to the transparent wall, where refraction and stereo mismatches are most likely.

4. **3D surface map**
   - Bin the surface into a 1 mm X/Y grid.
   - Use a median height per grid cell.
   - Replace isolated spatial spikes with a local median.
   - Assign missing in-circle cells the value of the nearest cleaned, observed
     cell in the reliable region. This nearest-neighbour imputation fills
     internal gaps and the excluded wall annulus; it is not a fitted radial
     extrapolation.
   - Compute direct coverage and maximum internal-hole radius only inside the
     reliable region, before filling. The excluded annulus is therefore not
     counted as measured surface or as an internal hole.
   - Save the map as NumPy, PNG, and PLY files.

5. **Volume and decision**
   - Integrate surface height over the known circular cross-section:

     \[
     V_{material} = \iint_A h(x,y)\,dA
     \]

   - Calculate `empty volume = tube capacity - material volume`.
   - A single result below 10% is marked low. Two consecutive low results activate the refill warning. The alarm clears above 12% to avoid rapid on/off switching near 10%.
   - If surface coverage, temporal validity, calibration quality, or hole size is unacceptable, the result is marked invalid and **no refill decision is issued**.

## Physical setup

1. Rigidly mount the D405 above the open tube. Keep the optical axis close to the tube axis and the opening centered.
2. Use diffuse, stable lighting. Avoid direct reflections on the transparent wall or glossy material.
3. Do not put a transparent lid between the camera and the material.
4. For calibration, put a flat matte disk at the inner bottom. A permanently installed, chemically compatible matte liner is even more repeatable.
5. If the calibration disk is removable and has measurable thickness, enter it as `calibration.reference_target_thickness_mm`. If it stays permanently, measure `usable_height_mm` from its top surface to the fill limit and set the reference thickness to zero.
6. Do not move the tube or camera after calibration. Recalibrate after any movement, focus/profile change, or tube replacement.

## Run from PowerShell in Visual Studio Code

Install the complete system once from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe .\3d_camera\validate_setup.py `
  --config .\3d_camera\config.local.yaml
```

Dependency and build metadata are maintained only in the root `pyproject.toml`.

### Configure the detection-circle centre

Close Intel RealSense Viewer, connect the D405, and run this from the repository root:

```powershell
.\.venv\Scripts\python.exe .\3d_camera\set_up\center\configure_center.py `
  --config .\3d_camera\config.local.yaml
```

Enter the inner measurement-circle diameter in millimetres and its required free
image margin per side, expressed as a percentage of the circle diameter. The
centre session always requests the D405 depth profile at
`1280x720@30`. In the preview, left-click the desired centre and press `Enter`;
press `S` to cycle through the depth map, left infrared imager, and normal camera
view, right-click to reset the point, or `Esc` to cancel without saving. All views
use the depth-frame dimensions and the normal view is aligned to depth, so centre
coordinates remain in the measurement pipeline's native depth coordinate system.
The accepted configuration saves `1280x720@30` as the measurement depth profile.
The displayed physical circle is projected at
the local depth beneath the selected point; if that patch has no valid depth,
the configured empty-bottom/reference distance is used as a fallback.
The tool lets you repeat the complete selection, asks for final confirmation,
then creates a timestamped `config.yaml` backup before an atomic update. Run the
empty-tube calibration again after changing the centre or diameter.

### Test without the camera

```powershell
.\.venv\Scripts\python.exe .\3d_camera\run_measurement.py --config .\3d_camera\config.yaml --source synthetic --fill-percent 35
.\.venv\Scripts\python.exe .\3d_camera\run_measurement.py --config .\3d_camera\config.yaml --source synthetic --fill-percent 5 --continuous --max-measurements 2
```

The second command demonstrates the two-measurement refill confirmation.

### Calibrate the real empty tube

Close Intel RealSense Viewer first and run:

```powershell
.\.venv\Scripts\python.exe .\3d_camera\calibrate_empty.py --config .\3d_camera\config.local.yaml
```

The command asks whether to create a configured virtual surface or run the
existing automatic empty-tube calibration. The virtual option captures one frame
only to read the active depth intrinsics; it does not inspect or fit the physical
bottom. It creates a camera-aligned plane at
`camera.distance_to_rim_mm + tube.usable_height_mm` and therefore assumes the
tube axis is parallel to the camera optical axis. The automatic option retains
the existing workflow: empty the tube and install a flat matte bottom reference.

The calibration is saved at `calibration\empty_tube_calibration.npz`. Recalibration backs up the previous file as `empty_tube_calibration.previous.npz`.

### Measure once

```powershell
.\.venv\Scripts\python.exe .\3d_camera\run_measurement.py --config .\3d_camera\config.local.yaml
```

### Save a session-aware research measurement

Session mode is opt-in: supply an experiment ID. From the parent repository
root, a hardware-free example is:

```powershell
.\.venv\Scripts\python.exe .\3d_camera\run_measurement.py --source synthetic --fill-percent 35 --experiment-id polymer-study-001 --purpose "repeatability pilot" --material-name "operator supplied name" --storage-profile research
```

Without `--experiment-id`, the two existing commands keep their original
timestamped `output` behavior and numerical path. Session metadata is always
non-interactive, which allows a later acquisition orchestrator to supply it.
Use `--output-root C:\absolute\path\with spaces` to place sessions outside the
default `3d_camera\research_records` directory. A custom root must be absolute;
Unicode path names are supported.

Each session has this layout:

```text
<output_root>/<experiment_id>/
|-- session_manifest.json
|-- effective_config.yaml
|-- depth_measurements.xlsx
|-- depth_measurements.csv
|-- provenance/
|   |-- depth_effective_config.yaml
|   |-- environment.json
|   `-- calibration/...
`-- measurements/<measurement_id>/
    |-- capture_manifest.json
    `-- depth/
        |-- result.json
        |-- depth_diagnostics.json
        |-- artifact_manifest.json
        `-- profile-dependent evidence
```

The primary Excel sheet is the method-neutral common record contract and uses
`method=depth`. Depth-only coverage, temporal MAD, hole radius, plane RMS,
minimum/mean/maximum level, refill state, capture settings, and raw-artifact
metadata remain in `depth_diagnostics.json`. The legacy
`quality.uncertainty_percent` is recorded as a heuristic variability indicator;
it is not GUM uncertainty.

Evidence profiles are frozen per session:

- `compact`: result JSON, height-map PNG, diagnostics, manifests, hashes, and
  configuration/calibration references. It omits large reconstructable arrays.
- `research` (default): compact evidence plus the height-map NPY, PLY surface,
  optional color snapshot, immutable configuration snapshot, calibration copy
  or reference with hash, camera/stream/settings metadata, environment versions,
  timestamps, quality reasons, and artifact hashes.
- `full_raw`: research evidence plus every lossless source Z16 depth frame in a
  compressed NPZ. This can be large and fails explicitly if a source cannot
  provide the raw frames; it never silently substitutes processed frames.

The reusable one-measurement interface is
`material_volume.session.PreparedDepthMethod`. Construct it after opening and
warming the camera source. For synchronized acquisition, call `reserve()` once
to allocate the shared ID and evidence directories, then call `capture()` in
the acquisition worker and `process(..., reservation=reservation)` after the
barrier completes. An external orchestrator that owns the shared
`capture_manifest.json` must also pass `manage_capture_manifest=False` so the
depth method cannot replace it. `run_one()` remains the standalone convenience
path. The source owner remains responsible for startup, warm-up, and cleanup.

Session evidence supports audit and later research; it does not establish
scientific or metrological validation. No YOLO inference or method comparison is
performed by this depth interface.

### Measure every five seconds

```powershell
.\.venv\Scripts\python.exe .\3d_camera\run_measurement.py --config .\3d_camera\config.local.yaml --continuous
```

Stop with `Ctrl+C`.

### Replay a RealSense recording

```powershell
.\.venv\Scripts\python.exe .\3d_camera\calibrate_empty.py --config .\3d_camera\config.local.yaml --source bag --bag D:\path\to\empty_tube.bag
.\.venv\Scripts\python.exe .\3d_camera\run_measurement.py --config .\3d_camera\config.local.yaml --source bag --bag D:\path\to\filled_tube.bag
```

The recording must use a stream profile compatible with `config.yaml`.

### Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q .\3d_camera\tests .\3d_camera\set_up\tests .\3d_camera\set_up\center\tests
```

All tests use synthetic data and require no camera.

## Output

Each accepted measurement creates a timestamped folder under `output` containing the enabled artifacts:

- `result.json`: volumes, fill percentage, warning state, and quality metrics.
- `height_map_mm.npy`: metric 3D surface height grid in millimetres.
- `height_map.png`: top-view height visualization; missing edge/gap cells were
  filled from the nearest cleaned, observed reliable-region cell.
- `material_surface_mm.ply`: 3D surface in tube coordinates, suitable for MeshLab or CloudCompare.
- `color_snapshot.png`: latest diagnostic color image when enabled.

## Most useful configuration fields

| YAML field | Meaning |
|---|---|
| `tube.inner_diameter_mm` | Actual usable internal diameter; directly controls capacity. |
| `tube.usable_height_mm` | Actual inner bottom-to-fill-limit height. |
| `tube.wall_edge_exclusion_mm` | Transparent-wall annulus excluded from direct depth. |
| `camera.distance_to_rim_mm` | Camera optical origin to tube rim/maximum-fill plane. |
| `camera.depth_width`, `depth_height`, `fps` | Requested RealSense depth profile. |
| `camera.visual_preset` | D400 visual-preset number; default 3 requests High Accuracy when supported. |
| `fusion.burst_frames` | Number of static frames combined per result. |
| `calibration.center_x_px`, `center_y_px` | Legacy tube-centre fields, synchronized by the centre setup tool. |
| `detection_roi.*` | Authoritative manual centre, physical diameter, safety margin, and native depth-stream coordinate metadata; a null centre uses the legacy fields or principal point. |
| `reconstruction.grid_cell_size_mm` | X/Y resolution of the exported 3D height map. |
| `reconstruction.minimum_surface_coverage` | Minimum direct measured grid coverage for a valid decision. |
| `decision.refill_threshold_percent` | Low-material threshold; default 10%. |
| `decision.confirmations_required` | Consecutive low measurements before the warning activates. |
| `runtime.update_interval_seconds` | Update period; default 5 seconds. |
| `research.directory` | Default root for opt-in named research sessions. |
| `research.storage_profile` | Default evidence policy: `compact`, `research`, or `full_raw`. |

## Practical limitations

- This measures **bulk occupied volume under the visible top surface**. For beads, it includes air gaps between beads. It does not measure true solid polymer volume or mass.
- A depth camera cannot guarantee valid depth from every transparent, very dark, glossy, or textureless polymer. The quality gate is therefore mandatory.
- Cropping the wall avoids measuring through transparent plastic; the algorithm observes the material through the open top.
- A strongly conical pile, deep central cavity, or hidden void can only be estimated from the visible surface. Multiple viewpoints would be needed to measure hidden geometry.
- Accuracy depends more on real geometry, camera rigidity, surface depth quality, and calibration than on adding an AI model.
- The JSON field `quality.uncertainty_percent` is a backward-compatible heuristic variability indicator, not a GUM-compliant measurement uncertainty.
- RGB and depth are not combined in this version. Volume stays in the native depth frame to avoid alignment resampling. If a future YOLO mask is added, align depth and color before applying that mask.

Before using the 10% warning operationally, validate the complete setup with known fill levels such as 0%, 5%, 10%, 25%, 50%, 75%, and 100%. Repeat each condition, record the error distribution, and adjust coverage/filter thresholds only from those measurements.

## File layout

```text
./
├── config.yaml
├── calibrate_empty.py
├── run_measurement.py
├── validate_setup.py
├── set_up/
│   ├── configure_height.py
│   ├── setup_optimizer.py
│   ├── setup_config.py
│   ├── precision_test.py
│   ├── result.md
│   └── tests/
│       └── test_setup_optimizer.py
├── material_volume/
│   ├── calibration.py
│   ├── capture.py
│   ├── config.py
│   ├── export.py
│   ├── geometry.py
│   ├── models.py
│   ├── pipeline.py
│   ├── processing.py
│   ├── synthetic.py
│   ├── session.py
│   └── volume.py
└── tests/
```

## Technical references

- [RealSense D405 product page](https://realsenseai.com/products/stereo-depth-camera-d405/)
- [D400-series datasheet](https://www.intelrealsense.com/wp-content/uploads/2022/11/Intel-RealSense-D400-Series-Datasheet-November-2022.pdf)
- [RealSense post-processing filters](https://dev.realsenseai.com/docs/post-processing-filters/)
- [Official librealsense measurement example](https://github.com/realsenseai/librealsense/blob/master/examples/measure/rs-measure.cpp)
- [JCGM Guide to the Expression of Uncertainty in Measurement](https://doi.org/10.59161/JCGM100-2008E)
