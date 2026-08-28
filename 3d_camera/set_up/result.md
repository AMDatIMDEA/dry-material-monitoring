# D405 Camera-Placement Optimizer

## What was implemented

The independent `3d_camera` package now includes an interactive setup
optimizer:

```powershell
.\.venv\Scripts\python.exe 3d_camera\set_up\configure_height.py
```

The implementation adds:

- `3d_camera/set_up/configure_height.py`: interactive input, output-mode
  selection, reporting, acceptance,
  exact-distance evaluation, and optional saving.
- `3d_camera/set_up/setup_optimizer.py`: hardware-independent geometry, profile
  discovery, candidate evaluation, and ranking.
- `3d_camera/set_up/setup_config.py`: comment-preserving targeted YAML updates,
  timestamped backup creation, validation, and atomic replacement.
- `3d_camera/set_up/precision_test.py`: optional connected-camera flat-matte-target
  temporal repeatability and plane-quality test.
- `3d_camera/set_up/tests/test_setup_optimizer.py`: hardware-independent geometry,
  ranking, fallback, custom-distance, and configuration tests.

The existing meaning of `camera.distance_to_rim_mm` was preserved, so
`calibrate_empty.py`, `run_measurement.py`, and `validate_setup.py` retain their
current configuration interface.

## Distance definitions

All distances begin at the RealSense depth optical origin, not at the front
glass or camera housing.

- `camera_to_rim_mm` is the distance to the tube opening/rim plane.
- `camera_to_bottom_mm` is:

  ```text
  camera_to_bottom_mm = camera_to_rim_mm + usable_height_mm
  ```

- `nearest_surface_distance_mm` is the distance to the highest permitted
  material surface:

  ```text
  nearest_surface_distance_mm =
      camera_to_bottom_mm - maximum_fill_height_mm
  ```

  When maximum fill equals usable height, this equals `camera_to_rim_mm`.

- `farthest_surface_distance_mm` is normally the empty inner bottom and equals
  `camera_to_bottom_mm`.

## How the optimizer works

### 1. Collect and validate inputs

The CLI collects outer diameter, inner diameter, usable internal height,
maximum fill height, free image margin, expected centring error, maximum tilt,
camera availability, and optional mechanical mounting limits.

Dimensions must be positive, inner diameter cannot exceed outer diameter,
maximum fill cannot exceed usable height, and allowances/ranges cannot be
negative or contradictory. Ordinary input errors are shown without a
traceback.

### 2. Obtain camera profiles

If a D405 is connected, the code queries its Z16 depth stream profiles through
`pyrealsense2`. It reports the device model and serial number and reads each
profile's width, height, FPS, and calibrated `fx`, `fy`, `cx`, and `cy`.
Geometry-equivalent FPS variants are reduced to one measurement profile per
resolution, preferring 30 FPS to retain the existing 45-frame acquisition
workflow.

If hardware cannot be queried, the optimizer uses one documented offline
table containing these 30 FPS D401/D405 modes and Min-Z values:

| Resolution | Min-Z |
|---|---:|
| 1280 x 720 | 100 mm |
| 848 x 480 | 70 mm |
| 640 x 360 | 55 mm |
| 480 x 270 | 45 mm |
| 424 x 240 | 40 mm |

The offline table uses the nominal D405 87 x 58 degree FOV to construct
centred fallback intrinsics. Device-specific lens calibration can differ, so
all offline FOV, coverage, projection, and sampling results are labelled
approximate. Connected-camera intrinsics always take precedence.

The profile modes and Min-Z values come from the RealSense D400 Series Product
Family Datasheet. The configured `filters.min_distance_mm` can make the
effective near limit more conservative. The configured
`filters.max_distance_mm` is treated as the far reliable limit.

### 3. Evaluate the complete fill range

For each profile and each candidate camera-to-rim distance, every material
height from maximum fill to the empty bottom is checked at 1 mm increments.

At each plane, full visible image dimensions are calculated directly from
intrinsics. Pixel-edge spans are measured independently on both sides of the
principal point:

```text
left span   = cx + 0.5
right span  = width - 0.5 - cx
top span    = cy + 0.5
bottom span = height - 0.5 - cy
```

This accounts for an asymmetric principal point. A centred circular tube is
limited by twice the smaller edge span in each image axis.

Stereo overlap is checked separately. The D405 fallback model uses its
documented 18 mm stereo baseline. Because the depth origin is tied to one
imager, the conservative centred common width is:

```text
2 * max(0, limiting_horizontal_half_field_mm - stereo_baseline_mm)
```

The real outer tube must fit this common left/right field, not merely the
single-imager FOV.

Tube-rim/wall visibility is checked by projecting the complete inner material
disk back through the inner rim opening. The nominal centred ray bundle and a
conservative bundle using centring plus tilt-origin allowance are both
evaluated at every material height. Negative inner-rim clearance is treated as
occlusion.

The required radius at a plane includes:

1. The outer tube radius enlarged conservatively by `1 / cos(tilt)`.
2. The requested free image margin on each side.
3. Expected centring error.
4. The worst-direction lateral shift caused by tilt at that tube depth.

The tilt shift is:

```text
distance_below_rim_mm * tan(maximum_tilt_degrees)
```

Depth-range checks also include the conservative axial edge excursion of a
tilted circular plane:

```text
(outer_diameter_mm / 2) * sin(maximum_tilt_degrees)
```

The named nearest/farthest distances remain the required plane-centre
definitions; the report additionally shows the nearer/farther tilt-envelope
points used for boundary validation.

The optimizer classifies each candidate:

- `VALID` means nominal depth/FOV/stereo/rim physics and every requested
  margin, centring, and tilt allowance pass.
- `CONDITIONALLY_USABLE` means nominal physics pass, but only a requested
  conservative allowance fails.
- `PHYSICALLY_INVALID` means nominal near/far depth, real tube FOV, stereo
  overlap, projected-pixel usefulness, rim/wall visibility, or mechanical
  mounting fails.

The 40-pixel threshold is an explicit workflow heuristic, not a D405 accuracy
specification.

### 4. Calculate coverage and sampling

For each setup, the report includes:

- full visible width and height at the nearest surface and bottom;
- principal-point-centred available width and height;
- required safety diameter at both planes;
- conservative stereo-overlap width and rim/wall clearance;
- horizontal and vertical remaining margin;
- the limiting maximum supported outer tube diameter over the full range;
- physically measurable and fully safety-compliant material-height/percentage
  ranges;
- projected outer and inner diameters in X and Y pixels; and
- geometric X/Y lateral sampling at the surface and bottom.

Lateral sampling is calculated as:

```text
sampling_x_mm_per_pixel = distance_mm / fx
sampling_y_mm_per_pixel = distance_mm / fy
```

Sampling becomes coarser as distance increases.

### 5. Rank physically usable candidates

The distance search uses 1 mm camera-to-rim increments. Valid candidates are
ranked for three different objectives:

- `HIGHEST_SPATIAL_SAMPLING`: minimizes the worse X/Y mm-per-pixel value,
  with closer distance as a tie-breaker.
- `SAFEST_SETUP`: emphasizes both depth-boundary clearance and remaining
  field-of-view margin.
- `BEST_COMPROMISE`: balances projected pixels, depth clearance, and field
  clearance.

Fully `VALID` candidates are preferred. `CONDITIONALLY_USABLE` candidates are
recommended only when no fully valid combination exists. Candidates exactly on a depth boundary remain visible as valid boundary
conditions, but a candidate with at least 1 mm depth clearance is preferred
for recommendations whenever one exists.

The objectives can legitimately choose the same physical setup. The CLI makes
that duplication explicit instead of pretending that three different
solutions always exist.

### Output modes

Before results are printed, the normal interactive command asks:

```text
Choose output mode:
1. ABSTRACT – practical recommendation only
2. LONG REVIEW – complete technical evaluation
Selection [1]:
```

`ABSTRACT` is the default. Its approximately ten-line result contains only the
best distances/profile, valid mounting range, measurable fill range, projected
pixels, geometric mm/pixel, final status, one warning, and the next action.
`LONG REVIEW` contains all profiles, calculations, model assumptions, and
failed constraints.

### Hardware-free 58/55/115 mm example

With maximum fill equal to 115 mm, 10% free margin per side, 3 mm centring
allowance, 2 degree tilt allowance, the current 45-250 mm configured filter
range, and nominal offline intrinsics, the implementation reports:

| Objective | Profile | Camera to rim | Camera to bottom |
|---|---|---:|---:|
| Highest spatial sampling | 1280 x 720 @ 30 FPS | 103 mm | 218 mm |
| Safest setup | 640 x 360 @ 30 FPS | 97 mm | 212 mm |
| Best compromise | 1280 x 720 @ 30 FPS | 118 mm | 233 mm |

For the best compromise, approximate visible area is 223.96 x 130.82 mm at
the full-fill surface and 442.22 x 258.31 mm at the bottom. The limiting
supported outer diameter is 103.96 mm. Approximate X/Y sampling is
0.1750/0.1817 mm/pixel at the full-fill surface and 0.3455/0.3588 mm/pixel at
the bottom.

These numbers are fallback-model results, not connected-camera measurements.
The actual D405 intrinsics may change the recommendation.

### 59/55/115 mm tube at exactly 65 mm

The regression case uses the same documented fallback model rather than a
hardcoded exception. For 640 x 360, the nominal surface/bottom distances are
65-180 mm and the real 59 mm tube fits the horizontal, vertical, and
conservative stereo-overlap fields. Rim/wall visibility also passes. The
requested 10% margin, 3 mm centring tolerance, and 2 degree tilt envelope do
not retain positive vertical safety margin, so the computed status is
`CONDITIONALLY_USABLE`, not `PHYSICALLY_INVALID`.

### 6. Accept, override, or save

After recommendations are displayed, the user is asked whether the
best-compromise camera-to-rim distance is mechanically suitable.

If it is not suitable, the user enters an exact preferred distance. Every
profile is evaluated at that exact distance. Invalid results list the failed
constraints and nearby valid alternatives; an invalid setup is never silently
accepted.

A valid accepted setup is shown in a complete final summary. Saving requires a
separate confirmation. Until that confirmation, `config.yaml` is not changed.
On confirmation:

1. A timestamped `config.backup-<UTC timestamp>.yaml` copy is created.
2. Existing tube diameter/height and camera distance/profile scalars are
   updated without rewriting unrelated YAML content. `wall_thickness_mm` is
   derived as `(outer_diameter_mm - inner_diameter_mm) / 2`.
3. Reproducibility data is added under `setup_optimizer`, including distances,
   dimensions, profile, FPS, intrinsics/FOV source, allowances, camera source,
   model, serial number, and approximate/connected state.
4. The candidate YAML is typed-validated before atomic replacement.
5. The user is told to repeat empty-tube calibration.

## Interpreting measurement terms

`lateral_sampling_mm_per_pixel` is only geometric X/Y sample spacing. It must
not be described as camera precision.

`estimated_depth_uncertainty_mm` is reported as:

> Not available from geometry alone.

The project does not contain documented D405 empirical coefficients that map
distance/profile geometry to a defensible numerical depth-uncertainty value.

`empirical_repeatability_mm` is unavailable until the installed camera and
target are tested. The optional `--precision-test` mode captures repeated
frames of a flat matte target and reports robust frame-median repeatability,
median pixel robust standard deviation, plane residual RMS, and valid-depth
coverage. Absolute error is only reported when the user supplies a known
reference distance with `--known-distance-mm`.

Repeatability is not absolute accuracy. Real measurement quality also depends
on material texture, transparency, reflections, lighting, surface angle,
stereo matching, calibration, camera temperature, and missing depth pixels.

## Workflow summary

```text
User input
   -> validation
   -> connected D405 query or approximate fallback profiles
   -> every profile x every 1 mm mounting distance
   -> every permitted fill plane checked
   -> nominal physics separated from conservative allowances
   -> VALID / CONDITIONALLY_USABLE / PHYSICALLY_INVALID status
   -> output-mode selection
   -> physically usable candidates ranked for three objectives
   -> user accepts recommendation or tests an exact distance
   -> complete final summary
   -> optional empirical repeatability test
   -> optional confirmed backup and config save
   -> mandatory empty-tube recalibration
```

The optimizer proves only geometric and configured-range feasibility. The
installed system must still be validated with the real camera, real material,
and known fill levels before operational use.
