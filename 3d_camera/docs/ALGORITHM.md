# Algorithm and coordinate definitions

This document describes what the implementation computes. It is a technical
description, not a claim that the system is a calibrated metrology instrument.

## Measurand

The reported material volume is the bulk volume below the visible upper surface
inside a cylindrical tube. It includes inter-particle voids and cannot recover
hidden cavities or true polymer mass/solid volume.

## Coordinates and units

- Raw Z16 samples are multiplied by the connected sensor's depth scale and held
  internally in metres.
- Pixel `(0, 0)` is the centre of the top-left depth pixel. The manually selected
  ROI centre is stored in the native processed depth-frame coordinates.
- Device-specific `fx`, `fy`, `ppx`, `ppy`, distortion model, and coefficients
  are captured from the active RealSense stream. Pixel/depth deprojection follows
  the RealSense camera model.
- User-facing tube geometry and reconstructed maps use millimetres; volumes use
  millilitres, with `1 mL = 1000 mm^3`.

For an undistorted pixel `(u, v)` at depth `z`, the camera-frame point is

```text
X = (u - ppx) z / fx
Y = (v - ppy) z / fy
Z = z
```

Brown-Conrady profiles are undistorted before applying this equation. The active
D405 calibration observed during development reports Brown-Conrady with zero
coefficients, but the implementation does not assume that all devices do.

## Empty-tube calibration

1. Capture and filter a static burst of a flat matte reference at the inner bottom.
2. Apply a per-pixel median/MAD temporal fusion.
3. Restrict candidates to the configured circular interior around the manual centre.
4. Fit a plane by SVD with iterative median-centred MAD trimming.
5. Intersect the selected centre ray with the fitted reference plane.
6. Correct for a configured reference-target thickness.
7. Define the tube Z axis as the plane normal toward the camera and construct two
   orthonormal in-plane axes.

Calibration records the full intrinsics, distortion metadata, tube dimensions,
plane residual, centre, and camera-to-rim consistency check. Measurement rejects
calibration captured with incompatible intrinsics, dimensions, or centre.

## Measurement reconstruction

1. Apply the configured RealSense threshold, disparity-domain spatial/temporal,
   and optional hole filters.
2. Fuse the burst by a median and median-absolute-deviation inlier rule.
3. Deproject valid depth pixels and transform them into the calibrated tube frame.
4. Reject points outside the physical height interval and the reliable inner radius.
5. Bin surface heights into an X/Y grid using the median per occupied cell.
6. Replace isolated spatial outliers with a local median.
7. Assign each remaining in-circle missing cell the value of its nearest
   cleaned, observed cell from the reliable region. This nearest-neighbour
   imputation covers both internal gaps and the excluded wall annulus; it is not
   a separately fitted radial or wall-surface extrapolation.
8. Clip reconstructed height to `[0, usable_height]` and integrate over the known
   circular cross-section.

With circular cross-section `A` and surface height `h(x,y)`, the target integral is

```text
V = integral_A h(x,y) dA
```

The code evaluates this with an equal-area Cartesian midpoint grid and scales the
mean in-circle height by the exact configured cylinder capacity. Constant-height
surfaces are therefore exact apart from depth/calibration error; nonuniform
surfaces carry grid and interpolation error.

Surface coverage and maximum internal-hole radius are calculated only within
the reliable radius, before the excluded wall annulus is filled. Consequently,
the annulus contributes imputed values to the final integral but is neither
claimed as directly measured surface nor counted as an internal data hole.

## Quality and decision logic

A result is invalid when coverage, temporal validity, internal-hole size, or the
weighted quality score fails its configured gate. Invalid results never issue a
refill decision. A valid low result must repeat for the configured number of
confirmations; a higher clear threshold supplies hysteresis.

The quality-score weights and the JSON field `uncertainty_percent` are engineering
heuristics. The latter is retained for schema compatibility but represents a
variability indicator made from temporal MAD, plane RMS, grid sampling, and a
missing-data penalty. It is **not** a standard or expanded measurement uncertainty.

## Primary technical references

- [RealSense D405 product specifications](https://www.realsenseai.com/products/stereo-depth-camera-d405/)
- [RealSense D400-series datasheet](https://www.intelrealsense.com/wp-content/uploads/2022/11/Intel-RealSense-D400-Series-Datasheet-November-2022.pdf)
- [RealSense SDK projection model](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)
- [RealSense SDK post-processing filters](https://dev.realsenseai.com/docs/post-processing-filters/)
