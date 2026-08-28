# Scientific validation plan

Passing unit tests establishes software consistency; it does not establish
measurement accuracy. There is no universal body that “scientifically approves”
this algorithm. A paper should report the method, calibration, reference standards,
error distribution, limitations, and acceptance criteria so reviewers can judge it.

## Current evidence level

The repository contains:

- exact geometry/unit tests;
- deterministic synthetic flat-surface tests;
- robust-fusion and plane-fit tests;
- configuration and calibration-compatibility gates; and
- optional connected-camera flat-target repeatability measurements.

Synthetic tests cannot model stereo correspondence failures, transparency,
specular reflection, temperature drift, tube-wall refraction, occlusion, packing
topology, or systematic camera bias. The current project is therefore a validated
software prototype, not yet a validated measurement method.

## Minimum experimental study before publication

Freeze the camera mounting, firmware, SDK version, stream profile, exposure,
filters, tube dimensions, and software revision. Archive the exact `config.yaml`,
calibration artifact hash, commit hash, environmental conditions, and reference
instrument calibration information.

Use independently established reference volumes across the intended range,
including points around the refill threshold. Include at least empty, low, middle,
and high levels and the real material classes of interest. Randomize measurement
order where practical. Repeat captures without movement to estimate repeatability,
and repeat after remounting/recalibration to expose reproducibility effects.

Report, at minimum:

- signed error and relative error at each reference level;
- mean bias with confidence interval;
- MAE and RMSE;
- repeatability standard deviation or robust equivalent;
- valid-depth/accepted-measurement rate;
- false refill and missed-refill rates around the decision threshold; and
- results stratified by material, level, lighting, and remount/calibration session.

Predefine acceptance limits from the research question before inspecting final
results. Do not tune quality thresholds and evaluate performance on the same data;
reserve independent validation runs.

## Measurement uncertainty

For a defensible uncertainty statement, construct a measurement model and budget
covering at least depth systematic error, temporal noise, tube inner-diameter and
height uncertainty, centre/tilt calibration, reference-plane residual, gridding,
missing-surface interpolation, material topology, and reference-volume uncertainty.
Propagate components and state coverage probability/factor according to the GUM.
Until that work is complete, report the software field `uncertainty_percent` only
as a heuristic variability indicator.

## Device constraints relevant to this setup

The D405 is specified for an ideal range of 7–50 cm and up to 1280x720 depth. The
D400-series datasheet lists D401/D405 minimum Z values that depend on resolution:
100 mm at 1280x720, 70 mm at 848x480, 55 mm at 640x360, 45 mm at 480x270, and
40 mm at 424x240. The configured nearest and farthest physical surfaces must remain
inside both this profile limit and the project's threshold filter interval.

The present active configuration should be reviewed with `validate_setup.py`.
Warnings are experimental blockers until deliberately resolved or justified.

## Metrology references

- [JCGM 100:2008, Guide to the Expression of Uncertainty in Measurement](https://doi.org/10.59161/JCGM100-2008E)
- [NIST Technical Note 1297](https://www.nist.gov/pml/nist-technical-note-1297)
- [RealSense D405 product specifications](https://www.realsenseai.com/products/stereo-depth-camera-d405/)
- [RealSense D400-series datasheet](https://www.intelrealsense.com/wp-content/uploads/2022/11/Intel-RealSense-D400-Series-Datasheet-November-2022.pdf)
