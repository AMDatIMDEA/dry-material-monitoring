# Multi-tube material measurement

A retained component of the Polymer Specimen Vision System that converts
Ultralytics YOLO segmentation results into explainable fill measurements for
images containing multiple transparent tubes. The detector is expected to
produce two region classes—`Empty` and `Polymer`/`Material`—rather than a
separate tube class.

The system supports images, image folders, and videos; writes annotated media plus JSON and CSV; and rejects incomplete or ambiguous masks instead of silently reporting false 0% or 100% measurements.

## Design analysis

The central difficulty is not the area formula. It is deciding which independently segmented regions belong to the same physical tube and deciding whether a lone class mask describes a truly full/empty tube or a missed detection.

This implementation uses an enhanced version of the suggested geometric baseline:

1. Normalize configured class aliases and lightly open/close each mask.
2. Generate only geometrically compatible Empty–Material candidates. A candidate is scored from horizontal overlap, center-X alignment, per-column interface distance, and vertical order.
3. Use a global Hungarian assignment, so each mask can belong to at most one tube. Global matching is more reliable than choosing the nearest mask independently when tubes are close together.
4. Compute effective areas after splitting minor overlap pixels equally between the two classes. This avoids double-counting while making no unsupported claim about which class owns a small overlap. Small gaps do not invalidate a pair when the configured interface tolerance permits them.
5. Estimate full-tube geometry from `expected_full_height_px` when configured, otherwise from the median geometry of accepted two-class pairs in the same image/frame.
6. Accept an unmatched Empty mask as 0% or unmatched Material mask as 100% only when its height (and inferred width, when available) matches full-tube geometry. Without a height reference, a lone mask is invalid.

The reported formula remains:

```text
material_percentage = 100 * material_effective_area
                      / (material_effective_area + empty_effective_area)
```

Validity and percentage are separate. Invalid records have `percentage: null` and a machine-readable rejection reason.

## Assumptions

- Empty is above Material in a normally oriented image.
- Both class masks are expressed in the original image coordinate system.
- Corresponding regions have substantial horizontal overlap and meet at approximately the same interface.
- Camera perspective and tube orientation are limited enough for image-space height to be meaningful.
- A single frame does not require persistent tracking IDs. Tube IDs are assigned left-to-right per image or video frame.
- At least one of these is available to recognize a genuinely 0%/100% tube:
  - a calibrated `expected_full_height_px`; or
  - enough accepted two-class pairs in the same frame to infer tube geometry.

For a fixed camera, configuring the expected height is the safest way to validate full and empty tubes.

## Installation

The integrated repository supports Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

No model weights or datasets are included. The example uses `device: auto`,
which prefers CUDA and otherwise uses CPU.

## Configuration

Copy `config.example.yaml` and tune it for the acquisition setup. Important settings are:

- `classes.empty` and `classes.material`: model class-name aliases, matched case-insensitively.
- `morphology.close_kernel`: fills tiny holes and notches inside a class mask. Use `0` or `1` to disable.
- `pairing.min_horizontal_overlap`: intersection width divided by the narrower mask width.
- `pairing.max_center_x_distance_ratio`: maximum center difference relative to mean mask width.
- `pairing.max_interface_distance_px` and `max_interface_distance_ratio`: allowed median gap/overlap at the interface. The larger resulting tolerance is used.
- `pairing.min_pair_score`: lowest weighted compatibility score accepted by global matching.
- `completeness.expected_full_height_px`: calibrated full tube height, or `null` for per-frame inference.
- `completeness.height_tolerance_ratio`: accepted relative error from the reference height.
- `measurement.max_overlap_fraction`: rejects pairs whose overlap is too large to be explained as segmentation noise.

Odd morphology kernel sizes are required. A kernel of `0` or `1` disables that operation.

## CLI

The same command handles one image, a folder (searched recursively), or a video:

```bash
tube-measure \
  --model path/to/segmentation_model.pt \
  --source path/to/image_or_folder_or_video \
  --config config.example.yaml \
  --output outputs/run_01
```

Optional overrides:

```bash
tube-measure ... --conf 0.35 --device 0
tube-measure ... --device cpu
```

Without installation, run from this directory with:

```bash
PYTHONPATH=src python -m tube_measure \
  --model path/to/model.pt \
  --source path/to/input \
  --config config.example.yaml \
  --output outputs/run_01
```

On PowerShell, set the module path for the current process with `$env:PYTHONPATH = "src"` before the `python -m tube_measure` command.

## Outputs

Every run creates:

- `results.json`: nested records per image/frame, including reference geometry and all tube records.
- `results.csv`: one flat row per tube measurement.
- `annotated/`: annotated images for image and folder inputs.
- `<video_name>_annotated.mp4`: annotated video for video input.

Each tube record contains:

- `tube_id`
- `percentage` (`null` when invalid)
- `confidence`
- `valid`
- `rejection_reason`
- bounding box and effective class areas
- pair score and measurement source (`paired`, `single_empty`, or `single_material`)

Confidence is an explainable quality score rather than a calibrated probability. For a pair it combines geometric pair quality, detector confidence, and completeness. For a single-class full tube it combines detector confidence and geometric agreement.

## Library usage with existing masks

The analysis layer does not depend on Ultralytics:

```python
import numpy as np
from tube_measure import MaskInstance, TubeAnalyzer, load_config

config = load_config("config.example.yaml")
analyzer = TubeAnalyzer(config)

instances = [
    MaskInstance("Empty", empty_mask.astype(bool), confidence=0.94),
    MaskInstance("Material", material_mask.astype(bool), confidence=0.91),
]
analysis = analyzer.analyze(instances)
print(analysis.to_dict())
```

All masks passed to one `analyze` call must have the same two-dimensional shape.

## Tests

The tests use generated NumPy masks only; no dataset or weights are needed. They cover:

- multiple tubes and deterministic left-to-right IDs;
- close neighboring tubes;
- irregular interfaces and small gaps;
- minor class overlaps;
- missing labels and partial single masks;
- false spatial matches;
- valid full Material (100%) and full Empty (0%) tubes;
- full-tube geometry inferred from a valid pair;
- JSON and CSV fields.

Run:

```bash
pytest
```

## Rejection reasons

Typical values include:

- `no_full_tube_height_reference`
- `single_class_mask_is_incomplete`
- `single_class_width_is_inconsistent`
- `one_to_one_pairing_conflict`
- `incomplete_combined_height`
- `excessive_class_overlap`
- `combined_area_too_small`
- `mask_area_below_pairing_minimum`

Unknown classes are counted in `ignored_instances` but are not emitted as tubes.

## Known limitations

- IDs are per frame; this package does not track tubes through time.
- Strong tube tilt, severe perspective variation, or curved tubes can violate the vertical-order and image-height assumptions. Rectifying the camera view before inference is preferable.
- If all accepted pairs in a frame are truncated by a similar amount, median inference cannot discover the missing height. Configure a calibrated expected height for safety-critical measurements.
- A lone full/empty mask cannot be distinguished from a missed complementary detection without geometric reference information and is therefore rejected.
- Equal splitting of overlap pixels is deliberately neutral. Large overlaps are invalid because their class ownership is ambiguous.
- The method measures segmented pixel area, not physical volume. Perspective, tube shape, and optical refraction can require separate calibration.
- Folder annotations are flattened and prefixed with an index to avoid duplicate filenames from nested directories.
