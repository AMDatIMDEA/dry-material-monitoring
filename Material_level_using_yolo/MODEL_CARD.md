# Model card: material-level segmentation models

This card must be completed for each released weight file. Do not publish or
cite a model as validated while required fields remain marked **TO COMPLETE**.

## Model inventory

| Profile | Portable filename | Task | Semantic roles | SHA-256 | Release/archive |
|---|---|---|---|---|---|
| powder | `powder_best.pt` | segmentation | `Powder`, `Empty` | **TO COMPLETE** | **TO COMPLETE** |
| polymer | `polymer_best.pt` | segmentation | `polymer`, `Empty` | **TO COMPLETE** | **TO COMPLETE** |

The runtime reads `model.names` and fails before inference if a configured role
is missing, ambiguous, duplicated, or mapped to an incompatible explicit ID.
Historical dataset IDs are documentation only and are not trusted at runtime.

## Intended use

- Estimate a material/empty interface from deliberately captured still images
  of the frozen laboratory tube/camera arrangement.
- Process saved C920 images after acquisition, never live preview frames.
- Support the material profiles and acquisition domain represented in the
  documented validation dataset.

## Out-of-scope use

- Unvalidated materials, tubes, viewpoints, lighting, cameras, or image
  resolutions.
- Safety-critical or autonomous refill control without an independently
  validated control system.
- Treating mask area as physical volume outside the implemented calibrated
  interface method.
- Assuming the model generalizes to other laboratories without transfer
  validation.

## Training information — complete before release

For each model, report:

- exact Ultralytics model family/checkpoint and software version;
- initialization/pretraining source and its license;
- dataset archive DOI/version and annotation protocol;
- specimen/material lots, camera, resolution, distance, ROI, lighting, and
  acquisition dates;
- train/validation/test split method, independence unit, counts, class balance,
  and leakage checks;
- augmentation and preprocessing parameters;
- optimizer, learning-rate schedule, epochs, batch size, image size, random
  seeds, early-stopping/model-selection rule, and compute hardware;
- frozen inference confidence, IoU, NMS, and aggregation thresholds.

## Evaluation — complete before release

Report results on an independent, frozen test set. Include confidence intervals
and per-material/per-level strata where defensible:

- mask mAP and class-specific precision/recall;
- interface or fill-percentage error against the declared reference;
- bias, MAE/RMSE, repeatability, invalid/rejection rate, and failure categories;
- robustness checks for lighting, fill range, material lot, and mount variation;
- clearly separated development, tuning, and final evaluation results.

Do not select thresholds on the final test set. Preserve excluded and invalid
samples with documented reasons.

## Limitations and risks

Performance depends on image domain, camera geometry, ROI/axis calibration,
lighting, transparent-wall reflections, material appearance, interface
visibility, annotation quality, and dataset representativeness. `device: auto`
can select different compute kernels across machines; archive the requested and
resolved device alongside dependency and weight hashes.

## Licensing and access

State the license and access conditions for weights, datasets, annotations,
pretrained checkpoints, and Ultralytics use. The repository's MIT license does
not automatically grant rights to separately distributed models or data.
