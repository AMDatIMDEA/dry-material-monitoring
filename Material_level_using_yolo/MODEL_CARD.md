# Model card: material-level segmentation models

No model weights are distributed in this Git repository. The paper's trained
weights and datasets are archived separately in the [Zenodo record](https://zenodo.org/records/22697983), DOI
[`10.5281/zenodo.22697983`](https://doi.org/10.5281/zenodo.22697983). Retrieve a
weight from that archive and verify its SHA-256 before use.

## Paper model inventory

The following entries are the three paper models. Archive locations are the
sanitized `archived_session` identifiers recorded in the corresponding
[`paper_configs/`](../paper_configs/) files. The weight files themselves remain
excluded from Git.

### Black PP

- **Archive location:** `black_polymer_E2`
- **Model SHA-256:** `a7d8f6e4e02039bb7c35d03773c66ee97a5f90f4f20c6707ad32a92666634b64`
- **YOLO effective-configuration SHA-256:** `9ae124c189a7c110e1817ee041ef0d13fe9912eb8607a87ef52e247b40bcc58c`
- **D405 effective-configuration SHA-256:** `09aac3a5920f98aed14f311dd20448e41b80bb2efd587a30c9717e8454277345`
- **D405 calibration SHA-256:** `7d1744ed1c6b264ab45e105e895493dfd1fe545e628893347ddc416305d96c9e`
- **Configuration record:** [`paper_configs/black_pp.yaml`](../paper_configs/black_pp.yaml)

### MCC

- **Archive location:** `white_powder_E3`
- **Model SHA-256:** `348c5fe9a4bb99b47c6e81ff31e3e969b66b2d6e8fc61c452da05aea307ef472`
- **YOLO effective-configuration SHA-256:** `cd9cfeb9ffff64993836f8ae59653a86ad97e3c756e3cad785c87361cfe20e55`
- **D405 effective-configuration SHA-256:** `81b606f9cc347e5182ef6233e88a4573c97652710ad0c2fa0fa798091f5bd8f5`
- **D405 calibration SHA-256:** `9d842eba8a44d768ddebf68740352a482245739288a1099e3f5e8c5ecc574077`
- **Configuration record:** [`paper_configs/mcc.yaml`](../paper_configs/mcc.yaml)

### Wood powder

- **Archive location:** `wood_powder_E4`
- **Model SHA-256:** `a5d46126319ccce7e59e73d30c7b8983b05ac9c5be3c92963ec5125a066ba235`
- **YOLO effective-configuration SHA-256:** `de962a971fa8555549c78a5737bb804e6fc2c080d169b7bc84bd104b3e7db870`
- **D405 effective-configuration SHA-256:** `5c5365fa99c60ff24e6e4f95a9578d18b5847506b677dafd0699f55d6962d487`
- **D405 calibration SHA-256:** `2a5a9f327ed54f167e867249029942d4fbd8de14eab2dacbde9b0435f9d92148`
- **Configuration record:** [`paper_configs/wood_powder.yaml`](../paper_configs/wood_powder.yaml)

The archived model files are not copied into this repository and no model
weights are added by this release. Their separate archive license and access
terms must be followed.

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

## Training information required for a model release

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

## Evaluation required for a model release

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
