# Changelog

All notable repository changes are recorded here. Version numbers follow
Semantic Versioning.

## [Unreleased]

## [0.1.2] - 2026-09-15

- Consolidated independently packaged components into one installable monorepo
  while preserving the established source-directory names and script wrappers.
- Added repository-level citation, authorship, contribution, deployment,
  reproducibility, CI, and privacy guidance.
- Replaced tracked laboratory-specific paths and device identity with portable
  examples while retaining ignored local configuration copies.
- Added automatic CUDA selection with CPU fallback and device provenance for
  YOLO inference.
- Kept generated evidence, calibration data, trained weights, and large local
  research datasets outside Git.
- Renamed the publication-facing project to `dry-material-monitoring` and the
  paper-facing title to “Non-Contact Dry-Material Monitoring for Self-Driving
  Laboratories.”
- Corrected analysis labels to distinguish mass-based reference volume from
  directly measured true volume.
- Released the D405 component under the MIT License with Mostafa Abd Al Kader
  and IMDEA Materials Institute as copyright holders.
- Made publication documentation self-contained, removed obsolete apparatus
  examples, and recorded the current model-release status explicitly.
- Added the public Zenodo record and DOI to the release metadata and model
  documentation.
- Documented the Black PP, MCC, and wood-powder paper model hashes and archived
  session locations without adding model weights to Git.

## [0.1.0]

- Initial integrated beta research-software release.
