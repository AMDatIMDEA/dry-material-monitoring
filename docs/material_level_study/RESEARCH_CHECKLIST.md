# Research data-collection checklist

This checklist prepares a study; it does not certify metrological validity.
Record deviations in the session notes before continuing.

## Apparatus and frozen software

- [ ] Identify the tube, inner diameter, usable internal height, capacity, and
  any liner/reference-target thickness with units and instrument resolution.
- [ ] Rigidly freeze and photograph the D405, C920, tube, supports, optical axis,
  working distances, focus, lighting, background, and cable arrangement.
- [ ] Freeze all three effective YAML configurations. Record the Git revision or
  dirty state and retain their SHA-256 hashes.
- [ ] Confirm the C920 ROI/full/empty limits and tube-axis orientation are
  calibrated for the mounted camera; the supplied whole-image ROI is only a
  placeholder.
- [ ] Record camera models, serial numbers, requested/active stream profiles,
  backend, warm-up, exposure/gain/focus behavior, and D405 depth scale.
- [ ] Verify sufficient storage for the chosen `compact`, `research`, or
  `full_raw` policy. Do not silently downgrade retention.

## Calibration and identity

- [ ] Calibrate the empty tube after the final mount/profile/geometry change.
  Record calibration procedure, date, operator, reference target, file, hash,
  plane residual, and acceptance result.
- [ ] Choose a portable, unique `experiment_id`; verify the output root and
  automatic measurement index start before acquisition.
- [ ] Confirm both workbooks use schema `1.0.0`, identical common headers, and
  one shared measurement ID per trigger.
- [ ] Validate each YOLO weights file, task, semantic `model.names`, profile,
  effective configuration hash, and model hash before processing study data.

## Material and independent references

- [ ] Record material name/grade, supplier, lot/batch, particle form and size
  distribution, conditioning, color/appearance, and handling history.
- [ ] Record the bulk-density value only when explicitly measured or sourced.
  Document source, method, packing/compaction protocol, temperature, replicate
  count, units, and uncertainty or stated limitations. Never infer density from
  a material name.
- [ ] Identify every independent balance, height gauge, caliper, reference
  vessel, or other reference instrument by asset/serial number, range,
  resolution, calibration status/date, traceability, and operating procedure.
- [ ] Record independent remaining mass and/or manual level without using either
  vision method to create that reference.

## Experimental design and environment

- [ ] Predefine levels, repeats, acceptance/exclusion rules, warm-up, settling
  time, refill/emptying procedure, and treatment of failed/partial captures.
- [ ] Randomize or counterbalance fill-level/order effects where practical.
  Include repeated measurements, independent remount/recalibration blocks when
  relevant, and reserved validation specimens/runs not used to tune thresholds.
- [ ] Record date/time, operator, ambient temperature, humidity, vibration,
  illumination, camera temperature/warm-up, material settling/compaction, and
  unusual events for every block.
- [ ] Do not tune ROI, thresholds, filters, calibration, or exclusion rules from
  the same reserved observations later reported as independent validation.

## Evidence and review

- [ ] Retain session/capture/artifact/processing manifests, effective configs,
  environment metadata, hashes, common Excel/CSV files, and all required
  method-specific diagnostics.
- [ ] Retain all lossless C920 source stills. Retain D405 raw Z16 bursts only
  when `full_raw` was selected; record the explicit policy otherwise.
- [ ] Verify artifact hashes, workbook/CSV parity, blank invalid estimates,
  valid-result conservation formulas, and recovery status before backing up.
- [ ] Make at least two controlled backups and record storage location,
  permissions, retention period, privacy restrictions, and checksum procedure.
- [ ] Perform accuracy, repeatability, reproducibility, robustness, and any
  uncertainty evaluation against independent references under a predeclared
  protocol. Software test success alone is not validation.
- [ ] Keep any future depth-versus-YOLO comparison in a separate analysis plan
  and implementation. The acquisition repositories do not compute it.
