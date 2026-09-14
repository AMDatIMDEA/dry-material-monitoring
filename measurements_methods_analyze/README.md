# Measurement methods analysis

This repository component compares three estimates—human/manual, offline YOLO,
and D405 depth—against a mass-based volume reference for a `Run_Experiment`
session. It reads the existing method workbooks without modifying them and
matches records by the existing `measurement_id` (while also verifying shared
experiment/reference fields).

This is descriptive software analysis. It does not establish metrological or
scientific validation and does not change any acquisition or estimation code.

> **Manuscript scope:** this general-purpose utility was not used to generate
> the manuscript's final statistical analysis, tables, or figures. It remains
> available for exploratory, traceable comparison of compatible experiment
> records and is not a paper-specific analysis pipeline.

## Configure and run

Copy [`config.yaml`](config.yaml) to the ignored `config.local.yaml` and change
only `experiment_directory` for a new session. Both spaces and accented
characters are supported. Relative `output_directory` values resolve from the
selected YAML file; the portable default creates
`measurements_methods_analyze\outputs`.

From the repository root:

```powershell
.\.venv\Scripts\analyze-measurements.exe --config .\measurements_methods_analyze\config.local.yaml
```

To use another YAML file:

```powershell
.\.venv\Scripts\python.exe .\measurements_methods_analyze\analyze.py `
  --config "C:\Research records\analysis config.yaml"
```

The tool prefers the configured format (`csv` by default), falls back to XLSX,
and automatically looks for `depth_measurements` and `yolo_measurements` in the
session root. Source files are opened read-only.

## Reference and inclusion rules

The mass-based volume is always the comparison reference. It is a derived
comparison quantity, not directly measured true material volume:

1. If `reference_volume_from_weight_ml` is populated, that explicitly recorded
   mass-derived volume is used directly.
2. If it is blank and `allow_full_mass_fraction_fallback` is enabled, the tool
   uses only explicitly recorded inputs:

   ```text
   nominal mass-fraction-based reference volume =
       total_capacity_ml × reference_material_weight_g
       / total_possible_weight_g
   ```

   This fallback is a nominal mass-fraction-based estimate. It assumes
   `total_possible_weight_g` is the net material mass at the same full-tube
   capacity and packing condition; it is not a direct measurement of true
   volume. The output report records the reference source for every row.
   Disable the fallback if that assumption is unsuitable.

No value is inferred from a material name. Missing or non-finite inputs stay
missing. A mass greater than the recorded full mass is rejected. YOLO and D405
rows are included only when `status=complete_valid`, `valid=true`, and a finite
estimated material volume exists. Manual results use an explicitly stored
`reference_volume_from_manual_ml`, or the stored manual percentage and capacity
when the volume field is absent. Missing and invalid comparisons are listed in
`excluded_comparisons.csv` with their reason.

## Graphs

Each figure is written as high-resolution PNG and vector PDF:

1. **Estimated versus reference volume** — Human, YOLO, and D405 estimates
   against the mass-based reference volume. The dashed `y=x` line represents
   ideal equality.
2. **Signed error versus reference fill** — `estimate − reference`; positive
   values overestimate and negative values underestimate. The zero line marks
   no error.
3. **Absolute-error distribution** — box plots summarize magnitude without
   retaining the error sign.
4. **Bland–Altman agreement** — each panel plots method/reference difference
   against their mean. Bias is the mean difference; 95% limits of agreement are
   `bias ± 1.96 × sample SD`. With fewer than two pairs, limits are unavailable.
5. **Repeatability across reference fill levels** — transparent points are
   individual measurements; solid points/error bars are group mean ± sample SD.
   Reference values within `grouping_tolerance_ml` form a repeat group. A
   single-observation group has no estimable SD and is drawn with a zero-length
   bar, not claimed as demonstrated repeatability.
6. **Measurement-by-measurement volume comparison** — shows the mass-based
   reference, human visual estimate, YOLO estimate, and D405 depth estimate in
   measurement-index order. Markers are connected by thin lines only to help
   visually track consecutive experimental measurements; the lines do not
   represent interpolation between measurements. Missing or invalid values
   create visible gaps and are never replaced with zero or interpolated.

## Numerical outputs

`outputs` contains:

- `summary_statistics.csv` and the `summary` XLSX sheet: MAE, RMSE, mean bias,
  sample SD of error, prediction R² (`1 − SSE/SST`), and valid pair count;
- `matched_measurements.csv`: every numerical pair and its `measurement_id`;
- `excluded_comparisons.csv`: missing/invalid pairs and reasons;
- `analysis_results.xlsx`: summary, matched, and excluded tables;
- `analysis_report.json`: input files, reference-source counts, assumptions,
  and exclusion count;
- six PNG/PDF figure pairs.

R² is blank for fewer than two observations or constant reference values. Error
SD is blank for fewer than two pairs. These blanks are deliberate and are never
replaced by zero.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest .\measurements_methods_analyze\tests -q
```
