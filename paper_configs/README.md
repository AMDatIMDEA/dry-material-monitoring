# Paper configuration records

These YAML files are sanitized, non-runtime provenance records of the settings
used for the three paper materials. They were transcribed from the archived
effective configurations and cross-checked against the configuration hashes in
the corresponding measurement records. They do not change the active examples
or laboratory configuration.

| Material | Archived session | YOLO confidence | D405 internal-hole limit | Tube capacity |
|---|---|---:|---:|---:|
| Black PP | `black_polymer_E2` | 0.60 | 10 mm | 211.5273603570178 mL |
| MCC | `white_powder_E3` | 0.38 | 15 mm | 197.0406912331518 mL |
| Wood powder | `wood_powder_E4` | 0.75 | 10 mm | 197.0406912331518 mL |

The Black PP capacity belongs to a 45 mm internal-diameter, 133 mm usable-height
tube. MCC and wood powder used a different tube with 56 mm internal diameter and
80 mm usable height. The capacities are therefore intentionally different.

All three archived YOLO profiles used `whole_image`. In that mode the complete
image is supplied to inference and the implementation uses the detected
dual-class semantic extent as uncalibrated provisional level limits. Model
weights, private weight paths, output paths, the real D405 serial number, and
ROI-selection timestamps are deliberately absent from these public records.
Consequently, these files document verified paper parameters but are not
directly executable YOLO configurations.

For another fixed deployment, `configured` ROI mode is available as an
alternative. A user must select and freeze a reviewed tube rectangle from the
final camera mount. Its axial ends act as the full/empty limits and its lateral
bounds exclude background. That alternative requires its own calibration and
validation and was not the mode recorded in these paper configurations.

Example of the paper's explicit mode after supplying an ignored local runtime
configuration and separately obtained model weights:

```powershell
.\.venv\Scripts\python.exe .\Material_level_using_yolo\process_synchronized.py `
  "C:\Research records\mcc-session" `
  --config .\Material_level_using_yolo\config.local.yaml `
  --profile powder --roi-mode whole_image
```

The source archives and experimental outputs remain ignored and are not part of
the public repository. The hashes in each YAML identify the exact archived
effective configuration and calibration/model artifacts used for verification.
