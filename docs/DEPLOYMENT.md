# Deployment in another laboratory

## Required equipment

- Windows workstation with Python 3.11.
- Intel RealSense D405 and a supported `pyrealsense2`/firmware combination.
- Logitech C920 or a configured replacement camera.
- A rigid, repeatable camera/tube mount and the tube geometry used by the
  experiment protocol.
- Released YOLO weights, their checksums, and any accompanying model card.

The acquisition code is currently Windows-oriented because it uses DirectShow
camera discovery. The saved-image analysis and most hardware-free tests are
portable, but another operating system requires its camera backend to be
configured and manually validated.

## Deployment sequence

1. Clone or unpack a tagged release and follow `REPRODUCING.md` to install it.
2. Copy the tracked YAML files to ignored `config.local.yaml` files.
3. Run the D405 setup tools to write the local camera serial, ROI, stream
   profile, height, and calibration. Do not move the mount afterward.
4. Place the two model files outside Git and set their local paths.
5. Freeze the C920 ROI, axis, exposure/lighting, class names, thresholds, and
   capture settings.
6. Run all unit tests, the synthetic D405 check, and the connected-camera
   manual smoke test.
7. Run a validation dataset with known references before collecting research
   measurements.

## Privacy and paths

Tracked configurations are portable examples. Real laboratory values belong
in `config.local.yaml`, which is ignored by the root `.gitignore`. Before every
push, use:

```powershell
git grep -n -I -E '[A-Z]:\\Users\\|/Users/|/home/|OneDrive'
git status --short --ignored
```

The second command intentionally lists many ignored research assets. Review
the status; do not delete those files. A file is protected from Git only while
it remains ignored and untracked. If sensitive content was committed earlier,
adding an ignore rule is insufficient: remove it from Git history before the
repository becomes public and rotate any exposed secret.

## Release checklist

- All tests and manual hardware checks pass on the target release commit.
- Citation metadata, author names, ORCIDs, repository URL, archive DOI, and
  paper DOI are current.
- The paper and repository use the same geometry, model hashes, thresholds,
  exclusion rules, software version, and data dictionary.
- The root MIT license and the matching `3d_camera/LICENSE` have been reviewed
  by the institution; third-party model and dependency licenses are documented.
- Public sample data are de-identified, consented/approved, checksum-verified,
  and small enough for the selected archive.
