from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from material_volume.calibration import create_virtual_bottom_calibration
from material_volume.config import load_config
from material_volume.models import CalibrationData, Intrinsics
from material_volume.pipeline import MaterialVolumePipeline, RefillWarningLatch
from material_volume.synthetic import SyntheticDepthSource


class SyntheticPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = deepcopy(load_config(PACKAGE_ROOT / "config.yaml"))
        # Hardware-specific pixel coordinates are not applicable to synthetic intrinsics.
        cls.config.calibration.center_x_px = None
        cls.config.calibration.center_y_px = None
        cls.config.detection_roi.center_x_px = None
        cls.config.detection_roi.center_y_px = None
        cls.config.detection_roi.frame_width_px = None
        cls.config.detection_roi.frame_height_px = None
        cls.config.fusion.burst_frames = 15
        with SyntheticDepthSource(cls.config, fill_percent=0.0) as source:
            burst = source.capture_burst(cls.config.fusion.burst_frames)
        cls.calibration = MaterialVolumePipeline.calibrate(burst, cls.config)

    def measure(self, fill_percent: float):
        with SyntheticDepthSource(self.config, fill_percent=fill_percent) as source:
            burst = source.capture_burst(self.config.fusion.burst_frames)
        return MaterialVolumePipeline(self.config, self.calibration).measure(burst)

    def test_known_flat_surface_volume(self) -> None:
        estimate = self.measure(35.0)
        self.assertTrue(estimate.result.quality.valid)
        self.assertAlmostEqual(estimate.result.fill_percent, 35.0, delta=0.6)
        self.assertAlmostEqual(
            estimate.result.mean_level_mm,
            self.config.tube.usable_height_mm * 0.35,
            delta=0.7,
        )
        self.assertAlmostEqual(
            estimate.result.material_volume_ml,
            self.config.tube.capacity_ml * 0.35,
            delta=1.5,
        )

    def test_refill_threshold_requires_two_confirmations(self) -> None:
        first = self.measure(5.0)
        second = self.measure(5.0)
        latch = RefillWarningLatch(self.config)
        self.assertTrue(latch.update(first).startswith("LOW_LEVEL_PENDING"))
        self.assertEqual(latch.update(second), "REFILL_WARNING_ACTIVE")

    def test_poor_depth_coverage_never_triggers_refill(self) -> None:
        bad_config = deepcopy(self.config)
        bad_config.synthetic.invalid_pixel_fraction = 0.70
        with SyntheticDepthSource(bad_config, fill_percent=5.0) as source:
            burst = source.capture_burst(bad_config.fusion.burst_frames)
        estimate = MaterialVolumePipeline(bad_config, self.calibration).measure(burst)
        self.assertFalse(estimate.result.quality.valid)
        self.assertFalse(estimate.result.refill_required)

    def test_calibration_round_trip_and_distortion_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.npz"
            self.calibration.save(path)
            restored = CalibrationData.load(path)
        self.assertEqual(restored.intrinsics, self.calibration.intrinsics)
        changed = replace(restored.intrinsics, coefficients=(0.01, 0.0, 0.0, 0.0, 0.0))
        errors = restored.compatibility_errors(changed, self.config.tube)
        self.assertIn("Camera distortion coefficients changed since calibration.", errors)

    def test_virtual_bottom_uses_configured_geometry_without_depth_points(self) -> None:
        config = deepcopy(self.config)
        config.camera.distance_to_rim_mm = 109.0
        config.tube.usable_height_mm = 133.0
        intrinsics = Intrinsics(640, 360, 400.0, 410.0, 319.5, 179.5)

        calibration = create_virtual_bottom_calibration(intrinsics, config)

        self.assertEqual(calibration.calibration_method, "virtual_from_config")
        self.assertAlmostEqual(calibration.bottom_center_m[2], 0.242, places=12)
        self.assertAlmostEqual(calibration.measured_rim_distance_mm, 109.0, places=12)
        self.assertEqual(calibration.plane_rms_mm, 0.0)
        self.assertEqual(calibration.plane_coverage, 1.0)
        self.assertEqual(calibration.tube_axis_toward_camera.tolist(), [0.0, 0.0, -1.0])

    def test_virtual_calibration_method_survives_round_trip(self) -> None:
        intrinsics = Intrinsics(640, 360, 400.0, 410.0, 319.5, 179.5)
        calibration = create_virtual_bottom_calibration(intrinsics, self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "virtual_calibration.npz"
            calibration.save(path)
            restored = CalibrationData.load(path)
        self.assertEqual(restored.calibration_method, "virtual_from_config")


if __name__ == "__main__":
    unittest.main()
