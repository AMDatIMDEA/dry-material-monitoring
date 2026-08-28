from __future__ import annotations

from math import isclose
from pathlib import Path
import sys
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from material_volume.config import approximate_d405_min_z_mm, load_config


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(PACKAGE_ROOT / "config.yaml")

    def test_capacity_uses_explicit_inner_diameter(self) -> None:
        expected = (
            3.141592653589793
            * (self.config.tube.inner_diameter_mm / 2.0) ** 2
            * self.config.tube.usable_height_mm
            / 1000.0
        )
        self.assertTrue(isclose(self.config.tube.capacity_ml, expected, rel_tol=1e-12))

    def test_dimension_conflict_is_reported(self) -> None:
        warnings = self.config.validate()
        tube = self.config.tube
        implied_inner = tube.outer_diameter_mm - 2.0 * tube.wall_thickness_mm
        conflicts = [item for item in warnings if "Tube geometry conflict" in item]
        self.assertEqual(
            bool(conflicts),
            abs(implied_inner - tube.inner_diameter_mm)
            > tube.geometry_warning_tolerance_mm,
        )

    def test_saved_roi_resolution_mismatch_is_reported(self) -> None:
        warnings = self.config.validate()
        roi = self.config.detection_roi
        expected = (
            roi.frame_width_px is not None
            and not self.config.filters.decimation_enabled
            and (roi.frame_width_px, roi.frame_height_px)
            != (self.config.camera.depth_width, self.config.camera.depth_height)
        )
        self.assertEqual(
            any("saved detection ROI" in item for item in warnings),
            expected,
        )

    def test_d405_profile_minimum_z_table_matches_datasheet(self) -> None:
        self.assertIsNone(approximate_d405_min_z_mm(640, 480))
        self.assertEqual(approximate_d405_min_z_mm(424, 240), 40.0)
        self.assertEqual(approximate_d405_min_z_mm(640, 360), 55.0)
        self.assertEqual(approximate_d405_min_z_mm(848, 480), 70.0)
        self.assertEqual(approximate_d405_min_z_mm(1280, 720), 100.0)


if __name__ == "__main__":
    unittest.main()
