from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))

from material_volume.config import load_config
from material_volume.models import Intrinsics
from set_up.center.center_config import (
    CenterSelection,
    render_updated_config,
    save_center_config,
)
from set_up.center.preview import (
    CENTER_DEPTH_PROFILE,
    ellipse_fits_frame,
    local_depth_mm,
    margin_radii_px,
    projected_radii_px,
)


CONFIG_PATH = PACKAGE_ROOT / "config.yaml"


def _selection() -> CenterSelection:
    return CenterSelection(
        center_x_px=301.0,
        center_y_px=172.0,
        diameter_mm=55.0,
        free_image_margin_percent=12.5,
        frame_width_px=640,
        frame_height_px=360,
        fps=30,
        depth_width=640,
        depth_height=360,
        camera_serial_number="123456",
        selected_utc="2026-08-03T12:00:00+00:00",
    )


class ProjectionTests(unittest.TestCase):
    def test_center_session_uses_requested_high_resolution_profile(self) -> None:
        self.assertEqual(CENTER_DEPTH_PROFILE, (1280, 720, 30))

    def test_physical_circle_projection_and_margin(self) -> None:
        intrinsics = Intrinsics(640, 360, 400.0, 420.0, 319.5, 179.5)
        radii = projected_radii_px(50.0, 100.0, intrinsics)
        self.assertEqual(radii, (100.0, 105.0))
        margin = margin_radii_px(radii, 10.0)
        self.assertAlmostEqual(margin[0], 120.0)
        self.assertAlmostEqual(margin[1], 126.0)

    def test_margin_must_remain_inside_native_depth_frame(self) -> None:
        self.assertTrue(ellipse_fits_frame((320.0, 180.0), (100.0, 100.0), 640, 360))
        self.assertFalse(ellipse_fits_frame((50.0, 180.0), (100.0, 100.0), 640, 360))

    def test_local_depth_uses_valid_neighbourhood_median(self) -> None:
        depth = np.full((9, 9), 0.165, dtype=np.float32)
        depth[4, 4] = np.nan
        depth[3, 4] = 0.0
        self.assertAlmostEqual(local_depth_mm(depth, (4, 4)), 165.0, places=3)
        self.assertIsNone(local_depth_mm(np.full((3, 3), np.nan), (1, 1)))


class ConfigUpdateTests(unittest.TestCase):
    def test_render_preserves_unrelated_values_and_comments(self) -> None:
        original = CONFIG_PATH.read_text(encoding="utf-8")
        updated = render_updated_config(original, _selection())
        self.assertIn("  center_x_px: 301.0", updated)
        self.assertIn("  free_image_margin_percent: 12.5", updated)
        self.assertIn("  directory: output", updated)
        self.assertIn("# Decimation is disabled", updated)

    def test_save_creates_timestamped_backup_and_loadable_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.yaml"
            shutil.copy2(CONFIG_PATH, target)
            original = target.read_text(encoding="utf-8")
            result = save_center_config(target, _selection())
            loaded = load_config(target)
            self.assertTrue(result.backup_path.is_file())
            self.assertEqual(result.backup_path.read_text(encoding="utf-8"), original)
            self.assertEqual(loaded.detection_roi.center_x_px, 301.0)
            self.assertEqual(loaded.detection_roi.center_y_px, 172.0)
            self.assertEqual(loaded.detection_roi.frame_width_px, 640)
            self.assertEqual(loaded.detection_roi.stream, "depth")
            self.assertEqual(loaded.tube.inner_diameter_mm, 55.0)


if __name__ == "__main__":
    unittest.main()
