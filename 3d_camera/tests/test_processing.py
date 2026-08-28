from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from material_volume.config import FilterConfig, FusionConfig
from material_volume.models import DepthBurst, Intrinsics
from material_volume.processing import fuse_depth_burst


class FusionTests(unittest.TestCase):
    def test_median_mad_rejects_single_frame_spike(self) -> None:
        frames = [np.full((5, 6), 0.120, dtype=np.float32) for _ in range(9)]
        frames[2][2, 3] = 0.180
        frames[4][1, 1] = np.nan
        burst = DepthBurst(
            frames_m=frames,
            intrinsics=Intrinsics(6, 5, 5.0, 5.0, 2.5, 2.0),
            depth_scale_m=0.001,
            source_name="unit-test",
        )
        fused = fuse_depth_burst(burst, FusionConfig(), FilterConfig())
        self.assertAlmostEqual(float(fused.depth_m[2, 3]), 0.120, places=5)
        self.assertGreater(float(fused.valid_fraction[2, 3]), 0.80)


if __name__ == "__main__":
    unittest.main()
