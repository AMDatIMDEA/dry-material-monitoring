from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from material_volume.geometry import deproject_pixels, fit_plane_robust
from material_volume.models import Intrinsics


class GeometryTests(unittest.TestCase):
    def test_principal_point_deprojects_to_optical_axis(self) -> None:
        intrinsics = Intrinsics(640, 360, 400.0, 400.0, 319.5, 179.5)
        point = deproject_pixels(
            np.array([319.5]), np.array([179.5]), np.array([0.2]), intrinsics
        )[0]
        np.testing.assert_allclose(point, (0.0, 0.0, 0.2), atol=1e-12)

    def test_brown_conrady_deprojection_inverts_forward_distortion(self) -> None:
        coefficients = (0.08, -0.015, 0.001, -0.002, 0.003)
        xu, yu = 0.20, -0.12
        k1, k2, p1, p2, k3 = coefficients
        radius_squared = xu**2 + yu**2
        radial = 1 + k1 * radius_squared + k2 * radius_squared**2 + k3 * radius_squared**3
        xd = xu * radial + 2 * p1 * xu * yu + p2 * (radius_squared + 2 * xu**2)
        yd = yu * radial + 2 * p2 * xu * yu + p1 * (radius_squared + 2 * yu**2)
        intrinsics = Intrinsics(
            640,
            360,
            400.0,
            410.0,
            319.5,
            179.5,
            distortion_model="distortion.brown_conrady",
            coefficients=coefficients,
        )
        point = deproject_pixels(
            np.array([319.5 + xd * 400.0]),
            np.array([179.5 + yd * 410.0]),
            np.array([0.25]),
            intrinsics,
        )[0]
        np.testing.assert_allclose(point, (xu * 0.25, yu * 0.25, 0.25), atol=1e-10)

    def test_robust_plane_fit_keeps_majority_parallel_plane(self) -> None:
        rng = np.random.default_rng(11)
        xy = rng.uniform(-0.02, 0.02, size=(100, 2))
        main_once = np.column_stack((xy, np.full(100, 0.200)))
        main = np.vstack((main_once, main_once, main_once))
        outliers = np.column_stack((xy, np.full(100, 0.202)))
        fit = fit_plane_robust(
            np.vstack((main, outliers)),
            trim_sigma=3.5,
            iterations=4,
            minimum_band_m=0.00025,
        )
        self.assertAlmostEqual(float(fit.point_m[2]), 0.200, places=6)
        self.assertGreaterEqual(int(np.count_nonzero(fit.inlier_mask)), 300)
        self.assertLess(float(fit.rms_m), 1e-10)


if __name__ == "__main__":
    unittest.main()
