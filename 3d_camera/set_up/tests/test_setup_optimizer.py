from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import yaml


SETUP_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from set_up.configure_height import OutputMode, run_example, run_interactive
from material_volume.config import load_config
from material_volume.models import Intrinsics
from set_up.setup_config import render_updated_config, save_setup_to_config
from set_up.setup_optimizer import (
    CameraProfile,
    MountingRange,
    ProfileDiscovery,
    SafetyAllowances,
    SetupStatus,
    TubeDimensions,
    best_valid_at_exact_distance,
    camera_to_bottom_mm,
    centred_available_diameter_mm,
    evaluate_exact_distance,
    evaluate_setup,
    lateral_sampling_mm_per_pixel,
    make_fallback_profiles,
    optimize_setups,
    visible_area_mm,
)


def profile(
    *,
    width: int = 640,
    height: int = 360,
    fx: float = 400.0,
    fy: float = 400.0,
    ppx: float | None = None,
    ppy: float | None = None,
    min_depth: float = 50.0,
    max_depth: float = 250.0,
    approximate: bool = False,
) -> CameraProfile:
    intrinsics = Intrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        ppx=(width - 1) / 2.0 if ppx is None else ppx,
        ppy=(height - 1) / 2.0 if ppy is None else ppy,
    )
    return CameraProfile(
        width=width,
        height=height,
        fps=30,
        intrinsics=intrinsics,
        minimum_usable_depth_mm=min_depth,
        maximum_reliable_depth_mm=max_depth,
        intrinsics_source="unit-test",
        approximate=approximate,
    )


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tube = TubeDimensions(40.0, 36.0, 50.0, 50.0)
        self.no_allowance = SafetyAllowances(
            free_margin_fraction=0.0,
            centring_error_mm=0.0,
            maximum_tilt_degrees=0.0,
            minimum_projected_diameter_px=5.0,
        )
        self.profile = profile()

    def test_camera_to_rim_and_bottom_distance_definitions(self) -> None:
        tube = TubeDimensions(40.0, 36.0, 100.0, 80.0)
        result = evaluate_setup(tube, self.profile, 60.0, self.no_allowance)
        self.assertEqual(camera_to_bottom_mm(60.0, 100.0), 160.0)
        self.assertEqual(result.camera_to_rim_mm, 60.0)
        self.assertEqual(result.camera_to_bottom_mm, 160.0)
        self.assertEqual(result.nearest_surface_distance_mm, 80.0)
        self.assertEqual(result.farthest_surface_distance_mm, 160.0)

    def test_visible_field_increases_with_distance(self) -> None:
        near = visible_area_mm(self.profile, 80.0)
        far = visible_area_mm(self.profile, 160.0)
        self.assertGreater(far[0], near[0])
        self.assertGreater(far[1], near[1])
        self.assertAlmostEqual(far[0], near[0] * 2.0)

    def test_sampling_becomes_coarser_with_distance(self) -> None:
        near = lateral_sampling_mm_per_pixel(self.profile, 80.0)
        far = lateral_sampling_mm_per_pixel(self.profile, 160.0)
        self.assertGreater(far[0], near[0])
        self.assertGreater(far[1], near[1])

    def test_tube_fits_horizontally_and_vertically(self) -> None:
        result = evaluate_setup(self.tube, self.profile, 70.0, self.no_allowance)
        self.assertTrue(result.valid)
        self.assertGreater(result.minimum_horizontal_remaining_margin_mm, 0.0)
        self.assertGreater(result.minimum_vertical_remaining_margin_mm, 0.0)

    def test_tube_can_fit_horizontally_but_fail_vertically(self) -> None:
        wide = TubeDimensions(95.0, 90.0, 20.0, 20.0)
        result = evaluate_setup(wide, self.profile, 100.0, self.no_allowance)
        self.assertGreater(result.minimum_horizontal_remaining_margin_mm, 0.0)
        self.assertLessEqual(result.minimum_vertical_remaining_margin_mm, 0.0)
        self.assertFalse(result.valid)
        self.assertTrue(any("vertical" in reason for reason in result.reasons))

    def test_asymmetric_principal_point_reduces_centred_coverage(self) -> None:
        asymmetric = profile(ppx=200.0, ppy=100.0)
        full = visible_area_mm(asymmetric, 100.0)
        centred = centred_available_diameter_mm(asymmetric, 100.0)
        self.assertLess(centred[0], full[0])
        self.assertLess(centred[1], full[1])

    def test_tall_tube_exceeds_far_depth_limit(self) -> None:
        tall = TubeDimensions(30.0, 25.0, 190.0, 190.0)
        result = evaluate_setup(tall, self.profile, 70.0, self.no_allowance)
        self.assertFalse(result.valid)
        self.assertLess(result.far_depth_margin_mm, 0.0)
        self.assertTrue(any("bottom" in reason.lower() for reason in result.reasons))

    def test_large_diameter_exceeds_image_coverage(self) -> None:
        large = TubeDimensions(200.0, 180.0, 20.0, 20.0)
        result = evaluate_setup(large, self.profile, 70.0, self.no_allowance)
        self.assertFalse(result.valid)
        self.assertTrue(
            any("image field" in reason.lower() for reason in result.reasons)
        )

    def test_minimum_depth_boundary_is_valid_but_warned(self) -> None:
        small = TubeDimensions(10.0, 8.0, 20.0, 20.0)
        result = evaluate_setup(small, self.profile, 50.0, self.no_allowance)
        self.assertTrue(result.valid)
        self.assertEqual(result.near_depth_margin_mm, 0.0)
        self.assertTrue(any("exactly" in warning for warning in result.warnings))

    def test_safety_margin_and_centring_reduce_remaining_field(self) -> None:
        baseline = evaluate_setup(self.tube, self.profile, 70.0, self.no_allowance)
        guarded = evaluate_setup(
            self.tube,
            self.profile,
            70.0,
            SafetyAllowances(
                free_margin_fraction=0.10,
                centring_error_mm=3.0,
                maximum_tilt_degrees=2.0,
                minimum_projected_diameter_px=5.0,
            ),
        )
        self.assertLess(
            guarded.minimum_vertical_remaining_margin_mm,
            baseline.minimum_vertical_remaining_margin_mm,
        )
        self.assertLess(
            guarded.maximum_supported_tube_diameter_mm,
            baseline.maximum_supported_tube_diameter_mm,
        )

    def test_tilt_envelope_is_included_in_depth_boundaries(self) -> None:
        tilted = evaluate_setup(
            TubeDimensions(40.0, 36.0, 20.0, 20.0),
            self.profile,
            50.0,
            SafetyAllowances(
                free_margin_fraction=0.0,
                centring_error_mm=0.0,
                maximum_tilt_degrees=2.0,
                minimum_projected_diameter_px=5.0,
            ),
        )
        self.assertEqual(tilted.nearest_surface_distance_mm, 50.0)
        self.assertLess(tilted.nearest_tilt_envelope_distance_mm, 50.0)
        self.assertFalse(tilted.valid)
        self.assertEqual(tilted.status, SetupStatus.CONDITIONALLY_USABLE)
        self.assertTrue(
            any("tilt envelope" in reason for reason in tilted.allowance_reasons)
        )

    def test_partial_measurable_fill_range_is_reported(self) -> None:
        partial = evaluate_setup(
            TubeDimensions(40.0, 36.0, 100.0, 100.0),
            profile(min_depth=80.0),
            60.0,
            self.no_allowance,
        )
        self.assertEqual(partial.status, SetupStatus.PHYSICALLY_INVALID)
        self.assertEqual(partial.physical_measurable_fill_min_mm, 0.0)
        self.assertEqual(partial.physical_measurable_fill_max_mm, 80.0)

    def test_stereo_overlap_can_be_limiting_physical_constraint(self) -> None:
        stereo_limited = profile(fy=300.0)
        result = evaluate_setup(
            TubeDimensions(80.0, 75.0, 20.0, 20.0),
            stereo_limited,
            70.0,
            self.no_allowance,
        )
        self.assertGreaterEqual(result.rim.nominal_horizontal_margin_mm, 0.0)
        self.assertGreaterEqual(result.rim.nominal_vertical_margin_mm, 0.0)
        self.assertLess(result.rim.nominal_stereo_overlap_margin_mm, 0.0)
        self.assertEqual(result.status, SetupStatus.PHYSICALLY_INVALID)
        self.assertTrue(
            any("stereo-overlap" in reason for reason in result.physical_reasons)
        )

    def test_rim_occlusion_is_a_physical_failure(self) -> None:
        result = evaluate_setup(
            TubeDimensions(40.0, 20.0, 50.0, 50.0),
            self.profile,
            70.0,
            SafetyAllowances(
                free_margin_fraction=0.0,
                centring_error_mm=30.0,
                maximum_tilt_degrees=0.0,
                minimum_projected_diameter_px=5.0,
            ),
        )
        self.assertLess(result.bottom.conservative_rim_clearance_mm, 0.0)
        self.assertEqual(result.status, SetupStatus.PHYSICALLY_INVALID)
        self.assertTrue(
            any("rim/wall occlusion" in reason for reason in result.physical_reasons)
        )


class OptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tube = TubeDimensions(40.0, 36.0, 50.0, 50.0)
        self.safety = SafetyAllowances(
            free_margin_fraction=0.05,
            centring_error_mm=1.0,
            maximum_tilt_degrees=1.0,
            minimum_projected_diameter_px=10.0,
        )

    def test_multiple_mocked_profiles_and_best_profile_selection(self) -> None:
        low = profile(width=320, height=180, fx=200.0, fy=200.0, min_depth=50.0)
        high = profile(width=640, height=360, fx=400.0, fy=400.0, min_depth=80.0)
        result = optimize_setups(
            self.tube,
            (low, high),
            self.safety,
            mounting_range=MountingRange(80.0, 150.0),
        )
        self.assertTrue(result.valid)
        spatial = next(
            rec for rec in result.recommendations
            if rec.category == "HIGHEST_SPATIAL_SAMPLING"
        )
        self.assertEqual(spatial.evaluation.profile.width, 640)
        self.assertGreater(spatial.evaluation.near_depth_margin_mm, 0.0)

    def test_no_valid_candidate(self) -> None:
        tall = TubeDimensions(40.0, 36.0, 230.0, 230.0)
        result = optimize_setups(tall, (profile(),), self.safety)
        self.assertFalse(result.valid)
        self.assertEqual(result.recommendations, ())

    def test_rejected_custom_distance(self) -> None:
        evaluations = evaluate_exact_distance(
            self.tube,
            (profile(min_depth=90.0),),
            60.0,
            self.safety,
        )
        self.assertIsNone(best_valid_at_exact_distance(evaluations))
        self.assertTrue(any("minimum" in reason for reason in evaluations[0].reasons))

    def test_mounting_range_can_remove_all_candidates(self) -> None:
        result = optimize_setups(
            self.tube,
            (profile(min_depth=100.0),),
            self.safety,
            mounting_range=MountingRange(20.0, 40.0),
        )
        self.assertFalse(result.valid)

    def test_offline_fallback_mode(self) -> None:
        discovery = make_fallback_profiles(250.0)
        self.assertTrue(discovery.approximate)
        self.assertEqual(len(discovery.profiles), 5)
        self.assertTrue(all(item.approximate for item in discovery.profiles))
        self.assertIn((640, 360, 30), {
            (item.width, item.height, item.fps) for item in discovery.profiles
        })

    def test_representative_58_55_115_tube_uses_actual_constraints(self) -> None:
        discovery = make_fallback_profiles(250.0)
        representative = TubeDimensions(58.0, 55.0, 115.0, 115.0)
        result = optimize_setups(
            representative,
            discovery.profiles,
            SafetyAllowances(),
            configured_minimum_depth_mm=45.0,
            configured_maximum_depth_mm=250.0,
        )
        self.assertGreater(result.evaluated_count, 0)
        if result.valid:
            self.assertTrue(all(rec.evaluation.valid for rec in result.recommendations))
        else:
            self.assertEqual(result.recommendations, ())

    def test_59_55_115_at_65_mm_is_conditional_when_only_allowance_fails(self) -> None:
        discovery = make_fallback_profiles(250.0)
        profile_640 = next(
            item
            for item in discovery.profiles
            if (item.width, item.height, item.fps) == (640, 360, 30)
        )
        evaluation = evaluate_setup(
            TubeDimensions(59.0, 55.0, 115.0, 115.0),
            profile_640,
            65.0,
            SafetyAllowances(),
            configured_minimum_depth_mm=45.0,
            configured_maximum_depth_mm=250.0,
        )
        self.assertLessEqual(evaluation.effective_minimum_depth_mm, 65.0)
        self.assertGreaterEqual(evaluation.effective_maximum_depth_mm, 180.0)
        self.assertGreaterEqual(evaluation.rim.nominal_horizontal_margin_mm, 0.0)
        self.assertGreaterEqual(evaluation.rim.nominal_vertical_margin_mm, 0.0)
        self.assertGreaterEqual(
            evaluation.rim.nominal_stereo_overlap_margin_mm, 0.0
        )
        self.assertEqual(evaluation.physical_reasons, [])
        self.assertTrue(evaluation.allowance_reasons)
        self.assertEqual(evaluation.status, SetupStatus.CONDITIONALLY_USABLE)


class ConfigurationSaveTests(unittest.TestCase):
    def _evaluation(self) -> tuple[SetupEvaluation, ProfileDiscovery]:
        discovery = make_fallback_profiles(250.0)
        tube = TubeDimensions(58.0, 55.0, 115.0, 115.0)
        result = optimize_setups(
            tube,
            discovery.profiles,
            SafetyAllowances(),
            configured_minimum_depth_mm=45.0,
            configured_maximum_depth_mm=250.0,
        )
        self.assertIsNotNone(result.best_compromise)
        return result.best_compromise, discovery  # type: ignore[return-value]

    def test_render_does_not_change_source_file_before_save(self) -> None:
        evaluation, discovery = self._evaluation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            original = (PACKAGE_ROOT / "config.yaml").read_text(encoding="utf-8")
            path.write_text(original, encoding="utf-8")
            rendered = render_updated_config(
                original,
                evaluation,
                discovery,
                generated_utc="2026-07-30T00:00:00+00:00",
            )
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertIn("setup_optimizer:", rendered)

    def test_configuration_backup_and_targeted_update(self) -> None:
        evaluation, discovery = self._evaluation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            shutil.copy2(PACKAGE_ROOT / "config.yaml", path)
            original = path.read_text(encoding="utf-8")
            saved = save_setup_to_config(
                path,
                evaluation,
                discovery,
                generated_utc="2026-07-30T00:00:00+00:00",
            )
            self.assertTrue(saved.backup_path.is_file())
            self.assertEqual(saved.backup_path.read_text(encoding="utf-8"), original)
            updated_text = path.read_text(encoding="utf-8")
            self.assertIn("# Standalone Intel RealSense", updated_text)
            data = yaml.safe_load(updated_text)
            self.assertEqual(
                data["camera"]["distance_to_rim_mm"],
                evaluation.camera_to_rim_mm,
            )
            self.assertEqual(data["tube"]["wall_thickness_mm"], 1.5)
            self.assertEqual(
                data["detection_roi"]["diameter_mm"],
                evaluation.tube.inner_diameter_mm,
            )
            self.assertEqual(data["camera"]["depth_width"], evaluation.profile.width)
            self.assertEqual(
                data["setup_optimizer"]["camera_to_bottom_mm"],
                evaluation.camera_to_bottom_mm,
            )
            load_config(path)

    def test_interactive_no_save_leaves_configuration_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            shutil.copy2(PACKAGE_ROOT / "config.yaml", path)
            original = path.read_bytes()
            config = load_config(path)
            answers = iter(
                [
                    "58",
                    "55",
                    "115",
                    "",
                    "",
                    "",
                    "",
                    "n",
                    "",
                    "",
                    "",
                    "",
                    "n",
                ]
            )
            args = argparse.Namespace(
                precision_test=False,
                precision_frames=60,
                known_distance_mm=None,
            )
            result = run_interactive(
                config,
                args,
                input_fn=lambda _prompt: next(answers),
                output=StringIO(),
            )
            self.assertEqual(result, 0)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.backup-*.yaml")), [])

    def test_abstract_and_long_review_outputs_are_separated(self) -> None:
        config = load_config(PACKAGE_ROOT / "config.yaml")
        abstract = StringIO()
        self.assertEqual(
            run_example(config, output=abstract, output_mode=OutputMode.ABSTRACT),
            0,
        )
        abstract_text = abstract.getvalue()
        self.assertLessEqual(
            len([line for line in abstract_text.splitlines() if line.strip()]),
            12,
        )
        self.assertIn("Final status:", abstract_text)
        self.assertNotIn("CAMERA INFORMATION", abstract_text)
        self.assertNotIn("Depth range:", abstract_text)

        long_review = StringIO()
        self.assertEqual(
            run_example(
                config,
                output=long_review,
                output_mode=OutputMode.LONG_REVIEW,
            ),
            0,
        )
        long_text = long_review.getvalue()
        self.assertIn("CAMERA INFORMATION", long_text)
        self.assertIn("Depth range:", long_text)
        self.assertIn("stereo-overlap", long_text)


if __name__ == "__main__":
    unittest.main()
