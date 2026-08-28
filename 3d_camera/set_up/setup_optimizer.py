"""Hardware-independent D405 tube-setup geometry and candidate optimization.

Distances in this module are measured from the RealSense depth optical origin:

* ``camera_to_rim_mm`` is the distance to the tube opening/rim plane.
* ``camera_to_bottom_mm`` is ``camera_to_rim_mm + usable_height_mm``.
* ``nearest_surface_distance_mm`` is the maximum-fill surface.
* ``farthest_surface_distance_mm`` is the empty tube bottom.

The offline table is based on the D401/D405 Z16 modes and Min-Z values in the
RealSense D400 Series Product Family Datasheet (revision 017, tables 4-4 and
4-11).  The nominal 87 x 58 degree D405 FOV comes from the D405 product
specification.  Per-device intrinsics vary, so fallback geometry is explicitly
approximate and connected-camera intrinsics are always preferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import atan, cos, degrees, floor, radians, sin, tan
from typing import Any, Iterable, Sequence

from material_volume.config import approximate_d405_min_z_mm
from material_volume.models import Intrinsics


D405_NOMINAL_HORIZONTAL_FOV_DEGREES = 87.0
D405_NOMINAL_VERTICAL_FOV_DEGREES = 58.0
D405_IDEAL_MIN_DISTANCE_MM = 70.0
D405_IDEAL_MAX_DISTANCE_MM = 500.0
DEFAULT_MIN_PROJECTED_DIAMETER_PX = 40.0
RECOMMENDATION_BOUNDARY_BUFFER_MM = 1.0
D405_STEREO_BASELINE_MM = 18.0

# One documented source of truth for offline profile specifications.  The
# optimizer uses 30 FPS because the existing measurement workflow fuses a
# roughly 1.5-second, 45-frame burst at 30 FPS.  A connected camera is queried
# instead of assuming this table.
D405_FALLBACK_PROFILE_SPECS: tuple[tuple[int, int, int, float], ...] = (
    (1280, 720, 30, 100.0),
    (848, 480, 30, 70.0),
    (640, 360, 30, 55.0),
    (480, 270, 30, 45.0),
    (424, 240, 30, 40.0),
)


class SetupValidationError(ValueError):
    """A user-supplied setup value is invalid."""


class CameraDiscoveryError(RuntimeError):
    """Connected-camera profiles could not be obtained safely."""


class SetupStatus(str, Enum):
    """Physical and conservative-safety outcome for one setup."""

    VALID = "VALID"
    CONDITIONALLY_USABLE = "CONDITIONALLY_USABLE"
    PHYSICALLY_INVALID = "PHYSICALLY_INVALID"


@dataclass(slots=True, frozen=True)
class TubeDimensions:
    outer_diameter_mm: float
    inner_diameter_mm: float
    usable_height_mm: float
    maximum_fill_height_mm: float

    def __post_init__(self) -> None:
        values = {
            "outer diameter": self.outer_diameter_mm,
            "inner diameter": self.inner_diameter_mm,
            "usable internal height": self.usable_height_mm,
        }
        invalid = [name for name, value in values.items() if value <= 0.0]
        if invalid:
            raise SetupValidationError(
                f"{', '.join(invalid)} must be greater than zero."
            )
        if self.inner_diameter_mm > self.outer_diameter_mm:
            raise SetupValidationError(
                "Tube inner diameter cannot be greater than outer diameter."
            )
        if self.maximum_fill_height_mm < 0.0:
            raise SetupValidationError("Maximum filling height cannot be negative.")
        if self.maximum_fill_height_mm > self.usable_height_mm:
            raise SetupValidationError(
                "Maximum filling height cannot exceed usable internal height."
            )


@dataclass(slots=True, frozen=True)
class SafetyAllowances:
    free_margin_fraction: float = 0.10
    centring_error_mm: float = 3.0
    maximum_tilt_degrees: float = 2.0
    minimum_projected_diameter_px: float = DEFAULT_MIN_PROJECTED_DIAMETER_PX

    def __post_init__(self) -> None:
        if self.free_margin_fraction < 0.0:
            raise SetupValidationError("Free image margin cannot be negative.")
        if self.free_margin_fraction >= 1.0:
            raise SetupValidationError("Free image margin must be less than 100%.")
        if self.centring_error_mm < 0.0:
            raise SetupValidationError("Expected centring error cannot be negative.")
        if not 0.0 <= self.maximum_tilt_degrees < 90.0:
            raise SetupValidationError(
                "Maximum camera tilt must be at least 0 and less than 90 degrees."
            )
        if self.minimum_projected_diameter_px <= 0.0:
            raise SetupValidationError(
                "Minimum projected tube diameter must be greater than zero."
            )


@dataclass(slots=True, frozen=True)
class MountingRange:
    minimum_camera_to_rim_mm: float | None = None
    maximum_camera_to_rim_mm: float | None = None

    def __post_init__(self) -> None:
        if (
            self.minimum_camera_to_rim_mm is not None
            and self.minimum_camera_to_rim_mm <= 0.0
        ):
            raise SetupValidationError(
                "Minimum mounting distance must be greater than zero."
            )
        if (
            self.maximum_camera_to_rim_mm is not None
            and self.maximum_camera_to_rim_mm <= 0.0
        ):
            raise SetupValidationError(
                "Maximum mounting distance must be greater than zero."
            )
        if (
            self.minimum_camera_to_rim_mm is not None
            and self.maximum_camera_to_rim_mm is not None
            and self.minimum_camera_to_rim_mm > self.maximum_camera_to_rim_mm
        ):
            raise SetupValidationError(
                "Minimum mounting distance cannot exceed maximum mounting distance."
            )

    def contains(self, distance_mm: float) -> bool:
        if (
            self.minimum_camera_to_rim_mm is not None
            and distance_mm < self.minimum_camera_to_rim_mm
        ):
            return False
        if (
            self.maximum_camera_to_rim_mm is not None
            and distance_mm > self.maximum_camera_to_rim_mm
        ):
            return False
        return True


@dataclass(slots=True, frozen=True)
class CameraProfile:
    width: int
    height: int
    fps: int
    intrinsics: Intrinsics
    minimum_usable_depth_mm: float
    maximum_reliable_depth_mm: float
    intrinsics_source: str
    approximate: bool = False
    stereo_baseline_mm: float = D405_STEREO_BASELINE_MM

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise SetupValidationError("Camera profile dimensions and FPS must be positive.")
        if (self.intrinsics.width, self.intrinsics.height) != (
            self.width,
            self.height,
        ):
            raise SetupValidationError(
                "Profile dimensions do not match its camera intrinsics."
            )
        if self.intrinsics.fx <= 0.0 or self.intrinsics.fy <= 0.0:
            raise SetupValidationError("Camera focal lengths must be positive.")
        if self.minimum_usable_depth_mm <= 0.0:
            raise SetupValidationError("Profile minimum usable depth must be positive.")
        if self.maximum_reliable_depth_mm <= self.minimum_usable_depth_mm:
            raise SetupValidationError(
                "Maximum reliable depth must exceed profile minimum usable depth."
            )
        if self.stereo_baseline_mm <= 0.0:
            raise SetupValidationError("Stereo baseline must be greater than zero.")

    @property
    def name(self) -> str:
        return f"{self.width}x{self.height}@{self.fps}"

    @property
    def horizontal_fov_degrees(self) -> float:
        left_px, right_px, _, _ = _edge_spans_px(self.intrinsics)
        return degrees(atan(left_px / self.intrinsics.fx) + atan(right_px / self.intrinsics.fx))

    @property
    def vertical_fov_degrees(self) -> float:
        _, _, top_px, bottom_px = _edge_spans_px(self.intrinsics)
        return degrees(atan(top_px / self.intrinsics.fy) + atan(bottom_px / self.intrinsics.fy))


@dataclass(slots=True, frozen=True)
class ProfileDiscovery:
    profiles: tuple[CameraProfile, ...]
    source: str
    device_model: str | None = None
    serial_number: str | None = None
    approximate: bool = False
    notes: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class PlaneCoverage:
    distance_mm: float
    fill_height_mm: float
    visible_width_mm: float
    visible_height_mm: float
    centred_width_mm: float
    centred_height_mm: float
    stereo_overlap_width_mm: float
    nominal_horizontal_margin_mm: float
    nominal_vertical_margin_mm: float
    nominal_stereo_overlap_margin_mm: float
    required_diameter_with_allowance_mm: float
    maximum_supported_diameter_mm: float
    horizontal_remaining_margin_mm: float
    vertical_remaining_margin_mm: float
    outer_pixels_x: float
    outer_pixels_y: float
    inner_pixels_x: float
    inner_pixels_y: float
    sampling_x_mm_per_pixel: float
    sampling_y_mm_per_pixel: float
    nominal_rim_clearance_mm: float
    conservative_rim_clearance_mm: float


@dataclass(slots=True)
class SetupEvaluation:
    profile: CameraProfile
    tube: TubeDimensions
    safety: SafetyAllowances
    camera_to_rim_mm: float
    camera_to_bottom_mm: float
    nearest_surface_distance_mm: float
    farthest_surface_distance_mm: float
    nearest_tilt_envelope_distance_mm: float
    farthest_tilt_envelope_distance_mm: float
    effective_minimum_depth_mm: float
    effective_maximum_depth_mm: float
    rim: PlaneCoverage
    nearest: PlaneCoverage
    bottom: PlaneCoverage
    maximum_supported_tube_diameter_mm: float
    minimum_horizontal_remaining_margin_mm: float
    minimum_vertical_remaining_margin_mm: float
    minimum_projected_outer_diameter_px: float
    near_depth_margin_mm: float
    far_depth_margin_mm: float
    status: SetupStatus
    physical_reasons: list[str] = field(default_factory=list)
    allowance_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    estimated_depth_quality: str = ""
    physical_measurable_fill_min_mm: float | None = None
    physical_measurable_fill_max_mm: float | None = None
    safe_measurable_fill_min_mm: float | None = None
    safe_measurable_fill_max_mm: float | None = None

    @property
    def valid(self) -> bool:
        return self.status is SetupStatus.VALID

    @property
    def usable(self) -> bool:
        return self.status is not SetupStatus.PHYSICALLY_INVALID

    @property
    def reasons(self) -> list[str]:
        return [*self.physical_reasons, *self.allowance_reasons]

    @property
    def measurable_fill_min_mm(self) -> float | None:
        if self.valid:
            return self.safe_measurable_fill_min_mm
        return self.physical_measurable_fill_min_mm

    @property
    def measurable_fill_max_mm(self) -> float | None:
        if self.valid:
            return self.safe_measurable_fill_max_mm
        return self.physical_measurable_fill_max_mm

    @property
    def setup_key(self) -> tuple[int, int, int, float]:
        return (
            self.profile.width,
            self.profile.height,
            self.profile.fps,
            self.camera_to_rim_mm,
        )


@dataclass(slots=True, frozen=True)
class Recommendation:
    category: str
    evaluation: SetupEvaluation
    ranking_score: float
    mounting_minimum_mm: float
    mounting_maximum_mm: float


@dataclass(slots=True, frozen=True)
class OptimizationResult:
    recommendations: tuple[Recommendation, ...]
    valid_candidates: tuple[SetupEvaluation, ...]
    evaluated_count: int
    search_minimum_camera_to_rim_mm: float
    search_maximum_camera_to_rim_mm: float

    @property
    def valid(self) -> bool:
        return bool(self.valid_candidates)

    @property
    def best_compromise(self) -> SetupEvaluation | None:
        for recommendation in self.recommendations:
            if recommendation.category == "BEST_COMPROMISE":
                return recommendation.evaluation
        return self.recommendations[0].evaluation if self.recommendations else None


def camera_to_bottom_mm(camera_to_rim_mm: float, usable_height_mm: float) -> float:
    """Return the optical-origin-to-inner-bottom distance."""
    if camera_to_rim_mm <= 0.0 or usable_height_mm <= 0.0:
        raise SetupValidationError(
            "Camera-to-rim distance and usable height must be greater than zero."
        )
    return float(camera_to_rim_mm + usable_height_mm)


def make_fallback_profiles(maximum_reliable_depth_mm: float) -> ProfileDiscovery:
    """Build approximate D405 profiles from the single documented fallback table."""
    profiles: list[CameraProfile] = []
    for width, height, fps, minimum_z_mm in D405_FALLBACK_PROFILE_SPECS:
        intrinsics = nominal_intrinsics(width, height)
        profiles.append(
            CameraProfile(
                width=width,
                height=height,
                fps=fps,
                intrinsics=intrinsics,
                minimum_usable_depth_mm=minimum_z_mm,
                maximum_reliable_depth_mm=maximum_reliable_depth_mm,
                intrinsics_source="D405 nominal 87 x 58 degree FOV",
                approximate=True,
            )
        )
    return ProfileDiscovery(
        profiles=tuple(profiles),
        source="documented D405 offline fallback table",
        device_model=None,
        serial_number=None,
        approximate=True,
        notes=(
            "No device-specific intrinsics are available; all coverage and sampling values are approximate.",
            "30 FPS is used to match the existing 45-frame measurement workflow.",
            "Stereo-overlap checks use the documented 18 mm D405 baseline.",
        ),
    )


def nominal_intrinsics(width: int, height: int) -> Intrinsics:
    """Convert documented nominal D405 FOV into approximate centred intrinsics."""
    fx = (float(width) / 2.0) / tan(radians(D405_NOMINAL_HORIZONTAL_FOV_DEGREES / 2.0))
    fy = (float(height) / 2.0) / tan(radians(D405_NOMINAL_VERTICAL_FOV_DEGREES / 2.0))
    return Intrinsics(
        width=int(width),
        height=int(height),
        fx=fx,
        fy=fy,
        ppx=(float(width) - 1.0) / 2.0,
        ppy=(float(height) - 1.0) / 2.0,
        distortion_model="nominal_fov_fallback",
        coefficients=(),
    )


def discover_connected_d405_profiles(
    maximum_reliable_depth_mm: float,
) -> ProfileDiscovery:
    """Query a connected D405 and return suitable unique Z16 depth profiles.

    Geometry-equivalent FPS variants are reduced to one profile per resolution:
    30 FPS is preferred to preserve the existing acquisition workflow; otherwise
    the highest available rate between 15 and 30 FPS is retained.
    """
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CameraDiscoveryError(
            "pyrealsense2 is unavailable in this Python environment."
        ) from exc

    try:
        devices = list(rs.context().query_devices())
    except RuntimeError as exc:
        raise CameraDiscoveryError(f"RealSense device query failed: {exc}") from exc
    if not devices:
        raise CameraDiscoveryError("No Intel RealSense camera is connected.")

    descriptions: list[str] = []
    d405_devices: list[Any] = []
    for device in devices:
        name = _device_info(device, rs, "name") or "Unknown RealSense"
        serial = _device_info(device, rs, "serial_number") or "unknown serial"
        descriptions.append(f"{name} ({serial})")
        if "D405" in name.upper():
            d405_devices.append(device)
    if not d405_devices:
        raise CameraDiscoveryError(
            "Connected RealSense device is not a D405: " + ", ".join(descriptions)
        )

    device = d405_devices[0]
    model = _device_info(device, rs, "name") or "Intel RealSense D405"
    serial = _device_info(device, rs, "serial_number")
    raw_profiles: list[CameraProfile] = []
    discovered_depth_profiles = 0
    skipped_unknown_min_z: set[tuple[int, int]] = set()

    try:
        sensors = list(device.query_sensors())
        for sensor in sensors:
            for stream_profile in sensor.get_stream_profiles():
                try:
                    if stream_profile.stream_type() != rs.stream.depth:
                        continue
                    if stream_profile.format() != rs.format.z16:
                        continue
                    video = stream_profile.as_video_stream_profile()
                    width = int(video.width())
                    height = int(video.height())
                    fps = int(stream_profile.fps())
                    discovered_depth_profiles += 1
                    min_z = approximate_d405_min_z_mm(width, height)
                    if min_z is None:
                        skipped_unknown_min_z.add((width, height))
                        continue
                    intrinsics_value = (
                        video.get_intrinsics()
                        if hasattr(video, "get_intrinsics")
                        else video.intrinsics
                    )
                    intrinsics = Intrinsics.from_realsense(intrinsics_value)
                    raw_profiles.append(
                        CameraProfile(
                            width=width,
                            height=height,
                            fps=fps,
                            intrinsics=intrinsics,
                            minimum_usable_depth_mm=min_z,
                            maximum_reliable_depth_mm=maximum_reliable_depth_mm,
                            intrinsics_source="connected D405 calibrated depth intrinsics",
                            approximate=False,
                        )
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    continue
    except RuntimeError as exc:
        raise CameraDiscoveryError(
            f"Could not enumerate D405 depth profiles: {exc}"
        ) from exc

    suitable = _select_measurement_profiles(raw_profiles)
    if not suitable:
        raise CameraDiscoveryError(
            "The D405 exposes no suitable Z16 depth profiles with documented Min-Z values."
        )
    notes: list[str] = [
        f"Queried {discovered_depth_profiles} Z16 depth stream variants; "
        f"retained {len(suitable)} measurement profiles.",
        "Stereo-overlap checks use the documented 18 mm D405 baseline.",
    ]
    if len(d405_devices) > 1:
        notes.append(
            f"Multiple D405 cameras are connected; using {model} ({serial or 'unknown serial'})."
        )
    if skipped_unknown_min_z:
        values = ", ".join(f"{w}x{h}" for w, h in sorted(skipped_unknown_min_z))
        notes.append(
            "Skipped profiles without a documented D405 profile-specific Min-Z: " + values
        )
    return ProfileDiscovery(
        profiles=tuple(suitable),
        source="connected D405",
        device_model=model,
        serial_number=serial,
        approximate=False,
        notes=tuple(notes),
    )


def visible_area_mm(profile: CameraProfile, distance_mm: float) -> tuple[float, float]:
    """Return full pinhole image width and height at a plane."""
    if distance_mm <= 0.0:
        raise SetupValidationError("Surface distance must be greater than zero.")
    left_px, right_px, top_px, bottom_px = _edge_spans_px(profile.intrinsics)
    width_mm = (left_px + right_px) * distance_mm / profile.intrinsics.fx
    height_mm = (top_px + bottom_px) * distance_mm / profile.intrinsics.fy
    return width_mm, height_mm


def centred_available_diameter_mm(
    profile: CameraProfile, distance_mm: float
) -> tuple[float, float]:
    """Return centred diameters limited by the asymmetric principal point."""
    left_px, right_px, top_px, bottom_px = _edge_spans_px(profile.intrinsics)
    horizontal = (
        2.0 * min(left_px, right_px) * distance_mm / profile.intrinsics.fx
    )
    vertical = (
        2.0 * min(top_px, bottom_px) * distance_mm / profile.intrinsics.fy
    )
    return horizontal, vertical


def stereo_overlap_diameter_mm(
    profile: CameraProfile, distance_mm: float
) -> float:
    """Return conservative centred horizontal common stereo coverage.

    The D405 depth origin is tied to one imager. A centred target must remain
    visible from the other imager, so one half-field is reduced by the
    documented 18 mm stereo baseline before forming a centred diameter.
    """
    left_px, right_px, _, _ = _edge_spans_px(profile.intrinsics)
    limiting_half_mm = (
        min(left_px, right_px) * distance_mm / profile.intrinsics.fx
    )
    return max(0.0, 2.0 * (limiting_half_mm - profile.stereo_baseline_mm))


def lateral_sampling_mm_per_pixel(
    profile: CameraProfile, distance_mm: float
) -> tuple[float, float]:
    """Return geometric X/Y sampling; this is not camera precision."""
    if distance_mm <= 0.0:
        raise SetupValidationError("Surface distance must be greater than zero.")
    return (
        distance_mm / profile.intrinsics.fx,
        distance_mm / profile.intrinsics.fy,
    )


def evaluate_setup(
    tube: TubeDimensions,
    profile: CameraProfile,
    camera_to_rim_distance_mm: float,
    safety: SafetyAllowances,
    configured_minimum_depth_mm: float = 0.0,
    configured_maximum_depth_mm: float | None = None,
    mounting_range: MountingRange | None = None,
    plane_step_mm: float = 1.0,
) -> SetupEvaluation:
    """Evaluate every required material plane and both depth boundaries."""
    if camera_to_rim_distance_mm <= 0.0:
        raise SetupValidationError("Camera-to-rim distance must be greater than zero.")
    if configured_minimum_depth_mm < 0.0:
        raise SetupValidationError("Configured minimum depth cannot be negative.")
    if plane_step_mm <= 0.0:
        raise SetupValidationError("Plane evaluation step must be positive.")

    camera_bottom = camera_to_bottom_mm(
        camera_to_rim_distance_mm, tube.usable_height_mm
    )
    nearest_distance = camera_bottom - tube.maximum_fill_height_mm
    farthest_distance = camera_bottom
    tilt_depth_excursion_mm = (
        tube.outer_diameter_mm
        / 2.0
        * sin(radians(safety.maximum_tilt_degrees))
    )
    nearest_tilt_envelope_distance = nearest_distance - tilt_depth_excursion_mm
    farthest_tilt_envelope_distance = farthest_distance + tilt_depth_excursion_mm
    effective_min = max(profile.minimum_usable_depth_mm, configured_minimum_depth_mm)
    effective_max = min(
        profile.maximum_reliable_depth_mm,
        configured_maximum_depth_mm
        if configured_maximum_depth_mm is not None
        else profile.maximum_reliable_depth_mm,
    )

    fill_heights = _inclusive_descending_values(
        tube.maximum_fill_height_mm, 0.0, plane_step_mm
    )
    plane_metrics = [
        _plane_coverage(
            tube=tube,
            profile=profile,
            safety=safety,
            fill_height_mm=fill_height,
            camera_to_bottom_distance_mm=camera_bottom,
        )
        for fill_height in fill_heights
    ]
    rim = _plane_coverage(
        tube=tube,
        profile=profile,
        safety=safety,
        fill_height_mm=tube.usable_height_mm,
        camera_to_bottom_distance_mm=camera_bottom,
    )
    nearest = plane_metrics[0]
    bottom = plane_metrics[-1]
    complete_metrics = [rim, *plane_metrics]
    min_horizontal = min(p.horizontal_remaining_margin_mm for p in complete_metrics)
    min_vertical = min(p.vertical_remaining_margin_mm for p in complete_metrics)
    max_supported = min(p.maximum_supported_diameter_mm for p in complete_metrics)
    min_projected_pixels = min(
        min(p.outer_pixels_x, p.outer_pixels_y) for p in plane_metrics
    )
    near_depth_margin = nearest_tilt_envelope_distance - effective_min
    far_depth_margin = effective_max - farthest_tilt_envelope_distance

    physical_reasons: list[str] = []
    allowance_reasons: list[str] = []
    warnings: list[str] = []
    if nearest_distance < effective_min:
        physical_reasons.append(
            f"Nearest material surface {nearest_distance:.1f} mm is below the "
            f"effective minimum usable depth {effective_min:.1f} mm."
        )
    if farthest_distance > effective_max:
        physical_reasons.append(
            f"Tube bottom {farthest_distance:.1f} mm is beyond the "
            f"effective maximum reliable depth {effective_max:.1f} mm."
        )
    if min(p.nominal_horizontal_margin_mm for p in complete_metrics) < 0.0:
        physical_reasons.append(
            "The real outer tube diameter does not fit the horizontal image field."
        )
    if min(p.nominal_vertical_margin_mm for p in complete_metrics) < 0.0:
        physical_reasons.append(
            "The real outer tube diameter does not fit the vertical image field."
        )
    if min(p.nominal_stereo_overlap_margin_mm for p in complete_metrics) < 0.0:
        physical_reasons.append(
            "The real outer tube diameter does not fit the conservative common "
            "left/right stereo-overlap field."
        )
    if min(p.conservative_rim_clearance_mm for p in plane_metrics) < -1e-9:
        physical_reasons.append(
            "Tube-rim/wall occlusion blocks part of the inner material surface "
            "under the requested alignment envelope."
        )
    if min_projected_pixels < safety.minimum_projected_diameter_px:
        physical_reasons.append(
            f"Outer tube projects to only {min_projected_pixels:.1f} px at the "
            f"least-sampled plane; at least {safety.minimum_projected_diameter_px:.1f} px "
            "is required by the measurement-usefulness heuristic."
        )
    if mounting_range is not None and not mounting_range.contains(
        camera_to_rim_distance_mm
    ):
        physical_reasons.append(
            f"Camera-to-rim distance {camera_to_rim_distance_mm:.1f} mm is outside "
            "the mechanically available mounting range."
        )
    if (
        nearest_distance >= effective_min
        and nearest_tilt_envelope_distance < effective_min
    ):
        allowance_reasons.append(
            f"Nominal near depth passes, but the requested tilt envelope reaches "
            f"{nearest_tilt_envelope_distance:.1f} mm below the "
            f"{effective_min:.1f} mm near limit."
        )
    if (
        farthest_distance <= effective_max
        and farthest_tilt_envelope_distance > effective_max
    ):
        allowance_reasons.append(
            f"Nominal far depth passes, but the requested tilt envelope reaches "
            f"{farthest_tilt_envelope_distance:.1f} mm beyond the "
            f"{effective_max:.1f} mm far limit."
        )
    if min_horizontal <= 0.0 and not any(
        "horizontal image field" in reason or "stereo-overlap" in reason
        for reason in physical_reasons
    ):
        allowance_reasons.append(
            "The tube physically fits horizontally, but the requested free margin, "
            "centring tolerance, or tilt allowance does not."
        )
    if min_vertical <= 0.0 and not any(
        "vertical image field" in reason for reason in physical_reasons
    ):
        allowance_reasons.append(
            "The tube physically fits vertically, but the requested free margin, "
            "centring tolerance, or tilt allowance does not."
        )

    if abs(near_depth_margin) <= 1e-9:
        warnings.append("Nearest surface lies exactly on the minimum-depth boundary.")
    elif 0.0 < near_depth_margin < 5.0:
        warnings.append(
            f"Only {near_depth_margin:.1f} mm near-depth clearance remains."
        )
    if abs(far_depth_margin) <= 1e-9:
        warnings.append("Tube bottom lies exactly on the maximum reliable-depth boundary.")
    elif 0.0 < far_depth_margin < 5.0:
        warnings.append(
            f"Only {far_depth_margin:.1f} mm far-depth clearance remains."
        )
    if profile.approximate:
        warnings.append(
            "Offline nominal FOV is in use; coverage and sampling are approximate."
        )
    if min_horizontal > 0.0 and min_horizontal < 5.0:
        warnings.append(
            f"Minimum horizontal image margin is only {min_horizontal:.1f} mm."
        )
    if min_vertical > 0.0 and min_vertical < 5.0:
        warnings.append(
            f"Minimum vertical image margin is only {min_vertical:.1f} mm."
        )
    if safety.maximum_tilt_degrees > 0.0:
        warnings.append(
            "Tilt allowance uses a conservative worst-direction envelope at every fill height."
        )

    if physical_reasons:
        status = SetupStatus.PHYSICALLY_INVALID
    elif allowance_reasons:
        status = SetupStatus.CONDITIONALLY_USABLE
    else:
        status = SetupStatus.VALID

    globally_physical = (
        farthest_distance <= effective_max
        and rim.nominal_horizontal_margin_mm >= 0.0
        and rim.nominal_vertical_margin_mm >= 0.0
        and rim.nominal_stereo_overlap_margin_mm >= 0.0
        and (
            mounting_range is None
            or mounting_range.contains(camera_to_rim_distance_mm)
        )
    )
    globally_safe = (
        globally_physical
        and farthest_tilt_envelope_distance <= effective_max
        and rim.horizontal_remaining_margin_mm > 0.0
        and rim.vertical_remaining_margin_mm > 0.0
    )
    physical_fill_heights: list[float] = []
    safe_fill_heights: list[float] = []
    for plane in plane_metrics:
        physical_plane = (
            globally_physical
            and effective_min <= plane.distance_mm <= effective_max
            and plane.nominal_horizontal_margin_mm >= 0.0
            and plane.nominal_vertical_margin_mm >= 0.0
            and plane.nominal_stereo_overlap_margin_mm >= 0.0
            and plane.conservative_rim_clearance_mm >= -1e-9
            and min(plane.outer_pixels_x, plane.outer_pixels_y)
            >= safety.minimum_projected_diameter_px
        )
        if physical_plane:
            physical_fill_heights.append(plane.fill_height_mm)
            tilt_excursion = (
                tube.outer_diameter_mm
                / 2.0
                * sin(radians(safety.maximum_tilt_degrees))
            )
            safe_plane = (
                globally_safe
                and plane.distance_mm - tilt_excursion >= effective_min
                and plane.distance_mm + tilt_excursion <= effective_max
                and plane.horizontal_remaining_margin_mm > 0.0
                and plane.vertical_remaining_margin_mm > 0.0
            )
            if safe_plane:
                safe_fill_heights.append(plane.fill_height_mm)
    physical_fill_range = _value_range(physical_fill_heights)
    safe_fill_range = _value_range(safe_fill_heights)

    quality = _estimated_depth_quality(
        status=status,
        approximate=profile.approximate,
        nearest_distance_mm=nearest_tilt_envelope_distance,
        farthest_distance_mm=farthest_tilt_envelope_distance,
        near_margin_mm=near_depth_margin,
        far_margin_mm=far_depth_margin,
    )
    return SetupEvaluation(
        profile=profile,
        tube=tube,
        safety=safety,
        camera_to_rim_mm=float(camera_to_rim_distance_mm),
        camera_to_bottom_mm=camera_bottom,
        nearest_surface_distance_mm=nearest_distance,
        farthest_surface_distance_mm=farthest_distance,
        nearest_tilt_envelope_distance_mm=nearest_tilt_envelope_distance,
        farthest_tilt_envelope_distance_mm=farthest_tilt_envelope_distance,
        effective_minimum_depth_mm=effective_min,
        effective_maximum_depth_mm=effective_max,
        rim=rim,
        nearest=nearest,
        bottom=bottom,
        maximum_supported_tube_diameter_mm=max_supported,
        minimum_horizontal_remaining_margin_mm=min_horizontal,
        minimum_vertical_remaining_margin_mm=min_vertical,
        minimum_projected_outer_diameter_px=min_projected_pixels,
        near_depth_margin_mm=near_depth_margin,
        far_depth_margin_mm=far_depth_margin,
        status=status,
        physical_reasons=physical_reasons,
        allowance_reasons=allowance_reasons,
        warnings=warnings,
        estimated_depth_quality=quality,
        physical_measurable_fill_min_mm=(
            physical_fill_range[0] if physical_fill_range else None
        ),
        physical_measurable_fill_max_mm=(
            physical_fill_range[1] if physical_fill_range else None
        ),
        safe_measurable_fill_min_mm=safe_fill_range[0] if safe_fill_range else None,
        safe_measurable_fill_max_mm=safe_fill_range[1] if safe_fill_range else None,
    )


def optimize_setups(
    tube: TubeDimensions,
    profiles: Sequence[CameraProfile],
    safety: SafetyAllowances,
    configured_minimum_depth_mm: float = 0.0,
    configured_maximum_depth_mm: float | None = None,
    mounting_range: MountingRange | None = None,
    distance_step_mm: float = 1.0,
) -> OptimizationResult:
    """Search all profile/distance combinations and rank three objectives."""
    if not profiles:
        raise SetupValidationError("At least one camera profile is required.")
    if distance_step_mm <= 0.0:
        raise SetupValidationError("Distance search step must be positive.")
    mounting = mounting_range or MountingRange()
    max_depth = min(
        profile.maximum_reliable_depth_mm for profile in profiles
    )
    if configured_maximum_depth_mm is not None:
        max_depth = min(max_depth, configured_maximum_depth_mm)
    search_min = mounting.minimum_camera_to_rim_mm or distance_step_mm
    physical_search_max = max_depth - tube.usable_height_mm
    search_max = min(
        mounting.maximum_camera_to_rim_mm
        if mounting.maximum_camera_to_rim_mm is not None
        else physical_search_max,
        physical_search_max,
    )
    if search_min > search_max:
        return OptimizationResult(
            recommendations=(),
            valid_candidates=(),
            evaluated_count=0,
            search_minimum_camera_to_rim_mm=search_min,
            search_maximum_camera_to_rim_mm=search_max,
        )

    step_count = max(
        0, floor((search_max - search_min) / distance_step_mm + 1e-12)
    )
    distances = [
        search_min + index * distance_step_mm
        for index in range(step_count + 1)
    ]
    if distances and search_max - distances[-1] > 1e-9:
        distances.append(search_max)
    evaluations: list[SetupEvaluation] = []
    for profile in profiles:
        for distance in distances:
            evaluations.append(
                evaluate_setup(
                    tube=tube,
                    profile=profile,
                    camera_to_rim_distance_mm=distance,
                    safety=safety,
                    configured_minimum_depth_mm=configured_minimum_depth_mm,
                    configured_maximum_depth_mm=configured_maximum_depth_mm,
                    mounting_range=mounting,
                )
            )
    usable = [evaluation for evaluation in evaluations if evaluation.usable]
    recommendations = _rank_recommendations(usable)
    return OptimizationResult(
        recommendations=tuple(recommendations),
        valid_candidates=tuple(usable),
        evaluated_count=len(evaluations),
        search_minimum_camera_to_rim_mm=search_min,
        search_maximum_camera_to_rim_mm=search_max,
    )


def evaluate_exact_distance(
    tube: TubeDimensions,
    profiles: Sequence[CameraProfile],
    camera_to_rim_distance_mm: float,
    safety: SafetyAllowances,
    configured_minimum_depth_mm: float = 0.0,
    configured_maximum_depth_mm: float | None = None,
    mounting_range: MountingRange | None = None,
) -> tuple[SetupEvaluation, ...]:
    """Evaluate all compatible profiles at one exact mechanical distance."""
    return tuple(
        evaluate_setup(
            tube=tube,
            profile=profile,
            camera_to_rim_distance_mm=camera_to_rim_distance_mm,
            safety=safety,
            configured_minimum_depth_mm=configured_minimum_depth_mm,
            configured_maximum_depth_mm=configured_maximum_depth_mm,
            mounting_range=mounting_range,
        )
        for profile in profiles
    )


def best_valid_at_exact_distance(
    evaluations: Iterable[SetupEvaluation],
) -> SetupEvaluation | None:
    """Select the highest spatial sampling among physically usable profiles."""
    usable = [evaluation for evaluation in evaluations if evaluation.usable]
    if not usable:
        return None
    fully_valid = [evaluation for evaluation in usable if evaluation.valid]
    selection_pool = fully_valid or usable
    return min(
        selection_pool,
        key=lambda evaluation: (
            max(
                evaluation.nearest.sampling_x_mm_per_pixel,
                evaluation.nearest.sampling_y_mm_per_pixel,
            ),
            -evaluation.minimum_projected_outer_diameter_px,
            -evaluation.profile.fps,
        ),
    )


def nearest_valid_alternatives(
    tube: TubeDimensions,
    profiles: Sequence[CameraProfile],
    requested_camera_to_rim_mm: float,
    safety: SafetyAllowances,
    configured_minimum_depth_mm: float = 0.0,
    configured_maximum_depth_mm: float | None = None,
    mounting_range: MountingRange | None = None,
    limit: int = 3,
) -> tuple[SetupEvaluation, ...]:
    """Return nearby valid 1 mm candidates, respecting the mounting range."""
    result = optimize_setups(
        tube=tube,
        profiles=profiles,
        safety=safety,
        configured_minimum_depth_mm=configured_minimum_depth_mm,
        configured_maximum_depth_mm=configured_maximum_depth_mm,
        mounting_range=mounting_range,
        distance_step_mm=1.0,
    )
    ordered = sorted(
        result.valid_candidates,
        key=lambda evaluation: (
            abs(evaluation.camera_to_rim_mm - requested_camera_to_rim_mm),
            max(
                evaluation.nearest.sampling_x_mm_per_pixel,
                evaluation.nearest.sampling_y_mm_per_pixel,
            ),
        ),
    )
    return tuple(ordered[: max(0, int(limit))])


def _plane_coverage(
    tube: TubeDimensions,
    profile: CameraProfile,
    safety: SafetyAllowances,
    fill_height_mm: float,
    camera_to_bottom_distance_mm: float,
) -> PlaneCoverage:
    distance_mm = camera_to_bottom_distance_mm - fill_height_mm
    visible_width, visible_height = visible_area_mm(profile, distance_mm)
    centred_width, centred_height = centred_available_diameter_mm(
        profile, distance_mm
    )
    stereo_overlap_width = stereo_overlap_diameter_mm(profile, distance_mm)
    below_rim_mm = tube.usable_height_mm - fill_height_mm
    tilt_shift_mm = below_rim_mm * tan(radians(safety.maximum_tilt_degrees))
    radius_coefficient = (
        1.0 / (2.0 * cos(radians(safety.maximum_tilt_degrees)))
        + safety.free_margin_fraction
    )
    fixed_allowance_mm = safety.centring_error_mm + tilt_shift_mm
    required_radius_mm = (
        tube.outer_diameter_mm * radius_coefficient + fixed_allowance_mm
    )
    required_diameter = 2.0 * required_radius_mm
    usable_horizontal = min(centred_width, stereo_overlap_width)
    horizontal_remaining = usable_horizontal - required_diameter
    vertical_remaining = centred_height - required_diameter
    maximum_horizontal = max(
        0.0, (usable_horizontal / 2.0 - fixed_allowance_mm) / radius_coefficient
    )
    maximum_vertical = max(
        0.0, (centred_height / 2.0 - fixed_allowance_mm) / radius_coefficient
    )
    maximum_supported = min(maximum_horizontal, maximum_vertical)
    sampling_x, sampling_y = lateral_sampling_mm_per_pixel(profile, distance_mm)
    camera_to_rim_distance_mm = (
        camera_to_bottom_distance_mm - tube.usable_height_mm
    )
    inner_radius_mm = tube.inner_diameter_mm / 2.0
    depth_below_rim_mm = max(0.0, below_rim_mm)
    denominator = max(
        camera_to_rim_distance_mm + depth_below_rim_mm, 1e-9
    )
    nominal_rim_crossing_radius_mm = (
        inner_radius_mm * camera_to_rim_distance_mm / denominator
    )
    origin_offset_allowance_mm = (
        safety.centring_error_mm
        + camera_to_rim_distance_mm
        * tan(radians(safety.maximum_tilt_degrees))
    )
    conservative_rim_crossing_radius_mm = (
        inner_radius_mm * camera_to_rim_distance_mm
        + origin_offset_allowance_mm * depth_below_rim_mm
    ) / denominator
    return PlaneCoverage(
        distance_mm=distance_mm,
        fill_height_mm=fill_height_mm,
        visible_width_mm=visible_width,
        visible_height_mm=visible_height,
        centred_width_mm=centred_width,
        centred_height_mm=centred_height,
        stereo_overlap_width_mm=stereo_overlap_width,
        nominal_horizontal_margin_mm=centred_width - tube.outer_diameter_mm,
        nominal_vertical_margin_mm=centred_height - tube.outer_diameter_mm,
        nominal_stereo_overlap_margin_mm=(
            stereo_overlap_width - tube.outer_diameter_mm
        ),
        required_diameter_with_allowance_mm=required_diameter,
        maximum_supported_diameter_mm=maximum_supported,
        horizontal_remaining_margin_mm=horizontal_remaining,
        vertical_remaining_margin_mm=vertical_remaining,
        outer_pixels_x=tube.outer_diameter_mm / sampling_x,
        outer_pixels_y=tube.outer_diameter_mm / sampling_y,
        inner_pixels_x=tube.inner_diameter_mm / sampling_x,
        inner_pixels_y=tube.inner_diameter_mm / sampling_y,
        sampling_x_mm_per_pixel=sampling_x,
        sampling_y_mm_per_pixel=sampling_y,
        nominal_rim_clearance_mm=(
            inner_radius_mm - nominal_rim_crossing_radius_mm
        ),
        conservative_rim_clearance_mm=(
            inner_radius_mm - conservative_rim_crossing_radius_mm
        ),
    )


def _rank_recommendations(
    valid_candidates: Sequence[SetupEvaluation],
) -> list[Recommendation]:
    if not valid_candidates:
        return []
    fully_valid = [candidate for candidate in valid_candidates if candidate.valid]
    status_pool = fully_valid or list(valid_candidates)
    away_from_boundary = [
        candidate
        for candidate in status_pool
        if candidate.near_depth_margin_mm >= RECOMMENDATION_BOUNDARY_BUFFER_MM
        and candidate.far_depth_margin_mm >= RECOMMENDATION_BOUNDARY_BUFFER_MM
    ]
    recommendation_pool = away_from_boundary or status_pool

    spatial = min(
        recommendation_pool,
        key=lambda candidate: (
            max(
                candidate.nearest.sampling_x_mm_per_pixel,
                candidate.nearest.sampling_y_mm_per_pixel,
            ),
            candidate.camera_to_rim_mm,
            -candidate.profile.fps,
        ),
    )

    safety_values = {
        candidate.setup_key: _safety_score(candidate)
        for candidate in recommendation_pool
    }
    safest = max(
        recommendation_pool,
        key=lambda candidate: (
            safety_values[candidate.setup_key],
            candidate.minimum_projected_outer_diameter_px,
        ),
    )

    max_pixels = max(
        candidate.minimum_projected_outer_diameter_px
        for candidate in recommendation_pool
    )
    max_depth_buffer = max(
        min(candidate.near_depth_margin_mm, candidate.far_depth_margin_mm)
        for candidate in recommendation_pool
    )
    max_fov_buffer = max(
        min(
            candidate.minimum_horizontal_remaining_margin_mm,
            candidate.minimum_vertical_remaining_margin_mm,
        )
        for candidate in recommendation_pool
    )

    def compromise_score(candidate: SetupEvaluation) -> float:
        sampling_score = candidate.minimum_projected_outer_diameter_px / max(
            max_pixels, 1e-9
        )
        depth_score = max(
            0.0,
            min(candidate.near_depth_margin_mm, candidate.far_depth_margin_mm)
            / max(max_depth_buffer, 1e-9),
        )
        fov_score = max(
            0.0,
            min(
                candidate.minimum_horizontal_remaining_margin_mm,
                candidate.minimum_vertical_remaining_margin_mm,
            )
            / max(max_fov_buffer, 1e-9),
        )
        return 0.45 * sampling_score + 0.30 * depth_score + 0.25 * fov_score

    compromise_scores = {
        candidate.setup_key: compromise_score(candidate)
        for candidate in recommendation_pool
    }
    compromise = max(
        recommendation_pool,
        key=lambda candidate: (
            compromise_scores[candidate.setup_key],
            candidate.minimum_projected_outer_diameter_px,
        ),
    )
    winners = (
        (
            "HIGHEST_SPATIAL_SAMPLING",
            spatial,
            spatial.minimum_projected_outer_diameter_px,
        ),
        ("SAFEST_SETUP", safest, safety_values[safest.setup_key]),
        (
            "BEST_COMPROMISE",
            compromise,
            compromise_scores[compromise.setup_key],
        ),
    )
    recommendations: list[Recommendation] = []
    for category, evaluation, score in winners:
        profile_candidates = [
            candidate
            for candidate in valid_candidates
            if (
                candidate.profile.width,
                candidate.profile.height,
                candidate.profile.fps,
            )
            == (
                evaluation.profile.width,
                evaluation.profile.height,
                evaluation.profile.fps,
            )
            and (
                candidate.valid
                if evaluation.valid
                else candidate.usable
            )
        ]
        recommendations.append(
            Recommendation(
                category=category,
                evaluation=evaluation,
                ranking_score=score,
                mounting_minimum_mm=min(
                    candidate.camera_to_rim_mm for candidate in profile_candidates
                ),
                mounting_maximum_mm=max(
                    candidate.camera_to_rim_mm for candidate in profile_candidates
                ),
            )
        )
    return recommendations


def _safety_score(candidate: SetupEvaluation) -> float:
    depth_span = max(
        candidate.effective_maximum_depth_mm
        - candidate.effective_minimum_depth_mm,
        1e-9,
    )
    depth_score = max(
        0.0,
        min(candidate.near_depth_margin_mm, candidate.far_depth_margin_mm)
        / (depth_span / 2.0),
    )
    required = max(
        candidate.nearest.required_diameter_with_allowance_mm,
        candidate.bottom.required_diameter_with_allowance_mm,
        1e-9,
    )
    fov_score = max(
        0.0,
        min(
            candidate.minimum_horizontal_remaining_margin_mm,
            candidate.minimum_vertical_remaining_margin_mm,
        )
        / required,
    )
    return min(depth_score, fov_score) + 0.20 * (depth_score + fov_score)


def _estimated_depth_quality(
    *,
    status: SetupStatus,
    approximate: bool,
    nearest_distance_mm: float,
    farthest_distance_mm: float,
    near_margin_mm: float,
    far_margin_mm: float,
) -> str:
    prefix = "APPROXIMATE GEOMETRIC ESTIMATE" if approximate else "GEOMETRIC ESTIMATE"
    if status is SetupStatus.PHYSICALLY_INVALID:
        return f"{prefix}: invalid range/coverage; usable depth quality is not expected."
    if status is SetupStatus.CONDITIONALLY_USABLE:
        return (
            f"{prefix}: nominal geometry is usable, but one or more requested "
            "conservative allowances are not met."
        )
    if near_margin_mm < 5.0 or far_margin_mm < 5.0:
        return (
            f"{prefix}: boundary-sensitive because less than 5 mm depth-range "
            "clearance remains."
        )
    if (
        nearest_distance_mm >= D405_IDEAL_MIN_DISTANCE_MM
        and farthest_distance_mm <= D405_IDEAL_MAX_DISTANCE_MM
    ):
        return (
            f"{prefix}: the full range is inside the documented D405 ideal range "
            "with geometric clearance. This is not an accuracy guarantee."
        )
    return (
        f"{prefix}: geometrically valid, but part of the range is outside the "
        "documented 70-500 mm ideal range. Validate with the real target."
    )


def _value_range(values: Sequence[float]) -> tuple[float, float] | None:
    if not values:
        return None
    return min(values), max(values)


def _edge_spans_px(intrinsics: Intrinsics) -> tuple[float, float, float, float]:
    # Pixel edges are half a pixel outside the first/last pixel centres.
    return (
        intrinsics.ppx + 0.5,
        intrinsics.width - 0.5 - intrinsics.ppx,
        intrinsics.ppy + 0.5,
        intrinsics.height - 0.5 - intrinsics.ppy,
    )


def _inclusive_descending_values(
    start: float, stop: float, step: float
) -> list[float]:
    if start < stop:
        raise SetupValidationError("Fill-height range is reversed.")
    values = [float(start)]
    current = start - step
    while current > stop + 1e-9:
        values.append(float(current))
        current -= step
    if abs(values[-1] - stop) > 1e-9:
        values.append(float(stop))
    return values


def _select_measurement_profiles(
    profiles: Sequence[CameraProfile],
) -> list[CameraProfile]:
    grouped: dict[tuple[int, int], list[CameraProfile]] = {}
    for profile in profiles:
        if 15 <= profile.fps <= 30:
            grouped.setdefault((profile.width, profile.height), []).append(profile)
    selected: list[CameraProfile] = []
    for candidates in grouped.values():
        selected.append(
            min(
                candidates,
                key=lambda profile: (
                    0 if profile.fps == 30 else 1,
                    -profile.fps,
                ),
            )
        )
    return sorted(
        selected,
        key=lambda profile: (-profile.width * profile.height, -profile.fps),
    )


def _device_info(device: Any, rs: Any, attribute: str) -> str | None:
    camera_info_value = getattr(rs.camera_info, attribute, None)
    if camera_info_value is None:
        return None
    try:
        if hasattr(device, "supports") and not device.supports(camera_info_value):
            return None
        return str(device.get_info(camera_info_value))
    except (AttributeError, RuntimeError, TypeError):
        return None
