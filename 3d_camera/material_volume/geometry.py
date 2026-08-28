"""Vectorized pinhole geometry and robust plane fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import CalibrationError
from .models import Intrinsics


@dataclass(slots=True, frozen=True)
class PlaneFit:
    point_m: np.ndarray
    normal_toward_camera: np.ndarray
    rms_m: float
    inlier_mask: np.ndarray


def deproject_pixels(
    u: np.ndarray,
    v: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
) -> np.ndarray:
    """Deproject depth pixels into camera XYZ coordinates.

    D405 depth profiles normally expose zero Brown-Conrady coefficients, making
    this the ordinary pinhole equation. The Brown-Conrady branches retain the
    device calibration if a profile supplies non-zero coefficients.
    """
    z = np.asarray(depth_m, dtype=np.float64)
    x_normalized = (np.asarray(u, dtype=np.float64) - intrinsics.ppx) / intrinsics.fx
    y_normalized = (np.asarray(v, dtype=np.float64) - intrinsics.ppy) / intrinsics.fy
    model = intrinsics.distortion_model.lower()
    coefficients = tuple(intrinsics.coefficients) + (0.0,) * max(
        0, 5 - len(intrinsics.coefficients)
    )
    k1, k2, p1, p2, k3 = coefficients[:5]
    has_distortion = any(abs(value) > 1e-15 for value in (k1, k2, p1, p2, k3))
    if has_distortion and "inverse_brown_conrady" in model:
        radius_squared = x_normalized**2 + y_normalized**2
        radial = 1.0 + k1 * radius_squared + k2 * radius_squared**2 + k3 * radius_squared**3
        original_x = x_normalized
        original_y = y_normalized
        x_normalized = (
            original_x * radial
            + 2.0 * p1 * original_x * original_y
            + p2 * (radius_squared + 2.0 * original_x**2)
        )
        y_normalized = (
            original_y * radial
            + 2.0 * p2 * original_x * original_y
            + p1 * (radius_squared + 2.0 * original_y**2)
        )
    elif has_distortion and (
        "brown_conrady" in model or "modified_brown_conrady" in model
    ):
        # Invert the forward Brown-Conrady mapping by fixed-point iteration,
        # matching the model used by librealsense deprojection.
        distorted_x = x_normalized.copy()
        distorted_y = y_normalized.copy()
        for _ in range(10):
            radius_squared = x_normalized**2 + y_normalized**2
            radial = 1.0 + k1 * radius_squared + k2 * radius_squared**2 + k3 * radius_squared**3
            delta_x = 2.0 * p1 * x_normalized * y_normalized + p2 * (
                radius_squared + 2.0 * x_normalized**2
            )
            delta_y = 2.0 * p2 * x_normalized * y_normalized + p1 * (
                radius_squared + 2.0 * y_normalized**2
            )
            x_normalized = (distorted_x - delta_x) / radial
            y_normalized = (distorted_y - delta_y) / radial
    elif has_distortion and not ("none" in model or "brown_conrady" in model):
        raise CalibrationError(
            f"Unsupported non-zero depth distortion model: {intrinsics.distortion_model}."
        )
    x = x_normalized * z
    y = y_normalized * z
    return np.column_stack((x, y, z))


def depth_image_to_points(
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid XYZ points and their source pixel coordinates."""
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    v, u = np.nonzero(valid)
    points = deproject_pixels(u, v, depth_m[v, u], intrinsics)
    return points, u, v


def fit_plane_robust(
    points_m: np.ndarray,
    trim_sigma: float,
    iterations: int,
    minimum_band_m: float = 0.00025,
) -> PlaneFit:
    """Fit a plane with iterative MAD trimming and orient it toward the camera."""
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 30:
        raise CalibrationError("At least 30 valid bottom-reference points are required.")
    finite = np.all(np.isfinite(points), axis=1)
    if np.count_nonzero(finite) < 30:
        raise CalibrationError("Too few finite 3D points are available for plane fitting.")

    inliers = finite.copy()
    point = np.zeros(3, dtype=np.float64)
    normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    for _ in range(max(1, int(iterations))):
        current = points[inliers]
        if current.shape[0] < 30:
            raise CalibrationError("Robust plane trimming rejected too many reference points.")
        point = np.mean(current, axis=0)
        _, _, vh = np.linalg.svd(current - point, full_matrices=False)
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        # A bottom-plane normal pointing toward the camera has a negative dot
        # product with the plane position vector (the camera is at the origin).
        if float(np.dot(normal, point)) > 0.0:
            normal = -normal
        residuals = (points - point) @ normal
        residual_center = float(np.median(residuals[finite]))
        centered = residuals[finite] - residual_center
        mad = float(np.median(np.abs(centered)))
        robust_sigma = 1.4826 * mad
        band = max(float(minimum_band_m), float(trim_sigma) * robust_sigma)
        updated = finite & (np.abs(residuals - residual_center) <= band)
        if np.array_equal(updated, inliers):
            break
        inliers = updated

    current = points[inliers]
    point = np.mean(current, axis=0)
    _, _, vh = np.linalg.svd(current - point, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if float(np.dot(normal, point)) > 0.0:
        normal = -normal
    rms = float(np.sqrt(np.mean(((current - point) @ normal) ** 2)))
    return PlaneFit(point_m=point, normal_toward_camera=normal, rms_m=rms, inlier_mask=inliers)


def ray_plane_intersection(
    u: float,
    v: float,
    intrinsics: Intrinsics,
    plane_point_m: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    ray = deproject_pixels(
        np.array([float(u)]),
        np.array([float(v)]),
        np.array([1.0]),
        intrinsics,
    )[0]
    denominator = float(np.dot(plane_normal, ray))
    if abs(denominator) < 1e-9:
        raise CalibrationError("The configured tube-center ray is parallel to the bottom plane.")
    distance = float(np.dot(plane_normal, plane_point_m) / denominator)
    if distance <= 0.0:
        raise CalibrationError("The fitted bottom plane lies behind the camera.")
    return distance * ray


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build stable right-handed X/Y axes in the bottom plane."""
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    camera_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    basis_x = camera_x - np.dot(camera_x, n) * n
    if np.linalg.norm(basis_x) < 1e-6:
        camera_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        basis_x = camera_x - np.dot(camera_x, n) * n
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(n, basis_x)
    basis_y /= np.linalg.norm(basis_y)
    return basis_x, basis_y


def camera_points_to_tube(
    points_m: np.ndarray,
    bottom_center_m: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    axis_toward_camera: np.ndarray,
) -> np.ndarray:
    relative = np.asarray(points_m, dtype=np.float64) - np.asarray(bottom_center_m)
    return np.column_stack(
        (
            relative @ np.asarray(basis_x),
            relative @ np.asarray(basis_y),
            relative @ np.asarray(axis_toward_camera),
        )
    )
