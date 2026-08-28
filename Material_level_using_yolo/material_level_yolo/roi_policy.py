"""Explicit runtime policy for calibrated-ROI or whole-image estimation."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .domain import ProjectConfig
from .errors import ConfigurationError


ROI_MODES = ("configured", "whole_image")


def apply_roi_mode(project: ProjectConfig, mode: str) -> ProjectConfig:
    """Return a config whose profiles use the explicitly selected ROI policy."""
    if mode not in ROI_MODES:
        raise ConfigurationError(
            f"roi_mode must be one of: {', '.join(ROI_MODES)}."
        )
    profiles = {}
    for name, profile in project.profiles.items():
        if mode == "configured":
            roi = replace(profile.tube_roi, usage_mode="configured")
        else:
            roi = replace(
                profile.tube_roi,
                left=0.0,
                top=0.0,
                right=1.0,
                bottom=1.0,
                frozen=False,
                calibration_reference=None,
                usage_mode="whole_image",
            )
        profiles[name] = replace(profile, tube_roi=roi)
    return replace(project, profiles=profiles)


def configured_roi_available(project: ProjectConfig) -> bool:
    """Return whether the shared configured ROI is frozen and readable."""
    profile = project.select_profile()
    roi = profile.tube_roi
    return bool(
        roi.frozen
        and roi.calibration_reference is not None
        and roi.calibration_reference.is_file()
        and (roi.left, roi.top, roi.right, roi.bottom) != (0.0, 0.0, 1.0, 1.0)
    )


def choose_roi_mode(
    project: ProjectConfig,
    requested: str | None,
    *,
    stdin_is_tty: bool,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> str:
    """Resolve an explicit CLI choice or ask the operator for y/n."""
    if requested is not None:
        if requested not in ROI_MODES:
            raise ConfigurationError(
                f"roi_mode must be one of: {', '.join(ROI_MODES)}."
            )
        return requested
    if not stdin_is_tty:
        raise ConfigurationError(
            "--roi-mode is required when stdin is noninteractive; choose configured "
            "or whole_image."
        )
    ask = input if input_fn is None else input_fn
    show = print if output_fn is None else output_fn
    available = configured_roi_available(project)
    show(
        "A frozen tube ROI is available."
        if available
        else "No frozen tube ROI is currently configured."
    )
    while True:
        answer = ask("Use the tube ROI for this YOLO calculation? [y/n]: ").strip().casefold()
        if answer in {"y", "yes"}:
            if not available:
                raise ConfigurationError(
                    "No frozen tube ROI is available. Run configure_roi.py first, "
                    "or answer n to use the whole image."
                )
            return "configured"
        if answer in {"n", "no"}:
            show(
                "WARNING: whole-image mode disables calibrated-ROI quality gates. "
                "Dual-class percentages use detected semantic top/bottom as provisional, "
                "uncalibrated limits."
            )
            return "whole_image"
        show("Please enter y or n.")
