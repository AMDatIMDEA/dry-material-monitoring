"""Publication-quality visualizations with explicit missing-data handling."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import AnalysisResult, Comparison, METHODS
from .config import AnalysisConfig


COLORS = {"Human": "#0072B2", "YOLO": "#D55E00", "D405": "#009E73"}
MARKERS = {"Human": "o", "YOLO": "s", "D405": "^"}


def create_all_plots(result: AnalysisResult, config: AnalysisConfig) -> list[Path]:
    try:
        plt.style.use(config.style)
    except OSError as exc:
        raise ValueError(f"Unknown matplotlib style: {config.style}") from exc
    outputs: list[Path] = []
    outputs += _save(_comparison_plot(result, config), "01_estimated_vs_reference_volume", config)
    outputs += _save(_signed_error_plot(result, config), "02_signed_error_vs_reference_fill", config)
    outputs += _save(_absolute_error_boxplot(result, config), "03_absolute_error_distribution", config)
    outputs += _save(_bland_altman_plot(result, config), "04_bland_altman_agreement", config)
    outputs += _save(_repeatability_plot(result, config), "05_repeatability_by_reference_fill", config)
    outputs += _save(_measurement_sequence_plot(result, config), "06_measurement_by_measurement_volume", config)
    return outputs


def _figure(config: AnalysisConfig, *, height_factor: float = 1.0):
    return plt.subplots(figsize=(config.figure_width_inches, config.figure_height_inches * height_factor), constrained_layout=True)


def _comparison_plot(result: AnalysisResult, config: AnalysisConfig):
    fig, ax = _figure(config)
    available = _plot_methods(ax, result, lambda item: item.reference_volume_ml, lambda item: item.estimate_volume_ml)
    values = [value for item in result.comparisons for value in (item.reference_volume_ml, item.estimate_volume_ml)]
    if values:
        low, high = _limits(values)
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.2, label="Ideal y = x")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
    _finish(ax, "Reference (gravimetric) material volume (mL)", "Estimated material volume (mL)", "Comparison of Estimated and Reference Material Volumes", available)
    return fig


def _signed_error_plot(result: AnalysisResult, config: AnalysisConfig):
    fig, ax = _figure(config)
    available = _plot_methods(ax, result, lambda item: item.reference_volume_ml, lambda item: item.error_ml)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, label="Zero error")
    _finish(ax, "Reference (gravimetric) material volume (mL)", "Signed error: estimate − reference (mL)", "Estimation Error as a Function of Reference Fill Level", available)
    return fig


def _absolute_error_boxplot(result: AnalysisResult, config: AnalysisConfig):
    fig, ax = _figure(config)
    labels, values = [], []
    for method in METHODS:
        data = [item.absolute_error_ml for item in result.comparisons if item.method == method]
        if data:
            labels.append(method)
            values.append(data)
    if values:
        boxes = ax.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=True)
        for patch, method in zip(boxes["boxes"], labels, strict=True):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.65)
    else:
        _no_data(ax)
    ax.set_ylabel("Absolute volume error (mL)")
    ax.set_title("Distribution of Absolute Volume Estimation Errors")
    return fig


def _bland_altman_plot(result: AnalysisResult, config: AnalysisConfig):
    fig, axes = plt.subplots(1, 3, figsize=(config.figure_width_inches * 1.7, config.figure_height_inches), constrained_layout=True, sharey=False)
    for ax, method in zip(axes, METHODS, strict=True):
        items = [item for item in result.comparisons if item.method == method]
        if not items:
            _no_data(ax)
            ax.set_title(f"{method} vs reference")
            continue
        means = np.asarray([(item.reference_volume_ml + item.estimate_volume_ml) / 2 for item in items])
        differences = np.asarray([item.error_ml for item in items])
        bias = float(np.mean(differences))
        sd = float(np.std(differences, ddof=1)) if len(items) >= 2 else None
        ax.scatter(means, differences, color=COLORS[method], marker=MARKERS[method], alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axhline(bias, color="black", linewidth=1.3, label=f"Bias {bias:.2f} mL")
        if sd is not None:
            lower, upper = bias - 1.96 * sd, bias + 1.96 * sd
            ax.axhline(lower, color="#666666", linestyle="--", linewidth=1.1, label=f"Lower LoA {lower:.2f}")
            ax.axhline(upper, color="#666666", linestyle="--", linewidth=1.1, label=f"Upper LoA {upper:.2f}")
        else:
            ax.text(0.02, 0.02, "95% LoA unavailable (n < 2)", transform=ax.transAxes, fontsize=8)
        ax.set_title(f"{method} vs reference (n={len(items)})")
        ax.set_xlabel("Mean of method and reference (mL)")
        ax.set_ylabel("Difference: method − reference (mL)")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Bland–Altman Agreement Between Each Method and the Reference", fontsize=13)
    return fig


def _repeatability_plot(result: AnalysisResult, config: AnalysisConfig):
    fig, ax = _figure(config)
    any_data = False
    for method in METHODS:
        items = sorted((item for item in result.comparisons if item.method == method), key=lambda item: item.reference_volume_ml)
        if not items:
            continue
        any_data = True
        ax.scatter([item.reference_volume_ml for item in items], [item.estimate_volume_ml for item in items], color=COLORS[method], marker=MARKERS[method], alpha=0.35, s=28, label=f"{method} individual")
        groups = _groups(items, config.grouping_tolerance_ml)
        centers, means, errors = [], [], []
        for group in groups:
            centers.append(float(np.mean([item.reference_volume_ml for item in group])))
            estimates = np.asarray([item.estimate_volume_ml for item in group])
            means.append(float(np.mean(estimates)))
            errors.append(float(np.std(estimates, ddof=1)) if len(group) >= 2 else 0.0)
        ax.errorbar(centers, means, yerr=errors, color=COLORS[method], marker=MARKERS[method], linewidth=1.4, capsize=3, label=f"{method} mean ± SD")
    if not any_data:
        _no_data(ax)
    _finish(ax, "Reference (gravimetric) material volume (mL)", "Measured material volume (mL)", "Repeatability Across Reference Fill Levels", any_data, legend=True)
    return fig


def _measurement_sequence_plot(result: AnalysisResult, config: AnalysisConfig):
    """Plot ordered measurements, retaining NaN gaps for unavailable methods."""

    fig, ax = _figure(config)
    references = sorted(result.references, key=lambda item: (item.measurement_index, item.measurement_id))
    if not references:
        _no_data(ax)
        ax.set_xlabel("Measurement index")
        ax.set_ylabel("Material volume (mL)")
        ax.set_title("Measurement-by-Measurement Comparison of Material Volume Estimates")
        return fig

    indices = [item.measurement_index for item in references]
    measurement_ids = [item.measurement_id for item in references]
    ax.plot(
        indices,
        [item.reference_volume_ml for item in references],
        color="#000000",
        marker="D",
        markersize=4.5,
        linewidth=1.0,
        label="Gravimetric Reference",
    )
    lookup = {
        (item.measurement_id, item.method): item.estimate_volume_ml
        for item in result.comparisons
    }
    for method in METHODS:
        # NaN is intentional: it breaks the line at an invalid/missing result.
        values = [lookup.get((measurement_id, method), np.nan) for measurement_id in measurement_ids]
        ax.plot(
            indices,
            values,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=4.5,
            linewidth=0.9,
            label={"Human": "Human Visual Estimation", "YOLO": "YOLO Estimation", "D405": "D405 Depth Estimation"}[method],
        )
    ax.set_xlabel("Measurement index")
    ax.set_ylabel("Material volume (mL)")
    ax.set_title("Measurement-by-Measurement Comparison of Material Volume Estimates")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(loc="best", frameon=True)
    return fig


def _groups(items: list[Comparison], tolerance: float) -> list[list[Comparison]]:
    groups: list[list[Comparison]] = []
    for item in items:
        if not groups or abs(item.reference_volume_ml - float(np.mean([member.reference_volume_ml for member in groups[-1]]))) > tolerance:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def _plot_methods(ax, result: AnalysisResult, x: Callable[[Comparison], float], y: Callable[[Comparison], float]) -> bool:
    available = False
    for method in METHODS:
        items = [item for item in result.comparisons if item.method == method]
        if not items:
            continue
        available = True
        ax.scatter([x(item) for item in items], [y(item) for item in items], label=f"{method} (n={len(items)})", color=COLORS[method], marker=MARKERS[method], s=38, alpha=0.8, edgecolor="white", linewidth=0.5)
    if not available:
        _no_data(ax)
    return available


def _finish(ax, xlabel: str, ylabel: str, title: str, available: bool, *, legend: bool = True) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend and available:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", frameon=True)


def _no_data(ax) -> None:
    ax.text(0.5, 0.5, "No valid comparisons available", ha="center", va="center", transform=ax.transAxes)


def _limits(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    span = high - low
    pad = max(span * 0.05, 1.0)
    return low - pad, high + pad


def _save(fig, stem: str, config: AnalysisConfig) -> list[Path]:
    paths = []
    for extension in config.formats:
        path = config.output_directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=config.dpi if extension == "png" else None, bbox_inches="tight", metadata={"Title": stem, "Creator": "measurements_methods_analyze"})
        paths.append(path)
    plt.close(fig)
    return paths
