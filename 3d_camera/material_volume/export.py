"""Measurement artifacts: JSON, height map, snapshot, and PLY surface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .config import AppConfig
from .models import VolumeEstimate


@dataclass(slots=True, frozen=True)
class ArtifactSelection:
    save_json: bool = True
    save_height_map_npy: bool = True
    save_height_map_png: bool = True
    save_surface_ply: bool = True
    save_color_snapshot: bool = True


def save_measurement(estimate: VolumeEstimate, config: AppConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = config.output_path / timestamp
    folder.mkdir(parents=True, exist_ok=False)
    settings = ArtifactSelection(
        save_json=config.output.save_json,
        save_height_map_npy=config.output.save_height_map_npy,
        save_height_map_png=config.output.save_height_map_png,
        save_surface_ply=config.output.save_surface_ply,
        save_color_snapshot=config.output.save_color_snapshot,
    )
    save_measurement_artifacts(estimate, config, folder, settings)
    return folder


def save_measurement_artifacts(
    estimate: VolumeEstimate,
    config: AppConfig,
    folder: str | Path,
    selection: ArtifactSelection,
) -> Path:
    """Save selected legacy artifacts into an explicit existing/new directory."""
    output = Path(folder)
    output.mkdir(parents=True, exist_ok=True)
    if selection.save_json:
        (output / "result.json").write_text(estimate.result.to_json() + "\n", encoding="utf-8")
    if selection.save_height_map_npy:
        np.save(output / "height_map_mm.npy", estimate.height_map_mm)
    if selection.save_height_map_png:
        _save_height_png(estimate, config, output / "height_map.png")
    if selection.save_surface_ply:
        _save_surface_ply(estimate, output / "material_surface_mm.ply")
    if selection.save_color_snapshot and estimate.color_bgr is not None:
        rgb = estimate.color_bgr[..., ::-1]
        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(output / "color_snapshot.png")
    return output


def _save_height_png(estimate: VolumeEstimate, config: AppConfig, path: Path) -> None:
    height = estimate.height_map_mm.astype(np.float64)
    normalized = np.nan_to_num(height / config.tube.usable_height_mm, nan=0.0)
    normalized = np.clip(normalized, 0.0, 1.0)
    # Vectorized blue-cyan-yellow-red map.
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    rgb[~estimate.inside_mask] = 0.0
    interpolated = estimate.inside_mask & ~estimate.observed_mask
    rgb[interpolated] *= 0.55
    pixels = np.rint(rgb * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    scale = max(1, int(np.ceil(600 / max(image.size))))
    image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST).save(path)


def _save_surface_ply(estimate: VolumeEstimate, path: Path) -> None:
    grid_x, grid_y = np.meshgrid(estimate.x_centers_mm, estimate.y_centers_mm)
    valid = estimate.inside_mask & np.isfinite(estimate.height_map_mm)
    x = grid_x[valid]
    y = grid_y[valid]
    z = estimate.height_map_mm[valid]
    z_range = max(float(np.max(z) - np.min(z)), 1e-9)
    normalized = (z - float(np.min(z))) / z_range
    red = np.rint(255.0 * normalized).astype(np.uint8)
    blue = np.rint(255.0 * (1.0 - normalized)).astype(np.uint8)
    green = np.rint(255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))).astype(np.uint8)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        "comment coordinates are in millimetres in the tube frame\n"
        f"element vertex {x.size}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(header)
        for values in zip(x, y, z, red, green, blue, strict=True):
            px, py, pz, pr, pg, pb = values
            stream.write(f"{px:.5f} {py:.5f} {pz:.5f} {int(pr)} {int(pg)} {int(pb)}\n")
