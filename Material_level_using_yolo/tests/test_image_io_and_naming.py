from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from material_level_yolo.errors import ImageIOError
from material_level_yolo.image_io import discover_images, read_image, write_image
from material_level_yolo.logging_utils import configure_logging
from material_level_yolo.naming import artifact_name


def test_unicode_image_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "Étude avec espaces" / "échantillon numéro 1.png"
    image = np.zeros((17, 23, 3), dtype=np.uint8)
    image[:, :, 1] = 173
    assert write_image(path, image) == path.resolve()
    assert np.array_equal(read_image(path), image)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_image(path, image)
    changed = image.copy()
    changed[:, :, 2] = 55
    write_image(path, changed, overwrite=True)
    assert np.array_equal(read_image(path), changed)


def test_bad_image_and_extension_fail_clearly(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(ImageIOError, match="could not decode"):
        read_image(corrupt)
    with pytest.raises(ImageIOError, match="Unsupported"):
        write_image(tmp_path / "output.xyz", np.zeros((2, 2, 3), dtype=np.uint8))


def test_discovery_and_artifact_names_are_deterministic(tmp_path: Path) -> None:
    folder = tmp_path / "Données triées"
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    second = write_image(folder / "B.png", image)
    first = write_image(folder / "á.png", image + 1)
    (folder / "ignore.txt").write_text("not an image", encoding="utf-8")
    discovered = discover_images(folder)
    assert set(discovered) == {first, second}
    name = artifact_name(first, index=1, kind="overlay", extension="png")
    assert name == artifact_name(first, index=1, kind="overlay", extension=".png")
    assert name.startswith("image_000001_")
    assert name.endswith("_overlay.png")
    assert artifact_name(second, index=1, kind="overlay") != name


def test_unicode_log_file_is_utf8_and_setup_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "Journaux de l'étude" / "exécution.log"
    logger = configure_logging("INFO", path)
    logger.info("résultat synthétique")
    assert "résultat synthétique" in path.read_text(encoding="utf-8")
    logger = configure_logging("WARNING", path)
    assert len(logger.handlers) == 2
