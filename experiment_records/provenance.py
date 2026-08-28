"""Streaming SHA-256 and non-failing Git provenance capture."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess

from .schema import SCHEMA_VERSION


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading a large model or capture wholly into memory."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repository: str | Path) -> str | None:
    """Return full revision plus '+dirty', or None when Git metadata is unavailable."""
    root = Path(repository)
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        revision = revision_result.stdout.strip().lower()
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            return None
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return revision + ("+dirty" if status_result.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass(frozen=True, slots=True)
class Provenance:
    schema_version: str
    config_sha256: str
    calibration_or_model_sha256: str
    software_commit: str | None


def capture_provenance(
    *,
    config_path: str | Path,
    calibration_or_model_path: str | Path,
    repository: str | Path,
) -> Provenance:
    """Capture common-row provenance without failing solely on missing Git metadata."""
    return Provenance(
        schema_version=SCHEMA_VERSION,
        config_sha256=sha256_file(config_path),
        calibration_or_model_sha256=sha256_file(calibration_or_model_path),
        software_commit=git_revision(repository),
    )

