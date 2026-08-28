"""Frozen common-record schema shared by every measurement method."""

from __future__ import annotations

from enum import StrEnum


SCHEMA_VERSION = "1.0.0"
MEASUREMENTS_SHEET = "measurements"

# This tuple is the single executable definition of the common workbook columns.
# Method packages must import it rather than copying the names.
COMMON_COLUMNS = (
    "schema_version",
    "experiment_id",
    "measurement_id",
    "measurement_index",
    "method",
    "acquisition_mode",
    "trigger_time_utc",
    "capture_start_utc",
    "capture_end_utc",
    "processing_time_utc",
    "material_name",
    "total_capacity_ml",
    "bulk_density_g_per_ml",
    "total_possible_weight_g",
    "reference_material_weight_g",
    "reference_volume_from_weight_ml",
    "manual_material_level_mm",
    "manual_material_percent",
    "reference_volume_from_manual_ml",
    "estimated_material_percent",
    "estimated_material_volume_ml",
    "estimated_empty_percent",
    "estimated_empty_volume_ml",
    "valid",
    "quality_score",
    "status",
    "notes",
    "artifact_directory",
    "source_artifact",
    "config_sha256",
    "calibration_or_model_sha256",
    "software_commit",
)

RESULT_KEY_COLUMNS = ("experiment_id", "measurement_id", "method")

PERCENT_TOLERANCE = 1e-6
ABSOLUTE_VOLUME_TOLERANCE_ML = 1e-6
RELATIVE_FORMULA_TOLERANCE = 1e-9


class Method(StrEnum):
    DEPTH = "depth"
    YOLO = "yolo"


class AcquisitionMode(StrEnum):
    STANDALONE_CAMERA = "standalone_camera"
    BAG = "bag"
    SYNTHETIC = "synthetic"
    SYNCHRONIZED = "synchronized"
    MANUAL_CAMERA = "manual_camera"
    TIMED_CAMERA = "timed_camera"
    FOLDER = "folder"


class RecordStatus(StrEnum):
    PENDING_OFFLINE_INFERENCE = "pending_offline_inference"
    COMPLETE_VALID = "complete_valid"
    COMPLETE_INVALID = "complete_invalid"
    PARTIAL_CAPTURE = "partial_capture"
    ACQUISITION_FAILED = "acquisition_failed"
    PROCESSING_FAILED = "processing_failed"
    CANCELLED = "cancelled"


METHOD_ACQUISITION_MODES = {
    Method.DEPTH: frozenset(
        {
            AcquisitionMode.STANDALONE_CAMERA,
            AcquisitionMode.BAG,
            AcquisitionMode.SYNTHETIC,
            AcquisitionMode.SYNCHRONIZED,
        }
    ),
    Method.YOLO: frozenset(
        {
            AcquisitionMode.SYNCHRONIZED,
            AcquisitionMode.MANUAL_CAMERA,
            AcquisitionMode.TIMED_CAMERA,
            AcquisitionMode.FOLDER,
        }
    ),
}

