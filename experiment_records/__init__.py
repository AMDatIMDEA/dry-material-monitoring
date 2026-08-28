"""Shared, method-neutral material-level experiment record contract."""

from .errors import (
    AtomicWriteError,
    ConcurrentWriteError,
    DuplicateRecordError,
    ExperimentRecordError,
    IdentifierError,
    RecordValidationError,
    WorkbookSchemaError,
)
from .export import MeasurementWorkbookStore
from .identifiers import (
    filesystem_utc,
    format_utc,
    make_measurement_id,
    next_measurement_identity,
    parse_measurement_id,
    parse_utc,
    validate_experiment_id,
    validate_measurement_identity,
)
from .provenance import Provenance, capture_provenance, git_revision, sha256_bytes, sha256_file
from .records import (
    CommonMeasurementRecord,
    derive_estimate,
    derive_manual_references,
    derive_weight_reference_ml,
    with_estimate,
)
from .schema import (
    ABSOLUTE_VOLUME_TOLERANCE_ML,
    COMMON_COLUMNS,
    MEASUREMENTS_SHEET,
    SCHEMA_VERSION,
    AcquisitionMode,
    Method,
    RecordStatus,
)
from .validation import (
    prepare_record,
    validate_aligned_records,
    validate_record,
    volume_tolerance,
)


__all__ = [
    "ABSOLUTE_VOLUME_TOLERANCE_ML",
    "AcquisitionMode",
    "AtomicWriteError",
    "COMMON_COLUMNS",
    "CommonMeasurementRecord",
    "ConcurrentWriteError",
    "DuplicateRecordError",
    "ExperimentRecordError",
    "IdentifierError",
    "MEASUREMENTS_SHEET",
    "MeasurementWorkbookStore",
    "Method",
    "Provenance",
    "RecordStatus",
    "RecordValidationError",
    "SCHEMA_VERSION",
    "WorkbookSchemaError",
    "capture_provenance",
    "derive_estimate",
    "derive_manual_references",
    "derive_weight_reference_ml",
    "filesystem_utc",
    "format_utc",
    "git_revision",
    "make_measurement_id",
    "next_measurement_identity",
    "parse_measurement_id",
    "parse_utc",
    "prepare_record",
    "sha256_bytes",
    "sha256_file",
    "validate_experiment_id",
    "validate_aligned_records",
    "validate_measurement_identity",
    "validate_record",
    "volume_tolerance",
    "with_estimate",
]

__version__ = SCHEMA_VERSION
