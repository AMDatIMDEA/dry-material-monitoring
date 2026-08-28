"""Errors raised by the shared experiment-record package."""


class ExperimentRecordError(RuntimeError):
    """Base error for common record operations."""


class RecordValidationError(ExperimentRecordError, ValueError):
    """A common measurement record violates the frozen contract."""


class IdentifierError(RecordValidationError):
    """An experiment or measurement identifier is invalid."""


class WorkbookSchemaError(ExperimentRecordError):
    """An existing workbook or CSV has an incompatible primary schema."""


class DuplicateRecordError(ExperimentRecordError):
    """Append was requested for an already-existing result key."""


class ConcurrentWriteError(ExperimentRecordError):
    """Another process owns the method workbook's write lock."""


class AtomicWriteError(ExperimentRecordError):
    """The workbook/CSV pair could not be completely replaced."""

