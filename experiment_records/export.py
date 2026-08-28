"""Locked, interruption-resistant Excel/CSV storage for common records."""

from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
from zipfile import BadZipFile

from openpyxl import Workbook, load_workbook
from openpyxl.workbook.workbook import Workbook as WorkbookType

from .errors import (
    AtomicWriteError,
    ConcurrentWriteError,
    DuplicateRecordError,
    WorkbookSchemaError,
)
from .records import CommonMeasurementRecord
from .identifiers import validate_experiment_id, validate_measurement_identity
from .schema import COMMON_COLUMNS, MEASUREMENTS_SHEET, SCHEMA_VERSION, Method
from .validation import prepare_record


def _blank_to_none(value: object) -> object | None:
    return None if value == "" else value


def _csv_value(column: str, value: str) -> object | None:
    if value == "":
        return None
    if column == "measurement_index":
        return int(value)
    if column == "valid":
        lowered = value.lower()
        if lowered not in {"true", "false"}:
            raise WorkbookSchemaError(f"Invalid CSV boolean for valid: {value!r}")
        return lowered == "true"
    if column in {
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
        "quality_score",
    }:
        return float(value)
    return value


class MeasurementWorkbookStore:
    """Single-method common workbook plus CSV mirror.

    Writes are serialized by an exclusive lock file. A caller must not bypass
    this store to write the primary sheet. The XLSX is replaced first and is the
    recovery authority if interruption occurs before CSV replacement; the next
    successful create/upsert rewrites and reconciles both files.
    """

    def __init__(
        self,
        workbook_path: str | Path,
        *,
        method: Method | str,
        csv_path: str | Path | None = None,
        lock_timeout_seconds: float = 10.0,
        lock_poll_seconds: float = 0.05,
    ) -> None:
        self.workbook_path = Path(workbook_path)
        self.csv_path = Path(csv_path) if csv_path is not None else self.workbook_path.with_suffix(".csv")
        if self.workbook_path.suffix.lower() != ".xlsx":
            raise ValueError("workbook_path must end in .xlsx")
        if self.csv_path.suffix.lower() != ".csv":
            raise ValueError("csv_path must end in .csv")
        if self.workbook_path == self.csv_path:
            raise ValueError("Workbook and CSV paths must be different.")
        self.method = Method(method)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.lock_poll_seconds = float(lock_poll_seconds)
        self.lock_path = self.workbook_path.with_suffix(self.workbook_path.suffix + ".lock")

    def create(
        self,
        records: Iterable[CommonMeasurementRecord] = (),
        *,
        usable_internal_height_mm: float | None = None,
    ) -> None:
        """Create a new workbook/CSV pair and refuse to overwrite either target."""
        with self._write_lock():
            if self.workbook_path.exists() or self.csv_path.exists():
                raise FileExistsError("Refusing to overwrite an existing workbook or CSV mirror.")
            prepared = [
                self._prepare(item, usable_internal_height_mm=usable_internal_height_mm)
                for item in records
            ]
            self._ensure_unique(prepared)
            self._write_pair(prepared, preserve_workbook=False)

    def append(
        self,
        record: CommonMeasurementRecord,
        *,
        usable_internal_height_mm: float | None = None,
    ) -> None:
        """Append one record and fail when its three-field result key exists."""
        with self._write_lock():
            existing = self._load_authoritative_records()
            prepared = self._prepare(record, usable_internal_height_mm=usable_internal_height_mm)
            if prepared.key() in {item.key() for item in existing}:
                raise DuplicateRecordError(f"Record already exists: {prepared.key()}")
            self._write_pair([*existing, prepared], preserve_workbook=True)

    def upsert(
        self,
        record: CommonMeasurementRecord,
        *,
        usable_internal_height_mm: float | None = None,
    ) -> None:
        """Create, append, or idempotently replace by experiment/measurement/method."""
        with self._write_lock():
            existing = self._load_authoritative_records()
            prepared = self._prepare(record, usable_internal_height_mm=usable_internal_height_mm)
            by_key = {item.key(): item for item in existing}
            by_key[prepared.key()] = prepared
            self._write_pair(list(by_key.values()), preserve_workbook=True)

    def upsert_with_diagnostic_rows(
        self,
        record: CommonMeasurementRecord,
        *,
        sheet_name: str,
        columns: Sequence[str],
        rows: Iterable[Sequence[Any]],
        replace_where: Mapping[str, Any],
        usable_internal_height_mm: float | None = None,
    ) -> None:
        """Atomically upsert a common row and one method-specific diagnostic group.

        Existing diagnostic rows matching every ``replace_where`` field are
        replaced. Other measurements and all unrelated workbook sheets remain.
        """
        diagnostic = self._validate_diagnostic_update(
            sheet_name,
            columns,
            rows,
            replace_where,
        )
        with self._write_lock():
            existing = self._load_authoritative_records()
            prepared = self._prepare(
                record,
                usable_internal_height_mm=usable_internal_height_mm,
            )
            by_key = {item.key(): item for item in existing}
            by_key[prepared.key()] = prepared
            self._write_pair(
                list(by_key.values()),
                preserve_workbook=True,
                diagnostic_update=diagnostic,
            )

    def load_records(self) -> list[CommonMeasurementRecord]:
        """Load the authoritative XLSX, falling back to CSV after XLSX corruption/absence."""
        return self._load_authoritative_records()

    def verify_mirror(self) -> bool:
        """Return whether XLSX and CSV contain identical common rows and headers."""
        if not self.workbook_path.is_file() or not self.csv_path.is_file():
            return False
        workbook_records, _ = self._read_workbook(self.workbook_path)
        csv_records = self._read_csv(self.csv_path)
        return [item.to_row() for item in workbook_records] == [item.to_row() for item in csv_records]

    def _prepare(
        self,
        record: CommonMeasurementRecord,
        *,
        usable_internal_height_mm: float | None,
    ) -> CommonMeasurementRecord:
        prepared = prepare_record(record, usable_internal_height_mm=usable_internal_height_mm)
        if prepared.method != self.method.value:
            raise WorkbookSchemaError(
                f"This {self.method.value} store cannot contain method={prepared.method!r}."
            )
        return prepared

    def _load_authoritative_records(self) -> list[CommonMeasurementRecord]:
        if self.workbook_path.is_file():
            try:
                records, _ = self._read_workbook(self.workbook_path)
                return records
            except (BadZipFile, OSError, EOFError):
                if not self.csv_path.is_file():
                    raise
        if self.csv_path.is_file():
            return self._read_csv(self.csv_path)
        return []

    def _read_workbook(self, path: Path) -> tuple[list[CommonMeasurementRecord], WorkbookType]:
        workbook = load_workbook(path)
        if MEASUREMENTS_SHEET not in workbook.sheetnames:
            raise WorkbookSchemaError(f"Workbook has no {MEASUREMENTS_SHEET!r} sheet: {path}")
        sheet = workbook[MEASUREMENTS_SHEET]
        headers = tuple(cell.value for cell in sheet[1])
        if headers != COMMON_COLUMNS:
            raise WorkbookSchemaError("Workbook measurements headers do not match schema 1.0.0.")
        records = []
        for values in sheet.iter_rows(min_row=2, max_col=len(COMMON_COLUMNS), values_only=True):
            if all(value is None for value in values):
                continue
            mapping = {column: _blank_to_none(value) for column, value in zip(COMMON_COLUMNS, values)}
            records.append(CommonMeasurementRecord.from_mapping(mapping))
        self._ensure_loaded_method(records)
        self._ensure_unique(records)
        return records, workbook

    def _read_csv(self, path: Path) -> list[CommonMeasurementRecord]:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != COMMON_COLUMNS:
                raise WorkbookSchemaError("CSV headers do not match schema 1.0.0.")
            records = [
                CommonMeasurementRecord.from_mapping(
                    {column: _csv_value(column, row[column]) for column in COMMON_COLUMNS}
                )
                for row in reader
                if any(row[column] != "" for column in COMMON_COLUMNS)
            ]
        self._ensure_loaded_method(records)
        self._ensure_unique(records)
        return records

    def _ensure_loaded_method(self, records: Iterable[CommonMeasurementRecord]) -> None:
        wrong_versions = [item.schema_version for item in records if item.schema_version != SCHEMA_VERSION]
        if wrong_versions:
            raise WorkbookSchemaError(
                f"Workbook rows must all use schema_version={SCHEMA_VERSION!r}: "
                f"{sorted(set(wrong_versions))}"
            )
        wrong = [item.method for item in records if item.method != self.method.value]
        if wrong:
            raise WorkbookSchemaError(
                f"A {self.method.value} workbook contains other method values: {sorted(set(wrong))}"
            )
        for item in records:
            try:
                validate_experiment_id(item.experiment_id)  # type: ignore[arg-type]
                validate_measurement_identity(
                    item.measurement_id,  # type: ignore[arg-type]
                    item.measurement_index,  # type: ignore[arg-type]
                    item.trigger_time_utc,  # type: ignore[arg-type]
                )
            except (TypeError, ValueError) as exc:
                raise WorkbookSchemaError(f"Stored row has an invalid stable identity: {item.key()}") from exc

    @staticmethod
    def _ensure_unique(records: Iterable[CommonMeasurementRecord]) -> None:
        seen: set[tuple[str | None, str | None, str | None]] = set()
        for item in records:
            if item.key() in seen:
                raise WorkbookSchemaError(f"Duplicate common result key: {item.key()}")
            seen.add(item.key())

    def _write_pair(
        self,
        records: list[CommonMeasurementRecord],
        *,
        preserve_workbook: bool,
        diagnostic_update: tuple[
            str,
            tuple[str, ...],
            tuple[tuple[Any, ...], ...],
            dict[str, Any],
        ]
        | None = None,
    ) -> None:
        ordered = sorted(records, key=lambda item: (item.experiment_id or "", item.measurement_index or 0))
        self.workbook_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        workbook_temp = self._temporary_path(self.workbook_path, ".tmp.xlsx")
        csv_temp = self._temporary_path(self.csv_path, ".tmp.csv")
        try:
            workbook = self._workbook_for_write(preserve_workbook)
            sheet = workbook[MEASUREMENTS_SHEET]
            if sheet.max_row:
                sheet.delete_rows(1, sheet.max_row)
            sheet.append(list(COMMON_COLUMNS))
            for item in ordered:
                sheet.append(list(item.to_row()))
            sheet.freeze_panes = "A2"
            expected_diagnostics = None
            if diagnostic_update is not None:
                expected_diagnostics = self._apply_diagnostic_update(
                    workbook,
                    diagnostic_update,
                )
            workbook.save(workbook_temp)
            check_records, checked_workbook = self._read_workbook(workbook_temp)
            if diagnostic_update is not None and expected_diagnostics is not None:
                diagnostic_sheet = checked_workbook[diagnostic_update[0]]
                actual_diagnostics = tuple(
                    tuple(row)
                    for row in diagnostic_sheet.iter_rows(values_only=True)
                )
                if not self._diagnostic_rows_equal(
                    actual_diagnostics,
                    expected_diagnostics,
                ):
                    raise WorkbookSchemaError(
                        "Temporary diagnostic sheet differs; targets were not replaced."
                    )
            # Excel is the recovery authority and may canonicalize floating-point
            # cells to its stored precision. Build the CSV from that verified
            # round-trip so both durable views contain identical values.
            self._write_csv(csv_temp, check_records)
            check_csv = self._read_csv(csv_temp)
            if [item.to_row() for item in check_records] != [item.to_row() for item in check_csv]:
                raise WorkbookSchemaError("Temporary XLSX and CSV rows differ; targets were not replaced.")

            # XLSX is the recovery authority. If CSV replacement is interrupted,
            # the next upsert reads this XLSX and regenerates both mirrors.
            os.replace(workbook_temp, self.workbook_path)
            os.replace(csv_temp, self.csv_path)
        except Exception as exc:
            if isinstance(exc, (WorkbookSchemaError, AtomicWriteError)):
                raise
            raise AtomicWriteError(
                "Could not atomically replace the workbook/CSV pair. The next successful "
                "upsert will reconcile from the authoritative XLSX when it exists."
            ) from exc
        finally:
            workbook_temp.unlink(missing_ok=True)
            csv_temp.unlink(missing_ok=True)

    @staticmethod
    def _validate_diagnostic_update(
        sheet_name: str,
        columns: Sequence[str],
        rows: Iterable[Sequence[Any]],
        replace_where: Mapping[str, Any],
    ) -> tuple[str, tuple[str, ...], tuple[tuple[Any, ...], ...], dict[str, Any]]:
        if (
            not isinstance(sheet_name, str)
            or not sheet_name.strip()
            or sheet_name == MEASUREMENTS_SHEET
            or len(sheet_name) > 31
            or any(character in sheet_name for character in "[]:*?/\\")
        ):
            raise ValueError("sheet_name must be a valid non-primary Excel sheet name.")
        headers = tuple(columns)
        if not headers or any(not isinstance(item, str) or not item for item in headers):
            raise ValueError("Diagnostic columns must be non-blank text.")
        if len(set(headers)) != len(headers):
            raise ValueError("Diagnostic columns must be unique.")
        replacements = dict(replace_where)
        if not replacements or not set(replacements) <= set(headers):
            raise ValueError("replace_where must name one or more diagnostic columns.")
        retained_rows = tuple(tuple(row) for row in rows)
        if any(len(row) != len(headers) for row in retained_rows):
            raise ValueError("Every diagnostic row must match the diagnostic column count.")
        return sheet_name, headers, retained_rows, replacements

    @staticmethod
    def _apply_diagnostic_update(
        workbook: WorkbookType,
        update: tuple[str, tuple[str, ...], tuple[tuple[Any, ...], ...], dict[str, Any]],
    ) -> tuple[tuple[Any, ...], ...]:
        sheet_name, headers, new_rows, replace_where = update
        if sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            existing_headers = tuple(cell.value for cell in sheet[1])
            if existing_headers != headers:
                raise WorkbookSchemaError(
                    f"Workbook {sheet_name!r} headers do not match the requested diagnostic schema."
                )
            existing_rows = [
                tuple(values)
                for values in sheet.iter_rows(
                    min_row=2,
                    max_col=len(headers),
                    values_only=True,
                )
                if any(value is not None for value in values)
            ]
            indexes = {name: headers.index(name) for name in replace_where}
            existing_rows = [
                row
                for row in existing_rows
                if not all(row[indexes[name]] == value for name, value in replace_where.items())
            ]
            sheet.delete_rows(1, sheet.max_row)
        else:
            sheet = workbook.create_sheet(sheet_name)
            existing_rows = []
        final_rows = [*existing_rows, *new_rows]
        sheet.append(list(headers))
        for row in final_rows:
            sheet.append(list(row))
        sheet.freeze_panes = "A2"
        return (tuple(headers), *(tuple(row) for row in final_rows))

    @staticmethod
    def _diagnostic_rows_equal(
        left: tuple[tuple[Any, ...], ...],
        right: tuple[tuple[Any, ...], ...],
    ) -> bool:
        if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right)):
            return False
        for left_row, right_row in zip(left, right):
            for left_value, right_value in zip(left_row, right_row):
                if (
                    (left_value is None or left_value == "")
                    and (right_value is None or right_value == "")
                ):
                    continue
                if (
                    isinstance(left_value, (int, float))
                    and not isinstance(left_value, bool)
                    and isinstance(right_value, (int, float))
                    and not isinstance(right_value, bool)
                ):
                    if not math.isclose(
                        float(left_value),
                        float(right_value),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        return False
                elif left_value != right_value:
                    return False
        return True

    def _workbook_for_write(self, preserve_workbook: bool) -> WorkbookType:
        if preserve_workbook and self.workbook_path.is_file():
            try:
                _, workbook = self._read_workbook(self.workbook_path)
                return workbook
            except (BadZipFile, OSError, EOFError):
                pass
        workbook = Workbook()
        workbook.active.title = MEASUREMENTS_SHEET
        return workbook

    @staticmethod
    def _temporary_path(target: Path, suffix: str) -> Path:
        descriptor, raw_path = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=suffix, dir=target.parent)
        os.close(descriptor)
        return Path(raw_path)

    @staticmethod
    def _write_csv(path: Path, records: Iterable[CommonMeasurementRecord]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(COMMON_COLUMNS), extrasaction="raise")
            writer.writeheader()
            for item in records:
                row = item.to_mapping()
                writer.writerow(
                    {
                        column: (
                            ""
                            if row[column] is None
                            else str(row[column]).lower()
                            if isinstance(row[column], bool)
                            else row[column]
                        )
                        for column in COMMON_COLUMNS
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.lock_timeout_seconds
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError as exc:
                if time.monotonic() >= deadline:
                    raise ConcurrentWriteError(
                        f"Timed out waiting for workbook writer lock: {self.lock_path}. "
                        "After a crash, verify that no writer is active before removing a stale lock."
                    ) from exc
                time.sleep(self.lock_poll_seconds)
        try:
            payload = json.dumps({"pid": os.getpid(), "created_unix": time.time()}).encode("utf-8")
            os.write(descriptor, payload)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
            self.lock_path.unlink(missing_ok=True)
