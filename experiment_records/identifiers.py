"""Portable experiment IDs, sortable measurement IDs, and UTC normalization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Iterable

from .errors import IdentifierError


EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MEASUREMENT_ID_PATTERN = re.compile(
    r"^(?P<index>[0-9]{6,})_taking_(?P<timestamp>[0-9]{8}T[0-9]{9}Z)$"
)
WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
)


def validate_experiment_id(value: str) -> str:
    """Validate and return a portable session-directory identifier."""
    if not isinstance(value, str) or not EXPERIMENT_ID_PATTERN.fullmatch(value):
        raise IdentifierError(
            "experiment_id must be 1-64 ASCII characters, start with an "
            "alphanumeric character, and contain only letters, digits, '.', '-', or '_'."
        )
    if ".." in value:
        raise IdentifierError("experiment_id must not contain '..'.")
    if value.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
        raise IdentifierError(f"experiment_id uses a reserved Windows name: {value}")
    return value


def parse_utc(value: datetime | str, *, field_name: str = "timestamp") -> datetime:
    """Parse a timezone-aware value and normalize it to UTC."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise IdentifierError(f"{field_name} must not be blank.")
        try:
            parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        except ValueError as exc:
            raise IdentifierError(f"{field_name} is not a valid ISO 8601 timestamp: {value}") from exc
    else:
        raise IdentifierError(f"{field_name} must be a timezone-aware datetime or ISO 8601 text.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise IdentifierError(f"{field_name} must include a timezone offset.")
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime | str) -> str:
    """Return the canonical UTC workbook/JSON representation."""
    return parse_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def filesystem_utc(value: datetime | str) -> str:
    """Return the filesystem-safe UTC timestamp used inside a measurement ID."""
    timestamp = parse_utc(value)
    milliseconds = timestamp.microsecond // 1000
    return f"{timestamp:%Y%m%dT%H%M%S}{milliseconds:03d}Z"


def make_measurement_id(index: int, trigger_time_utc: datetime | str) -> str:
    """Create a sortable, stable shared ID from an index and trigger instant."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise IdentifierError("measurement_index must be an integer greater than or equal to 1.")
    return f"{index:06d}_taking_{filesystem_utc(trigger_time_utc)}"


@dataclass(frozen=True, slots=True)
class ParsedMeasurementId:
    index: int
    trigger_time_utc: datetime


def parse_measurement_id(value: str) -> ParsedMeasurementId:
    """Parse a version-1 measurement ID and validate its calendar timestamp."""
    if not isinstance(value, str):
        raise IdentifierError("measurement_id must be text.")
    match = MEASUREMENT_ID_PATTERN.fullmatch(value)
    if match is None:
        raise IdentifierError(
            "measurement_id must match '<six-or-more digits>_taking_YYYYMMDDTHHMMSSfffZ'."
        )
    try:
        trigger = datetime.strptime(match.group("timestamp"), "%Y%m%dT%H%M%S%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise IdentifierError(f"measurement_id contains an invalid UTC timestamp: {value}") from exc
    return ParsedMeasurementId(index=int(match.group("index")), trigger_time_utc=trigger)


def validate_measurement_identity(
    measurement_id: str,
    measurement_index: int,
    trigger_time_utc: datetime | str,
) -> None:
    """Ensure an ID retains exactly the supplied index and trigger millisecond."""
    if isinstance(measurement_index, bool) or not isinstance(measurement_index, int):
        raise IdentifierError("measurement_index must be an integer.")
    parsed = parse_measurement_id(measurement_id)
    if parsed.index != measurement_index:
        raise IdentifierError(
            "measurement_index does not match the numeric measurement_id prefix."
        )
    expected = make_measurement_id(measurement_index, trigger_time_utc)
    if measurement_id != expected:
        raise IdentifierError(
            "measurement_id timestamp does not match trigger_time_utc truncated to milliseconds."
        )


def next_measurement_identity(
    existing_measurement_ids: Iterable[str],
    trigger_time_utc: datetime | str,
) -> tuple[int, str]:
    """Allocate the next index from durable IDs; caller must serialize allocation."""
    maximum = 0
    seen_indexes: set[int] = set()
    for value in existing_measurement_ids:
        parsed = parse_measurement_id(value)
        if parsed.index in seen_indexes:
            raise IdentifierError(f"Duplicate durable measurement index: {parsed.index}")
        seen_indexes.add(parsed.index)
        maximum = max(maximum, parsed.index)
    index = maximum + 1
    return index, make_measurement_id(index, trigger_time_utc)

