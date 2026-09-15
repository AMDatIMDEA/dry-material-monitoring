"""Synchronized warmed C920/D405 laboratory acquisition."""

from .config import load_config
from .core import SynchronizedExperiment, validate_inputs, validate_measurement_references
from .models import (
    ExperimentConfig,
    ExperimentInputs,
    ExperimentOutcome,
    ExperimentState,
    MeasurementReferenceInputs,
)

__version__ = "0.1.2"

__all__ = [
    "ExperimentConfig",
    "ExperimentInputs",
    "ExperimentOutcome",
    "ExperimentState",
    "MeasurementReferenceInputs",
    "SynchronizedExperiment",
    "__version__",
    "load_config",
    "validate_inputs",
    "validate_measurement_references",
]
