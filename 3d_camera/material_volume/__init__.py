"""Standalone D405 material-level and bulk-volume measurement package."""

from .config import AppConfig, load_config
from .errors import MaterialMeasurementError
from .models import CalibrationData, MeasurementResult
from .pipeline import MaterialVolumePipeline
from .session import (
    CapturedDepthBurst,
    DepthMeasurementOutcome,
    DepthMeasurementRequest,
    DepthMeasurementReservation,
    PreparedDepthMethod,
    StorageProfile,
)

__version__ = "0.1.2"

__all__ = [
    "AppConfig",
    "CalibrationData",
    "MaterialMeasurementError",
    "MaterialVolumePipeline",
    "MeasurementResult",
    "CapturedDepthBurst",
    "DepthMeasurementOutcome",
    "DepthMeasurementRequest",
    "DepthMeasurementReservation",
    "PreparedDepthMethod",
    "StorageProfile",
    "__version__",
    "load_config",
]
