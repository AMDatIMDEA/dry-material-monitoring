"""Domain-specific exceptions with user-facing messages."""


class MaterialMeasurementError(RuntimeError):
    """Base error for material measurement failures."""


class ConfigurationError(MaterialMeasurementError):
    """The YAML configuration is missing or physically inconsistent."""


class CameraUnavailableError(MaterialMeasurementError):
    """No usable RealSense device or recording is available."""


class FrameCaptureError(MaterialMeasurementError):
    """The camera did not supply the requested depth frames."""


class CalibrationError(MaterialMeasurementError):
    """An empty-tube calibration could not be created or used."""


class MeasurementQualityError(MaterialMeasurementError):
    """Depth quality is insufficient for a safe volume decision."""
