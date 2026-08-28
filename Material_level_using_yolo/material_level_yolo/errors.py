"""Project-specific failures with operator-facing messages."""


class YoloFoundationError(Exception):
    """Base class for expected configuration, model, and image failures."""


class ConfigurationError(YoloFoundationError):
    """The project YAML is missing, malformed, or internally inconsistent."""


class ProfileNotFoundError(ConfigurationError):
    """A requested named model profile does not exist."""


class WeightsError(YoloFoundationError):
    """Configured weights are absent or cannot be loaded."""


class ModelCompatibilityError(YoloFoundationError):
    """Model task or class metadata cannot satisfy the configured semantics."""


class InferenceError(YoloFoundationError):
    """A validated model failed while processing an offline still image."""


class ImageIOError(YoloFoundationError):
    """An image cannot be decoded, encoded, read, or written safely."""


class OperatorInputError(YoloFoundationError):
    """Operator-supplied experiment metadata is incomplete or inconsistent."""


class AcquisitionError(YoloFoundationError):
    """The C920 preview/capture layer could not acquire the requested stills."""
