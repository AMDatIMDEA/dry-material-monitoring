"""Offline YOLO tools with inference/operator exports loaded only on demand."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .camera import CameraPreviewProtocol, OpenCVCameraPreview, PreviewEvent, warm_up_preview
from .config import load_config
from .domain import ModelProfile, ProjectConfig, SemanticClassMap
from .estimation import (
    GroupLevelEstimate,
    ImageLevelEstimate,
    SemanticObservation,
    aggregate_image_estimates,
    estimate_image_level,
)

__version__ = "0.1.2"

_LAZY_EXPORTS = {
    "UltralyticsInferenceAdapter": ("inference", "UltralyticsInferenceAdapter"),
    "resolve_semantic_classes": ("inference", "resolve_semantic_classes"),
    "OfflineMeasurementOutcome": ("workflow", "OfflineMeasurementOutcome"),
    "OfflineMeasurementRequest": ("workflow", "OfflineMeasurementRequest"),
    "process_saved_images": ("workflow", "process_saved_images"),
    "OperatorInputs": ("operator", "OperatorInputs"),
    "OperatorWorkflowResult": ("operator", "OperatorWorkflowResult"),
    "discover_images": ("operator", "discover_images"),
    "run_operator_workflow": ("operator", "run_operator_workflow"),
    "validate_operator_inputs": ("operator", "validate_operator_inputs"),
    "SynchronizedProcessingReport": ("synchronized", "SynchronizedProcessingReport"),
    "discover_synchronized_groups": ("synchronized", "discover_synchronized_groups"),
    "process_synchronized_captures": ("synchronized", "process_synchronized_captures"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


__all__ = [
    "CameraPreviewProtocol",
    "GroupLevelEstimate",
    "ImageLevelEstimate",
    "ModelProfile",
    "OfflineMeasurementOutcome",
    "OfflineMeasurementRequest",
    "OpenCVCameraPreview",
    "OperatorInputs",
    "OperatorWorkflowResult",
    "PreviewEvent",
    "ProjectConfig",
    "SemanticClassMap",
    "SemanticObservation",
    "SynchronizedProcessingReport",
    "UltralyticsInferenceAdapter",
    "__version__",
    "aggregate_image_estimates",
    "discover_images",
    "discover_synchronized_groups",
    "estimate_image_level",
    "load_config",
    "process_saved_images",
    "process_synchronized_captures",
    "resolve_semantic_classes",
    "run_operator_workflow",
    "validate_operator_inputs",
    "warm_up_preview",
]
