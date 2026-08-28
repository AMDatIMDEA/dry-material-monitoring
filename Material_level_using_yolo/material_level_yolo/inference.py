"""Lazy Ultralytics adapter with fail-closed semantic and device validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from experiment_records import sha256_file

from .domain import InferenceOutput, ModelProfile, SemanticClassMap, SemanticRole
from .errors import InferenceError, ModelCompatibilityError, WeightsError


class ModelProtocol(Protocol):
    names: Mapping[int, str] | Sequence[str]
    task: str

    def predict(self, **kwargs: Any) -> Any: ...


ModelFactory = Callable[[Path, str], ModelProtocol]
CudaAvailability = Callable[[], bool]


def resolve_inference_device(
    requested: str,
    *,
    cuda_available: CudaAvailability | None = None,
) -> str:
    """Resolve ``auto`` to CUDA when usable, with a deterministic CPU fallback."""
    normalized = requested.strip().lower()
    if normalized != "auto":
        return "cuda:0" if normalized == "cuda" else normalized
    if cuda_available is None:
        try:
            import torch
        except (ImportError, OSError):
            return "cpu"
        cuda_available = torch.cuda.is_available
    try:
        return "cuda:0" if cuda_available() else "cpu"
    except Exception:
        # Device discovery is advisory in auto mode. Inference remains usable
        # on CPU even when a broken driver makes CUDA discovery raise.
        return "cpu"


def normalize_model_names(value: Any) -> dict[int, str]:
    """Normalize Ultralytics list/map metadata without trusting dataset IDs."""
    if isinstance(value, Mapping):
        normalized: dict[int, str] = {}
        for raw_id, raw_name in value.items():
            if isinstance(raw_id, bool) or not (
                isinstance(raw_id, Integral)
                or (isinstance(raw_id, str) and raw_id.strip().isdecimal())
            ):
                raise ModelCompatibilityError(
                    f"model.names class ID {raw_id!r} is not a nonnegative integer."
                )
            class_id = int(raw_id)
            if class_id < 0 or class_id in normalized:
                raise ModelCompatibilityError(
                    f"model.names contains an invalid or duplicate class ID: {class_id}."
                )
            normalized[class_id] = _model_class_name(raw_name, class_id)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized = {
            class_id: _model_class_name(raw_name, class_id)
            for class_id, raw_name in enumerate(value)
        }
    else:
        raise ModelCompatibilityError("model.names must be a class-name list or ID/name mapping.")
    if not normalized:
        raise ModelCompatibilityError("model.names is empty; semantic roles cannot be verified.")
    return dict(sorted(normalized.items()))


def resolve_semantic_classes(
    model_names: Mapping[int, str] | Sequence[str],
    material_role: SemanticRole,
    empty_role: SemanticRole,
) -> SemanticClassMap:
    """Resolve exactly one class for each semantic role or reject the model."""
    names = normalize_model_names(model_names)
    material_id = _resolve_role(names, material_role, "material")
    empty_id = _resolve_role(names, empty_role, "empty")
    if material_id == empty_id:
        raise ModelCompatibilityError(
            "Configured material and empty roles resolve to the same model class."
        )
    return SemanticClassMap(
        material_id=material_id,
        material_name=names[material_id],
        empty_id=empty_id,
        empty_name=names[empty_id],
        model_names=names,
    )


def validate_model_metadata(model: ModelProtocol, profile: ModelProfile) -> SemanticClassMap:
    """Validate task and semantics before any call to model.predict."""
    actual_task = getattr(model, "task", None)
    if not isinstance(actual_task, str) or not actual_task.strip():
        raise ModelCompatibilityError("Loaded model has no usable Ultralytics task metadata.")
    if actual_task.strip().lower() != profile.expected_task:
        raise ModelCompatibilityError(
            f"Profile {profile.name!r} expects Ultralytics task {profile.expected_task!r}, "
            f"but the loaded model reports {actual_task!r}."
        )
    if not hasattr(model, "names"):
        raise ModelCompatibilityError("Loaded model has no model.names metadata.")
    return resolve_semantic_classes(
        model.names,
        profile.material_role,
        profile.empty_role,
    )


class UltralyticsInferenceAdapter:
    """Thin adapter for offline still arrays; no camera or level formula lives here."""

    def __init__(
        self,
        profile: ModelProfile,
        *,
        model_factory: ModelFactory | None = None,
        cuda_available: CudaAvailability | None = None,
    ) -> None:
        self.profile = profile
        self._model_factory = model_factory or _default_model_factory
        self._resolved_device = resolve_inference_device(
            profile.inference.device,
            cuda_available=cuda_available,
        )
        self._model: ModelProtocol | None = None
        self._class_map: SemanticClassMap | None = None
        self._weights_sha256: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def class_map(self) -> SemanticClassMap | None:
        return self._class_map

    @property
    def weights_sha256(self) -> str | None:
        return self._weights_sha256

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

    def load(self) -> SemanticClassMap:
        """Load once, then fail closed on task/classes before inference is possible."""
        if self._model is not None and self._class_map is not None:
            return self._class_map
        weights = self.profile.require_weights()
        weights_hash_before = sha256_file(weights)
        try:
            candidate = self._model_factory(weights, self.profile.expected_task)
        except Exception as exc:
            raise WeightsError(
                f"Could not load weights for profile {self.profile.name!r} "
                f"from {weights}: {exc}"
            ) from exc
        class_map = validate_model_metadata(candidate, self.profile)
        weights_hash_after = sha256_file(weights)
        if weights_hash_after != weights_hash_before:
            raise WeightsError(
                f"Weights changed while profile {self.profile.name!r} was loading: {weights}"
            )
        self._weights_sha256 = weights_hash_before
        self._model = candidate
        self._class_map = class_map
        return class_map

    def predict(self, image: np.ndarray) -> InferenceOutput:
        """Run one already-captured still after metadata and device validation."""
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in {2, 3}:
            raise ValueError("image must be a non-empty NumPy image array.")
        class_map = self.load()
        assert self._model is not None
        assert self._weights_sha256 is not None
        settings = self.profile.inference
        kwargs: dict[str, Any] = {
            "source": image,
            "conf": settings.confidence,
            "iou": settings.iou,
            "imgsz": settings.image_size,
            "device": self._resolved_device,
            "max_det": settings.max_detections,
            "agnostic_nms": settings.nms.agnostic,
            "verbose": False,
        }
        if settings.nms.class_filter_only_semantic:
            kwargs["classes"] = [class_map.material_id, class_map.empty_id]
        try:
            raw = self._model.predict(**kwargs)
        except Exception as exc:
            raise InferenceError(
                f"Offline inference failed for profile {self.profile.name!r}: {exc}"
            ) from exc
        results = tuple(raw) if raw is not None else ()
        return InferenceOutput(
            raw_results=results,
            class_map=class_map,
            model_task=self.profile.expected_task,
            weights_sha256=self._weights_sha256,
            inference_device=self._resolved_device,
        )


def _resolve_role(names: Mapping[int, str], role: SemanticRole, label: str) -> int:
    expected = _semantic_name(role.class_name)
    matches = [class_id for class_id, name in names.items() if _semantic_name(name) == expected]
    if len(matches) > 1:
        raise ModelCompatibilityError(
            f"The configured {label} name {role.class_name!r} is ambiguous in model.names; "
            f"matching IDs: {matches}."
        )
    if role.explicit_id is not None:
        actual = names.get(role.explicit_id)
        if actual is None:
            raise ModelCompatibilityError(
                f"The configured {label} explicit ID {role.explicit_id} is absent from model.names."
            )
        if _semantic_name(actual) != expected:
            raise ModelCompatibilityError(
                f"The configured {label} explicit ID {role.explicit_id} maps to {actual!r}, "
                f"not {role.class_name!r}."
            )
        if matches != [role.explicit_id]:
            raise ModelCompatibilityError(
                f"The configured {label} name/ID pair does not resolve uniquely in model.names."
            )
        return role.explicit_id
    if not matches:
        available = ", ".join(f"{key}:{value}" for key, value in names.items())
        raise ModelCompatibilityError(
            f"The configured {label} class {role.class_name!r} is absent from model.names "
            f"({available})."
        )
    return matches[0]


def _model_class_name(value: Any, class_id: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelCompatibilityError(f"model.names[{class_id}] must be non-blank text.")
    return value.strip()


def _semantic_name(value: str) -> str:
    return value.strip().casefold()


def _default_model_factory(weights: Path, task: str) -> ModelProtocol:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise WeightsError(
            "Ultralytics is not installed. Install this project's runtime dependencies."
        ) from exc
    return YOLO(str(weights), task=task)
