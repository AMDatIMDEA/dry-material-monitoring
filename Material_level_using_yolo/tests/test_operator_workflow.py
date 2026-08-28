from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
import pytest

from experiment_records import COMMON_COLUMNS
from material_level_yolo.config import load_config
from material_level_yolo.domain import InferenceOutput, SemanticClassMap
from material_level_yolo.errors import AcquisitionError, InferenceError, OperatorInputError
from material_level_yolo.image_io import write_image
from material_level_yolo.operator import (
    OpenCVCameraPreview,
    OperatorInputs,
    PreviewEvent,
    run_operator_workflow,
    validate_operator_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRIGGER = datetime(2026, 8, 13, 16, 0, 0, tzinfo=timezone.utc)


class FakeBoxes:
    def __init__(self) -> None:
        self.cls = np.asarray([7, 3], dtype=np.float32)
        self.conf = np.asarray([0.95, 0.96], dtype=np.float32)
        self.xyxy = np.empty((0, 4), dtype=np.float32)


class FakeMasks:
    def __init__(self, height: int, width: int) -> None:
        material = np.zeros((height, width), dtype=np.float32)
        empty = np.zeros_like(material)
        material[height // 2 :] = 1.0
        empty[: height // 2] = 1.0
        self.data = np.asarray([material, empty])


class FakeResult:
    def __init__(self, height: int, width: int) -> None:
        self.boxes = FakeBoxes()
        self.masks = FakeMasks(height, width)


class FakeCamera:
    def __init__(self, frames, events=()) -> None:
        self.frames = deque(np.asarray(frame).copy() for frame in frames)
        self.events = deque(events)
        self.shown: list[np.ndarray] = []
        self.statuses: list[str] = []
        self.opened = False
        self.closed = False

    def open(self, settings) -> None:
        del settings
        self.opened = True

    def read(self):
        return None if not self.frames else self.frames.popleft().copy()

    def show(self, frame, status) -> None:
        self.shown.append(np.asarray(frame).copy())
        self.statuses.append(status)

    def poll_event(self, wait_ms):
        del wait_ms
        return self.events.popleft() if self.events else PreviewEvent.NONE

    def close(self) -> None:
        self.closed = True


class FakeAdapter:
    def __init__(self, camera: FakeCamera | None = None, fail_calls=()) -> None:
        self.mapping = SemanticClassMap(
            material_id=7,
            material_name="Powder",
            empty_id=3,
            empty_name="Empty",
            model_names={3: "Empty", 7: "Powder"},
        )
        self.weights_sha256 = "a" * 64
        self.camera = camera
        self.fail_calls = set(fail_calls)
        self.load_calls = 0
        self.predict_calls = 0
        self.predicted_means: list[float] = []

    def load(self):
        self.load_calls += 1
        return self.mapping

    def predict(self, image):
        self.predict_calls += 1
        if self.camera is not None:
            assert self.camera.closed, "Inference occurred while the preview camera was open."
        if self.predict_calls in self.fail_calls:
            raise InferenceError("mock per-image inference failure")
        self.predicted_means.append(float(np.mean(image)))
        return InferenceOutput(
            raw_results=(FakeResult(image.shape[0], image.shape[1]),),
            class_map=self.mapping,
            model_task="segment",
            weights_sha256=self.weights_sha256,
        )


class StepClock:
    def __init__(self, values) -> None:
        self.values = deque(values)
        self.last = values[-1]

    def __call__(self):
        if self.values:
            self.last = self.values.popleft()
        return self.last


@pytest.fixture
def project(tmp_path: Path):
    loaded = load_config(PROJECT_ROOT / "config.yaml")
    reference = tmp_path / "calibration" / "tube_roi.json"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("{}\n", encoding="utf-8")
    profiles = {}
    for name, profile in loaded.profiles.items():
        profiles[name] = replace(
            profile,
            camera=replace(profile.camera, warmup_frames=2, preview_wait_ms=1),
            tube_roi=replace(
                profile.tube_roi,
                frozen=True,
                left=0.05,
                right=0.95,
                calibration_reference=reference,
            ),
        )
    return replace(loaded, profiles=profiles)


def inputs(tmp_path: Path, mode: str, **updates) -> OperatorInputs:
    values = {
        "profile_name": "powder",
        "experiment_id": f"operator-{mode.replace('_', '-')}",
        "purpose": "hardware-free operator workflow test",
        "total_capacity_ml": 200.0,
        "acquisition_mode": mode,
        "output_root": (tmp_path / "Résultats avec espaces").resolve(),
        "material_name": "Powder",
        "operator_notes": "note opérateur",
        "software_repository": PROJECT_ROOT.parent,
    }
    values.update(updates)
    return OperatorInputs(**values)


def frames(count: int) -> list[np.ndarray]:
    return [np.full((100, 80, 3), index, dtype=np.uint8) for index in range(1, count + 1)]


def test_material_name_once_selects_profile_without_separate_profile_input(
    project, tmp_path: Path
) -> None:
    validated = validate_operator_inputs(
        project,
        inputs(tmp_path, "manual_camera", profile_name=None, material_name="polymer"),
    )
    assert validated.profile.name == "polymer"
    assert validated.material_name == "polymer"


def test_manual_key_capture_never_infers_on_warmup_or_preview_frames(project, tmp_path: Path) -> None:
    camera = FakeCamera(
        frames(4),
        [PreviewEvent.NONE, PreviewEvent.NONE, PreviewEvent.NONE, PreviewEvent.CAPTURE],
    )
    adapter = FakeAdapter(camera)
    summaries = []

    def confirm(summary):
        assert not camera.opened
        summaries.append(dict(summary))
        return True

    result = run_operator_workflow(
        project,
        inputs(tmp_path, "manual_camera"),
        confirm=confirm,
        camera=camera,
        adapter=adapter,
        utc_now=lambda: TRIGGER,
    )
    assert not result.cancelled
    assert result.captured_still_count == 1
    assert result.outcome is not None and result.outcome.group.valid
    assert result.outcome.record.acquisition_mode == "manual_camera"
    assert adapter.predict_calls == 1
    assert adapter.predicted_means == [4.0]
    assert len(camera.shown) == 4
    assert camera.closed
    assert summaries[0]["continuous_or_preview_inference"] is False
    assert summaries[0]["purpose"] == "hardware-free operator workflow test"


def test_left_click_event_captures_only_the_deliberate_still(project, tmp_path: Path) -> None:
    camera = FakeCamera(
        frames(3),
        [PreviewEvent.NONE, PreviewEvent.NONE, PreviewEvent.CAPTURE],
    )
    adapter = FakeAdapter(camera)
    result = run_operator_workflow(
        project,
        inputs(tmp_path, "manual_camera", experiment_id="operator-click"),
        confirm=lambda _summary: True,
        camera=camera,
        adapter=adapter,
        utc_now=lambda: TRIGGER,
    )
    assert result.outcome is not None
    assert adapter.predict_calls == 1
    assert adapter.predicted_means == [3.0]


def test_opencv_preview_maps_left_click_and_configured_key(monkeypatch, project) -> None:
    preview = OpenCVCameraPreview()
    preview._settings = project.select_profile("powder").camera
    monkeypatch.setattr("material_level_yolo.camera.cv2.waitKey", lambda _wait: -1)
    monkeypatch.setattr(
        "material_level_yolo.camera.cv2.getWindowProperty", lambda _name, _prop: 1.0
    )
    preview._mouse_callback(1, 0, 0, 0, None)
    assert preview.poll_event(1) is PreviewEvent.CAPTURE
    monkeypatch.setattr(
        "material_level_yolo.camera.cv2.waitKey",
        lambda _wait: ord(preview._settings.capture_key),
    )
    assert preview.poll_event(1) is PreviewEvent.CAPTURE
    preview._settings = None


def test_timed_count_schedule_spools_then_infers_without_backlog(project, tmp_path: Path) -> None:
    camera = FakeCamera(
        frames(5),
        [PreviewEvent.NONE] * 5,
    )
    adapter = FakeAdapter(camera)
    clock = StepClock([0.0, 0.0, 0.25, 0.5])
    result = run_operator_workflow(
        project,
        inputs(
            tmp_path,
            "timed_camera",
            timed_capture_count=3,
            capture_interval_seconds=0.25,
        ),
        confirm=lambda _summary: True,
        camera=camera,
        adapter=adapter,
        monotonic=clock,
        utc_now=lambda: TRIGGER,
    )
    assert result.outcome is not None
    assert result.captured_still_count == 3
    assert adapter.predict_calls == 3
    assert adapter.predicted_means == [3.0, 4.0, 5.0]
    assert result.outcome.record.acquisition_mode == "timed_camera"
    assert result.outcome.record.capture_start_utc is not None
    assert result.outcome.record.capture_end_utc is not None


def test_timed_duration_uses_start_and_end_schedule(project, tmp_path: Path) -> None:
    camera = FakeCamera(frames(5), [PreviewEvent.NONE] * 5)
    adapter = FakeAdapter(camera)
    result = run_operator_workflow(
        project,
        inputs(
            tmp_path,
            "timed_camera",
            experiment_id="operator-duration",
            capture_interval_seconds=0.5,
            capture_duration_seconds=1.0,
        ),
        confirm=lambda _summary: True,
        camera=camera,
        adapter=adapter,
        monotonic=StepClock([0.0, 0.0, 0.5, 1.0]),
        utc_now=lambda: TRIGGER,
    )
    assert result.outcome is not None
    assert result.captured_still_count == 3
    assert adapter.predict_calls == 3


def test_folder_mode_uses_only_supplied_extensions_and_no_camera(project, tmp_path: Path) -> None:
    folder = tmp_path / "Entrée dossier é"
    write_image(folder / "a.png", frames(1)[0])
    nested = folder / "nested"
    write_image(nested / "b.JPG", frames(1)[0])
    (folder / "ignore.txt").write_text("not an image", encoding="utf-8")
    adapter = FakeAdapter()
    result = run_operator_workflow(
        project,
        inputs(
            tmp_path,
            "folder",
            folder_path=folder,
            recursive=True,
        ),
        confirm=lambda summary: summary["folder_image_count"] == 2,
        adapter=adapter,
        utc_now=lambda: TRIGGER,
    )
    assert result.outcome is not None
    assert result.captured_still_count == 2
    assert adapter.predict_calls == 2
    assert result.outcome.record.acquisition_mode == "folder"


def test_confirmation_decline_never_opens_camera_loads_model_or_writes(project, tmp_path: Path) -> None:
    camera = FakeCamera(frames(1))
    adapter = FakeAdapter(camera)
    request = inputs(tmp_path, "manual_camera", experiment_id="operator-declined")
    result = run_operator_workflow(
        project,
        request,
        confirm=lambda _summary: False,
        camera=camera,
        adapter=adapter,
    )
    assert result.cancelled and result.cancellation_reason == "confirmation_declined"
    assert not camera.opened
    assert adapter.load_calls == adapter.predict_calls == 0
    assert not request.output_root.exists()


def test_model_validation_failure_occurs_before_camera_open(project, tmp_path: Path) -> None:
    camera = FakeCamera(frames(1))
    adapter = FakeAdapter(camera)

    def fail_load():
        adapter.load_calls += 1
        raise InferenceError("mock incompatible model")

    adapter.load = fail_load
    with pytest.raises(InferenceError, match="incompatible"):
        run_operator_workflow(
            project,
            inputs(tmp_path, "manual_camera", experiment_id="operator-model-failure"),
            confirm=lambda _summary: True,
            camera=camera,
            adapter=adapter,
        )
    assert not camera.opened
    assert adapter.predict_calls == 0


def test_window_close_after_partial_timed_capture_writes_no_measurement(project, tmp_path: Path) -> None:
    camera = FakeCamera(
        frames(3),
        [PreviewEvent.NONE, PreviewEvent.NONE, PreviewEvent.CLOSED],
    )
    adapter = FakeAdapter(camera)
    request = inputs(
        tmp_path,
        "timed_camera",
        experiment_id="operator-window-close",
        timed_capture_count=3,
        capture_interval_seconds=0.5,
    )
    result = run_operator_workflow(
        project,
        request,
        confirm=lambda _summary: True,
        camera=camera,
        adapter=adapter,
        monotonic=StepClock([0.0, 0.0]),
        utc_now=lambda: TRIGGER,
    )
    assert result.cancelled and result.cancellation_reason == "window_closed"
    assert result.captured_still_count == 1
    assert adapter.predict_calls == 0
    assert not request.output_root.exists()


def test_camera_failure_after_partial_capture_closes_and_writes_nothing(project, tmp_path: Path) -> None:
    camera = FakeCamera(frames(3), [PreviewEvent.NONE] * 3)
    adapter = FakeAdapter(camera)
    request = inputs(
        tmp_path,
        "timed_camera",
        experiment_id="operator-camera-failure",
        timed_capture_count=3,
        capture_interval_seconds=0.5,
    )
    with pytest.raises(AcquisitionError, match="after 1 of 3"):
        run_operator_workflow(
            project,
            request,
            confirm=lambda _summary: True,
            camera=camera,
            adapter=adapter,
            monotonic=StepClock([0.0, 0.0, 0.1]),
            utc_now=lambda: TRIGGER,
        )
    assert camera.closed
    assert adapter.predict_calls == 0
    assert not request.output_root.exists()


def test_per_image_inference_failure_is_retained_not_invented(project, tmp_path: Path) -> None:
    folder = tmp_path / "partial model inputs"
    paths = [write_image(folder / f"image_{index}.png", frame) for index, frame in enumerate(frames(3), 1)]
    adapter = FakeAdapter(fail_calls={2})
    result = run_operator_workflow(
        project,
        inputs(
            tmp_path,
            "folder",
            experiment_id="operator-partial-inference",
            folder_path=folder,
        ),
        confirm=lambda _summary: True,
        adapter=adapter,
        utc_now=lambda: TRIGGER,
    )
    assert len(paths) == 3
    assert result.outcome is not None
    assert result.outcome.group.valid
    assert result.outcome.group.accepted_count == 2
    assert result.outcome.group.rejected_count == 1
    assert result.outcome.record.estimated_material_percent == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"purpose": ""}, "purpose"),
        ({"total_capacity_ml": 0.0}, "greater than zero"),
        ({"bulk_density_g_per_ml": float("nan")}, "finite"),
        ({"remaining_weight_g": -1.0}, "nonnegative"),
        ({"manual_material_level": 2.0}, "explicit unit"),
        (
            {"manual_material_level": 2.0, "manual_material_level_unit": "cm"},
            "usable_internal_height_mm",
        ),
        ({"output_root": Path("relative-output")}, "absolute"),
        ({"measurement_id": "unsafe-id"}, "measurement_id"),
        ({"acquisition_mode": "live"}, "exactly one"),
        (
            {
                "timed_capture_count": 3,
                "capture_duration_seconds": 1.0,
                "capture_interval_seconds": 0.2,
            },
            "not both",
        ),
    ],
)
def test_all_operator_inputs_fail_closed_before_confirmation(
    project, tmp_path: Path, updates, message: str
) -> None:
    base_mode = "timed_camera" if "timed_capture_count" in updates else "manual_camera"
    with pytest.raises(OperatorInputError, match=message):
        validate_operator_inputs(project, inputs(tmp_path, base_mode, **updates))


def test_manual_units_references_manifest_auto_indexes_and_common_columns(project, tmp_path: Path) -> None:
    folder = tmp_path / "supplied images"
    for index, frame in enumerate(frames(3), 1):
        write_image(folder / f"source_{index}.png", frame)
    request = inputs(
        tmp_path,
        "folder",
        experiment_id="operator-indexing",
        folder_path=folder,
        bulk_density_g_per_ml=0.5,
        total_possible_weight_g=100.0,
        remaining_weight_g=25.0,
        manual_material_level=4.0,
        manual_material_level_unit="cm",
        usable_internal_height_mm=80.0,
    )
    first = run_operator_workflow(
        project, request, confirm=lambda _summary: True, adapter=FakeAdapter(), utc_now=lambda: TRIGGER
    )
    second = run_operator_workflow(
        project,
        replace(request, trigger_time_utc=TRIGGER + timedelta(seconds=1)),
        confirm=lambda _summary: True,
        adapter=FakeAdapter(),
        utc_now=lambda: TRIGGER + timedelta(seconds=1),
    )
    assert first.outcome is not None and second.outcome is not None
    assert first.outcome.record.measurement_index == 1
    assert second.outcome.record.measurement_index == 2
    assert first.outcome.record.reference_volume_from_weight_ml == 50.0
    assert first.outcome.record.manual_material_level_mm == 40.0
    assert first.outcome.record.manual_material_percent == 50.0
    workbook = load_workbook(first.outcome.session_directory / "yolo_measurements.xlsx")
    assert tuple(cell.value for cell in workbook["measurements"][1]) == COMMON_COLUMNS
    manifest = json.loads(
        (first.outcome.session_directory / "session_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["purpose"] == request.purpose
    assert [item["measurement_index"] for item in manifest["measurements"]] == [1, 2]
