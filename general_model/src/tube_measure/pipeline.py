from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2

from .analyzer import TubeAnalyzer
from .config import AppConfig
from .output import write_results
from .rendering import annotate_image
from .yolo_adapter import instances_from_result


def resolve_inference_device(
    device: Optional[str],
    *,
    cuda_available: Callable[[], bool] | None = None,
) -> str:
    """Prefer CUDA for ``auto``/unset configuration and otherwise use CPU."""
    requested = "auto" if device is None else device.strip().lower()
    if requested not in {"auto", "cpu", "cuda"} and not (
        requested.startswith("cuda:") and requested[5:].isdigit()
    ):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
    if requested == "cuda":
        return "cuda:0"
    if requested != "auto":
        return requested
    if cuda_available is None:
        try:
            import torch
        except (ImportError, OSError):
            return "cpu"
        cuda_available = torch.cuda.is_available
    try:
        return "cuda:0" if cuda_available() else "cpu"
    except Exception:
        return "cpu"


class InferencePipeline:
    def __init__(
        self,
        model_path: str | Path,
        config: AppConfig,
        confidence: Optional[float] = None,
        device: Optional[str] = None,
    ):
        # Keep Ultralytics optional for mask-only library users and unit tests.
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.config = config
        self.analyzer = TubeAnalyzer(config)
        self.confidence = config.inference.confidence if confidence is None else confidence
        requested_device = config.inference.device if device is None else device
        self.device = resolve_inference_device(requested_device)

    def run(self, source: str | Path, output_directory: str | Path) -> List[Dict[str, Any]]:
        source_path = Path(source)
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            records = self._run_folder(source_path, output_path)
        elif source_path.suffix.casefold() in self.config.inference.video_extensions:
            records = self._run_video(source_path, output_path)
        elif source_path.suffix.casefold() in self.config.inference.image_extensions:
            records = [self._run_image(source_path, output_path, source_path.name)]
        else:
            raise ValueError(f"Unsupported source: {source_path}")
        write_results(records, output_path)
        return records

    def _predict(self, image: Any) -> Any:
        arguments: Dict[str, Any] = {
            "source": image,
            "conf": self.confidence,
            "imgsz": self.config.inference.image_size,
            "verbose": False,
        }
        arguments["device"] = self.device
        results = self.model.predict(**arguments)
        if len(results) != 1:
            raise RuntimeError("Expected exactly one result for one image/frame")
        return results[0]

    def _analyze_image(self, image: Any) -> tuple[Any, Any]:
        model_result = self._predict(image)
        analysis = self.analyzer.analyze(instances_from_result(model_result))
        annotated = annotate_image(image, analysis, self.config.rendering)
        return analysis, annotated

    def _run_image(
        self, path: Path, output_directory: Path, source_label: str
    ) -> Dict[str, Any]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read image: {path}")
        analysis, annotated = self._analyze_image(image)
        annotated_directory = output_directory / "annotated"
        annotated_directory.mkdir(parents=True, exist_ok=True)
        output_name = f"{path.stem}_annotated{path.suffix}"
        if not cv2.imwrite(str(annotated_directory / output_name), annotated):
            raise OSError(f"Could not write annotated image for: {path}")
        return {"source": source_label, "frame_index": None, "analysis": analysis.to_dict()}

    def _run_folder(self, source: Path, output_directory: Path) -> List[Dict[str, Any]]:
        extensions = set(self.config.inference.image_extensions)
        paths = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.casefold() in extensions
        )
        if not paths:
            raise ValueError(f"No supported images found in: {source}")
        records: List[Dict[str, Any]] = []
        for index, path in enumerate(paths, start=1):
            relative = path.relative_to(source).as_posix()
            safe_label = f"{index:05d}_{path.name}"
            records.append(self._run_image(path, output_directory, relative))
            # _run_image uses the original name; rename only when collisions can occur.
            current = output_directory / "annotated" / f"{path.stem}_annotated{path.suffix}"
            target = output_directory / "annotated" / f"{Path(safe_label).stem}_annotated{path.suffix}"
            if current != target:
                current.replace(target)
        return records

    def _run_video(self, source: Path, output_directory: Path) -> List[Dict[str, Any]]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise OSError(f"Could not open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_path = output_directory / f"{source.stem}_annotated.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise OSError(f"Could not create output video: {video_path}")

        records: List[Dict[str, Any]] = []
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                analysis, annotated = self._analyze_image(frame)
                writer.write(annotated)
                records.append(
                    {
                        "source": source.name,
                        "frame_index": frame_index,
                        "analysis": analysis.to_dict(),
                    }
                )
                frame_index += 1
        finally:
            capture.release()
            writer.release()
        return records
