"""
Detector module for Real-Time Object Detection.
Encapsulates YOLO model loading, input validation, and inference logic.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None
    _import_error = exc
else:
    _import_error = None


class ObjectDetectionError(Exception):
    """Base exception for all object detection errors."""
    pass


class ModelLoadError(ObjectDetectionError):
    """Raised when the YOLO model cannot be loaded or initialized."""
    pass


class InferenceError(ObjectDetectionError):
    """Raised when inference on an input frame/image fails."""
    pass


class InvalidThresholdError(ObjectDetectionError, ValueError):
    """Raised when confidence or IOU threshold is outside valid bounds [0.0, 1.0]."""
    pass


@dataclass
class Detection:
    """Represents a single detected bounding box and classification."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary representation."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "xmin": self.bbox[0],
                "ymin": self.bbox[1],
                "xmax": self.bbox[2],
                "ymax": self.bbox[3],
            },
        }


@dataclass
class DetectionResult:
    """Encapsulates all results from an inference pass."""
    detections: List[Detection] = field(default_factory=list)
    total_count: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    inference_time_ms: float = 0.0

    @property
    def has_detections(self) -> bool:
        """True if one or more qualifying objects were detected."""
        return self.total_count > 0


class YOLODetector:
    """
    Modular wrapper around Ultralytics YOLO model.
    Maintains clean separation between model logic, image processing, and UI.
    """

    def __init__(self, model_path: Union[str, Path] = "yolov8n.pt"):
        """
        Initialize the YOLO detector with a pretrained weight file.

        Args:
            model_path: Path or identifier of the pretrained weights (e.g. 'yolov8n.pt').

        Raises:
            ModelLoadError: If the model cannot be loaded, file is corrupt, or dependencies missing.
        """
        if _import_error is not None:
            raise ModelLoadError(
                f"Ultralytics library is not installed: {_import_error}. "
                "Please run `pip install ultralytics`."
            )

        self.model_path = str(model_path)
        self.model = self._load_model(self.model_path)
        # Cache class names lookup
        self.class_names: Dict[int, str] = getattr(self.model, "names", {})

    def _load_model(self, path: str) -> Any:
        """Safely load YOLO model and handle missing/corrupt weights."""
        try:
            model = YOLO(path)
            return model
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load YOLO model from '{path}': {str(e)}. "
                "Ensure the model file exists, is compatible, or that internet access is available "
                "for automatic weight downloading."
            ) from e

    @staticmethod
    def validate_confidence_threshold(threshold: float) -> None:
        """
        Validate that the confidence threshold is a valid float between 0.0 and 1.0.

        Raises:
            InvalidThresholdError: If threshold is not a number or not in [0.0, 1.0].
        """
        if not isinstance(threshold, (int, float)):
            raise InvalidThresholdError(
                f"Confidence threshold must be numeric, received {type(threshold).__name__}: {threshold}"
            )
        if threshold < 0.0 or threshold > 1.0:
            raise InvalidThresholdError(
                f"Confidence threshold must be between 0.0 and 1.0, received {threshold}"
            )

    def detect(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> DetectionResult:
        """
        Run inference on an image input and return filtered detections.

        Args:
            image_input: NumPy array (BGR or RGB), PIL Image, or path to image.
            conf_threshold: Minimum confidence score [0.0, 1.0] to qualify.
            iou_threshold: Intersection over Union threshold for Non-Max Suppression.

        Returns:
            DetectionResult: Structured detection outputs and metrics.

        Raises:
            InvalidThresholdError: If thresholds are invalid.
            InferenceError: If model prediction fails.
        """
        self.validate_confidence_threshold(conf_threshold)

        if not isinstance(iou_threshold, (int, float)) or not (0.0 <= iou_threshold <= 1.0):
            raise InvalidThresholdError(
                f"IOU threshold must be between 0.0 and 1.0, received {iou_threshold}"
            )

        start_time = time.perf_counter()

        try:
            # Predict in inference mode (verbose=False suppresses terminal clutter)
            results = self.model.predict(
                source=image_input,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
        except Exception as e:
            raise InferenceError(f"YOLO model inference failed: {str(e)}") from e

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        if not results:
            return DetectionResult(
                detections=[],
                total_count=0,
                class_counts={},
                inference_time_ms=inference_time_ms,
            )

        first_result = results[0]
        detections: List[Detection] = []
        class_counts: Dict[str, int] = {}

        if first_result.boxes is not None and len(first_result.boxes) > 0:
            boxes = first_result.boxes
            coords = boxes.xyxy.cpu().numpy()  # [N, 4]
            confidences = boxes.conf.cpu().numpy()  # [N]
            class_ids = boxes.cls.cpu().numpy().astype(int)  # [N]

            for xyxy, conf, cls_id in zip(coords, confidences, class_ids):
                conf_val = float(conf)
                # Enforce confidence threshold strictly
                if conf_val < conf_threshold:
                    continue

                xmin, ymin, xmax, ymax = [int(round(coord)) for coord in xyxy]
                cls_name = self.class_names.get(int(cls_id), f"Class {cls_id}")

                det = Detection(
                    class_id=int(cls_id),
                    class_name=cls_name,
                    confidence=conf_val,
                    bbox=(xmin, ymin, xmax, ymax),
                )
                detections.append(det)
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        return DetectionResult(
            detections=detections,
            total_count=len(detections),
            class_counts=class_counts,
            inference_time_ms=round(inference_time_ms, 2),
        )
