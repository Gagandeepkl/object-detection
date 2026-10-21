"""
Automated test suite for YOLO detector and webcam components.
Implements test cases TC1 through TC6 matching the project specification.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from config import DEFAULT_MODEL_PATH, SAMPLE_IMAGES_DIR
from src.detector import (
    YOLODetector,
    DetectionResult,
    ModelLoadError,
    InvalidThresholdError,
)
from src.image_utils import load_image_from_path, ImageProcessingError
from src.webcam import WebcamManager, WebcamAccessError


@pytest.fixture(scope="module")
def shared_detector():
    """Load pretrained YOLO detector once for all module tests."""
    return YOLODetector(DEFAULT_MODEL_PATH)


class TestYOLODetector:
    """Core test cases for YOLO detection logic and validation."""

    def test_tc1_person_detection(self, shared_detector):
        """
        TC1: An image containing a visible person should detect a person
        when the selected pretrained model supports the class and the person is sufficiently visible.
        """
        person_image_path = SAMPLE_IMAGES_DIR / "person.jpg"
        assert person_image_path.exists(), f"Sample image {person_image_path} must exist"

        img = load_image_from_path(person_image_path)
        result = shared_detector.detect(img, conf_threshold=0.25)

        assert isinstance(result, DetectionResult)
        assert result.has_detections is True
        assert "person" in result.class_counts
        assert result.class_counts["person"] >= 1

        # Check each detected person has valid confidence and bbox
        person_detections = [d for d in result.detections if d.class_name == "person"]
        for p in person_detections:
            assert p.confidence >= 0.25
            xmin, ymin, xmax, ymax = p.bbox
            assert xmax > xmin
            assert ymax > ymin

    def test_tc2_car_or_dog_detection(self, shared_detector):
        """
        TC2: An image containing a car or dog should detect supported objects
        when confidence exceeds the configured threshold.
        """
        dog_image_path = SAMPLE_IMAGES_DIR / "dog.jpg"
        assert dog_image_path.exists(), f"Sample image {dog_image_path} must exist"

        img = load_image_from_path(dog_image_path)
        result = shared_detector.detect(img, conf_threshold=0.25)

        assert isinstance(result, DetectionResult)
        assert result.has_detections is True

        detected_classes = set(result.class_counts.keys())
        # The benchmark dog image contains a dog, bicycle, and car/truck
        has_car_or_dog = ("dog" in detected_classes) or ("car" in detected_classes)
        assert has_car_or_dog is True, f"Expected 'dog' or 'car' in detections, got {detected_classes}"

    def test_tc3_unrecognizable_objects_empty_detections(self, shared_detector):
        """
        TC3: An image with no recognizable supported objects should return
        an empty detection list or no qualifying boxes.
        """
        blank_image_path = SAMPLE_IMAGES_DIR / "blank_texture.jpg"
        assert blank_image_path.exists(), f"Sample image {blank_image_path} must exist"

        img = load_image_from_path(blank_image_path)
        result = shared_detector.detect(img, conf_threshold=0.25)

        assert isinstance(result, DetectionResult)
        assert result.total_count == 0
        assert len(result.detections) == 0
        assert result.has_detections is False
        assert result.class_counts == {}

    def test_tc4_high_confidence_threshold_filtering(self, shared_detector):
        """
        TC4: When confidence threshold is 0.90, detections below 0.90
        must not be displayed.
        """
        dog_image_path = SAMPLE_IMAGES_DIR / "dog.jpg"
        img = load_image_from_path(dog_image_path)

        # Baseline detection at lower threshold
        baseline_result = shared_detector.detect(img, conf_threshold=0.25)
        # Strict detection at 0.90 threshold
        strict_result = shared_detector.detect(img, conf_threshold=0.90)

        # All detections in strict_result MUST have confidence >= 0.90
        for det in strict_result.detections:
            assert det.confidence >= 0.90, f"Detection {det.class_name} has confidence {det.confidence} < 0.90"

        # Detections below 0.90 in baseline must not be present in strict
        strict_count = strict_result.total_count
        assert strict_count <= baseline_result.total_count

    def test_tc5_nonexistent_image_path_error(self):
        """
        TC5: A nonexistent image path must produce a controlled error.
        """
        fake_path = "non_existent_image_file_12345.jpg"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_image_from_path(fake_path)

        assert "not found" in str(exc_info.value).lower()

    def test_tc6_unavailable_webcam_error_handling(self):
        """
        TC6: An unavailable webcam must produce a useful error and exit cleanly.
        Uses an invalid device index (-999 or non-existent hardware).
        """
        # Using a nonexistent device index to simulate missing hardware
        manager = WebcamManager(camera_index=999)

        with pytest.raises(WebcamAccessError) as exc_info:
            manager.open_camera()

        # Check that error message is informative and guides the user
        error_msg = str(exc_info.value)
        assert "unavailable" in error_msg.lower() or "denied" in error_msg.lower()

        # Verify manager cleaned up its resources and is not active
        assert manager.is_active is False
        assert manager.cap is None

    def test_invalid_confidence_threshold_bounds(self, shared_detector):
        """Verify invalid confidence thresholds raise InvalidThresholdError."""
        dog_img = load_image_from_path(SAMPLE_IMAGES_DIR / "blank_texture.jpg")

        with pytest.raises(InvalidThresholdError):
            shared_detector.detect(dog_img, conf_threshold=-0.1)

        with pytest.raises(InvalidThresholdError):
            shared_detector.detect(dog_img, conf_threshold=1.5)

        with pytest.raises(InvalidThresholdError):
            shared_detector.detect(dog_img, conf_threshold="invalid")

    def test_invalid_model_path_error(self):
        """Verify attempting to load a nonexistent weights file raises ModelLoadError."""
        with pytest.raises(ModelLoadError) as exc_info:
            YOLODetector("non_existent_model_weights.pt")

        assert "failed to load yolo model" in str(exc_info.value).lower()
