"""
Automated unit tests for image validation, loading, decoding, and annotation utilities.
"""

from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from config import SAMPLE_IMAGES_DIR
from src.detector import Detection
from src.image_utils import (
    validate_uploaded_file,
    load_image_from_bytes,
    load_image_from_path,
    annotate_image,
    save_annotated_image,
    InvalidImageError,
    ImageDecodeError,
    ImageProcessingError,
)


class MockUploadedFile:
    """Mock Streamlit UploadedFile object."""

    def __init__(self, name: str, size: int, content: bytes = b""):
        self.name = name
        self.size = size
        self.content = content

    def read(self) -> bytes:
        return self.content


class TestImageUtils:
    """Tests for image utilities."""

    def test_validate_uploaded_file_valid(self):
        """Valid JPEG file passes validation."""
        valid_file = MockUploadedFile("test.jpg", 1024, b"test content")
        # Should not raise
        validate_uploaded_file(valid_file)

    def test_validate_uploaded_file_unsupported_extension(self):
        """Unsupported file extensions raise InvalidImageError."""
        invalid_ext_file = MockUploadedFile("document.pdf", 1024)
        with pytest.raises(InvalidImageError) as exc_info:
            validate_uploaded_file(invalid_ext_file)
        assert "unsupported file format" in str(exc_info.value).lower()

    def test_validate_uploaded_file_empty(self):
        """Empty file (0 bytes) raises InvalidImageError."""
        empty_file = MockUploadedFile("empty.png", 0)
        with pytest.raises(InvalidImageError) as exc_info:
            validate_uploaded_file(empty_file)
        assert "empty" in str(exc_info.value).lower()

    def test_validate_uploaded_file_none(self):
        """None uploaded file raises InvalidImageError."""
        with pytest.raises(InvalidImageError):
            validate_uploaded_file(None)

    def test_load_image_from_empty_bytes(self):
        """Empty byte string raises InvalidImageError."""
        with pytest.raises(InvalidImageError):
            load_image_from_bytes(b"")

    def test_load_image_from_corrupted_bytes(self):
        """Corrupted/random bytes raise ImageDecodeError."""
        corrupted_bytes = b"NOT_A_REAL_IMAGE_DATA_12345"
        with pytest.raises(ImageDecodeError):
            load_image_from_bytes(corrupted_bytes)

    def test_load_image_from_valid_bytes(self):
        """Valid PIL-generated image bytes decode to RGB NumPy array."""
        pil_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        raw_bytes = buffer.getvalue()

        arr = load_image_from_bytes(raw_bytes)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (100, 100, 3)
        # Check red color
        assert arr[50, 50, 0] > 200

    def test_load_image_from_path_missing(self):
        """Missing path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_from_path("non_existent_folder/fake_image.png")

    def test_load_image_from_path_empty_string(self):
        """Empty string path raises ImageProcessingError."""
        with pytest.raises(ImageProcessingError):
            load_image_from_path("")

    def test_annotate_image_structure(self):
        """Annotating an image preserves dimensions and draws overlays."""
        base_img = np.zeros((300, 400, 3), dtype=np.uint8)
        detections = [
            Detection(class_id=0, class_name="person", confidence=0.88, bbox=(20, 20, 100, 150)),
            Detection(class_id=1, class_name="bicycle", confidence=0.75, bbox=(120, 120, 220, 250)),
        ]

        annotated = annotate_image(base_img, detections)
        assert annotated.shape == base_img.shape
        # Pixels where annotations were drawn should now be non-zero
        assert np.any(annotated > 0)

    def test_save_annotated_image(self, tmp_path):
        """Saving annotated image writes file and returns Path."""
        test_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        saved_path = save_annotated_image(test_img, output_dir=tmp_path, prefix="test")

        assert isinstance(saved_path, Path)
        assert saved_path.exists()
        assert saved_path.is_file()
        assert saved_path.stat().st_size > 0
