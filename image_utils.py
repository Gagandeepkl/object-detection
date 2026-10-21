"""
Image utilities module for Real-Time Object Detection.
Handles validation, file loading, format conversion, and visual annotation.
"""

from io import BytesIO
from pathlib import Path
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from config import CLASS_COLORS, OUTPUTS_DIR, SUPPORTED_IMAGE_EXTENSIONS
from src.detector import Detection


class ImageProcessingError(Exception):
    """Base exception for image processing errors."""
    pass


class InvalidImageError(ImageProcessingError):
    """Raised when an uploaded or input image file is invalid or empty."""
    pass


class ImageDecodeError(ImageProcessingError):
    """Raised when image bytes cannot be decoded into a valid image."""
    pass


def validate_uploaded_file(uploaded_file) -> None:
    """
    Validate that an uploaded file object is non-empty and has a supported format.

    Args:
        uploaded_file: Streamlit UploadedFile or file-like object with name and size attributes.

    Raises:
        InvalidImageError: If the file is missing, empty, or has an unsupported extension.
    """
    if uploaded_file is None:
        raise InvalidImageError("No image file was provided.")

    filename = getattr(uploaded_file, "name", "")
    if not filename:
        raise InvalidImageError("Uploaded file has no filename.")

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        supported_str = ", ".join(SUPPORTED_IMAGE_EXTENSIONS)
        raise InvalidImageError(
            f"Unsupported file format '{ext}'. Supported formats are: {supported_str}"
        )

    # Check file size if available
    size = getattr(uploaded_file, "size", None)
    if size is not None and size <= 0:
        raise InvalidImageError("The uploaded file is empty (0 bytes).")


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Decode raw byte array into an RGB NumPy array.

    Args:
        file_bytes: Raw binary bytes of the image.

    Returns:
        np.ndarray: Image array in RGB format.

    Raises:
        InvalidImageError: If byte stream is empty.
        ImageDecodeError: If decoding fails or bytes are corrupted.
    """
    if not file_bytes:
        raise InvalidImageError("Cannot load image from empty byte stream.")

    try:
        # First attempt via Pillow
        with Image.open(BytesIO(file_bytes)) as pil_img:
            # Handle EXIF orientation if needed
            converted = pil_img.convert("RGB")
            return np.array(converted)
    except UnidentifiedImageError as e:
        # Fallback attempt via OpenCV
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ImageDecodeError(
                "Failed to decode image bytes. The file may be corrupt or not a valid image."
            ) from e
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise ImageDecodeError(f"Unexpected error while decoding image: {str(e)}") from e


def load_image_from_path(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from a filesystem path as an RGB NumPy array.

    Args:
        image_path: Path to the image file.

    Returns:
        np.ndarray: Image array in RGB format.

    Raises:
        FileNotFoundError: If the file does not exist.
        InvalidImageError: If the path is not a file or has unsupported extension.
        ImageDecodeError: If OpenCV fails to read the image.
    """
    if image_path is None or str(image_path).strip() == "":
        raise ImageProcessingError("Image path cannot be empty or None.")

    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found at: '{image_path}'")

    if not path.is_file():
        raise InvalidImageError(f"Path is not a regular file: '{image_path}'")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported extension '{ext}' for file '{path.name}'. Supported: {SUPPORTED_IMAGE_EXTENSIONS}"
        )

    # Read image via cv2 with UTF-8 / Windows safe imdecode
    try:
        with open(path, "rb") as f:
            raw_bytes = f.read()
        return load_image_from_bytes(raw_bytes)
    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageDecodeError(f"Failed to read image from path '{image_path}': {str(e)}") from e


def convert_cv2_to_pil(cv_image: np.ndarray, is_bgr: bool = False) -> Image.Image:
    """Convert OpenCV numpy array to PIL Image."""
    if is_bgr:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv_image)


def convert_pil_to_cv2(pil_image: Image.Image, to_bgr: bool = False) -> np.ndarray:
    """Convert PIL Image to NumPy array."""
    arr = np.array(pil_image.convert("RGB"))
    if to_bgr:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def annotate_image(
    image: np.ndarray,
    detections: List[Detection],
    is_bgr: bool = False,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw professional, high-contrast bounding boxes and label badges on an image.

    Args:
        image: Source image as NumPy array (RGB by default, or BGR if is_bgr=True).
        detections: List of Detection instances.
        is_bgr: Whether the input array format is BGR (default False = RGB).
        thickness: Line thickness for bounding boxes.

    Returns:
        np.ndarray: Annotated image in the same color space as input.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Dynamic scaling for fonts and padding based on image dimensions
    scale = max(0.45, min(1.0, max(h, w) / 1000.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = max(1, int(round(scale * 1.8)))

    for det in detections:
        xmin, ymin, xmax, ymax = det.bbox

        # Clamp box to image boundaries
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))

        # Distinct color per class
        color_rgb = CLASS_COLORS[det.class_id % len(CLASS_COLORS)]
        box_color = (color_rgb[2], color_rgb[1], color_rgb[0]) if is_bgr else color_rgb

        # Draw main bounding box
        cv2.rectangle(
            annotated,
            (xmin, ymin),
            (xmax, ymax),
            box_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        # Label text: "class_name 94.2%"
        label_text = f"{det.class_name} {det.confidence * 100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, font, scale, font_thickness
        )

        # Position label badge above box, or flip inside box if too close to image top
        padding = 4
        badge_y1 = ymin - text_h - (padding * 2)
        badge_y2 = ymin
        if badge_y1 < 0:
            badge_y1 = ymin
            badge_y2 = ymin + text_h + (padding * 2)

        badge_x1 = xmin
        badge_x2 = min(w - 1, xmin + text_w + (padding * 2))

        # Filled badge background
        cv2.rectangle(
            annotated,
            (badge_x1, badge_y1),
            (badge_x2, badge_y2),
            box_color,
            thickness=-1,
        )

        # Contrast text color (dark text on light badge)
        text_color = (0, 0, 0)
        text_origin_y = badge_y2 - padding - baseline + 2

        cv2.putText(
            annotated,
            label_text,
            (badge_x1 + padding, text_origin_y),
            font,
            scale,
            text_color,
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated


def save_annotated_image(
    image: np.ndarray,
    output_dir: Union[str, Path] = OUTPUTS_DIR,
    prefix: str = "detection",
    is_bgr: bool = False,
) -> Path:
    """
    Save an annotated image to disk in JPEG format with a timestamped filename.

    Args:
        image: RGB or BGR NumPy array.
        output_dir: Target directory.
        prefix: Filename prefix.
        is_bgr: True if array is BGR, False if RGB.

    Returns:
        Path: Path to the saved image file.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    out_path = out_dir / filename

    save_img = image if is_bgr else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(out_path), save_img)
    if not success:
        raise ImageProcessingError(f"Failed to write annotated image to: {out_path}")

    return out_path
