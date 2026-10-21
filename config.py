"""
Configuration module for the Real-Time Object Detection Application.
Centralizes all application constants, model parameters, paths, and UI settings.
"""

from pathlib import Path

# Base Paths (relative to project root, avoiding hardcoded OS-specific paths)
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"

# Model Configurations
DEFAULT_MODEL_NAME = "yolov8n.pt"
DEFAULT_MODEL_PATH = str(BASE_DIR / DEFAULT_MODEL_NAME)
AVAILABLE_MODELS = [
    "yolov8n.pt",  # Nano (Fastest, lightest ~6.5MB)
    "yolov8s.pt",  # Small (Balanced speed and accuracy ~22MB)
    "yolov8m.pt",  # Medium (Higher accuracy ~50MB)
]

# Detection Confidence Threshold Settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
MIN_CONFIDENCE_THRESHOLD = 0.0
MAX_CONFIDENCE_THRESHOLD = 1.0
CONFIDENCE_STEP = 0.05
DEFAULT_IOU_THRESHOLD = 0.45

# Supported File Formats
SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
SUPPORTED_MIME_TYPES = ["image/jpeg", "image/png", "image/jpg"]
MAX_IMAGE_FILE_SIZE_MB = 20

# Webcam Settings
DEFAULT_CAMERA_INDEX = 0
DEFAULT_WEBCAM_INDEX = 0
WEBCAM_FPS_BUFFER_SIZE = 15

# UI Theme & Presentation Settings
APP_TITLE = "Real-Time Object Detection"
APP_SUBTITLE = "Powered by Ultralytics YOLO & OpenCV"
APP_ICON = "🎯"

# Distinct Color Palette for Visual Bounding Boxes (RGB)
CLASS_COLORS = [
    (56, 189, 248),   # Sky Blue
    (248, 113, 113),  # Coral Red
    (74, 222, 128),   # Emerald Green
    (251, 191, 36),   # Amber Yellow
    (192, 132, 252),  # Purple Lavender
    (244, 114, 182),  # Pink Rose
    (45, 212, 191),   # Teal Cyan
    (251, 146, 60),   # Bright Orange
]
