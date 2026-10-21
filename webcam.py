"""
Webcam management module for Real-Time Object Detection.
Manages OpenCV VideoCapture, frame acquisition, camera release, and FPS calculation.
"""

from collections import deque
import logging
import platform
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WebcamError(Exception):
    """Base exception for webcam operations."""
    pass


class WebcamAccessError(WebcamError):
    """Raised when the camera device cannot be opened or permission is denied."""
    pass


class WebcamFrameError(WebcamError):
    """Raised when reading a frame from the webcam fails."""
    pass


class WebcamManager:
    """
    Manages OpenCV camera capture lifecycle safely.
    Ensures hardware devices are opened with proper backends and cleanly released.
    """

    def __init__(self, camera_index: int = 0, fps_window_size: int = 15):
        """
        Initialize the webcam manager.

        Args:
            camera_index: Device index (typically 0 for default integrated webcam).
            fps_window_size: Number of frames to average for stable FPS calculation.
        """
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps_window_size = fps_window_size
        self._frame_timestamps = deque(maxlen=fps_window_size)
        self._is_active = False

    def open_camera(self) -> None:
        """
        Open the video capture device using appropriate OS backends.

        Raises:
            WebcamAccessError: If the camera cannot be opened or is in use by another app.
        """
        if self.cap is not None and self.cap.isOpened():
            return

        # On Windows, cv2.CAP_DSHOW provides faster startup and reliable direct show access
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY

        try:
            self.cap = cv2.VideoCapture(self.camera_index, backend)
        except Exception as e:
            raise WebcamAccessError(
                f"Webcam initialization exception for camera index {self.camera_index}: {str(e)}"
            ) from e

        if not self.cap.isOpened():
            # Fallback to default backend if CAP_DSHOW failed
            if backend == cv2.CAP_DSHOW:
                try:
                    self.cap = cv2.VideoCapture(self.camera_index)
                except Exception:
                    pass

        if not self.cap or not self.cap.isOpened():
            self.release()
            raise WebcamAccessError(
                f"Webcam at device index {self.camera_index} is unavailable or access was denied. "
                "Please verify that: \n"
                "1. A webcam is securely connected.\n"
                "2. Camera privacy permissions are granted in Windows Settings.\n"
                "3. No other application (Zoom, Teams, Skype) is currently using the camera."
            )

        # Set standard resolution (640x480 for real-time responsiveness)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._is_active = True
        self._frame_timestamps.clear()

    def read_frame(self) -> np.ndarray:
        """
        Read a single frame from the camera and return in RGB format.

        Returns:
            np.ndarray: RGB image frame.

        Raises:
            WebcamAccessError: If camera is not opened.
            WebcamFrameError: If frame capture failed.
        """
        if not self._is_active or self.cap is None or not self.cap.isOpened():
            raise WebcamAccessError("Camera is not opened. Call open_camera() first.")

        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            raise WebcamFrameError(
                f"Failed to capture frame from webcam (device index {self.camera_index})."
            )

        # Track timestamp for FPS calculation
        now = time.perf_counter()
        self._frame_timestamps.append(now)

        # Convert to RGB for Streamlit and PIL consistency
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def get_fps(self) -> float:
        """Calculate smoothed FPS over recent frame intervals."""
        if len(self._frame_timestamps) < 2:
            return 0.0
        delta = self._frame_timestamps[-1] - self._frame_timestamps[0]
        if delta <= 0:
            return 0.0
        count = len(self._frame_timestamps) - 1
        return round(count / delta, 1)

    def release(self) -> None:
        """Safely release the camera hardware and clean up."""
        self._is_active = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning("Error releasing webcam: %s", e)
            finally:
                self.cap = None
        self._frame_timestamps.clear()

    @property
    def is_active(self) -> bool:
        """Check if camera is currently opened and active."""
        return self._is_active and self.cap is not None and self.cap.isOpened()

    def __enter__(self):
        self.open_camera()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
