import cv2
import numpy as np
import logging
from config.settings import IS_RASPBERRY_PI, get_settings

_CAM = get_settings().camera

CAMERA_WIDTH      = _CAM.width
CAMERA_HEIGHT     = _CAM.height
CAMERA_FPS        = _CAM.fps
CAMERA_INDEX      = _CAM.index
STREAM_BUFFER_SIZE = _CAM.buffer_size

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Cross-platform camera manager.
    - Raspberry Pi : uses picamera2 (IMX219 / Pi Camera V2)
    - Windows/Linux: uses OpenCV (webcam or any USB camera)

    All frames returned as: numpy array, shape (H, W, 3), BGR, uint8
    """

    def __init__(self):
        self._camera   = None
        self._is_open  = False
        self._backend  = "picamera2" if IS_RASPBERRY_PI else "opencv"
        logger.info(f"CameraManager initialized — backend: {self._backend}")

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def open(self) -> bool:
        """Open and initialize the camera. Returns True if successful."""
        if self._is_open:
            logger.warning("Camera is already open.")
            return True

        try:
            if self._backend == "picamera2":
                self._open_picamera()
            else:
                self._open_opencv()

            self._is_open = True
            logger.info(f"Camera opened — {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
            return True

        except Exception as e:
            logger.error(f"Failed to open camera: {e}")
            return False

    def close(self):
        """Release the camera and free all resources."""
        if not self._is_open:
            return

        try:
            if self._backend == "picamera2":
                self._camera.stop()
                self._camera.close()
            else:
                self._camera.release()

            self._camera  = None
            self._is_open = False
            logger.info("Camera closed.")

        except Exception as e:
            logger.error(f"Error closing camera: {e}")

    def get_frame(self) -> np.ndarray | None:
        """
        Capture and return a single frame.
        Returns: numpy array (H, W, 3) BGR uint8, or None on failure.
        """
        if not self._is_open:
            logger.error("Camera is not open. Call open() first.")
            return None

        try:
            if self._backend == "picamera2":
                return self._capture_picamera()
            else:
                return self._capture_opencv()

        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None

    def start_stream(self):
        """
        Generator that yields frames continuously.
        Used by obstacle detection module.

        Usage:
            for frame in camera.start_stream():
                process(frame)
                if done: break
        """
        if not self._is_open:
            logger.error("Camera is not open. Call open() first.")
            return

        logger.info("Stream started.")
        try:
            while True:
                # Read directly from the capture object to avoid
                # the is_open check inside get_frame() being affected
                # by context manager teardown timing
                frame = self._read_raw()
                if frame is not None:
                    yield frame
        except (GeneratorExit, KeyboardInterrupt):
            logger.info("Stream stopped.")

    def _read_raw(self) -> np.ndarray | None:
        """Internal: read one frame without open-state checks."""
        try:
            if self._backend == "picamera2":
                return self._capture_picamera()
            else:
                return self._capture_opencv()
        except Exception as e:
            logger.error(f"Raw read failed: {e}")
            return None

    def is_open(self) -> bool:
        """Returns True if camera is currently open."""
        return self._is_open

    # ──────────────────────────────────────────────
    # Context Manager support  (with CameraManager() as cam:)
    # ──────────────────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ──────────────────────────────────────────────
    # Internal — picamera2 (Raspberry Pi)
    # ──────────────────────────────────────────────

    def _open_picamera(self):
        from picamera2 import Picamera2
        self._camera = Picamera2()
        config = self._camera.create_preview_configuration(
            main={
                "size":   (CAMERA_WIDTH, CAMERA_HEIGHT),
                "format": "BGR888"          # Native BGR — no conversion needed
            },
            buffer_count=STREAM_BUFFER_SIZE
        )
        self._camera.configure(config)
        self._camera.start()

    def _capture_picamera(self) -> np.ndarray:
        # capture_array returns BGR888 directly — no conversion
        return self._camera.capture_array("main")

    # ──────────────────────────────────────────────
    # Internal — OpenCV (Windows / USB webcam)
    # ──────────────────────────────────────────────

    def _open_opencv(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open camera at index {CAMERA_INDEX}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   STREAM_BUFFER_SIZE)
        self._camera = cap

    def _capture_opencv(self) -> np.ndarray | None:
        ret, frame = self._camera.read()
        if not ret:
            logger.warning("OpenCV read() returned False — frame dropped.")
            return None
        return frame   # Already BGR