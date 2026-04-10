"""
vision/gesture.py — Gesture Control Module
Rule-based hand gesture detection via MediaPipe Tasks API.

Gestures:
    open_palm    → stop
    pointing_up  → continue
    none         → unrecognized
"""

import logging
import os
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

logger = logging.getLogger(__name__)

# Landmark indices
WRIST      = 0
INDEX_TIP  = 8;  INDEX_PIP  = 6
MIDDLE_TIP = 12; MIDDLE_PIP = 10
RING_TIP   = 16; RING_PIP   = 14
PINKY_TIP  = 20; PINKY_PIP  = 18

MODEL_PATH = os.getenv(
    "GESTURE_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "hand_landmarker.task")
)

GESTURE_TO_COMMAND = {
    "open_palm":   "stop",
    "pointing_up": "continue",
    "none":        "none",
}


class GestureDetector:
    """
    Detects open_palm / pointing_up / none from a BGR frame.

    Smoothing via a majority-vote buffer — a gesture only registers
    once it appears in `stability_frames` out of the last `buffer_size` frames.
    This kills the none↔gesture flickering without adding real latency.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.75,
        min_tracking_confidence:  float = 0.75,
        max_hands:                int   = 1,
        buffer_size:              int   = 5,
        stability_frames:         int   = 3,
    ):
        model_path = os.path.abspath(MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Download: curl -o hand_landmarker.task https://storage.googleapis.com/"
                "mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector        = HandLandmarker.create_from_options(options)
        self._buffer          = deque(maxlen=buffer_size)
        self._stability_frames = stability_frames
        self._stable_gesture  = "none"
        logger.info("GestureDetector ready.")

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Args:
            frame: BGR numpy array from core.camera

        Returns:
            gesture    – smoothed gesture string
            hand_found – whether a hand was detected this frame
            landmarks  – list of 21 (x, y) normalized points, or None
            command    – scheduler command ("stop" / "continue" / "none")
        """
        if frame is None:
            return self._result("none", False, None)

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detected = self._detector.detect(mp_image)

        if not detected.hand_landmarks:
            self._buffer.append("none")
            self._update_stable()
            return self._result(self._stable_gesture, False, None)

        lm     = detected.hand_landmarks[0]
        points = [(lm[i].x, lm[i].y) for i in range(21)]
        raw    = self._classify(points)

        self._buffer.append(raw)
        self._update_stable()

        return self._result(self._stable_gesture, True, points)

    def release(self):
        self._detector.close()
        logger.info("GestureDetector released.")

    # ── internals ────────────────────────────────────────────────────

    def _update_stable(self):
        # Only switch stable gesture when a candidate appears enough times
        for candidate in ("open_palm", "pointing_up", "none"):
            if self._buffer.count(candidate) >= self._stability_frames:
                self._stable_gesture = candidate
                return

    def _classify(self, points: list) -> str:
        index_up  = self._extended(points, INDEX_TIP,  INDEX_PIP)
        middle_up = self._extended(points, MIDDLE_TIP, MIDDLE_PIP)
        ring_up   = self._extended(points, RING_TIP,   RING_PIP)
        pinky_up  = self._extended(points, PINKY_TIP,  PINKY_PIP)

        if index_up and middle_up and ring_up and pinky_up:
            return "open_palm"
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "pointing_up"
        return "none"

    @staticmethod
    def _extended(points, tip_idx, pip_idx) -> bool:
        # Finger is extended if tip is meaningfully farther from wrist than PIP joint.
        # 20% margin (was 10%) reduces false positives on borderline curled fingers.
        wrist = np.array(points[WRIST])
        tip   = np.array(points[tip_idx])
        pip   = np.array(points[pip_idx])
        return np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist) * 1.2

    def _result(self, gesture, hand_found, landmarks) -> dict:
        return {
            "gesture":    gesture,
            "hand_found": hand_found,
            "landmarks":  landmarks,
            "command":    GESTURE_TO_COMMAND.get(gesture, "none"),
        }


def get_scheduler_command(gesture_result: dict) -> str:
    return gesture_result.get("command", "none")


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    try:
        from core.camera import CameraManager
        cam = CameraManager()
        cam.open()
        use_core = True
    except Exception:
        print("core.camera unavailable — falling back to OpenCV webcam.")
        use_core = False
        cap = cv2.VideoCapture(0)

    detector = GestureDetector()

    print("Open palm → STOP  |  Index finger → CONTINUE  |  Q to quit")

    while True:
        frame = cam.get_frame() if use_core else (lambda r, f: f if r else None)(*cap.read())
        if frame is None:
            break

        result = detector.process_frame(frame)

        if result["hand_found"] and result["landmarks"]:
            h, w = frame.shape[:2]
            for x, y in result["landmarks"]:
                cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)

        cv2.putText(frame,
                    f"Gesture: {result['gesture']}  |  Command: {result['command']}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        cv2.imshow("Gesture — Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.release()
    (cam.close() if use_core else cap.release())
    cv2.destroyAllWindows()