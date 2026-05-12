"""
vision/scene_segmentation/segmentor.py — Scene Segmentation Module

Uses YOLOv8n (detection-only) for fast object detection, then builds a
natural scene description from the detected objects to send to the LLM.

Switching from yolov8n-seg to yolov8n gives a major speed boost because
mask computation is skipped — the task only needs a description, not pixels.

No camera handling here — frames come from core/camera.py.
Optimized for Raspberry Pi 5.
"""

import logging
from collections import Counter, deque
from pathlib import Path

import numpy as np
from ultralytics import YOLO
import cv2

logger = logging.getLogger(__name__)

# Using detection model (not seg) — same accuracy, no mask overhead = faster
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "yolov8n.pt"

# Confidence threshold — 0.45 reduces ghost "person" detections from hands/shadows
CONFIDENCE_THRESHOLD = 0.45

# How many of the most dominant objects to include in the scene description
MAX_OBJECTS_IN_DESCRIPTION = 6

# Inference image size passed to YOLO (multiple of 32). 480 balances speed/accuracy.
INFERENCE_SIZE = 480

# Temporal smoothing: only report an object if it appeared in at least
# MIN_HITS of the last WINDOW_SIZE frames. Kills one-frame ghost detections.
WINDOW_SIZE = 4
MIN_HITS    = 2

# Center band — wider = less likely to wrongly label center as left/right
CENTER_LOW  = 0.30
CENTER_HIGH = 0.70


class SceneSegmentor:
    """
    Runs YOLOv8n on a single BGR frame and returns a natural scene
    description suitable for passing to the LLM.

    Uses a temporal smoothing window to suppress ghost detections:
    an object is only included in the description if it appeared in
    at least MIN_HITS of the last WINDOW_SIZE frames.

    Pi optimizations:
    - Detection-only model (no mask computation)
    - imgsz=480 passed to YOLO (no manual resize needed)
    - Temporal smoothing removes false positives without extra CPU cost
    """

    def __init__(
        self,
        model_path: str = str(MODEL_PATH),
        inference_size: int = INFERENCE_SIZE,
    ):
        self._model       = YOLO(model_path)
        self._infer_size  = inference_size

        # Rolling window: each entry is a set of class names seen in that frame
        self._window: deque = deque(maxlen=WINDOW_SIZE)

        logger.info(f"SceneSegmentor loaded model: {model_path}")
        logger.info(
            f"Pi mode — imgsz: {inference_size}, conf: {CONFIDENCE_THRESHOLD}, "
            f"smoothing: {MIN_HITS}/{WINDOW_SIZE} frames"
        )

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Detect objects in a BGR frame and return a smoothed scene description.

        Args:
            frame: BGR numpy array from core.camera

        Returns:
            dict:
                - "description": str  — natural language scene description
                - "objects":     list — smoothed list of detected class names
                - "has_content": bool — False if nothing stable was detected
        """
        if frame is None:
            self._window.append(set())
            return self._empty_result()

        results = self._model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=self._infer_size,
            verbose=False,
        )[0]

        # Build set of class names seen this frame
        if results.boxes is not None and len(results.boxes) > 0:
            class_ids   = results.boxes.cls.cpu().numpy().astype(int)
            frame_names = set(results.names[i] for i in class_ids)
        else:
            frame_names = set()

        self._window.append(frame_names)

        # Keep only objects that appear in at least MIN_HITS recent frames
        all_names = [name for frame_set in self._window for name in frame_set]
        counts    = Counter(all_names)
        stable    = [name for name, cnt in counts.items() if cnt >= MIN_HITS]

        if not stable:
            return self._empty_result()

        # For position, use the latest frame's boxes (if available)
        dominant    = self._dominant_region(frame, results, stable)
        description = self._build_description(stable, dominant)

        return {
            "description": description,
            "objects":     stable,
            "has_content": True,
        }

    def build_llm_prompt(self, scene_result: dict) -> str:
        """
        Wrap the scene description in a prompt for the LLM.
        Returns empty string if there's nothing to describe.
        """
        if not scene_result.get("has_content"):
            return ""

        description = scene_result["description"]
        return (
            f"[VISION] Scene description: {description}. "
            "Based on what you see, respond naturally and educationally."
        )

    # ── internals ────────────────────────────────────────────────────

    @staticmethod
    def _build_description(stable_names: list, dominant: str) -> str:
        """Build natural language description from stable object list."""
        counts = Counter(stable_names)
        top    = counts.most_common(MAX_OBJECTS_IN_DESCRIPTION)

        parts = []
        for name, count in top:
            parts.append(f"{count} {name}{'s' if count > 1 else ''}")

        if not parts:
            return "an empty scene"

        if len(parts) == 1:
            objects_str = parts[0]
        elif len(parts) == 2:
            objects_str = f"{parts[0]} and {parts[1]}"
        else:
            objects_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        if dominant:
            return f"a scene with {objects_str}, mostly in the {dominant} area"
        return f"a scene containing {objects_str}"

    @staticmethod
    def _dominant_region(frame: np.ndarray, results, stable_names: list) -> str:
        """
        Rough spatial hint based on bounding box centers of stable objects.
        Returns 'left', 'right', 'center', or '' if no boxes available.
        """
        if results.boxes is None or len(results.boxes) == 0:
            return ""

        w           = frame.shape[1]
        stable_set  = set(stable_names)
        left = center = right = 0

        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        xyxy      = results.boxes.xyxy.cpu().numpy()

        for i, cid in enumerate(class_ids):
            name = results.names[cid]
            if name not in stable_set:
                continue
            x1, _, x2, _ = xyxy[i]
            mean_x = ((x1 + x2) / 2) / w   # normalized 0-1
            if mean_x < CENTER_LOW:
                left += 1
            elif mean_x > CENTER_HIGH:
                right += 1
            else:
                center += 1

        dominant = max(
            ("left", left),
            ("center", center),
            ("right", right),
            key=lambda x: x[1],
        )
        return dominant[0] if dominant[1] > 0 else ""

    @staticmethod
    def _empty_result() -> dict:
        return {"description": "an empty scene", "objects": [], "has_content": False}


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    try:
        from core.camera import CameraManager
        cam = CameraManager()
        cam.open()
        use_core = True
    except Exception:
        print("core.camera unavailable — using OpenCV webcam.")
        use_core = False
        cap = cv2.VideoCapture(0)

    segmentor = SceneSegmentor()
    print("Scene Segmentation Test")
    print(">> CLICK ON THE CAMERA WINDOW FIRST, then press Q to quit, S to segment")

    while True:
        if use_core:
            frame = cam.get_frame()
        else:
            ret, frame = cap.read()
            if not ret:
                frame = None

        if frame is None:
            break

        cv2.imshow("Scene Segmentation — CLICK HERE, then Q/S", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            result   = segmentor.process_frame(frame)
            prompt   = segmentor.build_llm_prompt(result)
            print(f"\nObjects    : {result['objects']}")
            print(f"Description: {result['description']}")
            print(f"LLM prompt : {prompt}\n")

    if use_core:
        cam.close()
    else:
        cap.release()
    cv2.destroyAllWindows()



class SceneSegmentor:
    """
    Runs YOLOv8n-seg on a single BGR frame and returns a natural
    scene description suitable for passing to the LLM.

    The description focuses on the overall scene, not individual objects —
    e.g. "a classroom with students, chairs, and a whiteboard" rather than
    listing every detection separately.

    Pi optimizations:
    - Inference runs on a downscaled copy (INFERENCE_WIDTH px wide)
    - process_frame() skips frames via internal counter (PROCESS_EVERY_N_FRAMES)
    - Last valid result is cached and returned on skipped frames
    """

    def __init__(
        self,
        model_path: str = str(MODEL_PATH),
        process_every_n: int = PROCESS_EVERY_N_FRAMES,
        inference_size: int = INFERENCE_SIZE,
    ):
        self._model = YOLO(model_path)
        self._process_every_n = process_every_n
        self._inference_size = inference_size
        self._frame_counter = 0
        self._last_result = self._empty_result()
        logger.info(f"SceneSegmentor loaded model: {model_path}")
        logger.info(
            f"Pi mode — imgsz: {inference_size}, conf: {CONFIDENCE_THRESHOLD}, "
            f"process every {process_every_n} frame(s)"
        )

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Segment a single BGR frame and return a scene description.
        Skips inference on intermediate frames and returns the cached result.

        Args:
            frame: BGR numpy array from core.camera

        Returns:
            dict:
                - "description": str  — natural language scene description
                - "objects":     list — raw list of detected class names
                - "has_content": bool — False if nothing was detected
        """
        if frame is None:
            return self._empty_result()

        self._frame_counter += 1

        # Return cached result on skipped frames to save CPU
        if self._frame_counter % self._process_every_n != 0:
            return self._last_result

        # imgsz tells YOLO to resize internally — faster than doing it manually
        results = self._model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=self._inference_size,
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            self._last_result = self._empty_result()
            return self._last_result

        # Collect detected class names
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        class_names = [results.names[i] for i in class_ids]

        description = self._build_description(class_names, frame, results)

        self._last_result = {
            "description": description,
            "objects": class_names,
            "has_content": True,
        }
        return self._last_result

    def build_llm_prompt(self, scene_result: dict) -> str:
        """
        Wrap the scene description in a prompt for the LLM.
        Returns empty string if there's nothing to describe.
        """
        if not scene_result.get("has_content"):
            return ""

        description = scene_result["description"]
        return (
            f"[VISION] Scene description: {description}. "
            "Based on what you see, respond naturally and educationally."
        )

    # ── internals ────────────────────────────────────────────────────

    def _build_description(
        self,
        class_names: list,
        frame: np.ndarray,
        results,
    ) -> str:
        # Count how often each object appears and take the top N
        counts = Counter(class_names)
        top = counts.most_common(MAX_OBJECTS_IN_DESCRIPTION)
        dominant = self._dominant_region(frame, results)

        # Build object part: "2 persons, a chair, and a laptop"
        parts = []
        for name, count in top:
            parts.append(f"{count} {name}{'s' if count > 1 else ''}")

        if not parts:
            return "an empty scene"

        if len(parts) == 1:
            objects_str = parts[0]
        elif len(parts) == 2:
            objects_str = f"{parts[0]} and {parts[1]}"
        else:
            objects_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        if dominant:
            return f"a scene with {objects_str}, mostly in the {dominant} area"
        return f"a scene containing {objects_str}"

    @staticmethod
    def _dominant_region(frame: np.ndarray, results) -> str:
        """
        Rough spatial hint: where is most of the segmented area?
        Returns 'left', 'right', 'center', or '' if masks unavailable.
        """
        if results.masks is None:
            return ""

        left_mass = center_mass = right_mass = 0

        for mask in results.masks.data.cpu().numpy():
            cols = np.where(mask > 0)[1]
            if len(cols) == 0:
                continue
            mean_x = cols.mean() / mask.shape[1]  # normalized 0-1
            if mean_x < CENTER_LOW:
                left_mass += 1
            elif mean_x > CENTER_HIGH:
                right_mass += 1
            else:
                center_mass += 1

        dominant = max(
            ("left", left_mass),
            ("center", center_mass),
            ("right", right_mass),
            key=lambda x: x[1],
        )
        return dominant[0] if dominant[1] > 0 else ""

    @staticmethod
    def _empty_result() -> dict:
        return {"description": "an empty scene", "objects": [], "has_content": False}


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    try:
        from core.camera import CameraManager

        cam = CameraManager()
        cam.open()
        use_core = True
    except Exception:
        print("core.camera unavailable — using OpenCV webcam.")
        use_core = False
        cap = cv2.VideoCapture(0)

    segmentor = SceneSegmentor()
    print("Scene Segmentation Test")
    print(">> CLICK ON THE CAMERA WINDOW FIRST, then press Q to quit, S to segment")

    while True:
        if use_core:
            frame = cam.get_frame()
        else:
            ret, frame = cap.read()
            if not ret:
                frame = None

        if frame is None:
            break

        cv2.imshow("Scene Segmentation — CLICK HERE, then Q/S", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            result = segmentor.process_frame(frame)
            prompt = segmentor.build_llm_prompt(result)
            print(f"\nObjects    : {result['objects']}")
            print(f"Description: {result['description']}")
            print(f"LLM prompt : {prompt}\n")

    if use_core:
        cam.close()
    else:
        cap.release()
    cv2.destroyAllWindows()