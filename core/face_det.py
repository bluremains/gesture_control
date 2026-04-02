"""
Face Detection & Identity Tracking Module
-----------------------------------------
Detects faces, extracts 128-d embeddings, and tracks identities within a session.
All data kept in RAM only. Uses grayscale conversion for performance.
"""

from core.camera import CameraManager
import cv2
import face_recognition
import numpy as np
import uuid
from typing import List, Dict, Tuple, Optional


class FaceIdentityTracker:
    """
    Tracks face identities across frames using facial embeddings.
    Session-based: all known faces are forgotten when the instance is destroyed.
    """

    def __init__(self, threshold: float = 0.6, frame_skip: int = 3, scale_factor: float = 0.5):
        """
        Args:
            threshold: Euclidean distance threshold for identity matching (0.6 recommended).
            frame_skip: Process only 1 out of every `frame_skip` frames (performance).
            scale_factor: Downscale frame by this factor before processing (speed vs accuracy).
        """
        self.threshold = threshold
        self.frame_skip = frame_skip
        self.scale_factor = scale_factor
        self._frame_counter = 0

        # In-memory storage for the current session
        self._known_encodings: List[np.ndarray] = []  # list of 128-d vectors
        self._known_ids: List[str] = []               # corresponding UUID strings

    def process_frame(self, frame: np.ndarray) -> Optional[List[Dict]]:
        """
        Process a single BGR frame (from camera) and return face tracking results.

        Args:
            frame: BGR image (numpy array) from core.camera.

        Returns:
            List of dicts, each containing:
                - "face_id": str (UUID)
                - "status": "new_student" or "same_student"
                - "bbox": tuple (top, right, bottom, left) in original frame coordinates
            Returns None if frame is skipped, or empty list if no faces found.
        """
        self._frame_counter += 1
        if self._frame_counter % self.frame_skip != 0:
            return None  # skip this frame

        # 1. Convert to grayscale (performance + as requested)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Downscale if needed
        if self.scale_factor < 1.0:
            new_width = int(gray.shape[1] * self.scale_factor)
            new_height = int(gray.shape[0] * self.scale_factor)
            small_gray = cv2.resize(gray, (new_width, new_height))
        else:
            small_gray = gray

        # face_recognition requires 3-channel image (RGB or BGR), so convert grayscale back to 3-channel
        # by replicating the single channel.
        small_3ch = cv2.cvtColor(small_gray, cv2.COLOR_GRAY2BGR)

        # 3. Detect face locations (using HOG model – lighter than CNN)
        # Note: returns locations in (top, right, bottom, left) format
        face_locations = face_recognition.face_locations(small_3ch, model="hog")
        if not face_locations:
            return []

        # 4. Compute embeddings for each detected face
        encodings = face_recognition.face_encodings(small_3ch, face_locations)
        if not encodings:
            return []

        # 5. Scale back bbox coordinates to original frame size
        scale_inv = 1.0 / self.scale_factor if self.scale_factor < 1.0 else 1.0
        results = []

        for encoding, bbox_small in zip(encodings, face_locations):
            # Scale bbox back to original frame coordinates
            top, right, bottom, left = bbox_small
            if scale_inv != 1.0:
                top = int(top * scale_inv)
                right = int(right * scale_inv)
                bottom = int(bottom * scale_inv)
                left = int(left * scale_inv)

            face_id, status = self._identify_face(encoding)
            results.append({
                "face_id": face_id,
                "status": status,
                "bbox": (top, right, bottom, left)
            })

        return results

    def _identify_face(self, encoding: np.ndarray) -> Tuple[str, str]:
        """
        Compare a face encoding against known encodings in RAM.
        Returns (face_id, status) where status is "new_student" or "same_student".
        """
        if not self._known_encodings:
            # First face ever seen in this session
            new_id = str(uuid.uuid4())
            self._known_encodings.append(encoding)
            self._known_ids.append(new_id)
            return new_id, "new_student"

        # Compute Euclidean distances to all known encodings
        distances = np.linalg.norm(self._known_encodings - encoding, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist <= self.threshold:
            return self._known_ids[min_idx], "same_student"
        else:
            new_id = str(uuid.uuid4())
            self._known_encodings.append(encoding)
            self._known_ids.append(new_id)
            return new_id, "new_student"

    def reset_session(self) -> None:
        """Clear all stored identities (e.g., when robot shuts down)."""
        self._known_encodings.clear()
        self._known_ids.clear()
        self._frame_counter = 0

    def get_known_count(self) -> int:
        """Return number of unique faces seen so far in this session."""
        return len(self._known_ids)


# ---------------------------------------------------------------------
# Standalone test / example usage (run only when this file is executed)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from core.camera import CameraManager

    tracker = FaceIdentityTracker(threshold=0.6, frame_skip=2, scale_factor=0.5)
    cam = CameraManager()
    cam.open()   # CameraManager requires explicit open()

    print("Face Identity Tracker Test")
    print("Press 'q' to quit, 'r' to reset session")

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        result = tracker.process_frame(frame)
        if result is not None:  # None means frame skipped
            for face_info in result:
                print(f"ID: {face_info['face_id'][:8]}... | "
                      f"Status: {face_info['status']} | "
                      f"BBox: {face_info['bbox']}")

            # Optional: draw bounding boxes on frame for visual feedback
            for face_info in result:
                top, right, bottom, left = face_info['bbox']
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"{face_info['status'][:3]} {face_info['face_id'][:4]}"
                cv2.putText(frame, label, (left, top-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            cv2.imshow("Face Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset_session()
            print("Session reset. All identities cleared.")

    cam.close()
    cv2.destroyAllWindows()
    print("Test finished.")