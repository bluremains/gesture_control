"""
test_camera.py — Manual test for CameraManager

Run from project root:
    python tests/test_camera.py

Controls:
    Q  →  quit
    S  →  save current frame as test_frame.jpg
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import logging
from core.camera import CameraManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")


def test_single_frame():
    """Test: open → get one frame → close."""
    print("\n[ TEST 1 ] Single frame capture")
    cam = CameraManager()

    if not cam.open():
        print("FAIL — could not open camera.")
        return False

    frame = cam.get_frame()
    cam.close()

    if frame is None:
        print("FAIL — frame is None.")
        return False

    print(f"PASS — frame shape: {frame.shape}, dtype: {frame.dtype}")
    return True


def test_context_manager():
    """Test: using 'with' syntax."""
    print("\n[ TEST 2 ] Context manager (with statement)")
    with CameraManager() as cam:
        frame = cam.get_frame()
        if frame is not None:
            print(f"PASS — frame shape: {frame.shape}")
            return True

    print("FAIL — frame is None inside context.")
    return False


def test_live_stream():
    """Test: continuous stream with live preview window."""
    print("\n[ TEST 3 ] Live stream — press Q to quit, S to save frame")
    print("           (click on the camera window first to activate keys)")

    cam = CameraManager()
    if not cam.open():
        print("FAIL — could not open camera.")
        return

    # Create window upfront and bring it to front
    window_name = "Camera Test — Q:quit  S:save"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    try:
        for frame in cam.start_stream():
            cv2.imshow(window_name, frame)

            # waitKey > 1 makes keys more responsive on Windows
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q') or key == 27:   # Q or ESC
                print("Stream stopped by user.")
                break
            elif key == ord('s'):
                cv2.imwrite("test_frame.jpg", frame)
                print("Frame saved → test_frame.jpg")

    except KeyboardInterrupt:
        print("Stopped via Ctrl+C.")
    finally:
        cam.close()
        cv2.destroyAllWindows()

    print("PASS — stream completed.")


if __name__ == "__main__":
    print("=" * 50)
    print("  CameraManager — Test Suite")
    print("=" * 50)

    r1 = test_single_frame()
    r2 = test_context_manager()

    print("\n[ TEST 3 ] Starting live stream test...")
    test_live_stream()

    print("\n" + "=" * 50)
    print(f"  Results: {'PASS' if r1 else 'FAIL'} | {'PASS' if r2 else 'FAIL'} | See stream above")
    print("=" * 50)