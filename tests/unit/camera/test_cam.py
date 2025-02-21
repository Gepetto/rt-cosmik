import cv2
import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from camera.camera import * 

def test_single_camera():
    """Test capturing frames from a single camera."""
    cam = SingleCamera(camera_id=0, width=640, height=480, fps=30)  # Adjust ID if needed
    try:
        cam.open()
        print("Press 'q' to exit.")
        while True:
            frame = cam.read_frame()
            cv2.imshow("Single Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cam.release()

if __name__ == "__main__":
    test_single_camera()