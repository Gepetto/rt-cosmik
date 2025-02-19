import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from camera.multicamera import *
import multiprocessing as mp


if __name__ == "__main__":
    multi_cam = MultiCameraSystem(width=1280, height=720)

    if len(multi_cam.camera_ids) < 2:
        print("Error: At least two cameras are required!")
        sys.exit(1)

    multi_cam.start_processes()
    try:
        while True:
            frames = multi_cam.get_frames()
            for i, frame in enumerate(frames):
                cv2.imshow(f"Camera {multi_cam.camera_ids[i]}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        multi_cam.stop_processes()
