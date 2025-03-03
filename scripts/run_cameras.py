from settings import Settings
settings = Settings()

import time
from src.camera.cam_utils import list_cameras
from src.camera.camera import Camera
from src.multiprocessing import create_camera_shared_ressources

def main():
    cameras = list_cameras()
    NUM_CAMERAS = len(cameras)
    FRAME_SHAPE = (settings.camera["height"], settings.camera["width"], 3)

    camera_buffers, camera_timestamps, camera_locks = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)

    # Create camera processes
    camera_processes = [
        Camera(list(cameras.keys())[i], camera_buffers[i], camera_timestamps[i], camera_locks[i], FRAME_SHAPE, settings.fps, settings.fourcc)
        for i in range(NUM_CAMERAS)
    ]

    # Start processes
    for cam in camera_processes:
        cam.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop processes
        for cam in cameras:
            cam.stop()
            cam.join()

if __name__ == "__main__":
    main()