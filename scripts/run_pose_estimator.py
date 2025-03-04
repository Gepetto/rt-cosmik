from settings import Settings
settings = Settings()

import time
from src.camera.cam_utils import list_cameras
from src.camera.camera import Camera
from src.pose_estimator.pose_estimator import DisplayPoseTracker
from src.utils.mp_utils import create_camera_shared_ressources
from multiprocessing import set_start_method

def main():
    # rtmpose model paths
    DET_MODEL_PATH = "/root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano"
    POSE_MODEL_PATH = "/root/workspace/mmdeploy/rtmpose-trt/rtmpose-m"

    cameras = list_cameras()
    NUM_CAMERAS = len(cameras)
    FRAME_SHAPE = (settings.height, settings.width, 3)

    camera_buffers, camera_timestamps, camera_locks, barrier = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)
    
    # Create camera processes
    camera_processes = [
        Camera(list(cameras.keys())[i], camera_buffers[i], camera_timestamps[i], camera_locks[i], barrier, FRAME_SHAPE, settings.fps, settings.fourcc)
        for i in range(NUM_CAMERAS)
    ]

    # Create display consumer
    display = DisplayPoseTracker(
        DET_MODEL_PATH,
        POSE_MODEL_PATH,
        camera_buffers=camera_buffers,
        camera_locks=camera_locks,
        frame_shape=FRAME_SHAPE,
        num_cameras=NUM_CAMERAS
    )

    # Start processes
    for cam in camera_processes:
        cam.start()
    display.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop processes
        for cam in camera_processes:
            cam.stop()
            cam.join()
        display.stop()
        display.join()

if __name__ == "__main__":
    set_start_method('spawn')
    main()