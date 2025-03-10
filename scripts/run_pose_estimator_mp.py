from settings import Settings
settings = Settings()

import time
from src.rtcosmik.camera.cam_utils import list_cameras
from src.rtcosmik.camera.camera import Camera
from src.rtcosmik.pose_estimator.pose_estimator import DisplayPoseTracker, PoseTrackerProcess
from src.rtcosmik.utils.mp_utils import create_camera_shared_ressources, create_pose_estimator_shared_ressources
from multiprocessing import set_start_method

def main():
    # rtmpose model paths
    DET_MODEL_PATH = settings.det_model_path
    POSE_MODEL_PATH = settings.pose_model_path

    # List available cameras
    cameras = list_cameras()
    NUM_CAMERAS = len(cameras)
    FRAME_SHAPE = (settings.height, settings.width, 3)

    camera_buffers, camera_timestamps, camera_locks, frame_counters, camera_barrier, stop_event = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)
    pose_estimator_queues, pose_estimator_barrier = create_pose_estimator_shared_ressources(NUM_CAMERAS)

    # Create camera processes
    camera_processes = [
        Camera(list(cameras.keys())[i], 
               camera_buffers[i], 
               camera_timestamps[i], 
               camera_locks[i], 
               frame_counters[i], 
               camera_barrier, 
               stop_event, 
               FRAME_SHAPE, 
               settings.fps, 
               settings.fourcc)
        for i in range(NUM_CAMERAS)
    ]

    # Create pose estimator processes
    pose_estimators = [
        PoseTrackerProcess(
            DET_MODEL_PATH,
            POSE_MODEL_PATH,
            list(cameras.keys())[i],
            camera_buffers[i],
            camera_locks[i],
            frame_counters[i],
            pose_estimator_queues[i],
            pose_estimator_barrier,
            stop_event,
            FRAME_SHAPE)
        for i in range(NUM_CAMERAS)
    ]

    # Create display consumer
    display = DisplayPoseTracker(
        camera_buffers=camera_buffers,
        camera_locks=camera_locks,
        camera_frame_counters=frame_counters,
        result_queues=pose_estimator_queues,
        timestamp_buffers=camera_timestamps,
        stop_event=stop_event,
        frame_shape=FRAME_SHAPE,
        num_cameras=NUM_CAMERAS
    )

    processes = camera_processes + pose_estimators + [display]

    # Start processes
    for p in processes:
        p.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
        # Stop processes
        for process in processes:
            process.stop() if hasattr(process, 'stop') else None
            process.join(timeout=2)

if __name__ == "__main__":
    set_start_method('spawn')
    main()