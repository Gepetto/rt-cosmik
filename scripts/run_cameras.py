from settings import Settings
settings = Settings()

import time
from src.camera.cam_utils import list_cameras
from src.camera.camera import Camera, DisplayConsumer
from src.utils.mp_utils import create_camera_shared_ressources
from src.saver.video_saver import VideoSaverProcess

def main():
    cameras = list_cameras()
    NUM_CAMERAS = len(cameras)
    FRAME_SHAPE = (settings.height, settings.width, 3)

    camera_buffers, camera_timestamps, camera_locks, barrier, stop_event = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)
    
    # Create camera processes
    camera_processes = [
        Camera(list(cameras.keys())[i], camera_buffers[i], camera_timestamps[i], camera_locks[i], barrier, stop_event, FRAME_SHAPE, settings.fps, settings.fourcc)
        for i in range(NUM_CAMERAS)
    ]

    # Create display consumer
    display = DisplayConsumer(
        camera_buffers=camera_buffers,
        camera_locks=camera_locks,
        stop_event=stop_event,
        frame_shape=FRAME_SHAPE,
        num_cameras=NUM_CAMERAS
    )

    video_savers = []
    if settings.SAVE:
        for i in range(NUM_CAMERAS):
            vs = VideoSaverProcess(
                camera_id=list(cameras.keys())[i],
                shared_buffer=camera_buffers[i],
                lock=camera_locks[i],
                frame_shape=FRAME_SHAPE,
                save_dir=settings.SAVE_DIR,
                fps=settings.fps,
                stop_event=stop_event
            )
            video_savers.append(vs)

    processes = camera_processes + video_savers + [display]

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
    main()