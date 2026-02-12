import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")) # src dir

import time
from src.rtcosmik.config_loader import settings
from src.rtcosmik.camera.cam_utils import list_cameras
from src.rtcosmik.camera.camera import Camera, DisplayConsumer
from src.rtcosmik.utils.mp_utils import create_camera_shared_ressources
from src.rtcosmik.saver.video_saver import VideoSaverProcess2

def main():
    cameras = list_cameras()
    NUM_CAMERAS = len(cameras)
    FRAME_SHAPE = (settings.height, settings.width, 3)

    camera_buffers, camera_timestamps, camera_locks, frame_counters, camera_barrier, stop_event = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)
    
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
               settings.fs, 
               settings.fourcc,)
        for i in range(NUM_CAMERAS)
    ]

    # Create display consumer
    display = DisplayConsumer(frame_counters =frame_counters,
        camera_buffers=camera_buffers,
        camera_locks=camera_locks,
        timestamp_buffers=camera_timestamps,
        stop_event=stop_event,
        frame_shape=FRAME_SHAPE,
        num_cameras=NUM_CAMERAS
    )

    video_savers = []
    if settings.SAVE_VID:
        for i in range(NUM_CAMERAS):
            vs = VideoSaverProcess2(
                camera_id=list(cameras.keys())[i],
                shared_buffer=camera_buffers[i],
                lock=camera_locks[i],
                frame_counter=frame_counters[i],
                frame_shape=FRAME_SHAPE,
                save_dir=settings.SAVE_DIR,
                fps=settings.fs,
                stop_event=stop_event
            )
            video_savers.append(vs)

    processes = camera_processes   + [display]

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