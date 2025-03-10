from settings import Settings
settings = Settings()

from src.rtcosmik.camera.cam_utils import list_cameras
from src.rtcosmik.camera.camera import Camera, DisplayConsumer
from src.rtcosmik.utils.mp_utils import create_camera_shared_ressources, create_pipeline_shared_ressources
from src.rtcosmik.saver.video_saver import VideoSaverProcess
from src.rtcosmik.pipeline.pipeline import PipelineProcess

import time
from multiprocessing import set_start_method

def main():
    cameras = list_cameras()
    NUM_CAMERAS = len(cameras)
    FRAME_SHAPE = (settings.height, settings.width, 3)

    camera_buffers, camera_timestamps, camera_locks, frame_counters, camera_barrier, stop_event = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)
    results_queues = create_pipeline_shared_ressources()

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

    video_savers = []
    if settings.SAVE_VID:
        for i in range(NUM_CAMERAS):
            vs = VideoSaverProcess(
                camera_id=list(cameras.keys())[i],
                shared_buffer=camera_buffers[i],
                lock=camera_locks[i],
                frame_counter=frame_counters[i],
                frame_shape=FRAME_SHAPE,
                save_dir=settings.SAVE_DIR,
                fps=settings.fps,
                stop_event=stop_event
            )
            video_savers.append(vs)
    
    pipeline = PipelineProcess(settings,
                               camera_buffers,
                               camera_timestamps,
                               camera_locks,
                               frame_counters,
                               results_queues,
                               stop_event,
                               frame_shape=FRAME_SHAPE,
                               num_cameras=NUM_CAMERAS)

    processes = camera_processes + video_savers + [pipeline]

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