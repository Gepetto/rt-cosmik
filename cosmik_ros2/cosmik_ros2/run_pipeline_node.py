import time
from multiprocessing import set_start_method

from rtcosmik.config_loader import settings
from rtcosmik.camera.cam_utils import list_cameras
from rtcosmik.camera.camera import Camera
from rtcosmik.utils.mp_utils import (
    create_camera_shared_ressources,
    create_pipeline_shared_ressources,
)
from rtcosmik.saver.video_saver import VideoSaverProcess
from rtcosmik.pipeline.pipeline import PipelineProcess
from rtcosmik.viewer.viewer import ViewerProcess


def main() -> None:
    settings.viewer = "ros2"

    cameras = list_cameras()
    num_cameras = len(cameras)
    frame_shape = (settings.height, settings.width, 3)

    (
        camera_buffers,
        camera_timestamps,
        camera_locks,
        frame_counters,
        camera_barrier,
        stop_event,
    ) = create_camera_shared_ressources(num_cameras, frame_shape)

    results_queues = create_pipeline_shared_ressources()

    camera_processes = [
        Camera(
            list(cameras.keys())[i],
            camera_buffers[i],
            camera_timestamps[i],
            camera_locks[i],
            frame_counters[i],
            camera_barrier,
            stop_event,
            frame_shape,
            settings.fs,
            settings.fourcc,
        )
        for i in range(num_cameras)
    ]

    video_savers = []
    if settings.SAVE_VID:
        for i in range(num_cameras):
            video_savers.append(
                VideoSaverProcess(
                    camera_id=list(cameras.keys())[i],
                    shared_buffer=camera_buffers[i],
                    lock=camera_locks[i],
                    frame_counter=frame_counters[i],
                    frame_shape=frame_shape,
                    save_dir=settings.SAVE_DIR,
                    fps=settings.fs,
                    stop_event=stop_event,
                )
            )

    pipeline = PipelineProcess(
        settings,
        camera_buffers,
        camera_timestamps,
        camera_locks,
        frame_counters,
        results_queues,
        stop_event,
        frame_shape=frame_shape,
        num_cameras=num_cameras,
    )

    viewer = ViewerProcess(
        results_queues,
        stop_event,
        num_cameras=num_cameras,
        freeflyer=True,
    )

    processes = camera_processes + video_savers + [pipeline, viewer]

    for process in processes:
        process.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
        for process in processes:
            if hasattr(process, "stop"):
                process.stop()
            process.join(timeout=2)


if __name__ == "__main__":
    set_start_method("spawn")
    main()
