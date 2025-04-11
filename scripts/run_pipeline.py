import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")) # src dir

from src.rtcosmik.config_loader import settings
from src.rtcosmik.camera.cam_utils import list_cameras
from src.rtcosmik.camera.camera import Camera, DisplayConsumer, CameraUDP
from src.rtcosmik.utils.mp_utils import create_camera_shared_ressources, create_pipeline_shared_ressources, create_queue
from src.rtcosmik.saver.video_saver import VideoSaverProcess
from src.rtcosmik.pipeline.pipeline import PipelineProcess
from src.rtcosmik.viewer.viewer import ViewerProcess
from src.rtcosmik.vicon.vicon import UDPDataSaver, UDPDataSaverProcess
import time
from multiprocessing import set_start_method
from multiprocessing import Value, Array


def main():
    saving_enabled = Value('b', False)

    cameras = list_cameras()
    NUM_CAMERAS = len(cameras)
    FRAME_SHAPE = (settings.height, settings.width, 3)

    camera_buffers, camera_timestamps, camera_locks, frame_counters, camera_barrier, stop_event = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)
    results_queues = create_pipeline_shared_ressources()
    udp_data_buffer = Array('B', [0] * 1024)  # Shared UDP data buffer

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
               settings.fourcc)
        for i in range(NUM_CAMERAS)
    ]

    # camera_udp = [
    #     CameraUDP(list(cameras.keys())[i], 
    #            camera_buffers[i], 
    #            camera_timestamps[i], 
    #            camera_locks[i], 
    #            frame_counters[i], 
    #            camera_barrier, 
    #            stop_event, 
    #            udp_data_buffer,
    #            FRAME_SHAPE, 
    #            settings.fs, 
    #            settings.fourcc,
    #            udp_ip="172.20.167.86")
    #     for i in range(NUM_CAMERAS)
    # ]

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
                fps=settings.fs,
                stop_event=stop_event,
                saving_flag=saving_enabled 
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
    
    viewer = ViewerProcess(results_queues,
                           stop_event,
                           num_cameras=NUM_CAMERAS,
                           freeflyer=True,
                           saving_flag=saving_enabled)

    # vicon = UDPDataSaverProcess(ip= "172.20.167.86",
    #                              port=44445, output_dir= settings.SAVE_DIR,
    #                              stop_event= stop_event, saving_flag =saving_enabled)

    # udp_data_saver_process = UDPDataSaver(saving_flag=saving_enabled, 
    #                                       udp_data_buffer=udp_data_buffer, 
    #                                       save_dir=settings.SAVE_DIR, 
    #                                       stop_event=stop_event)

    processes = camera_processes  +video_savers+ [pipeline, viewer]

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