import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")) # src dir

from src.rtcosmik.config_loader import settings
from src.rtcosmik.camera.cam_utils import list_cameras
from src.rtcosmik.camera.camera import Camera, DisplayConsumer
from src.rtcosmik.utils.mp_utils import create_camera_shared_ressources, create_pipeline_shared_ressources, create_pipeline_shared_resources_with_buffers,create_udp_buffer
from src.rtcosmik.saver.video_saver import VideoSaverProcess
from src.rtcosmik.pipeline.pipeline import PipelineProcess
from src.rtcosmik.viewer.viewer import ViewerProcess
from src.rtcosmik.vicon.vicon import UDPDataSaver, UDPReceiver
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
    buffers = create_pipeline_shared_resources_with_buffers()
    shared_ts_udp,shared_values_udp,lock_udp,cam_event = create_udp_buffer(settings.marker_mocap_names)


     # Create camera processes
    camera_processes = [
        Camera(list(cameras.keys())[i], 
               camera_buffers[i], 
               camera_timestamps[i], 
               camera_locks[i], 
               frame_counters[i], 
               camera_barrier, 
               stop_event,
               cam_event, 
               FRAME_SHAPE, 
               settings.fs, 
               settings.fourcc,)
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
                fps=settings.fs,
                stop_event=stop_event,
                saving_flag=saving_enabled 
            )
            video_savers.append(vs)
    
    # pipeline = PipelineProcess(settings,
    #                            camera_buffers,
    #                            camera_timestamps,
    #                            camera_locks,
    #                            frame_counters,
    #                            buffers,
    #                            stop_event,
    #                            frame_shape=FRAME_SHAPE,
    #                            num_cameras=NUM_CAMERAS)

    pipeline = PipelineProcess(settings,
                               camera_buffers,
                               camera_timestamps,
                               camera_locks,
                               frame_counters,
                               results_queues,
                               stop_event,
                               frame_shape=FRAME_SHAPE,
                               num_cameras=NUM_CAMERAS)
    
    # viewer = ViewerProcess(buffers,
    #                        stop_event,
    #                        num_cameras=NUM_CAMERAS,
    #                        freeflyer=True,
    #                        saving_flag=saving_enabled)

    viewer = ViewerProcess(results_queues,
                           stop_event,
                           num_cameras=NUM_CAMERAS,
                           freeflyer=True,
                           saving_flag=saving_enabled)

    vicon = UDPReceiver(shared_values_udp,shared_ts_udp,lock_udp,
                                 ip= "172.20.183.220",
                                 port=44445, output_dir= settings.SAVE_DIR,
                                 stop_event= stop_event, markers_names= settings.marker_mocap_names)

    udp_data_saver_process = UDPDataSaver(saving_flag=saving_enabled, 
                                          shared_ts_udp=shared_ts_udp,
                                          shared_values_udp=shared_values_udp,
                                          lock_udp=lock_udp,
                                          cam_event=cam_event, 
                                          save_dir=settings.SAVE_DIR, 
                                          stop_event=stop_event)

    processes = camera_processes  +video_savers+ [pipeline, viewer,vicon,udp_data_saver_process]

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