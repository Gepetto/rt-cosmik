#python3 -m unittests.test_cam_process
from utils.settings import Settings
from utils.calib_utils import list_cameras_with_v4l2
from utils.process_utils import process_camera
import multiprocessing

if __name__ == "__main__":
    settings = Settings()
    camera_dict = list_cameras_with_v4l2()
    camera_ids = list(camera_dict.keys())

    # Create a barrier for the number of camera processes
    barrier = multiprocessing.Barrier(len(camera_ids))

    processes = []
    for cam_id in camera_ids:
        p = multiprocessing.Process(
            target=process_camera,
            args=(cam_id, settings.width, settings.height, settings.fs, barrier)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
