import argparse
import cv2
import numpy as np
import time
import multiprocessing as mp
from utils.viz_utils import VISUALIZATION_CFG
from utils.calib_utils import list_cameras_with_v4l2
from utils.settings import Settings
from datetime import datetime

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pose Tracking with Multi-Camera Buffering")
    parser.add_argument("device_name", help="name of device, cuda or cpu")
    parser.add_argument("det_model", help="path of mmdeploy SDK model")
    parser.add_argument("pose_model", help="path of mmdeploy SDK model")
    parser.add_argument("--output_dir", default=None, help="output directory")
    parser.add_argument(
        "--skeleton", default="body26", choices=["coco", "coco_wholebody", "body26"], help="skeleton type for keypoints"
    )
    return parser.parse_args()


def initialize_cameras(settings):
    """Initialize cameras and create shared buffers."""
    camera_dict = list_cameras_with_v4l2()
    camera_ids = list(camera_dict.keys())

    if len(camera_ids) < 2:
        print("Error: At least two cameras are required!")
        return None, None, None, None

    shape = (settings.height, settings.width, 3)
    buffers = {cam_id: mp.Array("B", settings.width * settings.height * 3) for cam_id in camera_ids}
    locks = {cam_id: mp.Lock() for cam_id in camera_ids}
    capture_times = mp.Manager().dict()
    barrier = mp.Barrier(len(camera_ids))  # Synchronize camera start
    
    return camera_ids, shape, buffers, locks, capture_times, barrier

#read frames from cameras using multiprocessing
def process_camera(camera_id, width, height, fps, barrier):
    # Set up the camera...
    barrier.wait()

    # print('parent process:', os.getppid())
    # print('process id:', os.getpid())
    print(camera_id, " ", datetime.now())
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))#which format to deliver the frames

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)


    fps_reported = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f"reports FPS: {fps_reported}")

    timestamp= datetime.now()

    try:
        while True:
            # barrier.wait()
            timestamp2 = datetime.now()
            ret, frame = cap.read()
            print(camera_id," ", timestamp2 - timestamp)
            timestamp = timestamp2

            if not ret:
                break
            
            # Process or display the frame
            cv2.imshow(f"Camera {camera_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

#read frames using multiprocessing and copy it to buffer
def capture_frames_buffer(cam_id, buffer, lock, shape, settings, barrier, capture_times):
    """Capture frames from a camera and store them in a shared memory buffer."""
    barrier.wait()
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"Failed to open camera {cam_id}")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

    while True:
        # t0 = time.time()
        ret, frame = cap.read()
        # capture_time = time.time() - t0 

        if ret:
            with lock:
                # t = time.time()
                np_buffer = np.frombuffer(buffer.get_obj(), dtype=np.uint8).reshape(shape)
                np_buffer[:] = frame  # Copy frame to shared buffer
                # print("bufer", time.time()- t)
        else:
            print(f"Camera {cam_id} failed to capture frame")
            break

    cap.release()



