#python3 -m unittests.multiprocess

import multiprocessing
import argparse
import os
import cv2
import numpy as np
import time

from mmdeploy_runtime import PoseTracker
from utils.viz_utils import visualize, VISUALIZATION_CFG
from utils.calib_utils import list_cameras_with_v4l2
from utils.settings import Settings
from datetime import datetime


import multiprocessing

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
