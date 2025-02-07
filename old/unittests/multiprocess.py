#python3 -m unittests.test_pose_tracker_rgbcam_multiprocess cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m 

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
    
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1); 
    buffsuze = cap.get(cv2.CAP_PROP_BUFFERSIZE) 
    while True:
        ret, frame = cap.read()
        print("buffsize:", buffsuze)
        print(camera_id, " ", datetime.now())
    # Camera configuration...
    time.sleep(0.01)
    # try:
    #     while True:
    #         # Wait for all processes to be ready
            
    #         print(camera_id)
    #         # print('parent process:', os.getppid())
    #         # print('process id:', os.getpid())

    #         print(datetime.now())
            
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
            
    #         # Process or display the frame
    #         cv2.imshow(f"Camera {camera_id}", frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    # finally:
    #     cap.release()
    #     cv2.destroyAllWindows()

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
