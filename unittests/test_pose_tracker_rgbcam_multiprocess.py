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


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use SDK Python API')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument(
        'det_model',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument(
        'pose_model',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('--output_dir', help='output directory', default=None)
    parser.add_argument(
        '--skeleton',
        default='body26',
        choices=['coco', 'coco_wholebody','body26'],
        help='skeleton for keypoints')

    args = parser.parse_args()
    return args


def process_camera(camera_id, device_name, det_model, pose_model, output_dir, skeleton, width, height, fps, event):
    """Function to process a single camera feed."""
    event.wait()
    print(camera_id)
    # print('parent process:', os.getppid())
    # print('process id:', os.getpid())

    print(datetime.now())
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

    
    if not cap.isOpened():
        print(f"Failed to open camera {camera_id}")
        return
    
    # Apply settings
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # cap.set(cv2.CAP_PROP_FPS, fps)

    # fps_reported = cap.get(cv2.CAP_PROP_FPS)
    # print(f"Camera {camera_id} reports FPS: {fps_reported}")

    # tracker = PoseTracker(det_model=det_model, pose_model=pose_model, device_name=device_name)
    # sigmas = VISUALIZATION_CFG[skeleton]['sigmas']
    # state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas)

    # frame_idx = 0
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_video = cv2.VideoWriter(f'output_cam{camera_id}.mp4', fourcc, 25.0, (width, height), True)

    # try:
    #     while True:
    #         frames = []
    #         print(camera_id)
    #         print(datetime.now())
    #         ret, frame = cap.read()
    #         if not ret:
    #             print(f" Failed to read frame from camera {camera_id}")
    #             break
    #         # frames.append(frame)

    #         # for i, frame in enumerate(frames):
    #         #     cv2.imshow(f"Camera {i}", frame)

    #         # frame_idx += 1
    #         # t0 = time.time()

    #         # Apply RTMPose
    #         # print("ok")
    #         # print(camera_id)
    #         # print(datetime.now())
    #         # results = tracker(state, frame, detect=-1)
    #         # keypoints, bboxes, _ = results
    #         # keypoints = keypoints[..., :2].astype(float)

    #         # t1 = time.time()
    #         # print(f" Camera {camera_id}: Inference time = {t1 - t0:.4f} sec")

    #         # if not visualize(frame, results, output_dir, camera_id, frame_idx, output_video, skeleton_type=skeleton):
    #         #     break

    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             print(f"Camera {camera_id} - Quit signal received")
    #             break
    
    # finally:
    #     cap.release()
    #     output_video.release()
    #     cv2.destroyAllWindows()

if __name__ == "__main__":
    
    #multiprocessing.set_start_method('spawn') 
    args = parse_args()
    settings = Settings()

    # Get available cameras
    camera_dict = list_cameras_with_v4l2()
    camera_ids = list(camera_dict.keys())

    processes = []
    event = multiprocessing.Event()
    for cam_id in camera_ids:
        
        p = multiprocessing.Process(
            target=process_camera,
            args=(cam_id, args.device_name, args.det_model, args.pose_model, args.output_dir, args.skeleton, settings.width, settings.height, settings.fs, event)
        )
        p.start()

        time.sleep(5)  # Ensure process is ready
        print("Starting video capture...")
        event.set()

        processes.append(p)

    for p in processes:
        p.join()
