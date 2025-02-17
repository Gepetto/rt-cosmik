# To run the code : python3 unittests/test_pose_tracker_rgbcam.py cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m
# or python3 -m unittests.test_pose_tracker_rgbcam_concat cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m

import argparse
import os
import cv2
import numpy as np
from mmdeploy_runtime import PoseTracker
import time 
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


def main():
    # FIRST, PARAM LOADING
    settings = Settings()
    args = parse_args()

    ### Initialize cams stream
    camera_dict = list_cameras_with_v4l2()
    captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in camera_dict.keys()]

    for idx, cap in enumerate(captures):
        if not cap.isOpened():
            continue

        # Apply settings
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        cap.set(cv2.CAP_PROP_FPS, settings.fs)
        
    frame_idx = 0

    tracker = PoseTracker(
        det_model=args.det_model,
        pose_model=args.pose_model,
        device_name=args.device_name)

    # optionally use OKS for keypoints similarity comparison
    sigmas = VISUALIZATION_CFG[args.skeleton]['sigmas']
    state = tracker.create_state(
        det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas)
        
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    try : 
        while True:

            frames = [cap.read()[1] for cap in captures]
     
            if not all(frame is not None for frame in frames):
                continue
            
            frame_idx += 1  # Increment frame counter
            
            stacked_frame = cv2.hconcat([frames[0], frames[1]]) #concat images 

            t0 = time.time()
            results = tracker(state,stacked_frame, detect=-1)
            keypoints, bboxes, _ = results

            if keypoints is not None and len(keypoints) >= 2:
                # For each detected skeleton, compute its mean x-coordinate.
                mean_x_values = [kp[:, 0].mean() for kp in keypoints]

                # The left skeleton is the one with the smallest mean x,
                # and the right skeleton is the one with the largest mean x.
                left_idx = np.argmin(mean_x_values)
                right_idx = np.argmax(mean_x_values)

                # Extract the keypoints and bboxes for each skeleton.
                left_skeleton = keypoints[[left_idx]]
                left_bboxes = bboxes[left_idx]
                
                right_skeleton = keypoints[[right_idx]]
                right_bboxes = bboxes[right_idx]
                
                # Reproject the right skeleton from the global (stacked) coordinate system to the right frame’s coordinate system.
                # Since the right half of the stacked frame starts at "x = settings.width", subtract "settings.width" from all x coordinates.
                right_skeleton_reproj = right_skeleton.copy()
                right_skeleton_reproj[..., 0] -= settings.width

                left_result = left_skeleton, left_bboxes, _ 
                right_result = right_skeleton_reproj, right_bboxes, _
                
                print("inf time", time.time() - t0)

                if not visualize(
                        frames[0],
                        left_result,
                        args.output_dir,
                        0,
                        frame_idx + 0,
                        skeleton_type=args.skeleton):
                    break
                
                if not visualize(
                        frames[1],
                        right_result,
                        args.output_dir,
                        1,
                        frame_idx + 1,
                        skeleton_type=args.skeleton):
                    break

            #visualize stacked frame      
            if not visualize(
                    stacked_frame,
                    results,
                    args.output_dir,
                    2,
                    frame_idx + 2,
                    skeleton_type=args.skeleton):
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("quit")
                break    
            
    finally:
        # Release the camera captures
        for cap in captures:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
