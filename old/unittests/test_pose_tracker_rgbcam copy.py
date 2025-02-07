# To run the code : python3 unittests/test_pose_tracker_rgbcam.py cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m
# or python3 -m unittests.test_pose_tracker_rgbcam cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m

import argparse
import os
import cv2
import numpy as np
from mmdeploy_runtime import PoseTracker
import time 
from utils.viz_utils import visualize, VISUALIZATION_CFG
from utils.calib_utils import list_cameras_with_v4l2
from utils.settings import Settings

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
parent_directory = os.path.dirname(script_directory)

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

    record_video = False

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

    if record_video:
        # Define the codec and create VideoWriter objects for both RGB streams
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI files
        out_vid1 = cv2.VideoWriter(os.path.join(parent_directory,'output/cam1.mp4'), fourcc, 25.0, (int(settings.width), int(settings.height)), True)

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

            # Process each frame individually
            for idx, frame in enumerate(frames):
                t0 = time.time()
                results = tracker(state, frame, detect=-1)
                keypoints, bboxes, _ = results
                keypoints = (keypoints[..., :2] ).astype(float)
                t1 =time.time()
                print("Time of inference for one image",t1-t0)

                if idx == 0 and record_video:
                    if not visualize(
                            frame,
                            results,
                            args.output_dir,
                            idx,
                            frame_idx + idx,
                            out_vid1,
                            skeleton_type=args.skeleton):
                        break
                else : 
                    if not visualize(
                            frame,
                            results,
                            args.output_dir,
                            idx,
                            frame_idx + idx,
                            skeleton_type=args.skeleton):
                        break
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("quit")
                break    
            
    finally:
        # Release the camera captures
        for cap in captures:
            cap.release()
        if record_video:
            out_vid1.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
