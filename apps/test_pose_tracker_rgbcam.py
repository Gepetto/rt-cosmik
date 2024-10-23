# To run the code : python3 apps/test_pose_tracker_rgbcam.py cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m
# or python3 -m apps.test_pose_tracker_rgbcam cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m

import argparse
import os
import cv2
import numpy as np
from mmdeploy_runtime import PoseTracker
import time 
from utils.viz_utils import visualize, VISUALIZATION_CFG

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

def list_available_cameras(max_cameras=10):
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()  # Release the camera after checking
    return available_cameras

def main():
    args = parse_args()
    np.set_printoptions(precision=4, suppress=True)

    width = 1280
    height = 720
    resize=1280

    # kpt_thr = 0.7

    # Initialize cams stream
    camera_indices = list_available_cameras()

    # if no webcam
    # captures = [cv2.VideoCapture(idx) for idx in camera_indices]

    # if webcam remove it 
    captures = [cv2.VideoCapture(idx) for idx in camera_indices if idx !=2]
    
    for cap in captures: 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # HD
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # HD
        cap.set(cv2.CAP_PROP_FPS, 40)  # Set frame rate to 40fps

    
    # Check if cameras opened successfully
    for i, cap in enumerate(captures):
        if not cap.isOpened():
            print(f"Error: Camera {i} not opened.")
            return
        
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

            # Process each frame individually
            for idx, frame in enumerate(frames):
                t0 = time.time()
                results = tracker(state, frame, detect=-1)
                scale = resize / max(frame.shape[0], frame.shape[1])
                keypoints, bboxes, _ = results
                scores = keypoints[..., 2]
                # keypoints = (keypoints[..., :2] * scale).astype(float)
                keypoints = (keypoints[..., :2] ).astype(float)
                bboxes *= scale
                t1 =time.time()
                print("Time of inference for one image",t1-t0)

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
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
