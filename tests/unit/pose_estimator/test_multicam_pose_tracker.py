import os
import sys
import cv2
import multiprocessing as mp

# Ensure the `src` folder is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from camera.multicamera import MultiCameraSystem
from pose_estimator.pose_estimator import PoseTrackerEstimator
from utils.linear_algebra_utils import reproject, hconcat_frames
import time 

# Model paths
width = 640
height = 480
fps = 40

DET_MODEL_PATH = "/root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano"
POSE_MODEL_PATH = "/root/workspace/mmdeploy/rtmpose-trt/rtmpose-m"

def main():
    # Initialize Pose Tracker
    tracker = PoseTrackerEstimator(DET_MODEL_PATH, POSE_MODEL_PATH)

    # Initialize Multi-Camera System
    multi_cam = MultiCameraSystem(width=width, height=height, fps=fps)
    multi_cam.start_processes()

    try:
        while True:
            # Get frames from all cameras
            frames = multi_cam.get_frames()
            
            if frames is None or len(frames) < 2:
                continue  # Skip if no valid frames

            # Concatenate frames horizontally
            stacked_frame = hconcat_frames(frames)

            # Run pose estimation
            results = tracker.estimate(stacked_frame)

            # Reproject results to original frames
            first_result, second_result = reproject(results, width, axis="horizontal")

            # Visualize results on each frame
            for idx, (frame, result) in enumerate(zip(frames, [first_result, second_result])):
                if result is not None and not tracker.visualize(frame, result, idx=idx):
                    return  # Exit if 'q' is pressed

    except KeyboardInterrupt:
        print("Exiting gracefully...")
    finally:
        multi_cam.stop_processes()

if __name__ == "__main__":
    main()
