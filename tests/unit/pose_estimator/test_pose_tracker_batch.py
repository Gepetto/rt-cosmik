import cv2
import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from pose_estimator.pose_estimator import *
import cv2

det_model = "/root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano"  # Replace with your actual model
pose_model = "/root/workspace/mmdeploy/rtmpose-trt/rtmpose-m"      # Replace with your actual model
pose_tracker = BatchPoseTrackerEstimator(2,det_model, pose_model)

# Open video file or webcam
# video_path = "/root/workspace/ros_ws/src/rt-cosmik/old/output/saved/cam1.mp4"  

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

frame_idx = 0
while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break  

    frames = [frame1, frame2]

    # Estimate keypoints
    results = pose_tracker.estimate(frames)
    # print(results)

    # Visualize keypoints
    if not pose_tracker.visualize(frames, results):
        break 
  


cap1.release()
cap2.release()
cv2.destroyAllWindows()
