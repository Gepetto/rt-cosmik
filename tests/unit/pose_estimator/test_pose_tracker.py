import cv2
import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from pose_estimator.pose_estimator import *
import cv2

# Initialize the PoseTrackerEstimator
det_model = "/root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano"  # Replace with your actual model
pose_model = "/root/workspace/mmdeploy/rtmpose-trt/rtmpose-m"      # Replace with your actual model
pose_tracker = PoseTrackerEstimator(det_model, pose_model)

# Open video file or webcam
video_path = "/root/workspace/ros_ws/src/rt-cosmik/old/output/saved/cam1.mp4"  
cap = cv2.VideoCapture(video_path)# Use 0 for webcam

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Estimate keypoints
    results = pose_tracker.estimate(frame)

    # Visualize keypoints
    if not pose_tracker.visualize(frame, results, frame_idx):
        break  # Exit if user presses 'q'


cap.release()
cv2.destroyAllWindows()
