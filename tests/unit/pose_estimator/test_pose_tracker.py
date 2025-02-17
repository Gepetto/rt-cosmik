import cv2
import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from pose_estimator.pose_estimator import *
import cv2

# Initialize the PoseTrackerEstimator
det_model = "/root/workspace/mmdeploy/rtmpose-ort/rtmdet-nano/end2end.onnx"  # Replace with your actual model
pose_model = "/root/workspace/mmdeploy/rtmpose-ort/rtmpose-m/end2end.onnx"      # Replace with your actual model
pose_tracker = PoseTrackerEstimator(det_model, pose_model, device = "cpu")

# Open video file or webcam
video_path = "/root/workspace/ros_ws/src/rt-cosmik/tests/unit/pose_estimator/cam2.mp4"  # Use 0 for webcam
cap = cv2.VideoCapture(video_path)

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

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
