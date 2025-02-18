import cv2
import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
import time
import numpy as np
from pose_estimator.pose_estimator import *  
from utils.linear_algebra_utils import reproject_horizontally


det_model = "/root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano"
pose_model = "/root/workspace/mmdeploy/rtmpose-trt/rtmpose-m"

tracker = PoseTrackerEstimator(det_model, pose_model)


cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

frame_idx = 0

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break  
    frames = [frame1, frame2]
    stacked_frame = cv2.hconcat(frames)  

    results = tracker.estimate(stacked_frame)
    left_result, right_result = reproject_horizontally(results,frame_width=frame1.shape[1])


    if left_result is not None and right_result is not None: 
        if not tracker.visualize(frame1, left_result,idx= 0):
            break
        if not tracker.visualize(frame2, right_result,idx= 1):
            break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
