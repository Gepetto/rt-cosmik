import cv2
import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
import time
import numpy as np
from pose_estimator.pose_estimator import *  
from utils.linear_algebra_utils import  concat_frames, reproject, reproject_four_frames


det_model = "/root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano"
pose_model = "/root/workspace/mmdeploy/rtmpose-trt/rtmpose-m"

tracker = PoseTrackerEstimator(det_model, pose_model)

video_path = "/root/workspace/ros_ws/src/rt-cosmik/old/output/saved/cam1.mp4"  
cap1 = cv2.VideoCapture(video_path)
cap2 = cv2.VideoCapture(video_path)

width = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)

frame_idx = 0

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break  

    frames = [frame1, frame2, frame1.copy(), frame2.copy()]
    # frames = [frame1, frame2]

    f = concat_frames(frames)  
    results = tracker.estimate(f)

    # first_result, second_result = reproject(results, width, axis="horizontal")
    first_result, second_result, third_result, fourth_result = reproject_four_frames(results, frame_width=width, frame_height=height)

    if first_result is not None and second_result is not None: 
        if not tracker.visualize(frame1, first_result,idx= 0):
            break
        if not tracker.visualize(frame2, second_result,idx= 1):
            break

    if third_result is not None and fourth_result is not None: 
        if not tracker.visualize(frame1, third_result,idx= 2):
            break
        if not tracker.visualize(frame2, fourth_result,idx= 3):
            break
    
    # Visualize stacked frame results
    if not tracker.visualize(f, results, idx=9):
        break 

cap1.release()
cap2.release()
cv2.destroyAllWindows()
