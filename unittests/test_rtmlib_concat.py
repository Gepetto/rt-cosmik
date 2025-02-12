import cv2
import time
import numpy as np
import pandas as pd
from rtmlib import RTMPose, RTMO, draw_skeleton

# Configuration
device = 'cuda'  # Options: 'cpu', 'cuda', 'mps'
backend = 'onnxruntime'  # Options: 'opencv', 'onnxruntime', 'openvino'

# Initialize the RTMPose model
pose_model = RTMO(
    onnx_model='/root/workspace/ros_ws/src/rt-cosmik/models/rtmo-l.onnx',  # Replace with your .onnx model
    model_input_size=(640, 640),
    backend=backend,
    device=device
)

# Open the video file (or webcam)
cap = cv2.VideoCapture('/root/workspace/ros_ws/src/rt-cosmik/output/saved/cam1.mp4')
if not cap.isOpened():
    print("Error: Unable to access the video source.")
    exit()

print("Processing video... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to capture frame.")
        break

    # Resize the frame to HD resolution (1280x720)
    hd_frame = cv2.resize(frame, (1280, 720))
    
    # Create a stacked frame by concatenating two copies of the HD frame.
    # This means the left copy occupies x in [0, 1279] and the right copy occupies x in [1280, 2559].
    stacked_frame = cv2.hconcat([hd_frame.copy(), hd_frame.copy()])
    
    # Optionally, you can measure inference time:
    start_time = time.time()
    
    # Perform pose estimation on the stacked frame
    keypoints, scores = pose_model(stacked_frame)
    
    inference_time = (time.time() - start_time) * 1000  # in milliseconds
    print(f"Inference Time: {inference_time:.2f} ms")
    
    # Make sure we have detected at least two skeletons
    if keypoints is not None and len(keypoints) >= 2:
        # For each detected skeleton, compute its mean x-coordinate.
        # (Assuming keypoints[i] has shape (num_keypoints, 2))
        mean_x_values = [kp[:, 0].mean() for kp in keypoints]
        
        # The left skeleton is the one with the smallest mean x,
        # and the right skeleton is the one with the largest mean x.
        left_idx = np.argmin(mean_x_values)
        right_idx = np.argmax(mean_x_values)
        
        # Extract the keypoints and scores for each skeleton.
        left_skeleton = keypoints[left_idx]
        left_score    = scores[left_idx]
        right_skeleton = keypoints[right_idx]
        right_score    = scores[right_idx]
        
        # Reproject the right skeleton from the global (stacked) coordinate system to the right frame’s coordinate system.
        # Since the right half of the stacked frame starts at x = 1280, subtract 1280 from all x coordinates.
        right_skeleton_reproj = right_skeleton.copy()
        right_skeleton_reproj[:, 0] -= 1280
        
        # Now, create two separate copies for the left and right original HD frames.
        left_frame  = hd_frame.copy()
        right_frame = hd_frame.copy()
        
        # The draw_skeleton function (from rtmlib) expects the skeletons and scores in a batched format.
        # We add a new axis so that each is shape (1, num_keypoints, 2) and (1, …) respectively.
        left_frame  = draw_skeleton(left_frame,  left_skeleton[np.newaxis, :], np.array([left_score]),  kpt_thr=0.5)
        right_frame = draw_skeleton(right_frame, right_skeleton_reproj[np.newaxis, :], np.array([right_score]), kpt_thr=0.5)
        
        # Display the two frames in separate windows.
        cv2.imshow('Left Pose', left_frame)
        cv2.imshow('Right Pose', right_frame)
    else:
        # If fewer than 2 skeletons are detected, you can choose to show the stacked frame as a fallback.
        cv2.imshow('Stacked Frame', stacked_frame)
    
    # Break the loop on 'q' key press.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video source and close all windows.
cap.release()
cv2.destroyAllWindows()