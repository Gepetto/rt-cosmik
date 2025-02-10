import cv2
import time
import csv
from rtmlib import RTMPose, RTMO, draw_skeleton

# Configuration
device = 'cuda'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

# Keypoint labels based on COCO 17 format
coco17_labels = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# # Initialize the RTMPose model
# pose_model = RTMPose(
#     onnx_model='C:/Users/crist/Downloads/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504/20230831/rtmpose_onnx/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504/end2end.onnx',
#     model_input_size=(192, 256),
#     backend=backend,
#     device=device
# )

# Initialize the RTMPose model
# pose_model = RTMO(
#     # onnx_model='C:/Users/crist/Downloads/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211/end2end.onnx',
#     onnx_model='../../models/rtmo-s.onnx',
#     model_input_size=(640, 640),
#     backend=backend,
#     device=device
# )

pose_model = RTMO(
    # onnx_model='C:/Users/crist/Downloads/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211/end2end.onnx',
    onnx_model='/root/workspace/ros_ws/src/rt-cosmik/models/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211/end2end.onnx',
    model_input_size=(640, 640),
    backend=backend,
    device=device
)

# Open the video file or webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Confirm the resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if unable to get fps

print(f"Recording at {frame_width}x{frame_height} resolution with {fps} FPS.")

# Define the codec and create a VideoWriter object
output_file = 'output_poses-t.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mjpg')  # Codec for .mp4 files
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

if not cap.isOpened():
    print("Error: Unable to access the video source.")
    exit()

print("Processing video... Press 'q' to stop.")

import numpy as np
# Process the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to capture frame.")
        break

    # Get the resolution of the current frame
    resolution = (frame.shape[1], frame.shape[0])  # (width, height)

    # Measure inference start time
    start_time = time.time()

    # imgs = np.stack(frame, frame.copy())
    
    # Perform pose estimation
    keypoints, scores = pose_model(frame)

    # Measure inference end time
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Print resolution and inference time
    print(f"Resolution: {resolution[0]}x{resolution[1]} | Inference Time: {inference_time:.2f} ms")


    # Visualize the results
    frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

    # Display the frame
    cv2.imshow('Pose Estimation', frame)
    # out.write(frame)
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video source and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
