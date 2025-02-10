#python3 -m unittests.check_fps_camera
import cv2
import time
from datetime import datetime
from utils.calib_utils import list_cameras_with_v4l2


camera_dict = list_cameras_with_v4l2()
captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in camera_dict.keys()]

    # Apply settings
for idx, cap in enumerate(captures):
    if not cap.isOpened():
        continue

    # Apply settings
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 40.0)


count=0

while(count<505):
    for capture in captures:
        rval, frame = capture.read()
        count = count + 1

count=0
start_time = time.time()

while((time.time() - start_time) < 5):
    for capture in captures:
        rval, frame = capture.read()
        count = count + 1

print(count)