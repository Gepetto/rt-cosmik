# This script helps to verify in the camera frame the different frames of the setup : human, robot, world 
# To launch the script python3 -m cams_calibration.check_pinpointed_frames expe trial

import cv2
import numpy as np
from utils.calib_utils import load_cam_params, load_cam_pose_rpy, load_cam_pose, get_aruco_pose, get_relative_pose_robot_in_cam, save_pose_rpy_to_yaml, list_cameras_with_v4l2
import sys
import os
from scipy.spatial.transform import Rotation
from utils.settings import Settings

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
parent_directory = os.path.dirname(script_directory)

# Checking if at least two arguments are passed (including the script name)
if len(sys.argv) > 2:
    arg1 = sys.argv[1]  # First argument
    arg2 = sys.argv[2]  # Second argument

    # You can now use arg1 and arg2 in your script
    # Remember to convert them from strings if they represent other types
else:
    print("Not enough arguments provided. Usage: mycode.py <arg1> <arg2>")
    sys.exit(1)  # Exit the script

expe_no = str(arg1)
trial_no = str(arg2)

# FIRST, PARAM LOADING
settings = Settings()

### Initialize cams stream
camera_dict = list_cameras_with_v4l2()
captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in camera_dict.keys()]

for idx, cap in enumerate(captures):
    if not cap.isOpened():
        continue

    # Apply settings
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

K1, D1 = load_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c1_params_color_"+ expe_no + "_" + trial_no +".yml"))
K2, D2 = load_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c2_params_color_"+ expe_no + "_" + trial_no +".yml"))

cam_R1_world, cam_T1_world = load_cam_pose(os.path.join(parent_directory,"cams_calibration/cam_params/camera1_pose_"+ expe_no + "_" + trial_no +".yml"))
cam_R2_world, cam_T2_world = load_cam_pose(os.path.join(parent_directory,"cams_calibration/cam_params/camera2_pose_"+ expe_no + "_" + trial_no +".yml"))

world_rpy1_human, world_T1_human = load_cam_pose_rpy(os.path.join(parent_directory,"cams_calibration/human_params/c1_human_color_"+ expe_no + "_" + trial_no +".yml"))
world_R1_human = Rotation.from_euler('xyz', world_rpy1_human.T, degrees=False).as_matrix()[0]
world_rpy2_human, world_T2_human = load_cam_pose_rpy(os.path.join(parent_directory,"cams_calibration/human_params/c2_human_color_"+ expe_no + "_" + trial_no +".yml"))
world_R2_human = Rotation.from_euler('xyz', world_rpy2_human.T, degrees=False).as_matrix()[0]

world_rpy1_robot, world_T1_robot = load_cam_pose_rpy(os.path.join(parent_directory,"cams_calibration/robot_params/c1_robot_color_"+ expe_no + "_" + trial_no +".yml"))
world_R1_robot = Rotation.from_euler('xyz', world_rpy1_robot.T, degrees=False).as_matrix()[0]
world_rpy2_robot, world_T2_robot = load_cam_pose_rpy(os.path.join(parent_directory,"cams_calibration/robot_params/c2_robot_color_"+ expe_no + "_" + trial_no +".yml"))
world_R2_robot = Rotation.from_euler('xyz', world_rpy2_robot.T, degrees=False).as_matrix()[0]

cam_T1_human = cam_T1_world + cam_R1_world@world_T1_human
cam_T2_human = cam_T2_world + cam_R2_world@world_T2_human
cam_T1_robot = cam_T1_world + cam_R1_world@world_T1_robot
cam_T2_robot = cam_T2_world + cam_R2_world@world_T2_robot

cam_R1_human = cam_R1_world@world_R1_human
cam_R2_human = cam_R2_world@world_R2_human
cam_R1_robot = cam_R1_world@world_R1_robot
cam_R2_robot = cam_R2_world@world_R2_robot

# Camera intrinsic parameters (from your YAML file)
camera_matrix_1 = K1
camera_matrix_2 = K2

# Distortion coefficients (from your YAML file)
dist_coeffs_1 = D1
dist_coeffs_2 = D2

try : 
    while True:
        frames = [cap.read()[1] for cap in captures]
            
        if not all(frame is not None for frame in frames):
            continue

        color_frame_1 = frames[0]
        color_frame_2 = frames[1]

        # Convert images to numpy arrays
        frame_1 = np.asanyarray(color_frame_1.copy())
        frame_2 = np.asanyarray(color_frame_2.copy())

        cv2.drawFrameAxes(frame_1, K1, D1, cv2.Rodrigues(cam_R1_world)[0], cam_T1_world, 0.1)
        cv2.drawFrameAxes(frame_1, K1, D1, cv2.Rodrigues(cam_R1_human)[0], cam_T1_human, 0.1)
        cv2.drawFrameAxes(frame_1, K1, D1, cv2.Rodrigues(cam_R1_robot)[0], cam_T1_robot, 0.1)


        cv2.drawFrameAxes(frame_2, K2, D2, cv2.Rodrigues(cam_R2_world)[0], cam_T2_world, 0.1)
        cv2.drawFrameAxes(frame_2, K2, D2, cv2.Rodrigues(cam_R2_human)[0], cam_T2_human, 0.1)
        cv2.drawFrameAxes(frame_2, K2, D2, cv2.Rodrigues(cam_R2_robot)[0], cam_T2_robot, 0.1)

        # Display the frames for both cameras
        cv2.imshow('Camera 1 Pose Estimation', frame_1)
        cv2.imshow('Camera 2 Pose Estimation', frame_2)
        c = cv2.waitKey(10)
        if c == ord('q'):
            print("quit")
            break
finally : 
    # Release the camera captures
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()