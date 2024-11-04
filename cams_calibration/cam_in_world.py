# To run the code from RT-COSMIK root : python3 -m cams_calibration.cam_in_world expe trial

import cv2
import numpy as np
import pyrealsense2 as rs
from utils.calib_utils import load_cam_params
import yaml
import sys
import os 

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

width = 1280
height = 720

max_cameras = 10
available_cameras = []
for index in range(max_cameras):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        available_cameras.append(index)
        cap.release()  # Release the camera after checking

camera_indices = available_cameras

captures = [cv2.VideoCapture(idx) for idx in camera_indices]

for cap in captures: 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # HD
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # HD
    cap.set(cv2.CAP_PROP_FPS, 40)  # Set frame rate to x fps


# Check if cameras opened successfully
for i, cap in enumerate(captures):
    if not cap.isOpened():
        print(f"Error: Camera {i} not opened.")

# Define the ArUco dictionary and marker size
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_size = 0.176  # Marker size in meters (17.6 cm)

K1, D1 = load_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c1_params_color_test_test.yml"))
K2, D2 = load_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c2_params_color_test_test.yml"))

# Camera intrinsic parameters (from your YAML file)
camera_matrix_1 = K1
camera_matrix_2 = K2

# Distortion coefficients (from your YAML file)
dist_coeffs_1 = D1
dist_coeffs_2 = D2

# Initialize the ArUco detection parameters
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Function to detect the ArUco marker and estimate the camera pose
def get_camera_pose(frame, camera_matrix, dist_coeffs):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    
    # Detect the markers in the image
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None and len(corners) > 0:
        # Extract the corners of the first detected marker for pose estimation
        # Reshape the first marker's corners for solvePnP
        corners_for_solvePnP = corners[0].reshape(-1, 2)
        
        # Estimate the pose of each marker
        _, R, t = cv2.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv2.SOLVEPNP_IPPE_SQUARE)
        
        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(R)
        
        # Now we can form the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = t.flatten()
        
        return transformation_matrix, corners[0], R, t
    else:
        return None, None, None, None
    
# Function to save the rotation matrix and translation vector to a YAML file
def save_pose_to_yaml(rotation_matrix, translation_vector, filename):
    # Prepare the data to be saved in YAML format
    data = {
        'rotation_matrix': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': rotation_matrix.flatten().tolist()
        },
        'translation_vector': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': translation_vector.flatten().tolist()
        }
    }
    
    # Write to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file)

# Example usage with RealSense D435
saved_1 = False
saved_2 = False

while True:
    frames = [cap.read()[1] for cap in captures]
            
    if not all(frame is not None for frame in frames):
        continue

    # Get images
    frame_1 = frames[0]
    frame_2 = frames[1]

    # Get the camera pose relative to the global frame defined by the ArUco marker
    transformation_matrix_1, corners_1, rvec_1, tvec_1 = get_camera_pose(frame_1, K1, D1)
    transformation_matrix_2, corners_2, rvec_2, tvec_2 = get_camera_pose(frame_2, K2, D2)

    if transformation_matrix_1 is not None:
        print("Camera 1 Pose (Transformation Matrix):")
        print(transformation_matrix_1)
    
        # Draw the marker and its pose on the frame for Camera 1
        cv2.aruco.drawDetectedMarkers(frame_1, [corners_1])
        cv2.drawFrameAxes(frame_1, K1, D1, rvec_1, tvec_1, 0.1)

        if not(saved_1):
            # Save the rotation matrix and translation vector to a YAML file for Camera 1
            save_pose_to_yaml(transformation_matrix_1[:3, :3], transformation_matrix_1[:3, 3], os.path.join(parent_directory,f'cams_calibration/cam_params/camera1_pose_{expe_no}_{trial_no}.yml'))
            saved_1 = True  # Ensure we save only once for Camera 1

    if transformation_matrix_2 is not None:
        print("Camera 2 Pose (Transformation Matrix):")
        print(transformation_matrix_2)
    
        # Draw the marker and its pose on the frame for Camera 2
        cv2.aruco.drawDetectedMarkers(frame_2, [corners_2])
        cv2.drawFrameAxes(frame_2, K2, D2, rvec_2, tvec_2, 0.1)

        if not(saved_2):
            # Save the rotation matrix and translation vector to a YAML file for Camera 2
            save_pose_to_yaml(transformation_matrix_2[:3, :3], transformation_matrix_2[:3, 3], os.path.join(parent_directory,f'./cams_calibration/cam_params/camera2_pose_{expe_no}_{trial_no}.yml'))
            saved_2 = True  # Ensure we save only once for Camera 2

    # Display the frames for both cameras
    cv2.imshow('Camera 1 Pose Estimation', frame_1)
    cv2.imshow('Camera 2 Pose Estimation', frame_2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("quit")
        break

# Release the camera captures
for cap in captures:
    cap.release()
cv2.destroyAllWindows()
