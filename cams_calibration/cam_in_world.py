# To run the code from RT-COSMIK root : python -m cams_calibration.cam_in_world.py 

import cv2
import numpy as np
import pyrealsense2 as rs
from utils.calib_utils import load_cam_params
import yaml
import sys

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

# Configure depth and color streams for both cameras
pipeline_1 = rs.pipeline()
pipeline_2 = rs.pipeline()
config_1 = rs.config()
config_2 = rs.config()

sn_list = []
ctx = rs.context()
if len(ctx.devices) > 0:
    for d in ctx.devices:

        print ('Found device: ', \

                d.get_info(rs.camera_info.name), ' ', \

                d.get_info(rs.camera_info.serial_number))
        sn_list.append(d.get_info(rs.camera_info.serial_number))

else:
    print("No Intel Device connected")

# Get device product line for setting a supporting resolution
pipeline_wrapper_1 = rs.pipeline_wrapper(pipeline_1)
pipeline_wrapper_2 = rs.pipeline_wrapper(pipeline_2)
pipeline_profile_1 = config_1.resolve(pipeline_wrapper_1)
pipeline_profile_2 = config_2.resolve(pipeline_wrapper_2)
device_1 = pipeline_profile_1.get_device()
device_2 = pipeline_profile_2.get_device()
device_product_line_1 = str(device_1.get_info(rs.camera_info.product_line))
device_product_line_2 = str(device_2.get_info(rs.camera_info.product_line))

found_rgb_1 = False
found_rgb_2 = False
for s in device_1.sensors:
    if s.get_info(rs.camera_info.name) == 'Stereo Module':
        found_rgb_1 = True
        break
for s in device_2.sensors:
    if s.get_info(rs.camera_info.name) == 'Stereo Module':
        found_rgb_2 = True
        break
if not found_rgb_1 or not found_rgb_2:
    print("The demo requires Depth cameras with Color sensors")
    exit(0)

print(sn_list)

# Replace these with your camera serial numbers
serial_number_1 = sn_list[0]
serial_number_2 = sn_list[1]

# Enable the devices with the serial numbers
config_1.enable_device(serial_number_1)
config_2.enable_device(serial_number_2)

config_1.enable_stream(rs.stream.infrared, 1280, 720, rs.format.y8, 30)
config_2.enable_stream(rs.stream.infrared, 1280, 720, rs.format.y8, 30)
config_1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config_2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)

# Define the ArUco dictionary and marker size
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_length = 0.176  # Marker size in meters (17.6 cm)

K1, D1 = load_cam_params("cam_params/c1_params_color_test_test.yml")
K2, D2 = load_cam_params("cam_params/c2_params_color_test_test.yml")

# Camera intrinsic parameters (from your YAML file)
camera_matrix_1 = K1
camera_matrix_2 = K2

# Distortion coefficients (from your YAML file)
dist_coeffs_1 = D1
dist_coeffs_2 = D2

# Initialize the ArUco detection parameters
parameters = cv2.aruco.DetectorParameters()

# Function to detect the ArUco marker and estimate the camera pose
def get_camera_pose(frame, camera_matrix, dist_coeffs):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect the markers in the image
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None:
        # Estimate the pose of each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        
        # Assuming we're using one marker, get the rotation and translation vectors
        rvec = rvecs[0][0]
        tvec = tvecs[0][0]
        
        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Now we can form the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = tvec
        
        return transformation_matrix, corners, rvec, tvec
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
    # Wait for a coherent pair of frames from both cameras
    frames_1 = pipeline_1.wait_for_frames()
    frames_2 = pipeline_2.wait_for_frames()

    color_frame_1 = frames_1.get_color_frame()
    color_frame_2 = frames_2.get_color_frame()

    if not color_frame_1 or not color_frame_2:
        continue

    # Convert images to numpy arrays
    frame_1 = np.asanyarray(color_frame_1.get_data())
    frame_2 = np.asanyarray(color_frame_2.get_data())

    # Get the camera pose relative to the global frame defined by the ArUco marker
    transformation_matrix_1, corners_1, rvec_1, tvec_1 = get_camera_pose(frame_1, K1, D1)
    transformation_matrix_2, corners_2, rvec_2, tvec_2 = get_camera_pose(frame_2, K2, D2)

    if transformation_matrix_1 is not None and not saved_1:
        print("Camera 1 Pose (Transformation Matrix):")
        print(transformation_matrix_1)
        
        # Save the rotation matrix and translation vector to a YAML file for Camera 1
        save_pose_to_yaml(transformation_matrix_1[:3, :3], transformation_matrix_1[:3, 3], f'cam_params/camera1_pose_{expe_no}_{trial_no}.yml')
        saved_1 = True  # Ensure we save only once for Camera 1
    
        # Draw the marker and its pose on the frame for Camera 1
        cv2.aruco.drawDetectedMarkers(frame_1, corners_1)
        cv2.drawFrameAxes(frame_1, K1, D1, rvec_1, tvec_1, 0.1)

    if transformation_matrix_2 is not None and not saved_2:
        print("Camera 2 Pose (Transformation Matrix):")
        print(transformation_matrix_2)
        
        # Save the rotation matrix and translation vector to a YAML file for Camera 2
        save_pose_to_yaml(transformation_matrix_2[:3, :3], transformation_matrix_2[:3, 3], f'cam_params/camera2_pose_{expe_no}_{trial_no}.yml')
        saved_2 = True  # Ensure we save only once for Camera 2
    
        # Draw the marker and its pose on the frame for Camera 2
        cv2.aruco.drawDetectedMarkers(frame_2, corners_2)
        cv2.drawFrameAxes(frame_2, K2, D2, rvec_2, tvec_2, 0.1)

    # Display the frames for both cameras
    cv2.imshow('Camera 1 Pose Estimation', frame_1)
    cv2.imshow('Camera 2 Pose Estimation', frame_2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop streaming
pipeline_1.stop()
pipeline_2.stop()
cv2.destroyAllWindows()
