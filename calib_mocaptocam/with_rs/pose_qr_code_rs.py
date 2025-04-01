import os
import sys
import numpy as np
import cv2
import pyrealsense2 as rs
import csv
import time
from src.rtcosmik.config_loader import settings
from src.rtcosmik.camera.cam_utils import load_camera_parameters

def load_cam_params(path):
    """
    Loads camera parameters from a given file.
    Args:
        path (str): The path to the file containing the camera parameters.
    Returns:
        tuple: A tuple containing the camera matrix and distortion matrix.
            - camera_matrix (numpy.ndarray): The camera matrix.
            - dist_matrix (numpy.ndarray): The distortion matrix.
    """
    
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return camera_matrix, dist_matrix

# Get repo root path
script_path = os.path.abspath(__file__)
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, repo_path)

# Define the ArUco dictionary and marker size
marker_size = 0.176
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Load camera calibration parameters
# mtxs, dists, _, _, _ = load_camera_parameters(settings.cam_calib_path)

camera_matrix_1, dist_coeffs_1 = load_cam_params(os.path.join(settings.cam_calib_path, "c1_params_color.yaml"))
camera_matrix_2, dist_coeffs_2 = load_cam_params(os.path.join(settings.cam_calib_path, "c2_params_color.yaml"))


# Initialize the pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Enable the IR streams (IR1 and IR2)
config.enable_stream(rs.stream.infrared, 1)  # Enable IR1 (left)
config.enable_stream(rs.stream.infrared, 2)  # Enable IR2 (right)

pipeline.start(config)

# Create output directory if it doesn't exist
output_dir  = "/root/workspace/ros_ws/src/rt-cosmik/output"
csv_file_1 = os.path.join(output_dir, "aruco_pose_cam1.csv")
csv_file_2 = os.path.join(output_dir, "aruco_pose_cam2.csv")

# Write CSV headers if files do not exist
for csv_file in [csv_file_1, csv_file_2]:
    if not os.path.isfile(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "tvec_x", "tvec_y", "tvec_z", "rvec_x", "rvec_y", "rvec_z"])

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        ir_frame_1 = frames.get_infrared_frame(1)
        ir_frame_2 = frames.get_infrared_frame(2)

        # print(ir_frame_1.get_profile().format())
        
        if not ir_frame_1 or not ir_frame_2:
            continue

        # Convert images to numpy arrays
        ir_image_1 = np.asanyarray(ir_frame_1.get_data())
        ir_image_2 = np.asanyarray(ir_frame_2.get_data())

        if not ir_frame_1 or not ir_frame_2:
            continue

        img_1 = np.asanyarray(ir_frame_1.get_data())
        img_2 = np.asanyarray(ir_frame_2.get_data())

        for cam_idx, (img, camera_matrix, dist_coeffs, csv_file) in enumerate(
            [(img_1, camera_matrix_1, dist_coeffs_1, csv_file_1), (img_2, camera_matrix_2, dist_coeffs_2, csv_file_2)]
        ):
            corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
                for rvec, tvec in zip(rvecs, tvecs):
                    cv2.aruco.drawDetectedMarkers(img, corners)
                    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                    # print(f"Camera {cam_idx+1} - Translation: {tvec.flatten()} meters, Rotation: {rvec.flatten()} (Rodrigues format)")
                # Check for 's' key to save data
                key = cv2.waitKey(10) & 0xFF

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_filename = os.path.join(output_dir, f"aruco_cam{cam_idx+1}_{timestamp}.png")
                # cv2.imwrite(img_filename, img)
                
                with open(csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, *tvec.flatten(), *rvec.flatten()])

        cv2.imshow("IR Cam 1", img_1)
        cv2.imshow("IR Cam 2", img_2)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
