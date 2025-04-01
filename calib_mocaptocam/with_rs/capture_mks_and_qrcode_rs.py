import cv2
import socket
import numpy as np
import csv
import os
import time
from src.rtcosmik.camera.cam_utils import load_camera_parameters
from src.rtcosmik.config_loader import settings
import sys
import pyrealsense2 as rs
import select

def get_latest_message(sock):
    latest_data = None
    while True:
        # Use select to check if there is data available
        ready = select.select([sock], [], [], 0)
        if ready[0]:
            try:
                data, addr = sock.recvfrom(4096)
                latest_data = data  # keep updating, so last one wins
            except BlockingIOError:
                break
        else:
            break
    return latest_data

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



# UDP Configuration
ip = "172.20.164.200"  # The IP the receiver listens on
port = 44445  # The port to receive data on

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
sock.bind((ip, port))
sock.setblocking(0)
# Define the ArUco dictionary and marker size
marker_size = 0.176
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Load camera calibration parameters
# mtxs, dists, _, _, _ = load_camera_parameters(settings.cam_calib_path)

camera_matrix, dist_coeffs = load_cam_params(os.path.join(settings.cam_calib_path, "c1_params_color.yaml"))


# Initialize the pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Enable the IR streams (IR1 and IR2)
config.enable_stream(rs.stream.infrared, 1)  # Enable IR1 (left)

pipeline.start(config)

# Create output directory if it doesn't exist
no_test = 6
output_dir = f"/root/workspace/ros_ws/src/rt-cosmik/output/test{no_test}"
pose_csv_file = os.path.join(output_dir, "pose_aruco.csv")
udp_csv_file = os.path.join(output_dir, "mks_data.csv")

# Check if CSV files exist, if not, write the header
if not os.path.isfile(pose_csv_file):
    with open(pose_csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "tvec_x", "tvec_y", "tvec_z", "rvec_x", "rvec_y", "rvec_z"])

if not os.path.isfile(udp_csv_file):
    with open(udp_csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp","mks_data"])

while True:
    # print("ok")
    # print(time.time())
    # Receive UDP data
    # data, addr = sock.recvfrom(4096)  # Buffer size 1024 bytes
    # decoded_data = data.decode("utf-8")
    # Wait for a coherent pair of frames: depth and color
    data = get_latest_message(sock)
    decoded_data = data.decode("utf-8")
    frames = pipeline.wait_for_frames()

    ir_frame_1 = frames.get_infrared_frame(1)
    # print(ir_frame_1.get_profile().format())
    
    if not ir_frame_1 :
        continue

    # Convert images to numpy arrays
    frame = np.asanyarray(ir_frame_1.get_data())

    if not ir_frame_1:
        continue
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    raw_frame = frame.copy()

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(raw_frame, aruco_dict, parameters=parameters)


    # Keep only the corners of the marker with ID 20
    if ids is not None:
        selected_corners = [corners[i] for i in range(len(ids)) if ids[i] == 20]
    else:
        selected_corners = []

    corners = selected_corners
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            # Check for 's' key to save data
            key = cv2.waitKey(10) & 0xFF
            if True: #key == ord('s'):
                img_filename = os.path.join(output_dir, f"aruco_{timestamp}.jpg")

                # Save image
                # cv2.imwrite(img_filename, raw_frame)
                # print(f"Image saved: {img_filename}")

                # Save pose data to CSV
                with open(pose_csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, *tvec.flatten(), *rvec.flatten()])

                # Save UDP data to CSV
                with open(udp_csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, decoded_data])

    cv2.imshow("Real-Time Pose Estimation", frame)

    # Press 'q' to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
