import cv2
import socket
import numpy as np
import csv
import os
import time
from src.rtcosmik.camera.cam_utils import load_camera_parameters
from src.rtcosmik.config_loader import settings

# UDP Configuration
ip = "172.20.164.200"  # The IP the receiver listens on
port = 44445  # The port to receive data on

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))

# Define the ArUco dictionary and marker size
marker_size = 0.176
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Load camera calibration parameters
mtxs, dists, projections, rotations, translations = load_camera_parameters(settings.cam_calib_path)
camera_matrix = mtxs[1]  # Assuming you're using the second camera
dist_coeffs = dists[1]

# Open webcam
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
cap.set(cv2.CAP_PROP_FPS, settings.fs)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create output directory if it doesn't exist
output_dir = "/root/workspace/ros_ws/src/rt-cosmik/output"
pose_csv_file = os.path.join(output_dir, "sphere_pose_aruco.csv")
udp_csv_file = os.path.join(output_dir, "sphere_mks_data.csv")

# Check if CSV files exist, if not, write the header
if not os.path.isfile(pose_csv_file):
    with open(pose_csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "tvec_x", "tvec_y", "tvec_z", "rvec_x", "rvec_y", "rvec_z"])

if not os.path.isfile(udp_csv_file):
    with open(udp_csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "mks_data"])

while True:
    # print("ok")
    # print(time.time())
    # Receive UDP data
    data, addr = sock.recvfrom(1024)  # Buffer size 1024 bytes
    ret, frame = cap.read()
    # print(time.time())
    # print(time.time())
    decoded_data = data.decode("utf-8")

    raw_frame = frame.copy()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            # Check for 's' key to save data
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                # img_filename = os.path.join(output_dir, f"aruco_{timestamp}.jpg")

                # Save image
                # cv2.imwrite(img_filename, raw_frame)
                # print(f"Image saved: {img_filename}")

                # Save pose data to CSV
                with open(pose_csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, *tvec.flatten(), *rvec.flatten()])

                # Save UDP data to CSV
                if decoded_data:  # Ensure we are saving only valid UDP data
                    with open(udp_csv_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([timestamp, decoded_data])

    cv2.imshow("Real-Time Pose Estimation", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
