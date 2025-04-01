import cv2
import socket
import numpy as np
import csv
import os
import time
from src.rtcosmik.camera.cam_utils import load_camera_parameters
from src.rtcosmik.config_loader import settings
import select


# Create output directory if it doesn't exist
no_test = 8
output_dir = f"/root/workspace/ros_ws/src/rt-cosmik/output/test{no_test}"
pose_csv_file = os.path.join(output_dir, "pose_aruco.csv")
udp_csv_file = os.path.join(output_dir, "mks_data.csv")

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
camera_matrix = mtxs[0]  
dist_coeffs = dists[0]
print(camera_matrix)

# Open webcam
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
cap.set(cv2.CAP_PROP_FPS, settings.fs)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()



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
    data = get_latest_message(sock)
    decoded_data = data.decode("utf-8")
    ret, frame = cap.read()
    timestamp = time.strftime("%Y%m%d_%H%M%S")


    # print(time.time())
    # print(time.time())

    raw_frame = frame.copy()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

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
            if key == ord('s'):
                img_filename = os.path.join(output_dir, f"aruco_{timestamp}.jpg")

                # Save image
                cv2.imwrite(img_filename, raw_frame)
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
