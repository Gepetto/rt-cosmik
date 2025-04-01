#get qr code pose from 2 webcams (we aleardy calibrated "intr and ext"). do soder and compare to ext param
import cv2
import numpy as np
import csv
import os
import time
from src.rtcosmik.camera.cam_utils import load_camera_parameters
from src.rtcosmik.config_loader import settings

# Define the ArUco dictionary and marker size
marker_size = 0.176
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Load camera calibration parameters
mtxs, dists, projections, rotations, translations = load_camera_parameters(settings.cam_calib_path)
camera_matrix_1, dist_coeffs_1 = mtxs[0], dists[0]
camera_matrix_2, dist_coeffs_2 = mtxs[1], dists[1]

# Initialize webcams
cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(4)

# Set properties for webcams
for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# Create output directory if it doesn't exist
output_dir = "/root/workspace/ros_ws/src/rt-cosmik/output"
csv_file_1 = os.path.join(output_dir, "qrcode_pose_cam1.csv")
csv_file_2 = os.path.join(output_dir, "qrcode_pose_cam2.csv")

# Ensure CSV files have a header
for csv_file in [csv_file_1, csv_file_2]:
    if not os.path.isfile(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "tvec_x", "tvec_y", "tvec_z", "rvec_x", "rvec_y", "rvec_z"])

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    raw_frame1 = frame1.copy()
    raw_frame2 = frame2.copy()

    if not ret1 or not ret2:
        print("Failed to grab frames")
        break

    detected_poses = {}  # Store detected poses for both cameras

    for cam_idx, (frame, camera_matrix, dist_coeffs) in enumerate(
        [(frame1, camera_matrix_1, dist_coeffs_1), (frame2, camera_matrix_2, dist_coeffs_2)]
    ):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

                # Save pose data in dictionary
                detected_poses[cam_idx] = (tvec.flatten(), rvec.flatten())

    # Display the frames
    cv2.imshow("cam1", frame1)
    cv2.imshow("cam2", frame2)

    key = cv2.waitKey(1) & 0xFF

    # Save when 's' is pressed and both cameras detected the marker
    if key == ord('s') and 0 in detected_poses and 1 in detected_poses:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        (tvec1, rvec1) = detected_poses[0]
        (tvec2, rvec2) = detected_poses[1]

        print(f"Saving data for both cameras at {timestamp}")

        # Save image from cam1 as reference
        img_filename1 = os.path.join(output_dir, f"aruco_cam1_{timestamp}.jpg")
        cv2.imwrite(img_filename1, frame1)
        img_filename2 = os.path.join(output_dir, f"aruco_cam2_{timestamp}.jpg")
        cv2.imwrite(img_filename2, frame2)

        # Save pose data separately for each camera
        with open(csv_file_1, mode="a", newline="") as file1:
            writer = csv.writer(file1)
            writer.writerow([timestamp, *tvec1, *rvec1])

        with open(csv_file_2, mode="a", newline="") as file2:
            writer = csv.writer(file2)
            writer.writerow([timestamp, *tvec2, *rvec2])

        # print(f"Data saved to: {csv_file_1} and {csv_file_2}")

    # Press 'q' to exit
    if key == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
