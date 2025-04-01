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
camera_matrix = mtxs[0]  # Assuming you're using the second camera
dist_coeffs = dists[0]

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
csv_file = os.path.join(output_dir, "aruco_pose_cam1.csv")

# Check if CSV file exists, if not, write the header
if not os.path.isfile(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "tvec_x", "tvec_y", "tvec_z", "rvec_x", "rvec_y", "rvec_z"])

while True:
    # print(time.time())
    ret, frame = cap.read()
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
            print(time.time())

            print(f"Translation (tvec): {tvec.flatten()} (meters)")
            print(f"Rotation (rvec): {rvec.flatten()} (Rodrigues format)")

            # Check for 's' key to save data
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_filename = os.path.join(output_dir, f"aruco_{timestamp}.jpg")

                # Save image
                cv2.imwrite(img_filename, raw_frame)
                # print(f"Image saved: {img_filename}")

                # Save pose data to CSV
                with open(csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, *tvec.flatten(), *rvec.flatten()])
                # print("Pose data saved to CSV.")

    cv2.imshow("Real-Time Pose Estimation", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
