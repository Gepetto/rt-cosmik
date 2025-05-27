import cv2
import socket
import numpy as np
import csv
import os
import time
from src.rtcosmik.camera.cam_utils import load_camera_parameters, load_cam_params
from src.rtcosmik.config_loader import settings
from src.rtcosmik.camera.cam_utils import list_cameras
import select
from utils import *
import pandas as pd
from rigid_bodies_algorithms import *

# UDP Configuration
ip = "172.20.183.220"  # The IP the receiver listens on
port = 44445  # The port to receive data on
id_cam = 1
no_test = "calib_mocap_2_cam1"

cameras = list_cameras()
print(cameras)
captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in cameras.keys()]

for idx, cap in enumerate(captures):
    if not cap.isOpened():
        continue

    # Apply settings
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

path = f"/root/workspace/ros_ws/src/rt-cosmik/config/cam_params/c{id_cam+1}_params_color.yaml"
K, D = load_cam_params(path)
camera_matrix = K 
dist_coeffs = D
print(camera_matrix)


# Create output directory if it doesn't exist
output_dir = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_test}"
os.makedirs(output_dir, exist_ok=True)
pose_csv_file = os.path.join(output_dir, "pose_aruco.csv")
udp_csv_file = os.path.join(output_dir, "mks_data.csv")


####first get data from qr code et mocap

def get_latest_message(sock):
    latest_data = None
    while True:
        # Use select to check if there is data available
        ready = select.select([sock], [], [], 0)
        if ready[0]:
            try:
                data, addr = sock.recvfrom(4096)
                # print(data)
                # print(data)
                latest_data = data  # keep updating, so last one wins
            except BlockingIOError:
                break
        else:
            break
    return latest_data


# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))

# Define the ArUco dictionary and marker size
marker_size = 0.176
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


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

    # Receive UDP data
    data = get_latest_message(sock)
    if data is not None:
        decoded_data = data.decode("utf-8")
        
    frames = [cap.read()[1] for cap in captures]
    frame = frames[id_cam]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_frame = frame.copy()


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

#get the barycenter of two markers 
df = pd.read_csv(f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_test}/mks_data.csv")
mks_array = np.array([list(map(float, row.split(";"))) for row in df["mks_data"]])
barycenter_global_list = []
  
for i in range (len(mks_array)):
    A = mks_array[i, :3]
    B = mks_array[i, 3:6]
    C = mks_array[i, 6:9]
    R = calculate_frame(A, B, C)
    print(A)
    print(C)

    barycenter = (A + C) / 2
    # print(barycenter)
    barycenter_local_frame = transform_to_local_frame(barycenter, B, R)
    barycenter_local_frame[2] = barycenter_local_frame[2]- 0.01
    # print(barycenter_local_frame)


    barycenter_global_frame = transform_to_global_frame(barycenter_local_frame, B, R)
    # print(barycenter_global_frame)
    barycenter_global_list.append(barycenter_global_frame)

# Convert to DataFrame and save to CSV
barycenter_global_df = pd.DataFrame(barycenter_global_list, columns=["Bx", "By", "Bz"])
barycenter_global_df.to_csv(os.path.join(output_dir, "barycenter.csv"), index=False)


###get the transformation using soder or challis
mks_names = ['B']
mks_features_names = []
for m in mks_names:
    mks_features_names = mks_features_names + [f"{m}x", f"{m}y", f"{m}z"]

position_names = ['tvec']
position = []
for p in position_names:
    position = position + [f"{p}_x",f"{p}_y",f"{p}_z"]
mocap_df = pd.read_csv(f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_test}/barycenter.csv", usecols=mks_features_names).values
postion_aruco = pd.read_csv(f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_test}/pose_aruco.csv", usecols=position).values
res_file = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_test}/soder.txt"

mocap_df = np.array(mocap_df)
postion_aruco = np.array(postion_aruco)


print("-- Soder --")
R, d, rms = soder(postion_aruco,mocap_df) #mocap vers cam
print("R:\n", R)
print("d:\n", d)
print("rms:\n", rms)

save_transformation(res_file, R, d, 1.0, rms)

print("-- Challis --")
R, d, s, rms = challis(postion_aruco,mocap_df)
print("R:\n", R)
print("d:\n", d)
print("scale factor:\n", s)
print("rms:\n", rms)

# save_transformation("../data/challis.txt", R, d, 1.0, rms)
