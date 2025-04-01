import cv2 as cv
import cv2 as cv2
import numpy as np
import csv
import os
import time
from src.rtcosmik.camera.cam_utils import load_camera_parameters
from src.rtcosmik.config_loader import settings

def get_aruco_pose(frame, camera_matrix, dist_coeffs, detector, marker_size):
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

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
        rms, R, t = cv.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv.SOLVEPNP_IPPE_SQUARE)
        
        print("rms:", )
        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv.Rodrigues(R)
        
        # Now we can form the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = t.flatten()
        
        return transformation_matrix, corners[0], R, t
    else:
        return None, None, None, None


# Define the ArUco dictionary and marker size
marker_size = 0.176
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Load camera calibration parameters
mtxs, dists, projections, rotations, translations = load_camera_parameters(settings.cam_calib_path)
camera_matrix = mtxs[0]  # Assuming you're using the second camera
dist_coeffs = dists[0]
print(camera_matrix)

# Input directory containing images
input_dir = "/root/workspace/ros_ws/src/rt-cosmik/output/test4"  # Update with actual path
output_dir = "/root/workspace/ros_ws/src/rt-cosmik/output/estimated_4"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
csv_file = os.path.join(output_dir, "aruco_pose.csv")

# Check if CSV file exists, if not, write the header
if not os.path.isfile(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "image_name", "tvec_x", "tvec_y", "tvec_z", "rvec_x", "rvec_y", "rvec_z"])

# Process all images in the directory
image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

for image_name in image_files:
    image_path = os.path.join(input_dir, image_name)
    frame = cv2.imread(image_path)
    raw_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    transformation_matrix_1, corners_1, rvec_1, tvec_1 = get_aruco_pose(frame, camera_matrix, dist_coeffs, detector, marker_size)

    # Draw the marker and its pose on the frame for Camera 1
    cv2.aruco.drawDetectedMarkers(frame, [corners_1])
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_1, tvec_1, 0.1)
    if tvec_1 is not None:
        
        print(f"Processing {image_name}")
        print(f"Translation (tvec): {tvec_1.flatten()} (meters)")
        print(f"Rotation (rvec): {rvec_1.flatten()} (Rodrigues format)")

        # Save pose data to CSV
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([time.strftime("%Y%m%d_%H%M%S"), *tvec_1.flatten(), *rvec_1.flatten()])

    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
