import cv2
import numpy as np


# Define the marker size in meters (17.6 cm = 0.176 m)
marker_length = 0.176

# Load the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Camera calibration parameters (replace these with your own)
mtxs, dists, projections, rotations, translations = load_camera_parameters(CAM_CONFIG_PATH)

# Capture an image or use a preloaded image
image = cv2.imread("qr_code.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect ArUco markers
corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

if ids is not None:
    # Estimate pose of the marker
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    for rvec, tvec in zip(rvecs, tvecs):
        # Draw axis
        cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        # Print translation (position) and rotation
        print(f"Translation (tvec): {tvec.flatten()} (in meters)")
        print(f"Rotation (rvec): {rvec.flatten()} (Rodrigues format)")

    # Display the image
    cv2.imshow("Pose Estimation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No marker detected.")
