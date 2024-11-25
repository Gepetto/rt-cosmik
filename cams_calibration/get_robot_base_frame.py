# To run the code from RT-COSMIK root : python3 -m cams_calibration.get_robot_base_frame

# For the pinpointing when facing the robot by the long side (behind the x axis): 
# - First image should be the top left screw at the base of the panda
# - Second image should be the top right screw at the base of the panda
# - Third image should be at the ;iddle of the bottom screws at the base of the panda
# - Fourth image should be under the QR code at the base of the panda (front side)

# At NUS for third and fourth images, screws on the box mount can be used

import cv2
import numpy as np
import pyrealsense2 as rs
from utils.calib_utils import load_cam_params, load_cam_pose
import yaml
import sys
import os
import glob
from scipy.spatial.transform import Rotation

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
parent_directory = os.path.dirname(script_directory)

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

# Use os.makedirs() to create your directory; exist_ok=True means it won't throw an error if the directory already exists
os.makedirs(os.path.join(parent_directory,"cams_calibration/images_robot_base_cam_1/" + expe_no + "_" + trial_no + "/color"), exist_ok=True)
os.makedirs(os.path.join(parent_directory,"cams_calibration/images_robot_base_cam_2/" + expe_no + "_" + trial_no + "/color"), exist_ok=True)
os.makedirs(os.path.join(parent_directory,"cams_calibration/robot_params"), exist_ok=True)

c1_color_imgs_path = os.path.join(parent_directory,"cams_calibration/images_robot_base_cam_1/" + expe_no + "_" + trial_no + "/color/*")
c2_color_imgs_path = os.path.join(parent_directory,"cams_calibration/images_robot_base_cam_2/" + expe_no + "_" + trial_no + "/color/*")

c1_color_params_path = os.path.join(parent_directory,"cams_calibration/robot_params/c1_robot_color_" + expe_no + "_" + trial_no + ".yml")
c2_color_params_path = os.path.join(parent_directory,"cams_calibration/robot_params/c2_robot_color_" + expe_no + "_" + trial_no + ".yml")

width = 1280
height = 720

max_cameras = 10
available_cameras = []
for index in range(max_cameras):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        available_cameras.append(index)
        cap.release()  # Release the camera after checking

camera_indices = available_cameras
captures = [cv2.VideoCapture(idx) for idx in camera_indices]

for cap in captures: 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # HD
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # HD
    cap.set(cv2.CAP_PROP_FPS, 40)  # Set frame rate to x fps


# Check if cameras opened successfully
for i, cap in enumerate(captures):
    if not cap.isOpened():
        print(f"Error: Camera {i} not opened.")

# Define the ArUco dictionary and marker size
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_size = 0.176  # Marker size in meters (17.6 cm)

K1, D1 = load_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c1_params_color_test_test.yml"))
K2, D2 = load_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c2_params_color_test_test.yml"))

R1_global, T1_global = load_cam_pose(os.path.join(parent_directory,'cams_calibration/cam_params/camera1_pose_test_test.yml'))
R2_global, T2_global = load_cam_pose(os.path.join(parent_directory,'cams_calibration/cam_params/camera2_pose_test_test.yml'))

# Camera intrinsic parameters (from your YAML file)
camera_matrix_1 = K1
camera_matrix_2 = K2

# Distortion coefficients (from your YAML file)
dist_coeffs_1 = D1
dist_coeffs_2 = D2

# Initialize the ArUco detection parameters
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Function to detect the ArUco marker and estimate the camera pose
def get_aruco_pose(frame, camera_matrix, dist_coeffs):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        _, R, t = cv2.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv2.SOLVEPNP_IPPE_SQUARE)
        
        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(R)
        
        # Now we can form the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = t.flatten()
        
        return transformation_matrix, corners[0], R, t
    else:
        return None, None, None, None
    
def get_relative_pose(images_folder,camera_matrix,dist_coeffs, T_cam, R_cam):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv2.imread(imname, 1)
        images.append(im)

    assert len(images)==4, "number of images to get robot base must be 4"
    
    wand_local = np.array([[-0.00004],[0.262865],[-0.000009]])

    wand_pos_global = []
    for ii, frame in enumerate(images):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
            _, R, t = cv2.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv2.SOLVEPNP_IPPE_SQUARE)
            
            # Convert the rotation vector to a rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(R)
            
            # Now we can form the transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = t.flatten()
        
            wand_pos_cam_frame = (t+rotation_matrix@wand_local).flatten()
            wand_pos_global.append((R_cam@wand_pos_cam_frame + T_cam.flatten()).flatten())

    robot_center = (wand_pos_global[0]+wand_pos_global[1])/2

    x_axis = wand_pos_global[3]-wand_pos_global[2]
    x_axis = x_axis/np.linalg.norm(x_axis)

    y_axis = wand_pos_global[0]-wand_pos_global[1]
    y_axis = y_axis/np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)

    # 1. Construct the rotation matrix
    R_robot = np.column_stack((x_axis, y_axis, z_axis))

    # 2. Convert the rotation matrix to RPY Euler angles
    r = Rotation.from_matrix(R_robot)
    relative_rpy = r.as_euler('xyz', degrees=False)  # Adjust 'xyz' for your URDF convention

    return robot_center, relative_rpy
    
# Function to save the translation vector to a YAML file
def save_pose_to_yaml(translation_vector, rotation_sequence, filename):

    # Ensure inputs are 1D or column vectors of correct shape
    assert translation_vector.shape in [(3,), (3, 1)], "Translation vector must have shape (3,) or (3, 1)"
    assert rotation_sequence.shape in [(3,), (3, 1)], "Rotation sequence must have shape (3,) or (3, 1)"
    
    # Prepare the data to be saved in YAML format
    data = {
        'translation_vector': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': translation_vector.flatten().tolist()
        },
        'rotation_rpy': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': rotation_sequence.flatten().tolist()
        }
    }
    
    # Write to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)  # Use block style for readability

wand_local = np.array([[-0.00004],[0.262865],[-0.000009]])
img_idx=0

try : 
    while True:
        frames = [cap.read()[1] for cap in captures]
            
        if not all(frame is not None for frame in frames):
            continue

        color_frame_1 = frames[0]
        color_frame_2 = frames[1]

        # Convert images to numpy arrays
        frame_1 = np.asanyarray(color_frame_1)
        frame_2 = np.asanyarray(color_frame_2)

        # Get the camera pose relative to the global frame defined by the ArUco marker
        transformation_matrix_1, corners_1, rvec_1, tvec_1 = get_aruco_pose(frame_1, K1, D1)
        transformation_matrix_2, corners_2, rvec_2, tvec_2 = get_aruco_pose(frame_2, K2, D2)

        if transformation_matrix_1 is not None:
            print("Camera 1 Pose (Transformation Matrix):")
            print(transformation_matrix_1)

            tip_pos1=tvec_1 + transformation_matrix_1[:3, :3]@wand_local 

            # Project the 3D wand tip position to 2D image coordinates
            image_points1, _ = cv2.projectPoints(tip_pos1, np.zeros(3,), np.zeros(3,), camera_matrix_1, dist_coeffs_1)
            image_points1 = image_points1[0][0]
        
            # Draw the marker and its pose on the frame for Camera 1
            cv2.aruco.drawDetectedMarkers(frame_1, [corners_1])
            cv2.drawFrameAxes(frame_1, K1, D1, rvec_1, tvec_1, 0.1)

            # Draw the reprojected wand tip on the image
            frame_1 = cv2.circle(frame_1, (int(image_points1[0]), int(image_points1[1])), 5, (0, 0, 255), -1)

        if transformation_matrix_2 is not None:
            print("Camera 2 Pose (Transformation Matrix):")
            print(transformation_matrix_2)

            tip_pos2=tvec_2 + transformation_matrix_2[:3, :3]@wand_local 

            # Project the 3D wand tip position to 2D image coordinates
            image_points2, _ = cv2.projectPoints(tip_pos2, np.zeros(3,), np.zeros(3,), camera_matrix_2, dist_coeffs_2)
            image_points2=image_points2[0][0]
        
            # Draw the marker and its pose on the frame for Camera 2
            cv2.aruco.drawDetectedMarkers(frame_2, [corners_2])
            cv2.drawFrameAxes(frame_2, K2, D2, rvec_2, tvec_2, 0.1)

            # Draw the reprojected wand tip on the image
            frame_2 = cv2.circle(frame_2, (int(image_points2[0]), int(image_points2[1])), 5, (0, 0, 255), -1)


        # Display the frames for both cameras
        cv2.imshow('Camera 1 Pose Estimation', frame_1)
        cv2.imshow('Camera 2 Pose Estimation', frame_2)
        c = cv2.waitKey(10)
        if c == ord('s'):
            print('images taken')
            cv2.imwrite(os.path.join(parent_directory,"cams_calibration/images_robot_base_cam_1/" + expe_no + "_" + trial_no + "/color/img_" + str(img_idx) + ".png"), color_frame_1)
            cv2.imwrite(os.path.join(parent_directory,"cams_calibration/images_robot_base_cam_2/" + expe_no + "_" + trial_no + "/color/img_" + str(img_idx) + ".png"), color_frame_2)
            img_idx = img_idx + 1
        if c == ord('q'):
            print("quit")
            break
finally : 
    # Release the camera captures
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()


T_color1, R_color1 = get_relative_pose(c1_color_imgs_path,K1,D1,T1_global,R1_global)
print("computed translation RGB cam 1 : ")
print(T_color1)
print("computed rotation RGB cam 1 : ")
print(R_color1)

save_pose_to_yaml(T_color1, R_color1, c1_color_params_path)

T_color2, R_color2 = get_relative_pose(c2_color_imgs_path,K2,D2,T2_global,R2_global)
print("computed translation RGB cam 2 : ")
print(T_color2)
print("computed rotation RGB cam 2 : ")
print(R_color2)

save_pose_to_yaml(T_color2, R_color2, c2_color_params_path)