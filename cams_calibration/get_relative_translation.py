# To run the code from RT-COSMIK root : python3 -m cams_calibration.get_relative_translation

import cv2
import numpy as np
import pyrealsense2 as rs
from utils.calib_utils import load_cam_params, load_cam_pose
import yaml
import sys
import os
import glob
import pinocchio as pin 

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
os.makedirs("./cams_calibration/images_robot_base_cam_1/" + expe_no + "_" + trial_no + "/color", exist_ok=True)
os.makedirs("./cams_calibration/images_robot_base_cam_2/" + expe_no + "_" + trial_no + "/color", exist_ok=True)
os.makedirs("./cams_calibration/robot_params/", exist_ok=True)

c1_color_imgs_path = "./cams_calibration/images_robot_base_cam_1/" + expe_no + "_" + trial_no + "/color/*"
c2_color_imgs_path = "./cams_calibration/images_robot_base_cam_2/" + expe_no + "_" + trial_no + "/color/*"

c1_color_params_path = "./cams_calibration/robot_params/c1_robot_color_" + expe_no + "_" + trial_no + ".yml"
c2_color_params_path = "./cams_calibration/robot_params/c2_robot_color_" + expe_no + "_" + trial_no + ".yml"

# Configure depth and color streams for both cameras
pipeline_1 = rs.pipeline()
pipeline_2 = rs.pipeline()
config_1 = rs.config()
config_2 = rs.config()

sn_list = []
ctx = rs.context()
if len(ctx.devices) > 0:
    for d in ctx.devices:

        print ('Found device: ', \

                d.get_info(rs.camera_info.name), ' ', \

                d.get_info(rs.camera_info.serial_number))
        sn_list.append(d.get_info(rs.camera_info.serial_number))

else:
    print("No Intel Device connected")

# Get device product line for setting a supporting resolution
pipeline_wrapper_1 = rs.pipeline_wrapper(pipeline_1)
pipeline_wrapper_2 = rs.pipeline_wrapper(pipeline_2)
pipeline_profile_1 = config_1.resolve(pipeline_wrapper_1)
pipeline_profile_2 = config_2.resolve(pipeline_wrapper_2)
device_1 = pipeline_profile_1.get_device()
device_2 = pipeline_profile_2.get_device()
device_product_line_1 = str(device_1.get_info(rs.camera_info.product_line))
device_product_line_2 = str(device_2.get_info(rs.camera_info.product_line))

found_rgb_1 = False
found_rgb_2 = False
for s in device_1.sensors:
    if s.get_info(rs.camera_info.name) == 'Stereo Module':
        found_rgb_1 = True
        break
for s in device_2.sensors:
    if s.get_info(rs.camera_info.name) == 'Stereo Module':
        found_rgb_2 = True
        break
if not found_rgb_1 or not found_rgb_2:
    print("The demo requires Depth cameras with Color sensors")
    exit(0)

print(sn_list)

# Replace these with your camera serial numbers
serial_number_1 = sn_list[0]
serial_number_2 = sn_list[1]

# Enable the devices with the serial numbers
config_1.enable_device(serial_number_1)
config_2.enable_device(serial_number_2)

config_1.enable_stream(rs.stream.infrared, 1280, 720, rs.format.y8, 30)
config_2.enable_stream(rs.stream.infrared, 1280, 720, rs.format.y8, 30)
config_1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config_2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)

# Define the ArUco dictionary and marker size
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_size = 0.176  # Marker size in meters (17.6 cm)

K1, D1 = load_cam_params("./cams_calibration/cam_params/c1_params_color_test_test.yml")
K2, D2 = load_cam_params("./cams_calibration/cam_params/c2_params_color_test_test.yml")

R1_global, T1_global = load_cam_pose('cams_calibration/cam_params/camera1_pose_test_test.yml')
R2_global, T2_global = load_cam_pose('cams_calibration/cam_params/camera2_pose_test_test.yml')

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
    
def get_relative_translation(images_folder,camera_matrix,dist_coeffs, T_cam, R_cam):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv2.imread(imname, 1)
        images.append(im)
    
    wand_local = np.array([[-0.00004],[0.262865],[-0.000009]])
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
        if ii == 0:
            ankle_pos_cam_frame = t+rotation_matrix@wand_local
        else : 
            robot_pos_cam_frame = t+rotation_matrix@wand_local

    ankle_pos = R_cam@ankle_pos_cam_frame + T_cam
    robot_pos = R_cam@robot_pos_cam_frame + T_cam

    relative_T = robot_pos - ankle_pos
    return relative_T
    
# Function to save the translation vector to a YAML file
def save_translation_to_yaml(translation_vector, filename):
    # Prepare the data to be saved in YAML format
    data = {
        'translation_vector': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': translation_vector.flatten().tolist()
        }
    }
    
    # Write to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file)

wand_local = np.array([[-0.00004],[0.262865],[-0.000009]])
img_idx=0

try : 
    while True:
        # Wait for a coherent pair of frames from both cameras
        frames_1 = pipeline_1.wait_for_frames()
        frames_2 = pipeline_2.wait_for_frames()

        color_frame_1 = frames_1.get_color_frame()
        color_frame_2 = frames_2.get_color_frame()

        if not color_frame_1 or not color_frame_2:
            continue

        # Pure image 
        color_image_1 = np.copy(np.asanyarray(color_frame_1.get_data()))
        color_image_2 = np.copy(np.asanyarray(color_frame_2.get_data()))

        # Convert images to numpy arrays
        frame_1 = np.asanyarray(color_frame_1.get_data())
        frame_2 = np.asanyarray(color_frame_2.get_data())

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
            cv2.imwrite("./cams_calibration/images_robot_base_cam_1/" + expe_no + "_" + trial_no + "/color/img_" + str(img_idx) + ".png", color_image_1)
            cv2.imwrite("./cams_calibration/images_robot_base_cam_2/" + expe_no + "_" + trial_no + "/color/img_" + str(img_idx) + ".png", color_image_2)
            img_idx = img_idx + 1
        if c == ord('q'):
            print("quit")
            break
finally : 
    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()
    cv2.destroyAllWindows()


T_color1 = get_relative_translation(c1_color_imgs_path,K1,D1,T1_global,R1_global)
print("computed translation RGB cam 1 : ")
print(T_color1)

save_translation_to_yaml(T_color1, c1_color_params_path)

T_color2 = get_relative_translation(c2_color_imgs_path,K2,D2,T2_global,R2_global)
print("computed translation RGB cam 2 : ")
print(T_color2)

save_translation_to_yaml(T_color2, c2_color_params_path)