# To run the code from RT-COSMIK root : python3 -m cams_calibration.get_human_base_frame expe trial

import cv2
import numpy as np
from utils.calib_utils import load_cam_params, load_cam_pose, get_aruco_pose, get_relative_pose_human_in_cam, save_pose_rpy_to_yaml, list_cameras_with_v4l2
import sys
import os
from scipy.spatial.transform import Rotation
from utils.settings import Settings

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
os.makedirs(os.path.join(parent_directory,"cams_calibration/images_human_base_cam_1/" + expe_no + "_" + trial_no + "/color"), exist_ok=True)
os.makedirs(os.path.join(parent_directory,"cams_calibration/images_human_base_cam_2/" + expe_no + "_" + trial_no + "/color"), exist_ok=True)
os.makedirs(os.path.join(parent_directory,"cams_calibration/human_params"), exist_ok=True)

c1_color_imgs_path = os.path.join(parent_directory,"cams_calibration/images_human_base_cam_1/" + expe_no + "_" + trial_no + "/color/*")
c2_color_imgs_path = os.path.join(parent_directory,"cams_calibration/images_human_base_cam_2/" + expe_no + "_" + trial_no + "/color/*")

c1_color_params_path = os.path.join(parent_directory,"cams_calibration/human_params/c1_human_color_" + expe_no + "_" + trial_no + ".yml")
c2_color_params_path = os.path.join(parent_directory,"cams_calibration/human_params/c2_human_color_" + expe_no + "_" + trial_no + ".yml")

# FIRST, PARAM LOADING
settings = Settings()

### Initialize cams stream
camera_dict = list_cameras_with_v4l2()
captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in camera_dict.keys()]

for idx, cap in enumerate(captures):
    if not cap.isOpened():
        continue

    # Apply settings
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

# Define the ArUco dictionary and marker size
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_size = settings.wand_marker_size  # Marker size in meters (17.6 cm)

K1, D1 = load_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c1_params_color_"+ expe_no + "_" + trial_no +".yml"))
K2, D2 = load_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c2_params_color_"+ expe_no + "_" + trial_no +".yml"))

cam_R1_world, cam_T1_world = load_cam_pose(os.path.join(parent_directory,"cams_calibration/cam_params/camera1_pose_"+ expe_no + "_" + trial_no +".yml"))
cam_R2_world, cam_T2_world = load_cam_pose(os.path.join(parent_directory,"cams_calibration/cam_params/camera2_pose_"+ expe_no + "_" + trial_no +".yml"))

# Inverse the pose to get cam in world frame 
world_R1_cam = cam_R1_world.T
world_T1_cam = -cam_R1_world.T@cam_T1_world
world_T1_cam = world_T1_cam.reshape((3,))

world_R2_cam = cam_R2_world.T
world_T2_cam = -cam_R2_world.T@cam_T2_world
world_T2_cam = world_T2_cam.reshape((3,))

# Camera intrinsic parameters (from your YAML file)
camera_matrix_1 = K1
camera_matrix_2 = K2

# Distortion coefficients (from your YAML file)
dist_coeffs_1 = D1
dist_coeffs_2 = D2

# Initialize the ArUco detection parameters
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

wand_local = settings.wand_end_effector_local_pos
img_idx=0

try : 
    while True:
        frames = [cap.read()[1] for cap in captures]
            
        if not all(frame is not None for frame in frames):
            continue

        color_frame_1 = frames[0]
        color_frame_2 = frames[1]

        # Convert images to numpy arrays
        frame_1 = np.asanyarray(color_frame_1.copy())
        frame_2 = np.asanyarray(color_frame_2.copy())

        # Get the camera pose relative to the global frame defined by the ArUco marker
        transformation_matrix_1, corners_1, rvec_1, tvec_1 = get_aruco_pose(frame_1, K1, D1, detector, marker_size)
        transformation_matrix_2, corners_2, rvec_2, tvec_2 = get_aruco_pose(frame_2, K2, D2, detector, marker_size)

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


        resized_color_image_1 = cv2.resize(frame_1, (640, 480), interpolation = cv2.INTER_NEAREST)
        resized_color_image_2 = cv2.resize(frame_2, (640, 480), interpolation = cv2.INTER_NEAREST) 

        images_hstack_1 = np.hstack((resized_color_image_1, resized_color_image_2))
        # Display the frames for both cameras
        cv2.imshow('Stacked images', images_hstack_1)
        
        c = cv2.waitKey(10)
        if c == ord('s'):
            print('images taken')
            cv2.imwrite(os.path.join(parent_directory,"cams_calibration/images_human_base_cam_1/" + expe_no + "_" + trial_no + "/color/img_" + str(img_idx) + ".png"), color_frame_1)
            cv2.imwrite(os.path.join(parent_directory,"cams_calibration/images_human_base_cam_2/" + expe_no + "_" + trial_no + "/color/img_" + str(img_idx) + ".png"), color_frame_2)
            img_idx = img_idx + 1
        if c == ord('q'):
            print("quit")
            break
finally : 
    # Release the camera captures
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()


cam_T1_human, cam_R1_human = get_relative_pose_human_in_cam(c1_color_imgs_path,K1,D1,detector, marker_size)

world_T1_human = world_R1_cam@cam_T1_human + world_T1_cam
world_R1_human = world_R1_cam@cam_R1_human 

world_rpy1_human = Rotation.from_matrix(world_R1_human).as_euler('xyz', degrees=False)

print("computed translation RGB cam 1 : ")
print(world_T1_human)
print("computed rpy RGB cam 1 : ")
print(world_rpy1_human)

save_pose_rpy_to_yaml(world_T1_human, world_rpy1_human, c1_color_params_path)

cam_T2_human, cam_R2_human = get_relative_pose_human_in_cam(c2_color_imgs_path,K2,D2,detector, marker_size)

world_T2_human = world_R2_cam@cam_T2_human + world_T2_cam
world_R2_human = world_R2_cam@cam_R2_human 

world_rpy2_human = Rotation.from_matrix(world_R2_human).as_euler('xyz', degrees=False)

print("computed translation RGB cam 2 : ")
print(world_T2_human)
print("computed rpy RGB cam 2 : ")
print(world_rpy2_human)

save_pose_rpy_to_yaml(world_T2_human, world_rpy2_human, c2_color_params_path)

# Camera transformations 
camera_data = [
    {   "K": K1, "D": D1,
        "cam_T_world": cam_T1_world, "cam_R_world": cam_R1_world,
        "cam_T_human": cam_T1_human, "cam_R_human": cam_R1_human,
        "image": cv2.imread(os.path.join(parent_directory,"cams_calibration/images_human_base_cam_1/" + expe_no + "_" + trial_no + "/color/img_0.png"))
    },
    {
        "K": K2, "D": D2,
        "cam_T_world": cam_T2_world, "cam_R_world": cam_R2_world,
        "cam_T_human": cam_T2_human, "cam_R_human": cam_R2_human,
        "image": cv2.imread(os.path.join(parent_directory,"cams_calibration/images_human_base_cam_2/" + expe_no + "_" + trial_no + "/color/img_0.png"))
    }
]

for cam_data in camera_data:

    cam_R_human = cam_data["cam_R_human"]
    cam_rodrigues_human = cv2.Rodrigues(cam_R_human)[0]
    cam_T_human = cam_data["cam_T_human"]

    cam_R_world = cam_data["cam_R_world"]
    cam_rodrigues_world = cv2.Rodrigues(cam_R_world)[0]
    cam_T_world = cam_data["cam_T_world"]

    image = cam_data["image"]
    K= cam_data["K"]
    D= cam_data["D"]

    cv2.drawFrameAxes(image, K, D, cam_rodrigues_world, cam_T_world, 0.1)
    cv2.drawFrameAxes(image, K, D, cam_rodrigues_human, cam_T_human, 0.1)

    # Save or display the updated image
    cv2.imshow('Reprojected Image', image)
    cv2.waitKey(0)

    if c == ord('a'):
        break

cv2.destroyAllWindows()