# To run the code from RT-COSMIK root : python3 -m cams_calibration.display_wand_rviz test test

import cv2
import numpy as np
from utils.calib_utils import load_cam_pose, load_cam_params, save_pose_matrix_to_yaml, get_aruco_pose, get_relative_pose_world_in_cam, list_cameras_with_v4l2
import sys
import os 
from utils.settings import Settings
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
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

K1, D1 = load_cam_params(os.path.join(parent_directory,"config/cam_params/c1_params_color_"+ expe_no + "_" + trial_no +".yaml"))
K2, D2 = load_cam_params(os.path.join(parent_directory,"config/cam_params/c2_params_color_"+ expe_no + "_" + trial_no +".yaml"))

cam_R1_world, cam_T1_world = load_cam_pose(os.path.join(parent_directory,"config/cam_params/camera1_pose_"+ expe_no + "_" + trial_no +".yaml"))
cam_R2_world, cam_T2_world = load_cam_pose(os.path.join(parent_directory,"config/cam_params/camera2_pose_"+ expe_no + "_" + trial_no +".yaml"))

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

### Initialize ROS node 
rospy.init_node('wand_rviz', anonymous=True)
wand_pose_publisher1 = rospy.Publisher('/wand_pose1', PoseStamped, queue_size=10)
tip_publisher1 = rospy.Publisher('/tip_position1', Marker, queue_size=10)
wand_pose_publisher2 = rospy.Publisher('/wand_pose2', PoseStamped, queue_size=10)
tip_publisher2 = rospy.Publisher('/tip_position2', Marker, queue_size=10)

tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

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
            tip_pos1=tvec_1 + transformation_matrix_1[:3, :3]@wand_local 

            # Project the 3D wand tip position to 2D image coordinates
            image_points1, _ = cv2.projectPoints(tip_pos1, np.zeros(3,), np.zeros(3,), camera_matrix_1, dist_coeffs_1)
            image_points1 = image_points1[0][0]
        
            # Draw the marker and its pose on the frame for Camera 1
            cv2.aruco.drawDetectedMarkers(frame_1, [corners_1])
            cv2.drawFrameAxes(frame_1, K1, D1, rvec_1, tvec_1, 0.1)

            # Draw the reprojected wand tip on the image
            frame_1 = cv2.circle(frame_1, (int(image_points1[0]), int(image_points1[1])), 5, (0, 0, 255), -1)
            
            # Express wand frame + wand tip in world and send it to rviz
            # Compute wand frame pose in the world
            world_R1_wand = world_R1_cam @ transformation_matrix_1[:3, :3]
            world_T1_wand = world_R1_cam @ transformation_matrix_1[:3, 3] + world_T1_cam
            
            world_tip_pos1 = world_R1_wand @ wand_local.reshape(3,) + world_T1_wand
            
            # Create a TransformStamped message for the wand frame
            wand_transform = TransformStamped()
            wand_transform.header.stamp = rospy.Time.now()
            wand_transform.header.frame_id = "world"
            wand_transform.child_frame_id = "wand_frame_1"
            wand_transform.transform.translation.x = world_T1_wand[0]
            wand_transform.transform.translation.y = world_T1_wand[1]
            wand_transform.transform.translation.z = world_T1_wand[2]
            wand_quaternion = Rotation.from_matrix(world_R1_wand).as_quat()
            wand_transform.transform.rotation.x = wand_quaternion[0]
            wand_transform.transform.rotation.y = wand_quaternion[1]
            wand_transform.transform.rotation.z = wand_quaternion[2]
            wand_transform.transform.rotation.w = wand_quaternion[3]
            tf_broadcaster.sendTransform(wand_transform)
            
            # Publish the wand tip position as a Marker
            tip_marker1 = Marker()
            tip_marker1.header.stamp = rospy.Time.now()
            tip_marker1.header.frame_id = "world"
            tip_marker1.ns = "wand_tip_1"
            tip_marker1.id = 0
            tip_marker1.type = Marker.SPHERE
            tip_marker1.action = Marker.ADD
            tip_marker1.pose.position.x = world_tip_pos1[0]
            tip_marker1.pose.position.y = world_tip_pos1[1]
            tip_marker1.pose.position.z = world_tip_pos1[2]
            tip_marker1.pose.orientation.x = 0.0
            tip_marker1.pose.orientation.y = 0.0
            tip_marker1.pose.orientation.z = 0.0
            tip_marker1.pose.orientation.w = 1.0
            tip_marker1.scale.x = 0.05  # Diameter of the sphere
            tip_marker1.scale.y = 0.05
            tip_marker1.scale.z = 0.05
            tip_marker1.color.a = 1.0  # Alpha (transparency)
            tip_marker1.color.r = 1.0  # Red
            tip_marker1.color.g = 0.0  # Green
            tip_marker1.color.b = 0.0  # Blue
            tip_publisher1.publish(tip_marker1)

        if transformation_matrix_2 is not None:
            tip_pos2=tvec_2 + transformation_matrix_2[:3, :3]@wand_local 

            # Project the 3D wand tip position to 2D image coordinates
            image_points2, _ = cv2.projectPoints(tip_pos2, np.zeros(3,), np.zeros(3,), camera_matrix_2, dist_coeffs_2)
            image_points2=image_points2[0][0]
        
            # Draw the marker and its pose on the frame for Camera 2
            cv2.aruco.drawDetectedMarkers(frame_2, [corners_2])
            cv2.drawFrameAxes(frame_2, K2, D2, rvec_2, tvec_2, 0.1)

            # Draw the reprojected wand tip on the image
            frame_2 = cv2.circle(frame_2, (int(image_points2[0]), int(image_points2[1])), 5, (0, 0, 255), -1)
            
            # Express wand frame + wand tip in world and send it to rviz
            # Compute wand frame pose in the world
            world_R2_wand = world_R2_cam @ transformation_matrix_2[:3, :3]
            world_T2_wand = world_R2_cam @ transformation_matrix_2[:3, 3] + world_T2_cam
            
            world_tip_pos2 = world_R2_wand @ wand_local.reshape(3,) + world_T2_wand
            
            # Create a TransformStamped message for the wand frame
            wand_transform = TransformStamped()
            wand_transform.header.stamp = rospy.Time.now()
            wand_transform.header.frame_id = "world"
            wand_transform.child_frame_id = "wand_frame_2"
            wand_transform.transform.translation.x = world_T2_wand[0]
            wand_transform.transform.translation.y = world_T2_wand[1]
            wand_transform.transform.translation.z = world_T2_wand[2]
            wand_quaternion = Rotation.from_matrix(world_R2_wand).as_quat()
            wand_transform.transform.rotation.x = wand_quaternion[0]
            wand_transform.transform.rotation.y = wand_quaternion[1]
            wand_transform.transform.rotation.z = wand_quaternion[2]
            wand_transform.transform.rotation.w = wand_quaternion[3]
            tf_broadcaster.sendTransform(wand_transform)
            
            # Publish the wand tip position as a Marker
            tip_marker2 = Marker()
            tip_marker2.header.stamp = rospy.Time.now()
            tip_marker2.header.frame_id = "world"
            tip_marker2.ns = "wand_tip_2"
            tip_marker2.id = 0
            tip_marker2.type = Marker.SPHERE
            tip_marker2.action = Marker.ADD
            tip_marker2.pose.position.x = world_tip_pos2[0]
            tip_marker2.pose.position.y = world_tip_pos2[1]
            tip_marker2.pose.position.z = world_tip_pos2[2]
            tip_marker2.pose.orientation.x = 0.0
            tip_marker2.pose.orientation.y = 0.0
            tip_marker2.pose.orientation.z = 0.0
            tip_marker2.pose.orientation.w = 1.0
            tip_marker2.scale.x = 0.05  # Diameter of the sphere
            tip_marker2.scale.y = 0.05
            tip_marker2.scale.z = 0.05
            tip_marker2.color.a = 1.0  # Alpha (transparency)
            tip_marker2.color.r = 0.0  # Red
            tip_marker2.color.g = 0.0  # Green
            tip_marker2.color.b = 1.0  # Blue
            tip_publisher2.publish(tip_marker2)

        # Display the frames for both cameras
        cv2.imshow('Camera 1 Pose Estimation', frame_1)
        cv2.imshow('Camera 2 Pose Estimation', frame_2)
        
        c = cv2.waitKey(10)
        if c == ord('q'):
            print("quit")
            break
finally : 
    # Release the camera captures
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()