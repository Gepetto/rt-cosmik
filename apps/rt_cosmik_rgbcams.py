# To run the code : python3 apps/rt_cosmik_rgbcams.py cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m
# or python3 -m apps.rt_cosmik_rgbcams cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m

# rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map world 5

import argparse
import os
import cv2
import numpy as np
from mmdeploy_runtime import PoseTracker
import time 
import csv
import pinocchio as pin
from collections import deque
from utils.lstm_v2 import augmentTRC, loadModel
from datetime import datetime
import pandas as pd

# don't forget to source dependancies
import rospy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from utils.read_write_utils import formatting_keypoints, set_zero_data
from utils.model_utils import Robot, model_scaling
from utils.calib_utils import load_cam_params, load_cam_to_cam_params, load_cam_pose
from utils.triangulation_utils import triangulate_points
from utils.ik_utils import RT_Quadprog
from utils.iir import IIR

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
parent_directory = os.path.dirname(script_directory)

augmenter_path = os.path.join(parent_directory, 'augmentation_model')

keypoints_buffer = deque(maxlen=30)
warmed_models= loadModel(augmenterDir=augmenter_path, augmenterModelName="LSTM",augmenter_model='v0.3')


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use SDK Python API')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument(
        'det_model',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument(
        'pose_model',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('--output_dir', help='output directory', default=None)
    parser.add_argument(
        '--skeleton',
        default='body26',
        choices=['coco', 'coco_wholebody','body26'],
        help='skeleton for keypoints')

    args = parser.parse_args()
    return args

VISUALIZATION_CFG = dict(
    body26=dict(
        skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 2), (5, 18),(6, 18), (17,18), # Head, shoulders, and neck connections
                    (5, 7), (7, 9),                                                              # Right arm connections
                    (6, 8), (8, 10),                                                             # Left arm connections
                    (18, 19),                                                                    # Trunk connection
                    (11, 13), (13, 15), (15, 20), (15, 22), (15, 24),                            # Left leg and foot connections
                    (12, 14), (14, 16), (16, 21), (16, 23), (16, 25),                            # Right leg and foot connections
                    (12, 19), (11, 19)],                                                         # Hip connection

        # Updated palette
        palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]],
    
        # Updated link color
        link_color = [
            1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
            2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
            2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],

        # Updated point color
        point_color = [
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4,
            5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5,
            5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        sigmas = [0.026] * 26
    ),
    coco=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
        palette=[(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
                 (255, 153, 255), (153, 204, 255), (255, 102, 255),
                 (255, 51, 255), (102, 178, 255), (51, 153, 255),
                 (255, 153, 153), (255, 102, 102), (255, 51, 51),
                 (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
                 (0, 0, 255), (255, 0, 0), (255, 255, 255)],
        link_color=[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ],
        point_color=[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ]),
    coco_wholebody=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                  (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
                  (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                  (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
                  (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
                  (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
                  (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
                  (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
                  (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
                  (129, 130), (130, 131), (131, 132)],
        palette=[(51, 153, 255), (0, 255, 0), (255, 128, 0), (255, 255, 255),
                 (255, 153, 255), (102, 178, 255), (255, 51, 51)],
        link_color=[
            1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1,
            1, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2,
            2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1,
            1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068,
            0.066, 0.066, 0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043,
            0.040, 0.035, 0.031, 0.025, 0.020, 0.023, 0.029, 0.032, 0.037,
            0.038, 0.043, 0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012,
            0.012, 0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007, 0.007,
            0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011, 0.009, 0.011,
            0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010, 0.034, 0.008,
            0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009, 0.009,
            0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01,
            0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
            0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032,
            0.02, 0.019, 0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047,
            0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
            0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031
        ]))

def visualize(frame,
              results,
              output_dir,
              idx,
              frame_id,
              thr=0.5,
              resize=1280,
              skeleton_type='coco'):

    skeleton = VISUALIZATION_CFG[skeleton_type]['skeleton']
    palette = VISUALIZATION_CFG[skeleton_type]['palette']
    link_color = VISUALIZATION_CFG[skeleton_type]['link_color']
    point_color = VISUALIZATION_CFG[skeleton_type]['point_color']

    scale = resize / max(frame.shape[0], frame.shape[1])
    keypoints, bboxes, _ = results
    scores = keypoints[..., 2]
    keypoints = (keypoints[..., :2] * scale).astype(int)
    bboxes *= scale
    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    for kpts, score, bbox in zip(keypoints, scores, bboxes):
        show = [1] * len(kpts)
        for (u, v), color in zip(skeleton, link_color):
            if score[u] > thr and score[v] > thr:
                cv2.line(img, kpts[u], tuple(kpts[v]), palette[color], 1,
                         cv2.LINE_AA)
            else:
                show[u] = show[v] = 0
        for kpt, show, color in zip(kpts, show, point_color):
            if show:
                cv2.circle(img, kpt, 1, palette[color], 2, cv2.LINE_AA)
    if output_dir:
        cv2.imwrite(f'{output_dir}/{str(frame_id).zfill(6)}.jpg', img)
    else:
        cv2.imshow('pose_tracker'+str(idx), img)
        return cv2.waitKey(1) != 'q'
    return True

def list_available_cameras(max_cameras=10):
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()  # Release the camera after checking
    return available_cameras

def publish_keypoints_as_marker_array(keypoints, marker_pub, keypoint_names, frame_id="world"):
    marker_array = MarkerArray()
    marker_template = Marker()
    marker_template.header.frame_id = frame_id
    marker_template.header.stamp = rospy.Time.now()
    marker_template.ns = "keypoints"
    marker_template.type = Marker.SPHERE
    marker_template.action = Marker.ADD
    marker_template.scale.x = 0.05  # Adjust size as needed
    marker_template.scale.y = 0.05
    marker_template.scale.z = 0.05
    marker_template.color.a = 1.0  # Fully opaque
    
    palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]]
    
    keypoints_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 
            1, 2, 1, 2, 1, 2, 1, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2
        ]

    for i, keypoint in enumerate(keypoints):
        marker = Marker()
        # Copy attributes from the template to the new marker
        marker.header = marker_template.header
        marker.ns = marker_template.ns
        marker.type = marker_template.type
        marker.action = marker_template.action
        marker.scale = marker_template.scale
        marker.color.a = marker_template.color.a
        marker.id = i
        marker.text = keypoint_names[i] if i < len(keypoint_names) else f"keypoint_{i}"

        # Set color based on index 
        color_info = palette[keypoints_color[i]]

        if i < len(keypoints_color):
            marker.color.r = color_info[0]/255
            marker.color.g = color_info[1]/255
            marker.color.b = color_info[2]/255
        else:
            marker.color.r = 1.0  # Fallback color
            marker.color.g = 0.0
            marker.color.b = 0.0

        # Set position of the keypoint
        marker.pose.position = Point(x=keypoint[0], y=keypoint[1], z=keypoint[2])
        marker_array.markers.append(marker)

    marker_pub.publish(marker_array)

def publish_augmented_markers(keypoints, marker_pub, keypoint_names, frame_id="world"):
    marker_array = MarkerArray()
    marker_template = Marker()
    marker_template.header.frame_id = frame_id
    marker_template.header.stamp = rospy.Time.now()
    marker_template.ns = "markers"
    marker_template.type = Marker.SPHERE
    marker_template.action = Marker.ADD
    marker_template.scale.x = 0.05  # Adjust size as needed
    marker_template.scale.y = 0.05
    marker_template.scale.z = 0.05
    marker_template.color.a = 1.0  # Fully opaque
    marker_template.color.r = 0.0  # Fallback color
    marker_template.color.g = 0.0
    marker_template.color.b = 1.0

    for i, keypoint in enumerate(keypoints):
        marker = Marker()
        # Copy attributes from the template to the new marker
        marker.header = marker_template.header
        marker.ns = marker_template.ns
        marker.type = marker_template.type
        marker.action = marker_template.action
        marker.scale = marker_template.scale
        marker.color.a = marker_template.color.a
        marker.color.r = marker_template.color.r
        marker.color.g = marker_template.color.g
        marker.color.b = marker_template.color.b
        marker.id = i
        marker.text = keypoint_names[i] if i < len(keypoint_names) else f"marker_{i}"

        # Set position of the keypoint
        marker.pose.position = Point(x=keypoint[0], y=keypoint[1], z=keypoint[2])
        marker_array.markers.append(marker)

    marker_pub.publish(marker_array)

def main():
    args = parse_args()
    np.set_printoptions(precision=4, suppress=True)

    keypoints_csv_file_path = os.path.join(parent_directory,'output/keypoints_3d_positions.csv')
    augmented_csv_file_path = os.path.join(parent_directory, 'output/augmented_markers_positions.csv')
    q_csv_file_path = os.path.join(parent_directory,'output/q.csv')

    K1, D1 = load_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c1_params_color_test_test.yml"))
    K2, D2 = load_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c2_params_color_test_test.yml"))
    R,T = load_cam_to_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c1_to_c2_params_color_test_test.yml"))

    dict_cam = {
        "cam1": {
            "mtx":np.array(K1),
            "dist":D1,
            "rotation":np.eye(3),
            "translation":[
                0.,
                0.,
                0.,
            ],
        },
        "cam2": {
            "mtx":np.array(K2),
            "dist":D2,
            "rotation":R,
            "translation":T,
        },
    }

    rotations=[]
    translations=[]
    dists=[]
    mtxs=[]
    projections=[]

    for cam in dict_cam :
        rotation=np.array(dict_cam[cam]["rotation"])
        rotations.append(rotation)
        translation=np.array([dict_cam[cam]["translation"]]).reshape(3,1)
        translations.append(translation)
        projection = np.concatenate([rotation, translation], axis=-1)
        projections.append(projection)
        dict_cam[cam]["projection"] = projection
        dists.append(dict_cam[cam]["dist"])
        mtxs.append(dict_cam[cam]["mtx"])

    ### Loading camera pose 
    R1_global, T1_global = load_cam_pose(os.path.join(parent_directory,'cams_calibration/cam_params/camera1_pose_test_test.yml'))
    # R1_global = R1_global@pin.utils.rotate('z', np.pi) # aligns measurements to human model definition
    
    dof_names=['ankle_Z', 'knee_Z', 'lumbar_Z', 'shoulder_Z', 'elbow_Z'] 

    keypoint_names = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Head",
        "Neck", "Hip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"]

    marker_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']

    mapping = dict(zip(keypoint_names,[i for i in range(len(keypoint_names))]))

    ### Initialize CSV files
    with open(keypoints_csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        # Write the header row
        csv_writer.writerow(['Frame', 'Time','Keypoint', 'X', 'Y', 'Z'])

    with open(augmented_csv_file_path, mode='w', newline='') as file2:
        csv2_writer = csv.writer(file2)
        # Write the header row
        csv2_writer.writerow(['Frame', 'Time','Marker', 'X', 'Y', 'Z'])

    with open(q_csv_file_path, mode='w', newline='') as file3:
        csv3_writer = csv.writer(file3)
        # Write the header row
        csv3_writer.writerow(['Frame','Time','q0', 'q1','q2','q3','q4'])

    ### Initialize ROS node 
    rospy.init_node('human_rt_ik', anonymous=True)
    
    # pub = rospy.Publisher('/human_RT_joint_angles', JointState, queue_size=10)
    keypoints_pub = rospy.Publisher('/pose_keypoints', MarkerArray, queue_size=10)
    augmented_markers_pub = rospy.Publisher('/markers_pose', MarkerArray, queue_size=10)

    width = 1280
    height = 720
    resize=1280

    # kpt_thr = 0.7

    ### Initialize cams stream
    camera_indices = list_available_cameras()
    # print(camera_indices)

    # if no webcam
    # captures = [cv2.VideoCapture(idx) for idx in camera_indices]

    # if webcam remove it 
    captures = [cv2.VideoCapture(idx) for idx in camera_indices if idx !=2]
    
    width_vids = []
    height_vids = []

    for cap in captures: 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # HD
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # HD
        cap.set(cv2.CAP_PROP_FPS, 40)  # Set frame rate to 40fps
        width_vids.append(width)
        height_vids.append(height)

    
    # Check if cameras opened successfully
    for i, cap in enumerate(captures):
        if not cap.isOpened():
            print(f"Error: Camera {i} not opened.")
            return

    ### Loading human urdf
    human = Robot(os.path.join(parent_directory,'models/human_urdf/urdf/human.urdf'),os.path.join(parent_directory,'models')) 
    human_model = human.model

    subject_height = 1.81
    subject_mass = 73.0

    ### IK calculations 
    q = np.array([np.pi/2,0,0,-np.pi,0]) # init pos
    dt = 1/30
    keys_to_track_list = ["Right Knee","Right Hip","Right Shoulder","Right Elbow","Right Wrist"]
    dict_dof_to_keypoints = dict(zip(keys_to_track_list,['knee_Z', 'lumbar_Z', 'shoulder_Z', 'elbow_Z', 'hand_fixed']))

    ### Set up real time filter 
    # Constant
    fs = 40
    num_channel = 3*len(keypoint_names)

    # Creating IIR instance
    iir_filter = IIR(
        num_channel=num_channel,
        sampling_frequency=fs
    )

    iir_filter.add_filter(order=4, cutoff=7, filter_type='lowpass')
    
    first_sample = True 
    frame_idx = 0

    # Define the codec and create VideoWriter objects for both RGB streams
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI files
    out_vid1 = cv2.VideoWriter(os.path.join(parent_directory,'output/cam1.mp4'), fourcc, 30.0, (int(width_vids[0]), int(height_vids[0])), True)
    out_vid2 = cv2.VideoWriter(os.path.join(parent_directory,'output/cam2.mp4'), fourcc, 30.0, (int(width_vids[1]), int(height_vids[1])), True)
    out_vid = [out_vid1, out_vid2]

    tracker = PoseTracker(
        det_model=args.det_model,
        pose_model=args.pose_model,
        device_name=args.device_name)

    # optionally use OKS for keypoints similarity comparison
    sigmas = VISUALIZATION_CFG[args.skeleton]['sigmas']
    state = tracker.create_state(
        det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    try : 
        while not rospy.is_shutdown():
            timestamp=datetime.now()
            formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f ")
            
            frames = [cap.read()[1] for cap in captures]
            
            if not all(frame is not None for frame in frames):
                continue
            
            keypoints_list = []
            frame_idx += 1  # Increment frame counter

            # Process each frame individually
            for idx, frame in enumerate(frames):
                #Videos savings
                if idx == 0 : 
                    out_vid1.write(frame)
                elif idx == 1 : 
                    out_vid2.write(frame)

                t0 = time.time()
                results = tracker(state, frame, detect=-1)
                scale = resize / max(frame.shape[0], frame.shape[1])
                keypoints, bboxes, _ = results
                scores = keypoints[..., 2]
                # keypoints = (keypoints[..., :2] * scale).astype(float)
                keypoints = (keypoints[..., :2] ).astype(float)
                bboxes *= scale
                t1 =time.time()
                print("Time of inference for one image",t1-t0)

                if keypoints.size == 0 or keypoints.flatten().shape != (52,):
                    print('i')
                    
                else :
                    keypoints_list.append(keypoints.reshape((26,2)).flatten())
                    
                if not visualize(
                        frame,
                        results,
                        args.output_dir,
                        idx,
                        frame_idx + idx,
                        skeleton_type=args.skeleton):
                    break

            if len(keypoints_list)!=2: #number of cams
                pass
            else :
                p3d_frame = triangulate_points(keypoints_list, mtxs, dists, projections)
                keypoints_in_world = p3d_frame

                # Subtract the translation vector (shifting the origin)
                keypoints_shifted = p3d_frame - T1_global.T

                # Apply the rotation matrix to align the points
                keypoints_in_world = np.dot(keypoints_shifted,R1_global)
                # print(keypoints_in_world)

                # # For Kahina 
                # flattened_keypoints = keypoints_in_world.flatten()
                # #print(flattened_keypoints)
                # row = flattened_keypoints.tolist()
                # with open('keypoints_rt_kahina.csv', mode='a') as file:
                #     csv_writer = csv.writer(file)
                #     csv_writer.writerow(row)

                # publish_keypoints_as_marker_array(keypoints_in_world, keypoints_pub, keypoint_names)
                
                
            #     # Translate so that the right ankle is at 0 everytime
            #     #keypoints_in_world -= keypoints_in_world[mapping["Right Ankle"],:]
            #     flattened_keypoints = keypoints_in_world.flatten()
            #     #print(flattened_keypoints)
            #     row = flattened_keypoints.tolist()

                # Saving keypoints
                with open(keypoints_csv_file_path, mode='a', newline='') as file:
                    csv_writer = csv.writer(file)
                    for jj in range(len(keypoint_names)):
                        # Write to CSV
                        csv_writer.writerow([frame_idx, formatted_timestamp,keypoint_names[jj], keypoints_in_world[jj][0], keypoints_in_world[jj][1], keypoints_in_world[jj][2]])
                
                if first_sample:
                    for k in range(30):
                        keypoints_buffer.append(keypoints_in_world)  #add the 1st frame 30 times
                    first_sample = False  #put the flag to false 
                else:
                    keypoints_buffer.append(keypoints_in_world) #add the keypoints to the buffer normally 
                
                if len(keypoints_buffer) == 30:
                    keypoints_buffer_array = np.array(keypoints_buffer)

                    # Filter keypoints in world to remove noisy artefacts 
                    filtered_keypoints_buffer = iir_filter.filter(np.reshape(keypoints_buffer_array,(30, 3*len(keypoint_names))))

                    filtered_keypoints_buffer = np.reshape(filtered_keypoints_buffer,(30, len(keypoint_names), 3))

                    publish_keypoints_as_marker_array(filtered_keypoints_buffer[-1], keypoints_pub, keypoint_names)
                    
                    #call augmentTrc
                    augmented_markers = augmentTRC(keypoints_buffer_array, subject_mass=subject_mass, subject_height=subject_height, models = warmed_models,
                               augmenterDir=os.path.join(parent_directory,"augmentation_model")
                               ,augmenter_model='v0.3', offset=True)

                    # # Save for kahina 
                    # # Convert responses_all_conc to a pandas DataFrame
                    # df = pd.DataFrame([augmented_markers])
                    # df.to_csv("markers_rt_kahina.csv", mode='a',header= False, index=False)


                    if len(augmented_markers) % 3 != 0:
                        raise ValueError("The length of the list must be divisible by 3.")

                    augmented_markers = np.array(augmented_markers).reshape(-1, 3) 

                    # Saving keypoints
                    with open(augmented_csv_file_path, mode='a', newline='') as file:
                        csv_writer = csv.writer(file)
                        for jj in range(len(augmented_markers)):
                            # Write to CSV
                            csv_writer.writerow([frame_idx, formatted_timestamp,marker_names[jj], augmented_markers[jj][0], augmented_markers[jj][1], augmented_markers[jj][2]])
                

                    publish_augmented_markers(augmented_markers, augmented_markers_pub, marker_names)                
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("quit")
                break    
            
    finally:
        # Release the camera captures
        for cap in captures:
            cap.release()
        out_vid1.release()
        out_vid2.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
