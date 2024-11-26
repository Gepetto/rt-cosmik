# To run the code : python3 apps/rt_cosmik_rgbcams_wb_rviz.py cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m
# or python3 -m apps.rt_cosmik_rgbcams_wb_rviz cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m

# rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map world 5

import argparse
import os
import cv2
import numpy as np
from mmdeploy_runtime import PoseTracker
import time 
# time.sleep(15)
import csv
import pinocchio as pin
from collections import deque
from utils.lstm_v2 import augmentTRC, loadModel
from datetime import datetime
import pandas as pd

# don't forget to source dependancies
import rospy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray
import tf2_ros

from utils.read_write_utils import formatting_keypoints, set_zero_data
from utils.model_utils import build_model_challenge
from utils.calib_utils import load_cam_params, load_cam_to_cam_params, load_cam_pose, list_available_cameras
from utils.triangulation_utils import triangulate_points
from utils.ik_utils import RT_IK
from utils.iir import IIR
from utils.viz_utils import visualize, VISUALIZATION_CFG
from utils.ros_utils import publish_keypoints_as_marker_array, publish_augmented_markers, publish_kinematics

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
parent_directory = os.path.dirname(script_directory)

augmenter_path = os.path.join(parent_directory, 'augmentation_model')
meshes_folder_path = os.path.join(parent_directory, 'meshes')

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
    cam_R1_world, cam_T1_world = load_cam_pose(os.path.join(parent_directory,'cams_calibration/cam_params/camera1_pose_test_test.yml'))
    
    # Inverse the pose to get cam in world frame 
    world_R1_cam = cam_R1_world.T
    world_T1_cam = -cam_R1_world.T@cam_T1_world
    world_T1_cam = world_T1_cam.reshape((3,))

    fs = 40
    dt = 1/fs

    keys_to_track_list = ['C7_study',
                        'r.ASIS_study', 'L.ASIS_study', 
                        'r.PSIS_study', 'L.PSIS_study', 
                        
                        'r_shoulder_study',
                        'r_lelbow_study', 'r_melbow_study',
                        'r_lwrist_study', 'r_mwrist_study',
                        'r_ankle_study', 'r_mankle_study',
                        'r_toe_study','r_5meta_study', 'r_calc_study',
                        'r_knee_study', 'r_mknee_study',
                        'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study',
                        'r_sh1_study', 'r_sh2_study', 'r_sh3_study',
                        
                        'L_shoulder_study', 
                        'L_lelbow_study', 'L_melbow_study',
                        'L_lwrist_study','L_mwrist_study',
                        'L_ankle_study', 'L_mankle_study', 
                        'L_toe_study','L_5meta_study', 'L_calc_study',
                        'L_knee_study', 'L_mknee_study',
                        'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study',
                        'L_sh1_study', 'L_sh2_study', 'L_sh3_study']
    
    subject_height = 1.81
    subject_mass = 73.0

    dof_names=['middle_lumbar_Z', 'middle_lumbar_Y', 'right_shoulder_Z', 'right_shoulder_X', 'right_shoulder_Y', 'right_elbow_Z', 'right_elbow_Y', 'left_shoulder_Z', 'left_shoulder_X', 'left_shoulder_Y', 'left_elbow_Z', 'left_elbow_Y', 'right_hip_Z', 'right_hip_X', 'right_hip_Y', 'right_knee_Z', 'right_ankle_Z','left_hip_Z', 'left_hip_X', 'left_hip_Y', 'left_knee_Z', 'left_ankle_Z'] 

    keypoint_names = [
        "Nose", "LEye", "REye", "LEar", "REar", 
        "LShoulder", "RShoulder", "LElbow", "RElbow", 
        "LWrist", "RWrist", "LHip", "RHip", 
        "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
        "Neck", "midHip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"]

    marker_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']

    ### Initialize CSV files
    with open(keypoints_csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        # Write the header row
        csv_writer.writerow(['Frame', 'Time','Keypoint', 'X', 'Y', 'Z'])

    with open(augmented_csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        # Write the header row
        csv_writer.writerow(['Frame', 'Time','Marker', 'X', 'Y', 'Z'])

    with open(q_csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        # Write the header row
        csv_writer.writerow(['Frame','Time'] + ['FF_X', 'FF_Y', 'FF_Z', 'FF_QUAT_X', 'FF_QUAT_Y', 'FF_QUAT_Z','FF_QUAT_W' ] + dof_names)

    ### Initialize ROS node 
    rospy.init_node('human_rt_ik', anonymous=True)
    
    pub = rospy.Publisher('/human_RT_joint_angles', JointState, queue_size=10)
    keypoints_pub = rospy.Publisher('/pose_keypoints', MarkerArray, queue_size=10)
    augmented_markers_pub = rospy.Publisher('/markers_pose', MarkerArray, queue_size=10)
    br = tf2_ros.TransformBroadcaster()

    width = 1280
    height = 720
    resize=1280

    ### Initialize cams stream
    camera_indices = list_available_cameras()
    captures = [cv2.VideoCapture(idx) for idx in camera_indices]
    
    width_vids = []
    height_vids = []

    for cap in captures: 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # HD
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # HD
        cap.set(cv2.CAP_PROP_FPS, fs)  # Set frame rate to x fps
        width_vids.append(width)
        height_vids.append(height)

    # Check if cameras opened successfully
    for i, cap in enumerate(captures):
        if not cap.isOpened():
            print(f"Error: Camera {i} not opened.")
            return
    
    ### Set up real time filter 
    # Constant
    num_channel = 3*len(keypoint_names)

    # Creating IIR instance
    iir_filter = IIR(
        num_channel=num_channel,
        sampling_frequency=fs
    )

    iir_filter.add_filter(order=4, cutoff=15, filter_type='lowpass')
    
    first_sample = True 
    frame_idx = 0

    # Define the codec and create VideoWriter objects for both RGB streams
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI files
    out_vid1 = cv2.VideoWriter(os.path.join(parent_directory,'output/cam1.mp4'), fourcc, 30.0, (int(width_vids[0]), int(height_vids[0])), True)
    out_vid2 = cv2.VideoWriter(os.path.join(parent_directory,'output/cam2.mp4'), fourcc, 30.0, (int(width_vids[1]), int(height_vids[1])), True)

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
                # print("Time of inference for one image",t1-t0)

                if keypoints.size == 0 or keypoints.flatten().shape != (52,):
                    pass
                    
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
                keypoints_in_cam = p3d_frame

                # Apply the rotation matrix to align the points
                keypoints_in_world = np.array([np.dot(world_R1_cam,point) + world_T1_cam for point in keypoints_in_cam])
                
                # Saving keypoints
                with open(keypoints_csv_file_path, mode='a', newline='') as file:
                    csv_writer = csv.writer(file)
                    for jj in range(len(keypoint_names)):
                        # Write to CSV
                        csv_writer.writerow([frame_idx, formatted_timestamp,keypoint_names[jj], keypoints_in_world[jj][0], keypoints_in_world[jj][1], keypoints_in_world[jj][2]])
                
                if first_sample:
                    for k in range(30):
                        keypoints_buffer.append(keypoints_in_world)  #add the 1st frame 30 times
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
                               augmenterDir=augmenter_path, augmenter_model='v0.3', offset=True)


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


                    if first_sample:
                        lstm_dict = dict(zip(keypoint_names+marker_names, np.concatenate((filtered_keypoints_buffer[-1],augmented_markers),axis=0)))
                        ### Generate human model
                        human_model, human_geom_model, visuals_dict = build_model_challenge(lstm_dict, lstm_dict, meshes_folder_path)

                        ### IK init 
                        q = pin.neutral(human_model) # init pos

                        ### IK calculations
                        ik_class = RT_IK(human_model, lstm_dict, q, keys_to_track_list, dt)
                        q = ik_class.solve_ik_sample_casadi()
                        ik_class._q0=q

                        publish_kinematics(q, pub,dof_names,br)
                        
                        # Saving kinematics
                        with open(q_csv_file_path, mode='a', newline='') as file:
                            csv_writer = csv.writer(file)
                            # Write to CSV
                            csv_writer.writerow([frame_idx, formatted_timestamp]+q.tolist())

                        first_sample = False  #put the flag to false 
                    else:
                        lstm_dict = dict(zip(marker_names, augmented_markers))
                        ### IK calculations
                        ik_class._dict_m= lstm_dict
                        # q = ik_class.solve_ik_sample_quadprog() 
                        q = ik_class.solve_ik_sample_casadi()

                        ik_class._q0 = q

                        publish_kinematics(q, pub,dof_names, br )     

                        # Saving kinematics
                        with open(q_csv_file_path, mode='a', newline='') as file:
                            csv_writer = csv.writer(file)
                            # Write to CSV
                            csv_writer.writerow([frame_idx, formatted_timestamp]+q.tolist())
                
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
