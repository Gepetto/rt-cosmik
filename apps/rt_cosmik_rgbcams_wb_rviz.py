# To run the code : python3 apps/rt_cosmik_rgbcams_wb_rviz.py cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m
# or python3 -m apps.rt_cosmik_rgbcams_wb_rviz cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m

import argparse
import os
import cv2
import numpy as np
np.set_printoptions(precision=4, suppress=True)
from mmdeploy_runtime import PoseTracker
import time 
time.sleep(5)
import pinocchio as pin
from collections import deque
from datetime import datetime

# don't forget to source dependancies
import rospy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray
import tf2_ros

from utils.lstm_v2 import augmentTRC, loadModel
from utils.model_utils import build_model_challenge
from utils.calib_utils import get_cameras_params, load_cam_params, load_cam_to_cam_params, load_cam_pose, list_cameras_with_v4l2
from utils.triangulation_utils import triangulate_points
from utils.ik_utils import RT_IK
from utils.iir import IIR
from utils.viz_utils import visualize, VISUALIZATION_CFG
from utils.ros_utils import publish_keypoints_as_marker_array, publish_augmented_markers, publish_kinematics
from utils.read_write_utils import init_csv, save_3dpos_to_csv, save_q_to_csv
from utils.settings import Settings

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
    # FIRST, PARAM LOADING
    settings = Settings()
    args = parse_args()

    # CSV PATHS
    keypoints_csv_file_path = os.path.join(parent_directory,'output/keypoints_3d_positions.csv')
    augmented_csv_file_path = os.path.join(parent_directory, 'output/augmented_markers_positions.csv')
    q_csv_file_path = os.path.join(parent_directory,'output/q.csv')

    # TODO : ADJUST WITH THE NUMBER OF CAMERAS
    K1, D1 = load_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c1_params_color_test_test.yml"))
    K2, D2 = load_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c2_params_color_test_test.yml"))
    R,T = load_cam_to_cam_params(os.path.join(parent_directory,"cams_calibration/cam_params/c1_to_c2_params_color_test_test.yml"))
    mtxs, dists, projections, rotations, translations = get_cameras_params(K1, D1, K2, D2, R, T)

    ### Loading camera pose 
    cam_R1_world, cam_T1_world = load_cam_pose(os.path.join(parent_directory,'cams_calibration/cam_params/camera1_pose_test_test.yml'))
    # Inverse the pose to get cam in world frame 
    world_R1_cam = cam_R1_world.T
    world_T1_cam = -cam_R1_world.T@cam_T1_world
    world_T1_cam = world_T1_cam.reshape((3,))

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

    dof_names=['middle_lumbar_Z', 'middle_lumbar_Y', 'right_shoulder_Z', 'right_shoulder_X', 'right_shoulder_Y', 'right_elbow_Z', 'right_elbow_Y', 'left_shoulder_Z', 'left_shoulder_X', 'left_shoulder_Y', 'left_elbow_Z', 'left_elbow_Y', 'right_hip_Z', 'right_hip_X', 'right_hip_Y', 'right_knee_Z', 'right_ankle_Z','left_hip_Z', 'left_hip_X', 'left_hip_Y', 'left_knee_Z', 'left_ankle_Z'] 

    ### Initialize CSV files
    init_csv(keypoints_csv_file_path,['Frame', 'Time','Keypoint', 'X', 'Y', 'Z'])
    init_csv(augmented_csv_file_path,['Frame', 'Time','Marker', 'X', 'Y', 'Z'])
    init_csv(q_csv_file_path,['Frame','Time','q0', 'q1','q2','q3','q4'])

    ### Initialize ROS node 
    rospy.init_node('human_rt_ik', anonymous=True)
    pub = rospy.Publisher('/human_RT_joint_angles', JointState, queue_size=10)
    keypoints_pub = rospy.Publisher('/pose_keypoints', MarkerArray, queue_size=10)
    augmented_markers_pub = rospy.Publisher('/markers_pose', MarkerArray, queue_size=10)
    br = tf2_ros.TransformBroadcaster()

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

    
    ### Set up real time filter 
    # Constant
    num_channel = 3*len(settings.keypoints_names)

    # Creating IIR instance
    iir_filter = IIR(
        num_channel=num_channel,
        sampling_frequency=settings.system_freq
    )

    iir_filter.add_filter(order=settings.order, cutoff=settings.cutoff_freq, filter_type=settings.filter_type)
    
    first_sample = True 
    frame_idx = 0

    # Define the codec and create VideoWriter objects for both RGB streams
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI files
    out_vid1 = cv2.VideoWriter(os.path.join(parent_directory,'output/cam1.mp4'), fourcc, 40.0, (int(settings.width), int(settings.height)), True)
    out_vid2 = cv2.VideoWriter(os.path.join(parent_directory,'output/cam2.mp4'), fourcc, 40.0, (int(settings.width), int(settings.height)), True)

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
                keypoints, bboxes, _ = results
                keypoints = (keypoints[..., :2] ).astype(float)
                
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
                
                # Save keypoints to csv
                save_3dpos_to_csv(keypoints_csv_file_path,keypoints_in_world,settings.keypoints_names,frame_idx, formatted_timestamp)
                
                if first_sample:
                    for k in range(30):
                        keypoints_buffer.append(keypoints_in_world)  #add the 1st frame 30 times
                else:
                    keypoints_buffer.append(keypoints_in_world) #add the keypoints to the buffer normally 
                
                if len(keypoints_buffer) == 30:
                    keypoints_buffer_array = np.array(keypoints_buffer)

                    # Filter keypoints in world to remove noisy artefacts 
                    filtered_keypoints_buffer = iir_filter.filter(np.reshape(keypoints_buffer_array,(30, 3*len(settings.keypoints_names))))
                    filtered_keypoints_buffer = np.reshape(filtered_keypoints_buffer,(30, len(settings.keypoints_names), 3))

                    publish_keypoints_as_marker_array(filtered_keypoints_buffer[-1], keypoints_pub, settings.keypoints_names)
                    
                    #call augmentTrc
                    augmented_markers = augmentTRC(filtered_keypoints_buffer, subject_mass=settings.human_mass, subject_height=settings.human_height, models = warmed_models,
                               augmenterDir=augmenter_path, augmenter_model='v0.3', offset=True)


                    if len(augmented_markers) % 3 != 0:
                        raise ValueError("The length of the list must be divisible by 3.")

                    augmented_markers = np.array(augmented_markers).reshape(-1, 3) 

                    # Saving markers
                    save_3dpos_to_csv(augmented_csv_file_path,augmented_markers,settings.marker_names,frame_idx, formatted_timestamp)
                   
                    publish_augmented_markers(augmented_markers, augmented_markers_pub, settings.marker_names)

                    if first_sample:
                        lstm_dict = dict(zip(settings.keypoints_names+settings.marker_names, np.concatenate((filtered_keypoints_buffer[-1],augmented_markers),axis=0)))
                        ### Generate human model
                        human_model, human_geom_model, visuals_dict = build_model_challenge(lstm_dict, lstm_dict, meshes_folder_path)

                        ### IK init 
                        q = pin.neutral(human_model) # init pos

                        ### IK calculations
                        ik_class = RT_IK(human_model, lstm_dict, q, keys_to_track_list, settings.dt)
                        q = ik_class.solve_ik_sample_casadi()
                        ik_class._q0=q

                        publish_kinematics(q, pub,dof_names,br)
                        
                        # Saving kinematics
                        save_q_to_csv(q_csv_file_path,q,frame_idx, formatted_timestamp)     

                        first_sample = False  #put the flag to false 
                    else:
                        lstm_dict = dict(zip(settings.marker_names, augmented_markers))
                        ### IK calculations
                        ik_class._dict_m= lstm_dict
                        # q = ik_class.solve_ik_sample_quadprog() 
                        q = ik_class.solve_ik_sample_casadi()

                        ik_class._q0 = q

                        publish_kinematics(q, pub,dof_names, br )     

                        # Saving kinematics
                        save_q_to_csv(q_csv_file_path,q,frame_idx, formatted_timestamp)   
                
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
