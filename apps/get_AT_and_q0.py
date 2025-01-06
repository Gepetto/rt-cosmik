# Launch gepetto-gui then 
# To run the code : python3 apps/get_AT_and_q0.py cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m
# or python3 -m apps.get_AT_and_q0 cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m

import argparse
import os
import cv2
import numpy as np
np.set_printoptions(precision=4, suppress=True)
from mmdeploy_runtime import PoseTracker
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
import sys
from collections import deque
from datetime import datetime
import yaml
import time

from utils.lstm_v2 import augmentTRC, loadModel
from utils.model_utils import Robot, get_jcp_global_pos, calculate_segment_lengths_from_dict, model_scaling_from_dict
from utils.calib_utils import load_cam_params, load_cam_to_cam_params, load_cam_pose, list_cameras_with_v4l2, get_cameras_params
from utils.triangulation_utils import triangulate_points
from utils.ik_utils import RT_IK
from utils.iir import IIR
from utils.viz_utils import visualize, VISUALIZATION_CFG, place
from utils.read_write_utils import init_csv, save_3dpos_to_csv, save_q_to_csv
from utils.settings import Settings

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


def main():
    # FIRST, PARAM LOADING
    settings = Settings()
    args = parse_args()

    # CSV PATHS
    keypoints_csv_file_path = os.path.join(parent_directory,'output/calib_keypoints_3d_positions.csv')
    augmented_csv_file_path = os.path.join(parent_directory, 'output/calib_augmented_markers_positions.csv')
    q_csv_file_path = os.path.join(parent_directory,'output/calib_q.csv')
    calib_human_path = os.path.join(parent_directory, "cams_calibration/human_params/human_anthropometry.yaml")

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

    ### Initialize CSV files
    init_csv(keypoints_csv_file_path,['Frame', 'Time','Keypoint', 'X', 'Y', 'Z'])
    init_csv(augmented_csv_file_path,['Frame', 'Time','Marker', 'X', 'Y', 'Z'])
    init_csv(q_csv_file_path,['Frame','Time','q0', 'q1','q2','q3','q4'])

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


    ### Loading human urdf
    human = Robot(os.path.join(parent_directory,'urdf/human_5dof.urdf'),os.path.join(parent_directory,'meshes')) 
    human_model = human.model
    human_data = human.data

    pin.framesForwardKinematics(human_model,human_data, pin.neutral(human_model))
    pos_ankle_calib = human_data.oMi[human_model.getJointId('ankle_Z')].translation

    ### IK calculations 
    q = np.array([np.pi/2,0,0,-np.pi,0]) # init pos
    keys_to_track_list = ['Knee', 'midHip', 'Shoulder', 'Elbow', 'Wrist']
    dict_dof_to_keypoints = dict(zip(['knee_Z', 'lumbar_Z', 'shoulder_Z', 'elbow_Z', 'hand_fixed'],keys_to_track_list))

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
    not_calibrated = True
    frame_idx = 0

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

    segment_lengths = []

    try : 
        while True:
            start_time_all = time.perf_counter()
            timestamp=datetime.now()
            formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f ")
            
            start_time = time.perf_counter()
            frames = [cap.read()[1] for cap in captures]
            
            if not all(frame is not None for frame in frames):
                continue
            elapsed = time.perf_counter()-start_time
            print(f"Time for cam readings: {elapsed:.4f} seconds")
            
            keypoints_list = []
            frame_idx += 1  # Increment frame counter

            start_time_mmpose_all = time.perf_counter()
            # Process each frame individually
            for idx, frame in enumerate(frames):
                start_time = time.perf_counter()
                results = tracker(state, frame, detect=-1)
                keypoints, bboxes, _ = results
                keypoints = (keypoints[..., :2] ).astype(float)

                if keypoints.size == 0 or keypoints.flatten().shape != (52,):
                    pass
                    
                else :
                    keypoints_list.append(keypoints.reshape((26,2)).flatten())
                elapsed = time.perf_counter()-start_time
                print(f"Time for mmpose inference on one image: {elapsed:.4f} seconds")

                start_time = time.perf_counter()
                if not visualize(
                        frame,
                        results,
                        args.output_dir,
                        idx,
                        frame_idx + idx,
                        skeleton_type=args.skeleton):
                    break
                elapsed = time.perf_counter() - start_time
                print(f"Time for cam visu one image: {elapsed:.4f} seconds")

            elapsed_mmpose_all = time.perf_counter()-start_time_mmpose_all
            print(f"Time for mmpose + visu on 2 images: {elapsed_mmpose_all:.4f} seconds")

            if len(keypoints_list)!=2: #number of cams
                pass

            else :
                start_time = time.perf_counter()
                p3d_frame = triangulate_points(keypoints_list, mtxs, dists, projections)
                keypoints_in_cam = p3d_frame

                # Apply the rotation matrix to align the points
                keypoints_in_world = np.array([np.dot(world_R1_cam,point) + world_T1_cam for point in keypoints_in_cam])
                elapsed =time.perf_counter()-start_time
                print(f"Time for triangulation block: {elapsed:.4f} seconds")

                start_time =time.perf_counter()
                # Saving keypoints
                save_3dpos_to_csv(keypoints_csv_file_path,keypoints_in_world,settings.keypoints_names,frame_idx, formatted_timestamp)
                elapsed =time.perf_counter()-start_time
                print(f"Time for keypoints saving block: {elapsed:.4f} seconds")

                if first_sample:
                    for k in range(30):
                        keypoints_buffer.append(keypoints_in_world)  #add the 1st frame 30 times
                else:
                    keypoints_buffer.append(keypoints_in_world) #add the keypoints to the buffer normally 
                
                if len(keypoints_buffer) == 30:
                    start_time = time.perf_counter()
                    keypoints_buffer_array = np.array(keypoints_buffer)

                    # Filter keypoints in world to remove noisy artefacts 
                    filtered_keypoints_buffer = iir_filter.filter(np.reshape(keypoints_buffer_array,(30, 3*len(settings.keypoints_names))))

                    filtered_keypoints_buffer = np.reshape(filtered_keypoints_buffer,(30, len(settings.keypoints_names), 3))
                    elapsed = time.perf_counter()-start_time
                    print(f"Time for filter block: {elapsed:.4f} seconds")
                    
                    start_time = time.perf_counter()
                    #call augmentTrc
                    augmented_markers = augmentTRC(filtered_keypoints_buffer, subject_mass=settings.human_mass, subject_height=settings.human_height, models = warmed_models,
                               augmenterDir=augmenter_path, augmenter_model='v0.3', offset=True)


                    if len(augmented_markers) % 3 != 0:
                        raise ValueError("The length of the list must be divisible by 3.")

                    augmented_markers = np.array(augmented_markers).reshape(-1, 3)
                    elapsed =time.perf_counter()-start_time
                    print(f"Time for data augmenter block: {elapsed:.4f} seconds")

                    start_time = time.perf_counter()
                    # Saving markers
                    save_3dpos_to_csv(augmented_csv_file_path,augmented_markers,settings.marker_names,frame_idx, formatted_timestamp)
                    print(f"Time for augmenter data saving block: {elapsed:.4f} seconds")

                    if (len(segment_lengths)<100):
                        if first_sample:
                            lstm_dict = dict(zip(settings.keypoints_names+settings.marker_names, np.concatenate((filtered_keypoints_buffer[-1],augmented_markers),axis=0)))
                            jcp_dict = get_jcp_global_pos(lstm_dict,pos_ankle_calib, settings.side_to_track)
                            seg_lengths = calculate_segment_lengths_from_dict(jcp_dict)
                            segment_lengths.append(seg_lengths)
                            first_sample = False
                        else :
                            lstm_dict = dict(zip(settings.marker_names, augmented_markers))
                            jcp_dict = get_jcp_global_pos(lstm_dict,pos_ankle_calib,settings.side_to_track)
                            seg_lengths = calculate_segment_lengths_from_dict(jcp_dict)
                            segment_lengths.append(seg_lengths)
                    else :
                        if not_calibrated:
                            segment_lengths_array = np.array(segment_lengths)

                            dict_mean_segment_lengths = dict(zip(['Knee', 'Hip', 'Shoulder', 'Elbow', 'Wrist'], np.round(np.mean(segment_lengths_array,axis=0),4))) 
                            # Convert NumPy types to native Python types
                            dict_mean_segment_lengths = {key: float(value) for key, value in dict_mean_segment_lengths.items()}
                            
                            # Dump the dictionary to a YAML file
                            with open(calib_human_path, 'w') as file:
                                yaml.dump(dict_mean_segment_lengths, file, default_flow_style=False)

                            print(f"Dictionary successfully dumped to {calib_human_path}")

                            human_model = model_scaling_from_dict(human_model, dict_mean_segment_lengths)

                            # VISUALIZATION
                            viz = GepettoVisualizer(human_model,human.collision_model,human.visual_model)
                            try:
                                viz.initViewer()
                            except ImportError as err:
                                print(
                                    "Error while initializing the viewer. It seems you should install gepetto-viewer"
                                )
                                print(err)
                                sys.exit(0)

                            try:
                                viz.loadViewerModel("pinocchio")
                            except AttributeError as err:
                                print(
                                    "Error while loading the viewer human_model. It seems you should start gepetto-viewer"
                                )
                                print(err)
                                sys.exit(0)

                            for frame in human_model.frames.tolist():
                                viz.viewer.gui.addXYZaxis('world/'+frame.name,[1,0,0,1],0.01,0.1)

                            lstm_dict = dict(zip(settings.marker_names, augmented_markers))
                            jcp_dict = get_jcp_global_pos(lstm_dict,pos_ankle_calib,settings.side_to_track)
                            
                            for key in jcp_dict.keys():
                                viz.viewer.gui.addSphere('world/'+key,0.01,[0,0,1,1])
                                place(viz, 'world/'+key, pin.SE3(np.eye(3), np.array([jcp_dict[key][0],jcp_dict[key][1],jcp_dict[key][2]])))

                            ### IK calculations
                            ik_class = RT_IK(human_model, jcp_dict, q, keys_to_track_list, settings.dt, dict_dof_to_keypoints, False)
                            q = ik_class.solve_ik_sample_casadi()

                            pin.framesForwardKinematics(human_model,human_data, q)
                            for frame in human_model.frames.tolist():
                                place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])
                            
                            viz.display(q)
                            ik_class._q0=q
                            not_calibrated=False

                        else :
                            start_time = time.perf_counter()
                            lstm_dict = dict(zip(settings.marker_names, augmented_markers))
                            jcp_dict = get_jcp_global_pos(lstm_dict,pos_ankle_calib,settings.side_to_track)

                            for key in jcp_dict.keys():
                                place(viz, 'world/'+key, pin.SE3(np.eye(3), np.array([jcp_dict[key][0],jcp_dict[key][1],jcp_dict[key][2]])))

                            ### IK calculations
                            ik_class._dict_m= jcp_dict
                            q = ik_class.solve_ik_sample_casadi()
                            elapsed =time.perf_counter()-start_time
                            print(f"Time for ik block: {elapsed:.4f} seconds")

                            pin.framesForwardKinematics(human_model,human_data, q)

                            for frame in human_model.frames.tolist():
                                place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])

                            viz.display(q)

                            ik_class._q0 = q

                        start_time = time.perf_counter()
                        # Saving kinematics
                        save_q_to_csv(q_csv_file_path,q,frame_idx, formatted_timestamp)
                        elapsed =time.perf_counter()-start_time
                        print(f"Time for ik saving block: {elapsed:.4f} seconds")

            elapsed_all = time.perf_counter() - start_time_all
            print(f"Time for while block: {elapsed_all:.4f} seconds")
    finally:
        # Release the camera captures
        for cap in captures:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
