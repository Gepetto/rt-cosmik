# Launch gepetto-gui then 
# To run the code : python3 apps/get_AT_and_q0.py cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m
# or python3 -m apps.get_AT_and_q0 cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m

import argparse
import os
import cv2
import numpy as np
from mmdeploy_runtime import PoseTracker
import time 
import csv
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
import sys
from collections import deque
from datetime import datetime
import yaml

from utils.lstm_v2 import augmentTRC, loadModel
from utils.model_utils import Robot, get_jcp_global_pos, calculate_segment_lengths_from_dict, model_scaling_from_dict
from utils.calib_utils import load_cam_params, load_cam_to_cam_params, load_cam_pose, list_cameras_with_v4l2
from utils.triangulation_utils import triangulate_points
from utils.ik_utils import RT_IK
from utils.iir import IIR
from utils.viz_utils import visualize, VISUALIZATION_CFG, place

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
    args = parse_args()
    np.set_printoptions(precision=4, suppress=True)

    keypoints_csv_file_path = os.path.join(parent_directory,'output/calib_keypoints_3d_positions.csv')
    augmented_csv_file_path = os.path.join(parent_directory, 'output/calib_augmented_markers_positions.csv')
    q_csv_file_path = os.path.join(parent_directory,'output/calib_q.csv')

    calib_human_path = os.path.join(parent_directory, "cams_calibration/human_params/human_anthropometry.yaml")

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

    width = 1280
    height = 720
    resize=1280

    # kpt_thr = 0.7

    ### Initialize cams stream
    camera_dict = list_cameras_with_v4l2
    captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in camera_dict.keys()]
    print(captures)
    input()

    width_vids = []
    height_vids = []

    for idx, cap in enumerate(captures):
        if not cap.isOpened():
            continue

        # Apply settings
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fs)

        width_vids.append(width)
        height_vids.append(height)

        # Verify settings
        print(f"Camera {camera_indices[idx]} Settings:")
        print("FOURCC:", "".join([chr(int(cap.get(cv2.CAP_PROP_FOURCC)) >> (i * 8) & 0xFF) for i in range(4)]))
        print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("FPS:", cap.get(cv2.CAP_PROP_FPS))

    
    # Check if cameras opened successfully
    for i, cap in enumerate(captures):
        if not cap.isOpened():
            print(f"Error: Camera {i} not opened.")
            return

    ### Loading human urdf
    human = Robot(os.path.join(parent_directory,'urdf/human_5dof.urdf'),os.path.join(parent_directory,'meshes')) 
    human_model = human.model
    human_data = human.data

    pin.framesForwardKinematics(human_model,human_data, pin.neutral(human_model))
    pos_ankle_calib = human_data.oMi[human_model.getJointId('ankle_Z')].translation

    subject_height = 1.81
    subject_mass = 73.0

    ### IK calculations 
    q = np.array([np.pi/2,0,0,-np.pi,0]) # init pos
    keys_to_track_list = ['Knee', 'midHip', 'Shoulder', 'Elbow', 'Wrist']
    dict_dof_to_keypoints = dict(zip(['knee_Z', 'lumbar_Z', 'shoulder_Z', 'elbow_Z', 'hand_fixed'],keys_to_track_list))

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
            timestamp=datetime.now()
            formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f ")
            
            frames = [cap.read()[1] for cap in captures]
            
            if not all(frame is not None for frame in frames):
                continue
            
            keypoints_list = []
            frame_idx += 1  # Increment frame counter

            # Process each frame individually
            for idx, frame in enumerate(frames):

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
                    
                    if (len(segment_lengths)<100):
                        if first_sample:
                            lstm_dict = dict(zip(keypoint_names+marker_names, np.concatenate((filtered_keypoints_buffer[-1],augmented_markers),axis=0)))
                            jcp_dict = get_jcp_global_pos(lstm_dict,pos_ankle_calib)
                            seg_lengths = calculate_segment_lengths_from_dict(jcp_dict)
                            segment_lengths.append(seg_lengths)
                        else :
                            lstm_dict = dict(zip(marker_names, augmented_markers))
                            jcp_dict = get_jcp_global_pos(lstm_dict,pos_ankle_calib)
                            seg_lengths = calculate_segment_lengths_from_dict(jcp_dict)
                            segment_lengths.append(seg_lengths)
                    else :
                        if not_calibrated:
                            segment_lengths_array = np.array(segment_lengths)

                            dict_mean_segment_lengths = dict(zip(['Knee', 'Hip', 'Shoulder', 'Elbow', 'Wrist'], np.mean(segment_lengths_array,axis=0))) 
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

                            lstm_dict = dict(zip(marker_names, augmented_markers))
                            jcp_dict = get_jcp_global_pos(lstm_dict,pos_ankle_calib)
                            
                            for key in jcp_dict.keys():
                                viz.viewer.gui.addSphere('world/'+key,0.01,[0,0,1,1])
                                place(viz, 'world/'+key, pin.SE3(np.eye(3), np.array([jcp_dict[key][0],jcp_dict[key][1],jcp_dict[key][2]])))

                            ### IK calculations
                            t1 =time.time()
                            ik_class = RT_IK(human_model, jcp_dict, q, keys_to_track_list, dt, dict_dof_to_keypoints, False)
                            q = ik_class.solve_ik_sample_casadi()
                            t2 = time.time()
                            # print("Time for IK",t2-t1)

                            pin.framesForwardKinematics(human_model,human_data, q)
                            for frame in human_model.frames.tolist():
                                place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])
                            
                            viz.display(q)
                            ik_class._q0=q
                            not_calibrated=False

                        else :
                            lstm_dict = dict(zip(marker_names, augmented_markers))
                            jcp_dict = get_jcp_global_pos(lstm_dict,pos_ankle_calib)

                            for key in jcp_dict.keys():
                                place(viz, 'world/'+key, pin.SE3(np.eye(3), np.array([jcp_dict[key][0],jcp_dict[key][1],jcp_dict[key][2]])))

                            ### IK calculations
                            ik_class._dict_m= jcp_dict
                            t1 = time.time()
                            q = ik_class.solve_ik_sample_casadi()
                            t2 = time.time()
                            # print("Time for IK",t2-t1)

                            pin.framesForwardKinematics(human_model,human_data, q)

                            for frame in human_model.frames.tolist():
                                place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])

                            viz.display(q)

                            ik_class._q0 = q

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
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
