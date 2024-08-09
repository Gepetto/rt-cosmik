#! /home/gepetto/mmpose/mmpose_env/bin/python3

# To run the code from RT-COSMIK root : python -m apps.rtmpose_2realsense 

import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np
import csv
import pinocchio as pin
import rospy
from sensor_msgs.msg import JointState
from utils.read_write_utils import formatting_keypoints, set_zero_data
from utils.model_utils import Robot, model_scaling
from utils.calib_utils import load_cam_params, load_cam_to_cam_params, load_cam_pose
from utils.triangulation_utils import triangulate_points
from utils.ik_utils import RT_Quadprog

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

import pyrealsense2 as rs
import cv2

from datetime import datetime
import time

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# Initialize ROS node 
rospy.init_node('human_rt_ik', anonymous=True)
pub = rospy.Publisher('/human_RT_joint_angles', JointState, queue_size=10)

csv_file_path = './output/keypoints_3d_positions.csv'
csv2_file_path = './output/q.csv'

K1, D1 = load_cam_params("cams_calibration/cam_params/c1_params_color_test_test.yml")
K2, D2 = load_cam_params("cams_calibration/cam_params/c2_params_color_test_test.yml")
R,T = load_cam_to_cam_params("cams_calibration/cam_params/c1_to_c2_params_color_test_test.yml")

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

# Loading camera pose 
R1_global, T1_global = load_cam_pose('cams_calibration/cam_params/camera1_pose_test_test.yml')
R1_global = R1_global@pin.utils.rotate('z', np.pi) # aligns measurements to human model definition

local_runtime = True

det_config = 'config/rtmdet_nano_320-8xb32_coco-person.py'
det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
pose_config = 'config/rtmpose-s_8xb256-420e_coco-256x192.py'
# pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'
# pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth'

device = 'cuda:0'
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=False)))

# build detector
detector = init_detector(
    det_config,
    det_checkpoint,
    device=device
)

# build pose estimator
pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=cfg_options
)

# init visualizer
pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_pose_estimator
visualizer.set_dataset_meta(pose_estimator.dataset_meta)

dof_names=['ankle_Z', 'knee_Z', 'lumbar_Z', 'shoulder_Z', 'elbow_Z'] 

keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]

mapping = dict(zip(keypoint_names,[i for i in range(len(keypoint_names))]))

keypoint_colors = {
    "Nose": "blue", "Left Eye": "blue", "Right Eye": "blue",
    "Left Ear": "blue", "Right Ear": "blue",
    "Left Shoulder": "green", "Right Shoulder": "orange",
    "Left Elbow": "green", "Right Elbow": "orange",
    "Left Wrist": "green", "Right Wrist": "orange",
    "Left Hip": "green", "Right Hip": "orange",
    "Left Knee": "green", "Right Knee": "orange",
    "Left Ankle": "green", "Right Ankle": "orange"}


# Initialize CSV files
with open(csv_file_path, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    # Write the header row
    csv_writer.writerow(['Frame', 'Time','Keypoint', 'X', 'Y', 'Z'])

with open(csv2_file_path, mode='w', newline='') as file2:
    csv2_writer = csv.writer(file2)
    # Write the header row
    csv2_writer.writerow(['Frame','Time','q0', 'q1','q2','q3','q4'])

def get_device_serial_numbers():
    """Get a list of serial numbers for connected RealSense devices."""
    ctx = rs.context()
    return [device.get_info(rs.camera_info.serial_number) for device in ctx.query_devices()]

def process_realsense_multi(detector, pose_estimator, visualizer, show_interval=1):
    """Process frames from multiple Intel RealSense cameras and visualize predicted keypoints."""
    
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

    serial_numbers = get_device_serial_numbers()
    pipelines = []
    for serial in serial_numbers:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        pipelines.append(pipeline)

    # Loading human urdf
    human = Robot('models/human_urdf/urdf/human.urdf','models') 
    human_model = human.model

    # IK calculations 
    q = pin.neutral(human_model)
    dt = 1/30
    keys_to_track_list = ["Right Knee","Right Hip","Right Shoulder","Right Elbow","Right Wrist"]
    dict_dof_to_keypoints = dict(zip(keys_to_track_list,['knee_Z', 'lumbar_Z', 'shoulder_Z', 'elbow_Z', 'hand_fixed']))

    first_sample = True 
    frame_idx = 0
    output_root = './output'
    mmengine.mkdir_or_exist(output_root)

    try:
        while not rospy.is_shutdown():
            timestamp=datetime.now()
            formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f ")

            frames_list = [pipeline.wait_for_frames().get_color_frame() for pipeline in pipelines]
            if not all(frames_list):
                continue

            frame_idx += 1
            keypoints_list = []

            for idx, color_frame in enumerate(frames_list):
                time_init = time.time()
                frame = np.asanyarray(color_frame.get_data())
                
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Predict bbox
                scope = detector.cfg.get('default_scope', 'mmdet')
                if scope is not None:
                    init_default_scope(scope)
                detect_result = inference_detector(detector, frame_rgb)
                pred_instance = detect_result.pred_instances.cpu().numpy()
                bboxes = np.concatenate(
                    (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
                bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                            pred_instance.scores > 0.45)]
                bboxes = bboxes[nms(bboxes, 0.3)][:, :4]
                
                # Predict keypoints
                pose_results = inference_topdown(pose_estimator, frame_rgb, bboxes)
                data_samples = merge_data_samples(pose_results)

                # keypoints_list.append(data_samples.pred_instances.keypoints.reshape((26,2)).flatten())
                keypoints_list.append(pose_results[0].pred_instances.keypoints.reshape((17,2)).flatten())
                time_end = time.time()
                delta_time = (time_end-time_init)*1000          

                print('delta time for mmpose ', delta_time)

                # Show the results
                visualizer.add_datasample(
                    'result',
                    frame_rgb,
                    data_sample=data_samples,
                    draw_gt=False,
                    draw_heatmap=False,
                    draw_bbox=True,
                    show=False,
                    wait_time=show_interval,
                    out_file=None,
                    kpt_thr=0.4)
                
                # Retrieve the visualized image
                vis_result = visualizer.get_image()
                
                # Convert image from RGB to BGR for OpenCV
                vis_result_bgr = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)
                
                # Display the frame using OpenCV
                cv2.imshow(f'Visualization Result {idx}', vis_result_bgr)
            
            time_init = time.time()
            p3d_frame = triangulate_points(keypoints_list, mtxs, dists, projections)
            time_end = time.time()

            delta_time = (time_end-time_init)*1000          

            print('delta time for triangul ', delta_time)

            # Subtract the translation vector (shifting the origin)
            keypoints_shifted = p3d_frame - T1_global.T

            # Apply the rotation matrix to align the points
            keypoints_in_world = np.dot(keypoints_shifted,R1_global)
            
            if first_sample:
                x0_ankle, y0_ankle, z0_ankle = keypoints_in_world[mapping['Right Ankle']][0], keypoints_in_world[mapping['Right Ankle']][1], keypoints_in_world[mapping['Right Ankle']][2]
                keypoints_in_world=set_zero_data(keypoints_in_world,x0_ankle,y0_ankle,z0_ankle)
                # Scaling segments lengths 
                human_model, _ = model_scaling(human_model, keypoints_in_world)
                first_sample = False
            else :
                keypoints_in_world=set_zero_data(keypoints_in_world,x0_ankle,y0_ankle,z0_ankle)

            with open(csv_file_path, mode='a', newline='') as file:
                csv_writer = csv.writer(file)
                for jj in range(len(keypoint_names)):
                    # Write to CSV
                    csv_writer.writerow([frame_idx, formatted_timestamp,keypoint_names[jj], keypoints_in_world[jj][0], keypoints_in_world[jj][1], keypoints_in_world[jj][2]])
            
            dict_keypoints = formatting_keypoints(keypoints_in_world,keypoint_names)
            # print(dict_keypoints)
            time_init=time.time()
            ik_problem = RT_Quadprog(human_model, dict_keypoints, q, keys_to_track_list,dt,dict_dof_to_keypoints,False)
            q = ik_problem.solve_ik_sample()

            time_end = time.time()

            delta_time = (time_end-time_init)*1000          

            print('delta time for IK ', delta_time)

            with open(csv2_file_path, mode='a', newline='') as file2:
                csv2_writer = csv.writer(file2)
                csv2_writer.writerow([frame_idx,formatted_timestamp,q[0],q[1],q[2],q[3],q[4]])

            # Publish joint angles 
            joint_state_msg=JointState()
            joint_state_msg.header.stamp=rospy.Time.now()
            joint_state_msg.name = dof_names
            joint_state_msg.position = q.tolist()
            pub.publish(joint_state_msg)

            # print(q)
            # Press 'q' to exit the loop, 's' to start/stop saving
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
    finally:
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try :
        process_realsense_multi(
            detector,
            pose_estimator,
            visualizer,
            show_interval=1
        )
    except rospy.ROSInterruptException:
        pass