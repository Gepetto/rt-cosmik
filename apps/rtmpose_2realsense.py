# To run the code from RT-COSMIK root : python -m apps.rtmpose_2realsense.py  

import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np
from utils.calib_utils import load_cam_params, load_cam_to_cam_params
from utils.triangulation_utils import DLT_adaptive, triangulate_points

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

import pyrealsense2 as rs
import cv2
import os.path as osp
import json_tricks as json

import csv
import matplotlib.pyplot as plt
from scipy import linalg

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


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

local_runtime = True

det_config = 'config/rtmdet_nano_320-8xb32_coco-person.py'
det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
pose_config = 'config/rtmpose-s_8xb256-420e_coco-256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'


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

# Define the keypoint names in order as per the Halpe 26-keypoint format
# keypoint_names = [
#     "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
#     "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
#     "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
#     "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
#     "Pelvis", "Upper Neck", "Head Top", 
#     "Left Big Toe", "Left Small Toe", "Left Heel", 
#     "Right Big Toe", "Right Small Toe", "Right Heel"
# ]

keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]

# keypoint_colors = {
#     "Nose": "blue", "Left Eye": "blue", "Right Eye": "blue",
#     "Left Ear": "blue", "Right Ear": "blue",
#     "Left Shoulder": "green", "Right Shoulder": "orange",
#     "Left Elbow": "green", "Right Elbow": "orange",
#     "Left Wrist": "green", "Right Wrist": "orange",
#     "Left Hip": "green", "Right Hip": "orange",
#     "Left Knee": "green", "Right Knee": "orange",
#     "Left Ankle": "green", "Right Ankle": "orange",
#     "Pelvis": "blue", "Upper Neck": "orange", "Head Top":"orange", 
#     "Left Big Toe": "green", "Left Small Toe": "green", "Left Heel": "green", 
#     "Right Big Toe": "orange", "Right Small Toe": "orange", "Right Heel": "orange"
# }

keypoint_colors = {
    "Nose": "blue", "Left Eye": "blue", "Right Eye": "blue",
    "Left Ear": "blue", "Right Ear": "blue",
    "Left Shoulder": "green", "Right Shoulder": "orange",
    "Left Elbow": "green", "Right Elbow": "orange",
    "Left Wrist": "green", "Right Wrist": "orange",
    "Left Hip": "green", "Right Hip": "orange",
    "Left Knee": "green", "Right Knee": "orange",
    "Left Ankle": "green", "Right Ankle": "orange"}

# Initialize Matplotlib for real-time 3D visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')

# Initialize CSV file
csv_file_path = './output/keypoints_3d_positions.csv'
with open(csv_file_path, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    # Write the header row
    csv_writer.writerow(['Frame', 'Keypoint', 'X', 'Y', 'Z'])

def get_device_serial_numbers():
    """Get a list of serial numbers for connected RealSense devices."""
    ctx = rs.context()
    serial_numbers = []
    for device in ctx.query_devices():
        serial_numbers.append(device.get_info(rs.camera_info.serial_number))
    return serial_numbers

def process_realsense_multi(detector, pose_estimator, visualizer, show_interval=1):
    """Process frames from multiple Intel RealSense cameras and visualize predicted keypoints."""
    
    serial_numbers = get_device_serial_numbers()
    pipelines = []
    for serial in serial_numbers:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)
        pipelines.append(pipeline)
    
    saving = False
    pred_instances_list = []
    frame_idx = 0
    output_root = './output'
    mmengine.mkdir_or_exist(output_root)
    pred_save_path = f'{output_root}/results_realsense.json'
    
    try:
        while True:
            frames_list = [pipeline.wait_for_frames().get_color_frame() for pipeline in pipelines]
            if not all(frames_list):
                continue
            
            frame_idx += 1
            keypoints_list = []
            ax.clear()

            for idx, color_frame in enumerate(frames_list):
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
            
            p3d_frame = triangulate_points(keypoints_list, mtxs, dists, projections)
            print(p3d_frame)

            with open(csv_file_path, mode='a', newline='') as file:
                csv_writer = csv.writer(file)
                for jj in range(len(keypoint_names)):
                    # Write to CSV
                    csv_writer.writerow([frame_idx, keypoint_names[jj], p3d_frame[jj][0], p3d_frame[jj][1], p3d_frame[jj][2]])
            #         ax.scatter(p3d_frame[jj][0],p3d_frame[jj][1],p3d_frame[jj][2], color = keypoint_colors[keypoint_names[jj]], label = keypoint_names[jj])

            # ax.set_xlim([-2,2])
            # ax.set_ylim([-2,2])
            # ax.set_zlim([0,4])
            # ax.legend()
            # plt.draw()
            # plt.pause(0.001)

            # Press 'q' to exit the loop, 's' to start/stop saving
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
    finally:
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()
        
        if pred_instances_list:
            with open(pred_save_path, 'w') as f:
                json.dump(
                    dict(
                        meta_info=pose_estimator.dataset_meta,
                        instance_info=pred_instances_list),
                    f,
                    indent='\t')
            print(f'predictions have been saved at {pred_save_path}')

process_realsense_multi(
    detector,
    pose_estimator,
    visualizer,
    show_interval=1
)
