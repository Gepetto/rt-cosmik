import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np
import csv

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

import pyrealsense2 as rs
import cv2
import os.path as osp
import json_tricks as json

import matplotlib.pyplot as plt

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

det_config = 'projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'
det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
pose_config = 'projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-t_8xb1024-700e_body8-halpe26-256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth'

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
visualizer.set_dataset_meta(pose_estimator.dataset_meta)

# Define the keypoint names in order as per the Halpe 26-keypoint format
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
    "Pelvis", "Thorax", "Upper Neck", "Head Top", 
    "Left Big Toe", "Left Small Toe", "Left Heel", 
    "Right Big Toe", "Right Small Toe", "Right Heel"
]

keypoint_colors = {
    "Nose": "blue", "Left Eye": "blue", "Right Eye": "blue",
    "Left Ear": "blue", "Right Ear": "blue",
    "Left Shoulder": "green", "Right Shoulder": "orange",
    "Left Elbow": "green", "Right Elbow": "orange",
    "Left Wrist": "green", "Right Wrist": "orange",
    "Left Hip": "green", "Right Hip": "orange",
    "Left Knee": "green", "Right Knee": "orange",
    "Left Ankle": "green", "Right Ankle": "orange",
    "Pelvis": "blue", "Thorax": "blue", "Upper Neck": "orange", "Head Top":"orange", 
    "Left Big Toe": "green", "Left Small Toe": "green", "Left Heel": "green", 
    "Right Big Toe": "orange", "Right Small Toe": "orange", "Right Heel": "orange"
}

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

def process_realsense_single(detector, pose_estimator, visualizer, show_interval=1):
    """Process frames from a single Intel RealSense camera and visualize predicted keypoints with depth."""
    
    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)

    # Get the depth sensor's depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Get the intrinsics of the depth stream
    depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    
    saving = False
    pred_instances_list = []
    frame_idx = 0
    output_root = './output'
    mmengine.mkdir_or_exist(output_root)
    pred_save_path = f'{output_root}/results_realsense.json'
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            
            frame_idx += 1
            # Convert color frame to numpy array
            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
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
                                           pred_instance.scores > 0.6)]
            bboxes = bboxes[nms(bboxes, 0.3)][:, :4]
            
            # Predict keypoints
            pose_results = inference_topdown(pose_estimator, frame_rgb, bboxes)
            data_samples = merge_data_samples(pose_results)
            keypoints_with_depth = []
            ax.clear()
            if len(bboxes)>0:
                with open(csv_file_path, mode='a', newline='') as file:
                    csv_writer = csv.writer(file)
                    
                    for pose in data_samples.pred_instances.keypoints:
                        for i, keypoint in enumerate(pose):
                            x, y = int(keypoint[0]), int(keypoint[1])

                            # Ensure that (x, y) are within the bounds of the depth frame
                            if 0 <= x < depth_intrinsics.width and 0 <= y < depth_intrinsics.height:
                                depth = depth_frame.get_distance(x, y)
                                X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
                                keypoints_with_depth.append({
                                    'name': keypoint_names[i], 
                                    'x': x, 
                                    'y': y, 
                                    'depth': depth, 
                                    'X': X, 
                                    'Y': Y, 
                                    'Z': Z
                                })
                                # Write to CSV
                                csv_writer.writerow([frame_idx, keypoint_names[i], X, Y, Z])
                                print(f"{keypoint_names[i]} - Pixel (x, y): ({x}, {y}), Depth: {depth:.2f} meters, Real-world (X, Y, Z): ({X:.2f}, {Y:.2f}, {Z:.2f})")
                                ax.scatter(X,Y,Z, color = keypoint_colors[keypoint_names[i]], label = keypoint_names[i])
                            else:
                                print(f"{keypoint_names[i]} - Keypoint (x, y): ({x}, {y}) is out of range.")
            else :
                keypoints_with_depth = [[0,0,0] for _ in keypoint_names]
                for i in range(len(keypoint_names)):
                    ax.scatter(0,0,0, color = keypoint_colors[keypoint_names[i]], label = keypoint_names[i])

            ax.set_xlim([-5,5])
            ax.set_ylim([-5,5])
            ax.set_zlim([0,8])
            ax.legend()
            plt.draw()
            plt.pause(0.001)
            
            if saving:
                # save prediction results with depth information
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        camera_id=0,
                        instances=[{'keypoints_with_depth': keypoints_with_depth}]))
            
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
                kpt_thr=0.3)
            
            # Retrieve the visualized image
            vis_result = visualizer.get_image()
            
            # Convert image from RGB to BGR for OpenCV
            vis_result_bgr = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)
            
            # Display the frame using OpenCV
            cv2.imshow(f'Visualization Result', vis_result_bgr)
            
            # Press 'q' to exit the loop, 's' to start/stop saving
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                saving = not saving
                if saving:
                    print("Started saving keypoints.")
                else:
                    print("Stopped saving keypoints.")
        
    finally:
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

process_realsense_single(
    detector,
    pose_estimator,
    visualizer,
    show_interval=1
)