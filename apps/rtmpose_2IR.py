import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np
from calib_utils import load_cam_params, load_cam_to_cam_params

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

def DLT_adaptive(projections, points):
    A=[]
    for i in range(len(projections)):
        P=projections[i]
        point = points[i]

        for j in range (len(point)):
            A.append(point[j][1]*P[2,:] - P[1,:])
            A.append(P[0,:] - point[j][0]*P[2,:])

    A = np.array(A).reshape((-1,4))
    B = A.transpose() @ A
    _, _, Vh = linalg.svd(B, full_matrices = False)

    return Vh[3,0:3]/Vh[3,3]

def triangulate_points(keypoints_list, mtxs, dists, projections):
    p3ds_frame=[]
    undistorted_points = []

    for ii in range(len(keypoints_list)):
        points = keypoints_list[ii] 
        distCoeffs_mat = np.array([dists[ii]]).reshape(-1, 1)
        points_undistorted = cv2.undistortPoints(np.array(points).reshape(-1, 1, 2), mtxs[ii], distCoeffs_mat)
        undistorted_points.append(points_undistorted)

    for point_idx in range(26):
        points_per_point = [undistorted_points[i][point_idx] for i in range(len(undistorted_points))]
        _p3d = DLT_adaptive(projections, points_per_point)
        p3ds_frame.append(_p3d)

    return np.array(p3ds_frame)

K1, D1 = load_cam_params("cams_params/c1_params_ir_test_test.yml")
K2, D2 = load_cam_params("cams_params/c2_params_ir_test_test.yml")
R,T = load_cam_to_cam_params("cams_params/c1_to_c2_params_ir_test_test.yml")

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

det_config = 'projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'
det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
pose_config = 'projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-t_8xb1024-700e_body8-halpe26-256x192.py'
# pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/1/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth'
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
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_pose_estimator
visualizer.set_dataset_meta(pose_estimator.dataset_meta)

# Define the keypoint names in order as per the Halpe 26-keypoint format
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
    "Pelvis", "Upper Neck", "Head Top", 
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
    "Pelvis": "blue", "Upper Neck": "orange", "Head Top":"orange", 
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
    """Process frames from a single Intel RealSense camera and visualize predicted keypoints ."""

    # Initialize the pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Enable the IR streams (IR1 and IR2)
    config.enable_stream(rs.stream.infrared, 1)  # Enable IR1 (left)
    config.enable_stream(rs.stream.infrared, 2)  # Enable IR2 (right)

    # Start streaming
    pipeline.start(config)
    
    # Get the stream profile to extract the width and height for video writer
    profile = pipeline.get_active_profile()
    ir1_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir2_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()

    # Get the width and height
    width = ir1_profile.width()
    height = ir1_profile.height()

    saving = False
    pred_instances_list = []
    frame_idx = 0
    output_root = './output'
    mmengine.mkdir_or_exist(output_root)
    pred_save_path = f'{output_root}/results_realsense.json'

    # Define the codec and create VideoWriter objects for both IR streams
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI files
    out_ir1 = cv2.VideoWriter(f'{output_root}/IR1.avi', fourcc, 30.0, (width, height), False)
    out_ir2 = cv2.VideoWriter(f'{output_root}/IR2.avi', fourcc, 30.0, (width, height), False)

    try:
        while True:
            print("Frame")
            # Wait for a frame of data
            frames = pipeline.wait_for_frames()

            # Get the infrared frames
            ir1_frame = frames.get_infrared_frame(1)
            ir2_frame = frames.get_infrared_frame(2)

            if not ir1_frame or not ir2_frame:
                continue

            frames_list = [ir1_frame,ir2_frame]
            frame_idx += 1

            keypoints_list = []
            ax.clear()

            for idx, ir_frame in enumerate(frames_list):
                frame = np.asanyarray(ir_frame.get_data())
                if idx == 0 : 
                    out_ir1.write(frame)
                elif idx == 1 : 
                    out_ir2.write(frame)

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # Predict bbox
                scope = detector.cfg.get('default_scope', 'mmdet')
                if scope is not None:
                    init_default_scope(scope)
                detect_result = inference_detector(detector, frame_gray)
                pred_instance = detect_result.pred_instances.cpu().numpy()
                bboxes = np.concatenate(
                    (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
                bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                                pred_instance.scores > 0.5)]
                bboxes = bboxes[nms(bboxes, 0.3)][:, :4]
            
                # Predict keypoints
                pose_results = inference_topdown(pose_estimator, frame_gray, bboxes)
                data_samples = merge_data_samples(pose_results) 

                keypoints_list.append(data_samples.pred_instances.keypoints.reshape((26,2)).flatten())

                if saving:
                    # save prediction results
                    pred_instances_list.append(
                        dict(
                            frame_id=frame_idx,
                            camera_id=idx,
                            instances=split_instances(data_samples.get('pred_instances', None))))

                # Show the results
                visualizer.add_datasample(
                    'result',
                    frame_gray,
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
            elif key == ord('s'):
                saving = not saving
                if saving:
                    print("Started saving keypoints.")
                else:
                    print("Stopped saving keypoints.")
    finally:
        pipeline.stop()
        out_ir1.release()
        out_ir2.release()
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