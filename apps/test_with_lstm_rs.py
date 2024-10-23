# To run the code : python3 test_with_lstm_rs.py cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m
# or python3 -m apps.test_with_lstm_rs cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m

import argparse
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from mmdeploy_runtime import PoseTracker
import csv
from collections import deque
from utils.lstm_v2 import augmentTRC, loadModel
from datetime import datetime

from utils.calib_utils import load_cam_params, load_cam_to_cam_params, load_cam_pose
from utils.triangulation_utils import triangulate_points
from utils.viz_utils import visualize, VISUALIZATION_CFG

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
        default='coco',
        choices=['coco', 'coco_wholebody'],
        help='skeleton for keypoints')

    args = parser.parse_args()
    return args

def get_device_serial_numbers():
    """Get a list of serial numbers for connected RealSense devices."""
    ctx = rs.context()
    return [device.get_info(rs.camera_info.serial_number) for device in ctx.query_devices()]

def configure_realsense_pipeline(width,height):
    """Configures and starts the RealSense pipelines."""
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
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        pipeline.start(config)
        pipelines.append(pipeline)

    return pipelines

def main():
    args = parse_args()
    np.set_printoptions(precision=4, suppress=True)

    csv_file_path = os.path.join(parent_directory,'output/keypoints_3d_positions.csv')
    csv2_file_path = os.path.join(parent_directory,'output/q.csv')

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

    # Loading camera pose 
    #R1_global, T1_global = load_cam_pose('/home/kahina/mmdeploy-1.0.0-linux-x86_64-cxx11abi-cuda11.3/cam_calibration/cam_params/camera1_pose_test_test.yml')
    R1_global = np.eye(3)
    T1_global = np.zeros((3,1))
    # R1_global = R1_global@pin.utils.rotate('z', np.pi) # aligns measurements to human model definition
    
    dof_names=['ankle_Z', 'knee_Z', 'lumbar_Z', 'shoulder_Z', 'elbow_Z'] 

    keypoint_names = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Head",
        "Neck", "Hip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"]

    mapping = dict(zip(keypoint_names,[i for i in range(len(keypoint_names))]))
    # Initialize CSV files
    # with open(csv_file_path, mode='w', newline='') as file:
    #     csv_writer = csv.writer(file)
    #     # Write the header row
    #     csv_writer.writerow(['Frame', 'Time','Keypoint', 'X', 'Y', 'Z'])

    width = 1920
    height = 1080
    resize=1280

    # kpt_thr = 0.7

    # Initialize RealSense pipelines
    pipelines = configure_realsense_pipeline(width,height)

    first_sample = True 
    frame_idx = 0

    width_vids = []
    height_vids = []
    for pipeline in pipelines:
        # Get the stream profile to extract the width and height for video writer
        profile = pipeline.get_active_profile()
        rgb_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        width_vids.append(rgb_profile.width())
        height_vids.append(rgb_profile.height())

    # Define the codec and create VideoWriter objects for both RGB streams
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI files
    out_vid1 = cv2.VideoWriter('output/cam1.mp4', fourcc, 30.0, (int(width_vids[0]), int(height_vids[0])), True)
    out_vid2 = cv2.VideoWriter('output/cam2.mp4', fourcc, 30.0, (int(width_vids[1]), int(height_vids[1])), True)
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
        while 1:
            timestamp=datetime.now()
            formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f ")
            
            frames = [pipeline.wait_for_frames().get_color_frame() for pipeline in pipelines]
            if not all(frames):
                continue

            keypoints_list = []
            frame_idx += 1  # Increment frame counter
            print(frame_idx)


            # Process each frame individually
            for idx, color_frame in enumerate(frames):
                frame = np.asanyarray(color_frame.get_data())
                if idx == 0 : 
                    out_vid1.write(frame)
                elif idx == 1 : 
                    out_vid2.write(frame)

                results = tracker(state, frame, detect=-1)
                scale = resize / max(frame.shape[0], frame.shape[1])
                keypoints, bboxes, _ = results
                scores = keypoints[..., 2]
                # keypoints = (keypoints[..., :2] * scale).astype(float)
                keypoints = (keypoints[..., :2] ).astype(float)
                bboxes *= scale

                if keypoints.size == 0 or keypoints.flatten().shape != (52,):
                    print('i')
                    
                else :
                    #print("ok", keypoints)
                    keypoints_list.append(keypoints.reshape((26,2)).flatten())
                    #print("keypoints_list", keypoints_list)

                if not visualize(
                        frame,
                        results,
                        args.output_dir,
                        idx,
                        frame_idx + idx,
                        out_vid[idx],
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
                print("keypoints_in_world_triangulated", keypoints_in_world)
                # Translate so that the right ankle is at 0 everytime
                #keypoints_in_world -= keypoints_in_world[mapping["Right Ankle"],:]
                flattened_keypoints = keypoints_in_world.flatten()
                #print(flattened_keypoints)
                row = flattened_keypoints.tolist()
                with open(csv_file_path, mode='a') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(row)


                    # csv_writer = csv.writer(file)
                    # for jj in range(len(keypoint_names)):
                    #     # Write to CSV
                    #     csv_writer.writerow([frame_idx, formatted_timestamp,keypoint_names[jj], keypoints_in_world[jj][0], keypoints_in_world[jj][1], keypoints_in_world[jj][2]])

                keypoints_buffer.append(keypoints_in_world)
                print("keypoints_buffer", keypoints_buffer)
                if len(keypoints_buffer) == 2:
                    print("lstm")
                    
                    
                    keypoints_buffer_array = np.array(keypoints_buffer)
                    print("keypoints_buffer_array", keypoints_buffer_array)
                   # print("keypoints_buffer_array", keypoints_buffer_array)
                    
                    #call augmentTrc
                    augmentTRC(keypoints_buffer_array, subject_mass=60, subject_height=1.57, models = warmed_models,
                               augmenterDir=augmenter_path
                               ,augmenter_model='v0.3', offset=True)
                
    finally :
        # Stop the RealSense pipelines
        for pipeline in pipelines:
            pipeline.stop()
        out_vid1.release()
        out_vid2.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

