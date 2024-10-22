# To run the code : python3 test_with_lstm_rs.py cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m
# or python3 -m apps.test_with_lstm_rs cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m

import argparse
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from mmdeploy_runtime import PoseTracker
import time 
import csv
import pinocchio as pin
from collections import deque
from utils.lstm_v2 import augmentTRC, loadModel
from datetime import datetime


from utils.calib_utils import load_cam_params, load_cam_to_cam_params, load_cam_pose
from utils.triangulation_utils import triangulate_points

keypoints_buffer = deque(maxlen=2)
warmed_models= loadModel( augmenterDir="/home/kahina/mmdeploy-1.0.0-linux-x86_64-cxx11abi-cuda11.3/augmentation_model", augmenterModelName="LSTM",augmenter_model='v0.3')

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

VISUALIZATION_CFG = dict(
    coco=dict(
        skeleton= [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (17, 18), (1, 2), (5, 18),(6, 18), # Head, shoulders, and neck connections

    (5, 7), (7, 9),                                                              # Right arm connections

    (6, 8), (8, 10),                                                             # Left arm connections

    (18, 19), (19, 11), (19, 12),                                                      # Shoulders to hips connections

    (11, 13), (13, 15), (15, 20), (15, 22), (15, 24),                            # Left leg and foot connections

    (12, 14), (14, 16), (16, 21), (16, 23), (16, 25),                            # Right leg and foot connections
                                                      # Hip connection
],
        palette=[[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]],
        link_color=[
        1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
        2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
        2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
    ],
        point_color=[
        0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4,
        5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5,
        5, 6, 6, 6, 6, 1, 1, 1, 1
    ],
        sigmas=[0.026] * 26),

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
              out_vid,
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
        # out_vid.write(img)
        return cv2.waitKey(1) != 'q'
    return True

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

    csv_file_path = os.path.join(parent_directory,'/output/keypoints_3d_positions.csv')
    csv2_file_path = os.path.join(parent_directory,'/output/q.csv')

    K1, D1 = load_cam_params(os.path.join(parent_directory,"/cams_calibration/cam_params/c1_params_color_test_test.yml"))
    K2, D2 = load_cam_params(os.path.join(parent_directory,"/cams_calibration/cam_params/c2_params_color_test_test.yml"))
    R,T = load_cam_to_cam_params(os.path.join(parent_directory,"/cams_calibration/cam_params/c1_to_c2_params_color_test_test.yml"))

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
                               augmenterDir="/home/kahina/mmdeploy-1.0.0-linux-x86_64-cxx11abi-cuda11.3/augmentation_model"
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

