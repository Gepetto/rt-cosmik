import sys
import os
import cv2
import time
import csv
import numpy as np
import ctypes
from pynput import keyboard
from datetime import datetime
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
from multiprocessing import Process, Queue, Event, Array, Manager
from queue import Empty
from mmdeploy_runtime import PoseTracker
import torch
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")) # src dir

from src.rtcosmik.camera.cam_utils import list_cameras
from src.rtcosmik.config_loader import settings
from src.rtcosmik.utils.linear_algebra_utils import concat_frames
from src.rtcosmik.pose_estimator.config import VISUALIZATION_CFG
from src.rtcosmik.triangulation.triangulation import triangulate_points
from src.rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
from src.rtcosmik.filtering.iir import IIR
from src.rtcosmik.augmenter.marker_augmenter import augmentTRC, loadModel
from src.rtcosmik.ik.ik import RT_IK, RT_SWIKA
from src.rtcosmik.human_model.pin_model import build_model_no_visuals



def camera_process(idx_cam, stop_event, current_frame, timestamp):

    cap = cv2.VideoCapture(idx_cam, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: Could not open camera {idx_cam}")
        exit()
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

    while not stop_event.is_set():

        ret, frame = cap.read()
        if not ret:
            keyboard.Controller().press("q")
            raise Exception(f"Camera {idx_cam} has crashed, quitting the recording.")

        np.frombuffer(current_frame.get_obj(), dtype=np.uint8)[:] = frame.flatten()
        timestamp.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    cap.release()

    print(f"Camera {idx_cam} process terminated.")


# def display_rtcosmik_process(current_angles_list, stop_event):

#     # Displays rt-cosmik
    
#     print("Display process terminated.")


def hpe_process(current_frames, stop_event, timestamp, mks_dict, idx_cams):

    device = "cuda"
    frame_shape = (settings.height, settings.width, 3)
    batch_size = len(idx_cams)
    CAM_CONFIG_PATH = settings.cam_calib_path
    AUGMENTER_PATH = settings.augmenter_path
    mtxs, dists, projections, _, _ = load_camera_parameters(CAM_CONFIG_PATH)
    world_R1_cam, world_T1_cam = load_world_transformation(CAM_CONFIG_PATH)
    keypoints_stack_max_len = 30
    keypoints_stack = []
    keys_to_add = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']
    num_channel = 3*len(settings.keypoints_names)
    iir_filter = IIR(
            num_channel=num_channel,
            sampling_frequency=settings.fs
        )
    warmed_augmenter_model = loadModel(augmenterDir=AUGMENTER_PATH, augmenterModelName="LSTM",augmenter_model='v0.3')

    tracker = PoseTracker(settings.det_model_path, settings.pose_model_path, device)
    sigmas = VISUALIZATION_CFG["body26"]['sigmas']
    states =  [tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas) for _ in range(batch_size)]
    # Warmup
    _ = tracker.batch(states, [np.zeros(frame_shape, dtype=np.uint8) for _ in range(batch_size)], detects=[-1]*batch_size)

    first_sample = True

    while not stop_event.is_set():

        start_whole_HPE_process = time.time()

        keypoints_list = []

        processed_frames = []
        for image in current_frames:

            frame = np.frombuffer(image.get_obj(), dtype=np.uint8).reshape((settings.height, settings.width, 3))

            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Frame must be HWC with 3 channels, got {frame.shape}")
            processed_frames.append(frame)
        
        start_HPE = time.time()

        results = tracker.batch(states, processed_frames, detects=[-1]*batch_size)

        end_HPE = time.time()
        elapsed_HPE = end_HPE - start_HPE
        print("HPE :", f"{elapsed_HPE:.5f} sec")

        for res in results: 
            keypoints, _, _ = res
            keypoints = (keypoints[..., :2] ).astype(float)

            if keypoints.size == 0 or keypoints.flatten().shape != (52,):
                continue
            else :
                keypoints_list.append(keypoints.reshape((26,2)).flatten())

        if len(keypoints_list) != batch_size:
            continue
        else:
            # self.valid_event.set()
            start_triangul = time.time()
            keypoints_in_cam = triangulate_points(keypoints_list, mtxs, dists, projections)
            end_triangul = time.time()
            elapsed_triangul = end_triangul - start_triangul
            print("Triangul :", f"{elapsed_triangul:.5f} sec")
            keypoints_in_world = np.array([np.dot(world_R1_cam, point) + world_T1_cam for point in keypoints_in_cam])

            if first_sample:
                for _ in range(keypoints_stack_max_len):
                    keypoints_stack.append(keypoints_in_world)  #add the 1st frame 30 times
                first_sample = False
            else:
                keypoints_stack.append(keypoints_in_world) #add the keypoints to the buffer normally 
            
            if len(keypoints_stack) == keypoints_stack_max_len:
                keypoints_stack_array = np.array(keypoints_stack)

                # Filter keypoints in world to remove noisy artefacts 
                filtered_keypoints_stack = iir_filter.filter(np.reshape(keypoints_stack_array,(keypoints_stack_max_len, num_channel)))
                filtered_keypoints_stack = np.reshape(filtered_keypoints_stack, (keypoints_stack_max_len, int(num_channel/3), 3))

                start_LSTM = time.time()

                augmented_markers = augmentTRC(filtered_keypoints_stack, subject_mass=settings.human_mass, 
                                               subject_height=settings.human_height, models = warmed_augmenter_model,
                                               augmenterDir=AUGMENTER_PATH, augmenter_model='v0.3')
                
                end_LSTM = time.time()
                elapsed_LSTM = end_LSTM - start_LSTM
                print("LSTM :", f"{elapsed_LSTM:.5f} sec")
                
                if len(augmented_markers) % 3 != 0:
                    raise ValueError("The length of the list must be divisible by 3.")

                augmented_markers = np.array(augmented_markers).reshape(-1, 3)
                
                kp_dict = dict(zip(settings.keypoints_names, filtered_keypoints_stack[-1]))
                final_dict = dict(zip(settings.marker_names, augmented_markers))
                final_dict.update({key: kp_dict[key] for key in keys_to_add})
                mks_dict.put((timestamp.timestamp, final_dict))
                keypoints_stack.pop(0)
        
        end_whole_HPE_process = time.time()
        elapsed_whole_HPE_process = end_whole_HPE_process - start_whole_HPE_process
        print("Whole HPE Process :", f"{elapsed_whole_HPE_process:.5f} sec")

    print("HPE process terminated.")


def ik_process(mks_dict, angles, stop_event):

    first_sample = True

    while not stop_event.is_set():

        (timestamp, current_mks_dict) = mks_dict.get()

        start_whole_IK_process = time.time()

        if first_sample:
            human_model = build_model_no_visuals(current_mks_dict)
                        
            if settings.ik_type == 'sbs':
                q = pin.neutral(human_model)
                ik_class = RT_IK(human_model, current_mks_dict, q, settings.keys_to_track_list, settings.dt)

                q = ik_class.solve_ik_sample_casadi()
                ik_class._q0 = q

            elif settings.ik_type == 'mhe':
                ik_class = RT_SWIKA(human_model, settings.keys_to_track_list, settings.N, code=settings.ik_code)

                x_array = np.zeros((human_model.nq + human_model.nv, settings.N))
                x_array[6,:]=1
                u_array = np.zeros((human_model.nv, settings.N))
                list_lstm_dict = []
                for _ in range(settings.N):
                    list_lstm_dict.append(current_mks_dict)

                array_data = np.array([np.hstack([d[marker] for marker in settings.keys_to_track_list]) for d in list_lstm_dict]).T

                x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], settings.cost_weights, settings.dt)

            else : 
                raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")

            first_sample = False

        else:
            if settings.ik_type == 'sbs':
                ### IK calculations
                ik_class._dict_m = current_mks_dict
                q = ik_class.solve_ik_sample_quadprog() 
                angles.put((timestamp, q))
                ik_class._q0 = q
                
            elif settings.ik_type == 'mhe':
                list_lstm_dict.pop(0)
                list_lstm_dict.append(current_mks_dict)
                array_data = np.array([np.hstack([d[marker] for marker in settings.keys_to_track_list]) for d in list_lstm_dict]).T

                start_IK = time.time()
                
                x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], settings.cost_weights, settings.dt)

                end_IK = time.time()
                elapsed_IK = end_IK - start_IK
                print("IK :", f"{elapsed_IK:.5f} sec")

                q = pin.neutral(human_model)
                q[:] = np.array(x_array[:human_model.nq,-1]).flatten()
                angles.put((timestamp, q))
            else : 
                raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")
        
        end_whole_IK_process = time.time()
        elapsed_whole_IK_process = end_whole_IK_process - start_whole_IK_process
        print("Whole IK Process :", f"{elapsed_whole_IK_process:.5f} sec")

    print("IK process terminated.")


def saver_process(angles, stop_event):

    with open(os.path.join(settings.SAVE_DIR, f"angles_fsp_max.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamps"] + settings.joint_angles_names)

        while True:

            # start = time.time()

            try:
                oldest_angle = angles.get_nowait()
                writer.writerow([oldest_angle[0]] + list(oldest_angle[1]))

            except Empty:
                if stop_event.is_set():
                    break
                else: 
                    pass
            
            # end = time.time()
            # elapsed = end - start
            # print("Saver :", f"{elapsed:.2f} sec")
        
        print(f"Saver process terminated.")


def on_press(key):
    try:
        if key.char == 's' and not start_event.is_set():
            start_event.set()

        elif key.char == 'q':
            stop_event.set()
            return False  # Arrête le listener
    except AttributeError:
        # touches spéciales (ex: ctrl, alt...) qu'on ignore ici
        pass



if __name__ == "__main__":

    os.makedirs(settings.SAVE_DIR, exist_ok=True)

    cameras = list_cameras()

    print(cameras)

    global start_event
    global stop_event
    start_event = Event()
    stop_event = Event()

    frame_size = settings.width*settings.height*3

    num_dofs = len(settings.joint_angles_names)

    manager = Manager()
    timestamp = manager.Namespace()  # Shared timestamps for cameras
    mks_dict = Queue(maxsize=30)  # Queue to hold the markers data
    angles = Queue(maxsize=30)
    current_angles_list = Array(ctypes.c_float, num_dofs)

    for idx_cam in cameras.keys():
        globals()[f"current_frame_{idx_cam}"] = Array(ctypes.c_ubyte, frame_size)

    cameras_processes = [
        Process(
            target=camera_process, 
            args=(idx_cam, stop_event, globals()[f"current_frame_{idx_cam}"], timestamp),
            name=f"Process camera {idx_cam}",
        ) 
        for idx_cam in cameras.keys()
    ]

    # display_process = Process(
    #     target=display_rtcosmik_process, 
    #     args=(current_angles_list, stop_event),
    #     name="Display process"
    # )

    HPE_process = Process(
            target=hpe_process, 
            args=([globals()[f"current_frame_{idx_cam}"] for idx_cam in cameras.keys()], stop_event, timestamp, mks_dict, list(cameras.keys())),
            name="HPE process"
        )
    
    IK_process = Process(
            target=ik_process, 
            args=(mks_dict, angles, stop_event),
            name="IK process"
        )
    
    data_saver_process = Process(
            target=saver_process, 
            args=(angles, stop_event),
            name="Data Saver process"
        )

    all_processes = cameras_processes + [HPE_process] + [IK_process] + [data_saver_process] #+ [display_process]

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    unrestarter = True
    print("Waiting for 's' to be pressed to start trial...")
    while True:

        if start_event.is_set() and unrestarter:
                
            print("\nStarting all processes...")
            for process in all_processes:
                process.start()

            unrestarter = False

            print("Press 'q' to stop recording.")

        if stop_event.is_set():
                
            while data_saver_process.is_alive():
                print("Waiting for saver process to finish writing ...")
                time.sleep(2)
                
            break
    
    print("All saving processes terminated.")

