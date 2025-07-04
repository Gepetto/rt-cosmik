import sys
import os
import cv2
import time
import csv
import numpy as np
import ctypes
from pynput import keyboard
from datetime import datetime
import subprocess
from multiprocessing import Process, Barrier, Queue, Event, Array
from threading import BrokenBarrierError
from queue import Empty
from mmdeploy_runtime import PoseTracker
import torch
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")) # src dir

from src.rtcosmik.camera.cam_utils import list_cameras
from src.rtcosmik.config_loader import settings
from src.rtcosmik.utils.linear_algebra_utils import concat_frames
from src.rtcosmik.human_model.urdf_model import Robot
from src.rtcosmik.pose_estimator.config import VISUALIZATION_CFG
from src.rtcosmik.triangulation.triangulation import triangulate_points
from src.rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
from src.rtcosmik.filtering.iir import IIR
from src.rtcosmik.augmenter.marker_augmenter import augmentTRC, loadModel



def camera_process(idx_cam, stop_event, current_frame):

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

    cap.release()

    print(f"Camera {idx_cam} process terminated.")


def display_rtcosmik_process(q_list, stop_event):

    # Displays rt-cosmik
    
    print("Display process terminated.")


def hpe_process(current_frames, stop_event, augmented_mks, idx_cams):

    rt_cosmik_path = os.path.dirname(os.path.join(os.path.abspath(__file__), ".."))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frame_shape = (settings.height, settings.width, 3)
    batch_size = len(idx_cams)
    CAM_CONFIG_PATH = settings.cam_calib_path
    AUGMENTER_PATH = settings.augmenter_path
    mtxs, dists, projections, rotations, translations = load_camera_parameters(CAM_CONFIG_PATH)
    world_R1_cam, world_T1_cam = load_world_transformation(CAM_CONFIG_PATH)
    keypoints_stack_max_len = 30
    keypoints_stack = []
    num_channel = 3*len(settings.keypoints_names)
    iir_filter = IIR(
            num_channel=num_channel,
            sampling_frequency=settings.fs
        )
    warmed_augmenter_model = loadModel(augmenterDir=AUGMENTER_PATH, augmenterModelName="LSTM",augmenter_model='v0.3')

    tracker = PoseTracker(settings.det_model_path, settings.pose_model_path, device=device)
    sigmas = VISUALIZATION_CFG["body26"]['sigmas']
    states =  [tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas) for _ in range(batch_size)]
    # Warmup
    _ = tracker([np.zeros(frame_shape, dtype=np.uint8) for _ in range(batch_size)])

    first_sample = True

    while not stop_event.is_set():

        keypoints_list = []

        results = tracker.batch(states, current_frames, detects=[-1]*batch_size)

        for res in results: 
            keypoints, bboxes, _ = res
            keypoints = (keypoints[..., :2] ).astype(float)

            if keypoints.size == 0 or keypoints.flatten().shape != (52,):
                continue
            else :
                keypoints_list.append(keypoints.reshape((26,2)).flatten())

        if len(keypoints_list)!=batch_size:
            continue
        else:
            # self.valid_event.set()
            keypoints_in_cam = triangulate_points(keypoints_list, mtxs, dists, projections)
            keypoints_in_world = np.array([np.dot(world_R1_cam, point) + world_T1_cam for point in keypoints_in_cam])

            if first_sample:
                for k in range(keypoints_stack_max_len):
                    keypoints_stack.append(keypoints_in_world)  #add the 1st frame 30 times
            else:
                keypoints_stack.append(keypoints_in_world) #add the keypoints to the buffer normally 
            
            if len(keypoints_stack) == keypoints_stack_max_len:
                keypoints_stack_array = np.array(keypoints_stack)

                # Filter keypoints in world to remove noisy artefacts 
                filtered_keypoints_stack = iir_filter.filter(np.reshape(keypoints_stack_array,(keypoints_stack_max_len, num_channel)))
                filtered_keypoints_stack = np.reshape(filtered_keypoints_stack, (keypoints_stack_max_len, num_channel/3, 3))

                augmented_markers = augmentTRC(filtered_keypoints_stack, subject_mass=settings.subject_mass, 
                                               subject_height=settings.subject_height, models = warmed_augmenter_model,
                                               augmenterDir=AUGMENTER_PATH, augmenter_model='v0.3')
                
                if len(augmented_markers) % 3 != 0:
                    raise ValueError("The length of the list must be divisible by 3.")

                augmented_markers = np.array(augmented_markers).reshape(-1, 3)

                if first_sample:
                    kp_dict = dict(zip(settings.keypoints_names, filtered_keypoints_stack[-1]))
                    mks_dict = dict(zip(settings.marker_names, augmented_markers))

                    # Adds head keypoints in lstm output for head tracking
                    keys_to_add = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']
                    mks_dict.update({key: kp_dict[key] for key in keys_to_add})

                    # self.human_model = build_model_no_visuals(mks_dict)
                    
                    #load urdf
                    human = Robot('/root/workspace/ros_ws/src/rt-cosmik/urdf/human.urdf', rt_cosmik_path, isFext=True) 
                    human_model = human.model
                    human_data = human.data
                    human_collision_model = human.collision_model
                    human_visual_model = human.visual_model

                    #scale the model to data
                    human_model = scale_human_model(human_model, mks_dict, with_hand=True, gender='male', subject_height=settings.human_height)
                    print(human_model.nq)

                    human_model = mks_registration(human_model, mks_dict, with_hand=False)

                    human_data = pin.Data(human_model)

                    
                    if ik_type == 'sbs':
                        q = pin.neutral(human_model)
                        ik_class = RT_IK(human_model, mks_dict, q, keys_to_track_list, dt)

                        q = ik_class.solve_ik_sample_casadi()
                        ik_class._q0 = q

                    elif ik_type == 'mhe':
                        ik_class = RT_SWIKA(human_model, keys_to_track_list, N, code=ik_code)

                        x_array = np.zeros((human_model.nq + human_model.nv, N))
                        x_array[6,:]=1
                        u_array = np.zeros((human_model.nv, N))
                        deque_lstm_dict = deque(maxlen=N)
                        for k in range(self.N):
                            deque_lstm_dict.append(mks_dict)

                        array_data = np.array([np.hstack([d[marker] for marker in self.keys_to_track_list]) for d in deque_lstm_dict]).T

                        x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], self.cost_weights, self.dt)

                    else : 
                        raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")

                    self.first_sample = False
                
                else:
                    # print(new_counters)
                    kp_dict = dict(zip(self.keypoints_names,filtered_keypoints_buffer[-1]))
                    self.results_queues[0].put((timestamps, kp_dict))

                    mks_dict = dict(zip(self.marker_names, augmented_markers))
                    self.results_queues[1].put((timestamps, mks_dict))

                    # Adds head keypoints in lstm output for head tracking
                    keys_to_add = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']
                    mks_dict.update({key: kp_dict[key] for key in keys_to_add})
                    
                    if self.ik_type == 'sbs':
                        ### IK calculations
                        ik_class._dict_m = mks_dict
                        q = ik_class.solve_ik_sample_quadprog() 
                        self.results_queues[2].put((timestamps, q))
                        ik_class._q0 = q
                        
                    elif self.ik_type == 'mhe':
                        deque_lstm_dict.append(mks_dict)
                        array_data = np.array([np.hstack([d[marker] for marker in self.keys_to_track_list]) for d in deque_lstm_dict]).T
                        
                        x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], self.cost_weights, self.dt)

                        q = pin.neutral(self.human_model)
                        q[:] = np.array(x_array[:self.human_model.nq,-1]).flatten()
                        self.results_queues[2].put((timestamps, q))
                    else : 
                        raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")

    print("HPE process terminated.")


def ik_process(augmented_mks, stop_event):

    # Simulate IK processing
    while not stop_event.is_set():
        try:
            # Here you would process the augmented_mks
            time.sleep(0.1)  # Simulate processing time
        except Empty:
            continue

    print("IK process terminated.")



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
    num_lstm_mks = len(settings.marker_names)

    current_q_list = Array(ctypes.c_float, num_dofs)
    augmented_mks = Array(ctypes.c_float, num_lstm_mks * 3)

    for idx_cam in cameras.keys():
        globals()[f"current_frame_{idx_cam}"] = Array(ctypes.c_ubyte, frame_size)

    cameras_processes = [
        Process(
            target=camera_process, 
            args=(idx_cam, stop_event, globals()[f"current_frame_{idx_cam}"]),
            name=f"Process camera {idx_cam}"
        ) 
        for idx_cam in cameras.keys()
    ]

    display_process = Process(
        target=display_rtcosmik_process, 
        args=(current_q_list, stop_event),
        name="Display process"
    )

    HPE_process = Process(
            target=hpe_process, 
            args=([globals()[f"current_frame_{idx_cam}"] for idx_cam in cameras.keys], stop_event, augmented_mks, cameras.keys()),
            name="HPE process"
        )
    
    IK_process = Process(
            target=ik_process, 
            args=(augmented_mks, stop_event),
            name="IK process"
        )

    all_processes = cameras_processes + [display_process] + HPE_process + IK_process

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

            barrier.abort()
                
            while any([frames_saver_process.is_alive() for frames_saver_process in frames_saver_processes]):
                print("Waiting for saver processes to finish writing ...")
                time.sleep(2)
                
            break
    
    print("All saving processes terminated.")

    # # Trouver tous les processus Python
    # ps_output = subprocess.check_output(['ps', 'aux'])
    # pids = []
    # for line in ps_output.decode('utf-8').split('\n'):
    #     if 'python scripts/run_cameras_refacto.py' in line:
    #         # Extraire le PID
    #         pid = int(line.split()[1])
    #         pids.append(pid)

    # # Tuer les processus
    # for pid in pids:
    #     try:
    #         # Envoyer le signal SIGTERM pour terminer le processus
    #         subprocess.call(['kill', str(pid)])
    #     except Exception as e:
    #         print(f"Erreur lors de la tentative de tuer le processus {pid}: {e}")
