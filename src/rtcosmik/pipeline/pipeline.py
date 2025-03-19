from rtcosmik.triangulation.triangulation import triangulate_points
from rtcosmik.augmenter.marker_augmenter import augmentTRC, loadModel
from rtcosmik.pose_estimator.pose_estimator import BatchPoseTrackerEstimator
from rtcosmik.filtering.iir import IIR
from rtcosmik.ik.ik import RT_IK, RT_SWIKA
from rtcosmik.camera.cam_utils import load_camera_parameters,load_world_transformation
from rtcosmik.human_model.pin_model import build_model_no_visuals

from collections import deque
import torch
import numpy as np
import pinocchio as pin
from datetime import datetime
from multiprocessing import Process, Array, Lock, Value, Event, Queue
from typing import List
import time

class PipelineProcess(Process):
    def __init__(self, 
                 settings,
                 camera_buffers: List[Array],
                 camera_timestamps: List[Array], # Character array for timestamp
                 camera_locks: List[Lock],
                 camera_frame_counters: List[Value],
                 results_queues: List[Queue],
                 stop_event: Event,
                 frame_shape: tuple = (720, 1280, 3),
                 num_cameras: int = 2
                 ):
        super().__init__()
        # Settings related parameters
        self.DET_MODEL_PATH = settings.det_model_path
        self.POSE_MODEL_PATH = settings.pose_model_path
        self.AUGMENTER_PATH = settings.augmenter_path
        self.CAM_CONFIG_PATH = settings.cam_calib_path
        self.fs = settings.fs
        self.subject_mass=settings.human_mass
        self.subject_height=settings.human_height
        self.keypoints_names=settings.keypoints_names
        self.marker_names=settings.marker_names
        self.dt = settings.dt
        self.order = settings.order
        self.cutoff_freq = settings.cutoff_freq
        self.filter_type = settings.filter_type
        self.keys_to_track_list = settings.keys_to_track_list
        self.ik_type = settings.ik_type
        self.ik_code = settings.ik_code
        self.cost_weights = settings.cost_weights
        self.N = settings.N

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # MP
        self.camera_buffers = camera_buffers
        self.camera_timestamps = camera_timestamps
        self.camera_locks = camera_locks
        self.camera_frame_counters = camera_frame_counters
        self.stop_event = stop_event
        self.results_queues = results_queues

        self.frame_shape = frame_shape
        self.num_cameras = num_cameras

        self.first_sample = True
        self.last_frame_counters = [0] * self.num_cameras

    def run(self):
        self.buffer_max_len = 30
        self.keypoints_buffer = deque(maxlen=self.buffer_max_len)
        self.warmed_augmenter_model = loadModel(augmenterDir=self.AUGMENTER_PATH, augmenterModelName="LSTM",augmenter_model='v0.3')

        #load camera param and config
        self.mtxs, self.dists, self.projections, self.rotations, self.translations = load_camera_parameters(self.CAM_CONFIG_PATH)
        self.world_R1_cam, self.world_T1_cam = load_world_transformation(self.CAM_CONFIG_PATH)

        ### Set up real time filter 
        # Constant
        num_channel = 3*len(self.keypoints_names)

        # Creating IIR instance
        self.iir_filter = IIR(
            num_channel=num_channel,
            sampling_frequency=self.fs
        )

        self.iir_filter.add_filter(order=self.order, cutoff=self.cutoff_freq, filter_type=self.filter_type)

        
        self.tracker = BatchPoseTrackerEstimator(self.num_cameras,self.DET_MODEL_PATH, self.POSE_MODEL_PATH, device=self.device)
        # Warmup
        _ = self.tracker.estimate([np.zeros(self.frame_shape, dtype=np.uint8) for _ in range(self.num_cameras)])

        try:
            while not self.stop_event.is_set():
                frames = []
                keypoints_list = []
                new_counters = []
                for i, (lock, buffer, cam_ts, frame_counter) in enumerate(zip(self.camera_locks, self.camera_buffers, self.camera_timestamps, self.camera_frame_counters)):
                    with lock:
                        #  Only accept data if this camera has produced a new frame
                        if frame_counter.value > self.last_frame_counters[i]:
                            # Read and copy shared data atomically
                            arr = np.frombuffer(buffer, dtype=np.uint8)
                            frame = arr.reshape(self.frame_shape).copy()
                            # Get current timestamp
                            timestamp = bytes(cam_ts[:]).decode().strip('\x00')

                            if timestamp == '': # empty data
                                continue
                            else:
                                frames.append(frame)
                            new_counters.append(frame_counter.value)

                if len(frames)!=self.num_cameras:
                    continue

                # Update the last processed frame counters so the same frame is not processed twice
                self.last_frame_counters = new_counters.copy()

                results = self.tracker.estimate(frames)

                for res in results: 
                    keypoints, bboxes, _ = res
                    keypoints = (keypoints[..., :2] ).astype(float)

                    if keypoints.size == 0 or keypoints.flatten().shape != (52,):
                        continue
                    else :
                        keypoints_list.append(keypoints.reshape((26,2)).flatten())

                if len(keypoints_list)!=self.num_cameras:
                    continue
                else:
                    keypoints_in_cam = triangulate_points(keypoints_list, self.mtxs, self.dists, self.projections)
                    keypoints_in_world = np.array([np.dot(self.world_R1_cam,point) + self.world_T1_cam for point in keypoints_in_cam])

                    if self.first_sample:
                        for k in range(self.buffer_max_len):
                            self.keypoints_buffer.append(keypoints_in_world)  #add the 1st frame 30 times
                    else:
                        self.keypoints_buffer.append(keypoints_in_world) #add the keypoints to the buffer normally 
                    
                    if len(self.keypoints_buffer) == self.buffer_max_len:
                        keypoints_buffer_array = np.array(self.keypoints_buffer)

                        # Filter keypoints in world to remove noisy artefacts 
                        filtered_keypoints_buffer = self.iir_filter.filter(np.reshape(keypoints_buffer_array,(self.buffer_max_len, 3*len(self.keypoints_names))))
                        filtered_keypoints_buffer = np.reshape(filtered_keypoints_buffer,(self.buffer_max_len, len(self.keypoints_names), 3))

                        augmented_markers = augmentTRC(filtered_keypoints_buffer, subject_mass=self.subject_mass, subject_height=self.subject_height, models = self.warmed_augmenter_model,
                                    augmenterDir=self.AUGMENTER_PATH, augmenter_model='v0.3')
                        
                        if len(augmented_markers) % 3 != 0:
                            raise ValueError("The length of the list must be divisible by 3.")

                        augmented_markers = np.array(augmented_markers).reshape(-1, 3)

                        if self.first_sample:
                            kp_dict = dict(zip(self.keypoints_names,filtered_keypoints_buffer[-1]))
                            mks_dict = dict(zip(self.marker_names, augmented_markers))

                            # Adds head keypoints in lstm output for head tracking
                            keys_to_add = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']
                            mks_dict.update({key: kp_dict[key] for key in keys_to_add})

                            self.human_model = build_model_no_visuals(mks_dict)
                            
                            if self.ik_type == 'sbs':
                                q = pin.neutral(self.human_model)
                                ik_class = RT_IK(self.human_model, mks_dict, q, self.keys_to_track_list, self.dt)

                                q = ik_class.solve_ik_sample_casadi()
                                ik_class._q0 = q

                            elif self.ik_type == 'mhe':
                                ik_class = RT_SWIKA(self.human_model, self.keys_to_track_list, self.N, code = self.ik_code)

                                x_array = np.zeros((self.human_model.nq+self.human_model.nv, self.N))
                                x_array[6,:]=1
                                u_array = np.zeros((self.human_model.nv, self.N))
                                deque_lstm_dict = deque(maxlen=self.N)
                                for k in range(self.N):
                                    deque_lstm_dict.append(mks_dict)

                                array_data = np.array([np.hstack([d[marker] for marker in self.keys_to_track_list]) for d in deque_lstm_dict]).T

                                x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], self.cost_weights, self.dt)

                            else : 
                                raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")

                            self.first_sample = False
                        
                        else:
                            kp_dict = dict(zip(self.keypoints_names,filtered_keypoints_buffer[-1]))
                            self.results_queues[0].put((new_counters, kp_dict))

                            mks_dict = dict(zip(self.marker_names, augmented_markers))
                            self.results_queues[1].put((new_counters, mks_dict))

                            # Adds head keypoints in lstm output for head tracking
                            keys_to_add = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']
                            mks_dict.update({key: kp_dict[key] for key in keys_to_add})
                            
                            if self.ik_type == 'sbs':
                                ### IK calculations
                                ik_class._dict_m = mks_dict
                                q = ik_class.solve_ik_sample_quadprog() 
                                self.results_queues[2].put((new_counters, q))
                                ik_class._q0 = q
                                
                            elif self.ik_type == 'mhe':
                                deque_lstm_dict.append(mks_dict)
                                array_data = np.array([np.hstack([d[marker] for marker in self.keys_to_track_list]) for d in deque_lstm_dict]).T
                                
                                x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], self.cost_weights, self.dt)

                                q = pin.neutral(self.human_model)
                                q[:] = np.array(x_array[:self.human_model.nq,-1]).flatten()
                                self.results_queues[2].put((new_counters, q))
                            else : 
                                raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")
                            
        finally: 
            print("Pipeline process terminated")       


class BenchmarkedPipelineProcess(Process):
    def __init__(self, 
                 settings,
                 camera_buffers: List[Array],
                 camera_timestamps: List[Array], # Character array for timestamp
                 camera_locks: List[Lock],
                 camera_frame_counters: List[Value],
                 results_queues: List[Queue],
                 stop_event: Event,
                 frame_shape: tuple = (720, 1280, 3),
                 num_cameras: int = 2
                 ):
        super().__init__()
        # Settings related parameters
        self.DET_MODEL_PATH = settings.det_model_path
        self.POSE_MODEL_PATH = settings.pose_model_path
        self.AUGMENTER_PATH = settings.augmenter_path
        self.CAM_CONFIG_PATH = settings.cam_calib_path
        self.fs = settings.fs
        self.subject_mass=settings.human_mass
        self.subject_height=settings.human_height
        self.keypoints_names=settings.keypoints_names
        self.marker_names=settings.marker_names
        self.dt = settings.dt
        self.order = settings.order
        self.cutoff_freq = settings.cutoff_freq
        self.filter_type = settings.filter_type
        self.keys_to_track_list = settings.keys_to_track_list
        self.ik_type = settings.ik_type
        self.ik_code = settings.ik_code
        self.cost_weights = settings.cost_weights
        self.N = settings.N

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # MP
        self.camera_buffers = camera_buffers
        self.camera_timestamps = camera_timestamps
        self.camera_locks = camera_locks
        self.camera_frame_counters = camera_frame_counters
        self.stop_event = stop_event
        self.results_queues = results_queues

        self.frame_shape = frame_shape
        self.num_cameras = num_cameras

        self.first_sample = True
        self.last_frame_counters = [0] * self.num_cameras

    def run(self):
        self.buffer_max_len = 30
        self.keypoints_buffer = deque(maxlen=self.buffer_max_len)
        self.warmed_augmenter_model = loadModel(augmenterDir=self.AUGMENTER_PATH, augmenterModelName="LSTM",augmenter_model='v0.3')

        #load camera param and config
        self.mtxs, self.dists, self.projections, self.rotations, self.translations = load_camera_parameters(self.CAM_CONFIG_PATH)
        self.world_R1_cam, self.world_T1_cam = load_world_transformation(self.CAM_CONFIG_PATH)

        ### Set up real time filter 
        # Constant
        num_channel = 3*len(self.keypoints_names)

        # Creating IIR instance
        self.iir_filter = IIR(
            num_channel=num_channel,
            sampling_frequency=self.fs
        )

        self.iir_filter.add_filter(order=self.order, cutoff=self.cutoff_freq, filter_type=self.filter_type)

        
        self.tracker = BatchPoseTrackerEstimator(self.num_cameras,self.DET_MODEL_PATH, self.POSE_MODEL_PATH, device=self.device)
        # Warmup
        _ = self.tracker.estimate([np.zeros(self.frame_shape, dtype=np.uint8) for _ in range(self.num_cameras)])

        try:
            while not self.stop_event.is_set():
                time_init_pipe = time.perf_counter()
                frames = []
                timestamps = []
                keypoints_list = []
                new_counters = []
                for i, (lock, buffer, cam_ts, frame_counter) in enumerate(zip(self.camera_locks, self.camera_buffers, self.camera_timestamps, self.camera_frame_counters)):
                    with lock:
                        #  Only accept data if this camera has produced a new frame
                        if frame_counter.value > self.last_frame_counters[i]:
                            # Read and copy shared data atomically
                            arr = np.frombuffer(buffer, dtype=np.uint8)
                            frame = arr.reshape(self.frame_shape).copy()
                            # Get current timestamp
                            timestamp = bytes(cam_ts[:]).decode().strip('\x00')

                            if timestamp == '': # empty data
                                continue
                            else:
                                frames.append(frame)
                                timestamps.append(timestamp)
                            new_counters.append(frame_counter.value)

                if len(timestamps)!=self.num_cameras and len(frames)!=self.num_cameras:
                    continue

                print(timestamps, new_counters)

                # Update the last processed frame counters so the same frame is not processed twice
                self.last_frame_counters = new_counters.copy()

                # Convert to Unix timestamps (float)
                unix_timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").timestamp() for ts in timestamps]
                output_timestamp = max(unix_timestamps)
                # Convert back to a formatted string if needed
                output_time_str = datetime.fromtimestamp(output_timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")

                time_init_pt = time.perf_counter()
                results = self.tracker.estimate(frames)
                time_end_pt = time.perf_counter()
                print("Time for pose tracker batched = ", time_end_pt-time_init_pt)

                for res in results: 
                    keypoints, bboxes, _ = res
                    keypoints = (keypoints[..., :2] ).astype(float)

                    if keypoints.size == 0 or keypoints.flatten().shape != (52,):
                        continue
                    else :
                        keypoints_list.append(keypoints.reshape((26,2)).flatten())

                if len(keypoints_list)!=self.num_cameras:
                    continue
                else:
                    time_init_lstm = time.perf_counter()
                    keypoints_in_cam = triangulate_points(keypoints_list, self.mtxs, self.dists, self.projections)
                    keypoints_in_world = np.array([np.dot(self.world_R1_cam,point) + self.world_T1_cam for point in keypoints_in_cam])

                    if self.first_sample:
                        for k in range(self.buffer_max_len):
                            self.keypoints_buffer.append(keypoints_in_world)  #add the 1st frame 30 times
                    else:
                        self.keypoints_buffer.append(keypoints_in_world) #add the keypoints to the buffer normally 
                    
                    if len(self.keypoints_buffer) == self.buffer_max_len:
                        keypoints_buffer_array = np.array(self.keypoints_buffer)

                        # Filter keypoints in world to remove noisy artefacts 
                        filtered_keypoints_buffer = self.iir_filter.filter(np.reshape(keypoints_buffer_array,(self.buffer_max_len, 3*len(self.keypoints_names))))
                        filtered_keypoints_buffer = np.reshape(filtered_keypoints_buffer,(self.buffer_max_len, len(self.keypoints_names), 3))

                        augmented_markers = augmentTRC(filtered_keypoints_buffer, subject_mass=self.subject_mass, subject_height=self.subject_height, models = self.warmed_augmenter_model,
                                    augmenterDir=self.AUGMENTER_PATH, augmenter_model='v0.3')
                        
                        if len(augmented_markers) % 3 != 0:
                            raise ValueError("The length of the list must be divisible by 3.")

                        augmented_markers = np.array(augmented_markers).reshape(-1, 3)
                        time_end_lstm = time.perf_counter()
                        print("Time to perform triangul + filtering + lstm =", time_end_lstm-time_init_lstm)

                        if self.first_sample:
                            kp_dict = dict(zip(self.keypoints_names,filtered_keypoints_buffer[-1]))
                            mks_dict = dict(zip(self.marker_names, augmented_markers))
                            
                            self.human_model = build_model_no_visuals(mks_dict)
                            
                            if self.ik_type == 'sbs':
                                q = pin.neutral(self.human_model)
                                ik_class = RT_IK(self.human_model, mks_dict, q, self.keys_to_track_list, self.dt)

                                q = ik_class.solve_ik_sample_casadi()
                                ik_class._q0 = q

                            elif self.ik_type == 'mhe':
                                ik_class = RT_SWIKA(self.human_model, self.keys_to_track_list, self.N, code = self.ik_code)

                                x_array = np.zeros((self.human_model.nq+self.human_model.nv, self.N))
                                x_array[6,:]=1
                                u_array = np.zeros((self.human_model.nv, self.N))
                                deque_lstm_dict = deque(maxlen=self.N)
                                for k in range(self.N):
                                    deque_lstm_dict.append(mks_dict)

                                array_data = np.array([np.hstack([d[marker] for marker in self.keys_to_track_list]) for d in deque_lstm_dict]).T

                                x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], self.cost_weights, self.dt)

                            else : 
                                raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")

                            self.first_sample = False
                        
                        else:
                            kp_dict = dict(zip(self.keypoints_names,filtered_keypoints_buffer[-1]))
                            self.results_queues[0].put((output_time_str, kp_dict))

                            mks_dict = dict(zip(self.marker_names, augmented_markers))
                            self.results_queues[1].put((output_time_str, mks_dict))
                            
                            if self.ik_type == 'sbs':
                                time_init_ik_qp = time.perf_counter()
                                ### IK calculations
                                ik_class._dict_m = mks_dict
                                q = ik_class.solve_ik_sample_quadprog() 
                                self.results_queues[2].put((output_time_str, q))
                                ik_class._q0 = q
                                time_end_ik_qp = time.perf_counter()
                                print("Time to perform qp ik = ", time_end_ik_qp-time_init_ik_qp)
                                
                            elif self.ik_type == 'mhe':
                                time_init_swika = time.perf_counter()
                                deque_lstm_dict.append(mks_dict)
                                array_data = np.array([np.hstack([d[marker] for marker in self.keys_to_track_list]) for d in deque_lstm_dict]).T
                                x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], self.cost_weights, self.dt)

                                q = pin.neutral(self.human_model)
                                q[:] = np.array(x_array[:self.human_model.nq,-1])
                                self.results_queues[2].put((output_time_str, q))
                                time_end_swika = time.perf_counter()
                                print("Time to perform SWIKA = ", time_end_swika-time_init_swika)
                            else : 
                                raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")
                
                time_end_pipe = time.perf_counter()
                print("Total pipe time = ", time_end_pipe-time_init_pipe)            
        finally: 
            print("Pipeline process terminated")       