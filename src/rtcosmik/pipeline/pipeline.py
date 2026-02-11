from collections import deque
import torch
import numpy as np
import pinocchio as pin
import example_robot_data as robex
from datetime import datetime
from multiprocessing import Process, Array, Lock, Value, Event, Queue
from typing import List
import time

from src.rtcosmik.nlf.nlf import NLFEstimator
from src.rtcosmik.triangulation.triangulation import triangulate_points
from src.rtcosmik.filtering.iir import IIR
from src.rtcosmik.human_model.model_utils import scale_human_model, mks_registration, recalibrate_marker_frames_in_joint_space
from src.rtcosmik.ik.ik import RT_IK, RT_SWIKA
from src.rtcosmik.camera.cam_utils import load_camera_parameters,load_world_transformation

import logging

LOGGER = logging.getLogger(__name__)

class PipelineProcess(Process):
    def __init__(self, 
                 settings,
                 frame_counters,
                 camera_buffers, 
                 camera_locks, 
                 timestamp_buffers,
                 results_queues: List[Queue],
                 stop_event: Event,
                 mtxs,
                 dists,
                 projections,
                 world_R1_cam,
                 world_T1_cam,
                 frame_shape: tuple = (720, 1280, 3),
                 num_cameras: int = 2,
                 logger=None,
                 ):
        super().__init__()
        # MP
        self.camera_buffers = camera_buffers
        self.camera_locks = camera_locks
        self.timestamp_buffers = timestamp_buffers
        self.frame_shape = frame_shape  # (height, width, channels)
        self.num_cameras = num_cameras
        self.stop_event = stop_event
        self.results_queues = results_queues

        self.last_frame_counters = [0] * self.num_cameras
        self.frame_counters = frame_counters

        # Settings related parameters
        self.settings=settings

        # Others, cam parameters
        self.first_sample = True

        self.p3d_buffer=deque(maxlen=self.settings.N)

        self.mtxs=mtxs
        self.dists=dists
        self.projections=projections
        self.world_R1_cam=world_R1_cam
        self.world_T1_cam=world_T1_cam
        
        self.logger = logger or LOGGER

    def run(self):

        est = NLFEstimator(
            yolo_path=self.settings.yolo_path,
            nlf_path=self.settings.nlf_path,
            cano_path=self.settings.cano_path,
            image_size=(self.frame_shape[1], self.frame_shape[0]),
            cam_Ks=self.mtxs,
            indices=self.settings.nlf_indices,
            conf=self.settings.yolo_conf,
            imgsz=self.settings.yolo_imgsz,
            device=self.settings.device,
        )

        num_channel = 3*len(self.settings.marker_names)
        iir_filter = IIR(
            num_channel=num_channel,
            sampling_frequency=self.settings.fs
        )
        iir_filter.add_filter(order=self.settings.order, cutoff=self.settings.cutoff_freq, filter_type=self.settings.filter_type)

        try:
            while not self.stop_event.is_set():
                    frames = []
                    new_counters = []
                    for i, (lock, buffer, cam_ts, frame_counter) in enumerate(zip(self.camera_locks, self.camera_buffers, self.timestamp_buffers, self.frame_counters)):
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

                    self.last_frame_counters = new_counters.copy()

                    nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)

                    if nlf_out is None or len(nlf_out) < len(paths):
                        continue

                    keypoints_list = [None] * self.num_cameras
                    valid_cam_ids = []

                    for ii in range(self.num_cameras):
                        out_i = nlf_out[ii]
                        if out_i is None or not isinstance(out_i, dict):
                            continue
                        poses2d = out_i.get("poses2d", None)
                        if poses2d is None or len(poses2d) == 0 or poses2d[0] is None:
                            continue

                        keypoints_list[ii] = poses2d[0].detach().float().cpu().numpy()
                        valid_cam_ids.append(ii)

                    if len(valid_cam_ids) < 2:
                        continue
                    
                    p3d = triangulate_points(
                        keypoints_list=keypoints_list,
                        mtxs=self.mtxs,
                        dists=self.dists,
                        projections=self.projections,
                    )

                    p3d_np = torch.from_numpy(p3d).to(dtype=torch.float32)

                    p3d_in_world=np.array([np.dot(self.world_R1_cam,point) + self.world_T1_cam for point in p3d_np])

                    if self.first_sample:
                        for k in range(self.settings.N):
                            self.p3d_buffer.append(p3d_in_world)  # add the 1st frame 30 times
                    else:
                        self.p3d_buffer.append(p3d_in_world) # add the keypoints to the buffer normally

                    if len(self.p3d_buffer) == self.settings.N:
                        p3d_buffer_array = np.array(self.p3d_buffer)

                        # Filter keypoints in world to remove noisy artefacts 
                        filtered_p3d_buffer = iir_filter.filter(np.reshape(p3d_buffer_array,(self.settings.N, 3*len(self.settings.marker_names))))
                        filtered_p3d_buffer = np.reshape(filtered_p3d_buffer,(self.settings.N, len(self.settings.marker_names), 3))

                        augmented_markers=filtered_p3d_buffer[-1]

                        if self.first_sample:
                            mks_dict = dict(zip(self.settings.marker_names, augmented_markers))

                            human = robex.human.HumanLoader(height=self.settings.human_height, weight=self.settings.human_weight, gender=self.settings.human_gender).robot
                            human_model = human.model

                            #scale the model to data
                            human_model = scale_human_model(human_model, mks_dict, gender=self.settings.human_gender, subject_height=self.settings.human_height)
                            human_model= mks_registration(human_model, mks_dict, gender=self.settings.human_gender, subject_height=self.settings.human_height)

                            # IK
                            if self.settings.ik_type == 'sbs':
                                omega = {}
                                for key in self.settings.keys_to_track_list:
                                    omega[key] = 1
                                q = pin.neutral(human_model)
                                ik_class = RT_IK(human_model, mks_dict, q, self.settings.keys_to_track_list, self.settings.dt, omega)

                                q = ik_class.solve_ik_sample_casadi()
                                ik_class._q0 = q

                                # Recalibrate briefly the markers translation in joint frames
                                human_model=recalibrate_marker_frames_in_joint_space(human_model,q,mks_dict,self.settings.marker_names)

                                ik_class = RT_IK(human_model, mks_dict, q, self.settings.keys_to_track_list, self.settings.dt, omega)
                                self.logger.info("[INFO] Model calibration finished, ready to process...")

                            elif self.settings.ik_type == 'mhe':
                                ik_class = RT_SWIKA(human_model, self.settings.keys_to_track_list, self.settings.N, code = self.settings.ik_code)

                                x_array = np.zeros((human_model.nq+human_model.nv, self.settings.N))
                                x_array[6,:]=1
                                u_array = np.zeros((human_model.nv, self.settings.N))
                                deque_lstm_dict = deque(maxlen=self.settings.N)
                                for k in range(self.settings.N):
                                    deque_lstm_dict.append(mks_dict)

                                array_data = np.array([np.hstack([d[marker] for marker in self.settings.keys_to_track_list]) for d in deque_lstm_dict]).T

                                x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], self.settings.cost_weights, self.settings.dt)

                                q = pin.neutral(human_model)
                                q[:] = np.array(x_array[:human_model.nq,-1]).flatten()
                                viz_human.display(q)

                                # Recalibrate briefly the markers translation in joint frames
                                human_model=recalibrate_marker_frames_in_joint_space(human_model,q,mks_dict,self.settings.marker_names)

                                ik_class = RT_SWIKA(human_model, self.settings.keys_to_track_list, self.settings.N, code = self.settings.ik_code)
                                self.logger.info("[INFO] Model calibration finished, ready to process...")
                            else : 
                                raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")

                            self.first_sample = False

                        else: # Init phase finished
                            mks_dict = dict(zip(self.settings.marker_names, augmented_markers))

                            # IK directly 
                            if self.settings.ik_type == 'sbs':
                                ik_class._dict_m = mks_dict
                                q = ik_class.solve_ik_sample_quadprog() 
                                ik_class._q0 = q
                            elif settings.ik_type == 'mhe':
                                deque_lstm_dict.append(mks_dict)
                                array_data = np.array([np.hstack([d[marker] for marker in self.settings.keys_to_track_list]) for d in deque_lstm_dict]).T
                                
                                x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], self.settings.cost_weights, self.settings.dt)

                                q = pin.neutral(human_model)
                                q[:] = np.array(x_array[:human_model.nq,-1]).flatten()
                            else : 
                                raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")
        finally:        
            self.logger.info("[INFO] Pipeline Process terminated")
