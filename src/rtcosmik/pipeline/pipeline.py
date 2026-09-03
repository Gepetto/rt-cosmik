from collections import deque
import torch
import numpy as np
import pinocchio as pin
from datetime import datetime
from multiprocessing import Process, Array, Lock, Value, Event, Queue
from typing import List
import time

from rtcosmik.nlf.nlf import NLFEstimator, extract_views
from rtcosmik.triangulation.triangulation import reconstruct_3d
from rtcosmik.filtering.iir import IIR
from rtcosmik.pipeline.solver import HumanSolver
from rtcosmik.camera.cam_utils import load_camera_parameters,load_world_transformation
from rtcosmik.model_weights import resolve_detector_engine

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

        self.solver = HumanSolver(self.settings, logger=self.logger)

        est = NLFEstimator(
            yolo_path=resolve_detector_engine(self.settings.yolo_path, self.num_cameras),
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

                    views = extract_views(nlf_out, self.num_cameras)
                    p3d = reconstruct_3d(views, self.projections)
                    if len(p3d) == 0:
                        continue

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

                        mks_dict = dict(zip(self.settings.marker_names, augmented_markers))

                        if self.first_sample:
                            q = self.solver.calibrate(mks_dict)
                            self.first_sample = False
                        else:
                            # Only publish once the model is calibrated, so a
                            # consumer never sees poses from the init frame.
                            self.results_queues[0].put((new_counters, mks_dict))
                            q = self.solver.step(mks_dict)
                            self.results_queues[1].put((new_counters, q))

        finally:        
            self.logger.info("[INFO] Pipeline Process terminated")
