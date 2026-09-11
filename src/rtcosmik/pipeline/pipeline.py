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
from rtcosmik.saver.recorder import Recorder
from rtcosmik.viewer.async_display import AsyncDisplay
from rtcosmik.viewer.viewer import Viewer
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
                 stop_event: Event,
                 mtxs,
                 dists,
                 projections,
                 world_R1_cam,
                 world_T1_cam,
                 frame_shape: tuple = (720, 1280, 3),
                 num_cameras: int = 2,
                 saving_flag=None,
                 subject=None,   # (height, weight, gender)
                 calibrated_event=None,
                 report_every: float = 2.0,
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
        self.saving_flag = saving_flag
        self.subject = subject
        self.calibrated_event = calibrated_event
        self.report_every = report_every

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

    def _log_live(self, recent, window_s):
        """One line per reporting window, while the session runs."""
        loop = float(np.median(recent["loop"]))
        skip = float(np.mean(recent["skip"])) if recent["skip"] else 1.0
        self.logger.info(
            "[TIME] %5.1f turns/s | loop %5.1f ms (pose %4.1f, ik %4.1f) | "
            "kept %3.0f%% of camera frames | %d turns in %.1fs",
            1000.0 / max(loop, 1e-9), loop,
            float(np.median(recent["pose"])), float(np.median(recent["ik"])),
            100.0 / max(skip, 1e-9), len(recent["loop"]), window_s)

    def _log_timing(self, stage_ms, skipped):
        """What one pipeline turn cost, and whether it kept up with the cameras."""
        if not stage_ms["loop"]:
            return
        self.logger.info("[TIME] over %d turns (median / p95 / max, ms):",
                         len(stage_ms["loop"]))
        for name in ("wait", "pose", "reconstruct", "ik", "record", "display",
                     "loop"):
            vals = np.asarray(stage_ms[name], dtype=float)
            if vals.size:
                self.logger.info(
                    "[TIME]   %-11s %7.1f / %7.1f / %7.1f", name,
                    np.median(vals), np.percentile(vals, 95), vals.max())
        loop = float(np.median(stage_ms["loop"]))
        self.logger.info("[TIME]   -> %.1f turns/s (cameras run at %d fps)",
                         1000.0 / max(loop, 1e-9), self.settings.fs)
        if skipped:
            steps = np.asarray(skipped, dtype=float)
            kept = 100.0 / max(steps.mean(), 1e-9)
            self.logger.info(
                "[TIME]   camera frames per processed frame: median %.0f, "
                "p95 %.0f, max %.0f -> %.0f%% of frames processed",
                np.median(steps), np.percentile(steps, 95), steps.max(), kept)

    def run(self):

        height, weight, gender = self.subject or (None, None, None)
        self.solver = HumanSolver(self.settings, gender=gender, height=height,
                                  weight=weight, logger=self.logger)

        # Recording and display both live here now, with the data. The display
        # is a thread that drops frames when it falls behind, so a slow viewer
        # can never hold up the estimate; recording writes straight from the
        # calibrated results.
        # Same stage breakdown as the offline script, plus what only matters
        # online: how many camera frames went by between the ones processed.
        stage_ms = {"wait": [], "pose": [], "reconstruct": [], "ik": [],
                    "record": [], "display": [], "loop": []}
        skipped = []
        # Rolling window for the live line: a summary printed only at the end
        # cannot show *when* a stall happened, which is the thing worth seeing
        # while a session runs.
        recent = {"loop": [], "pose": [], "ik": [], "skip": []}
        t_report = time.perf_counter()

        recorder = Recorder(self.settings, self.num_cameras,
                            saving_flag=self.saving_flag, logger=self.logger).start()
        display = AsyncDisplay(logger=self.logger)
        display.__enter__()
        viewer = None

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
                    t_loop0 = time.perf_counter()
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

                    # How far the cameras moved on since the last processed
                    # frame: 1 means keeping up, more means frames were missed.
                    if self.last_frame_counters[0]:
                        skipped.append(new_counters[0] - self.last_frame_counters[0])
                    self.last_frame_counters = new_counters.copy()
                    t_wait = time.perf_counter()

                    nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)
                    t_pose = time.perf_counter()

                    views = extract_views(nlf_out, self.num_cameras)
                    p3d = reconstruct_3d(views, self.projections)
                    t_rec = time.perf_counter()
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
                            # Built here, from the *calibrated* model. The old
                            # viewer process rebuilt its own from settings
                            # defaults and drew a differently sized person.
                            # Releases the replay sources, which have been
                            # holding their first frame while this ran.
                            if self.calibrated_event is not None:
                                self.calibrated_event.set()
                            viewer = Viewer(self.solver.model,
                                            self.solver.collision_model,
                                            self.solver.visual_model,
                                            self.settings.marker_names)
                        else:
                            t_ik0 = time.perf_counter()
                            q = self.solver.step(mks_dict)
                            stage_ms["ik"].append((time.perf_counter()-t_ik0)*1e3)

                            t_rec0 = time.perf_counter()
                            recorder.record(new_counters, mks_dict, q)
                            stage_ms["record"].append((time.perf_counter()-t_rec0)*1e3)

                            t_disp0 = time.perf_counter()
                            if viewer is not None:
                                display.submit(
                                    lambda m=dict(mks_dict), qq=np.array(q, copy=True):
                                    (viewer.display_markers(m), viewer.display_q(qq)))
                            stage_ms["display"].append(
                                (time.perf_counter()-t_disp0)*1e3)
                            stage_ms["wait"].append((t_wait-t_loop0)*1e3)
                            stage_ms["pose"].append((t_pose-t_wait)*1e3)
                            stage_ms["reconstruct"].append((t_rec-t_pose)*1e3)
                            loop_ms = (time.perf_counter()-t_loop0)*1e3
                            stage_ms["loop"].append(loop_ms)

                            recent["loop"].append(loop_ms)
                            recent["pose"].append((t_pose-t_wait)*1e3)
                            recent["ik"].append(stage_ms["ik"][-1])
                            recent["skip"].append(skipped[-1] if skipped else 1)
                            now = time.perf_counter()
                            if now - t_report >= self.report_every:
                                self._log_live(recent, now - t_report)
                                for v in recent.values():
                                    v.clear()
                                t_report = now

        finally:
            self._log_timing(stage_ms, skipped)
            display.close()
            recorder.close()
            self.logger.info("[INFO] Pipeline Process terminated")
