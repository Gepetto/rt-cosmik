from mmdeploy_runtime import PoseTracker
from .config import VISUALIZATION_CFG
import cv2
import numpy as np 
from typing import List, Tuple
import time
from src.utils.linear_algebra_utils import reproject, concat_frames, reproject_four_frames, reproject
from multiprocessing import Process, Value
import torch

class PoseTrackerEstimator:
    def __init__(self, det_model, pose_model, device='cuda', thr=0.1, skeleton = 'body26'):
        self._det_model = det_model
        self._pose_model = pose_model
        self._device = device
        self._thr = thr
        self._skeleton = skeleton
        self.tracker = PoseTracker(det_model, pose_model, device)
        self.VISUALISATION_CFG = VISUALIZATION_CFG
        self.sigmas = VISUALIZATION_CFG[self._skeleton]['sigmas']
        self.state =  self.tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=self.sigmas)

    def estimate(self, frame):
        t0 = time.time()
        results = self.tracker(self.state, frame, detect=-1)
        print("time of inference :", time.time()-t0)
        # keypoints, bboxes, _ = results
        # keypoints = (keypoints[..., :2] ).astype(float)
        return results
    
    def visualize(self, 
                  frame,
                  results,
                  idx,
                  resize=1280):
        
        skeleton = self.VISUALISATION_CFG[self._skeleton]['skeleton']
        palette = self.VISUALISATION_CFG[self._skeleton]['palette']
        link_color = self.VISUALISATION_CFG[self._skeleton]['link_color']
        point_color = self.VISUALISATION_CFG[self._skeleton]['point_color']

        scale = resize / max(frame.shape[0], frame.shape[1])
        keypoints, bboxes, _ = results
        scores = keypoints[..., 2]
        keypoints = (keypoints[..., :2] * scale).astype(int)
        bboxes *= scale
        img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        for kpts, score, bbox in zip(keypoints, scores, bboxes):
            show = [1] * len(kpts)

            for (u, v), color in zip(skeleton, link_color):
                if score[u] > self._thr and score[v] > self._thr:
                    cv2.line(img, kpts[u], tuple(kpts[v]), palette[color], 1,
                            cv2.LINE_AA)
                else:
                    show[u] = show[v] = 0

            for kpt, show, color in zip(kpts, show, point_color):
                if show:
                    cv2.circle(img, kpt, 1, palette[color], 2, cv2.LINE_AA)
           
        cv2.imshow('pose_tracker'+str(idx), img)
        return cv2.waitKey(1) != 'q'
        return True
    
class BatchPoseTrackerEstimator:
    def __init__(self, batch_size: int, det_model, pose_model, device='cuda', thr=0.1, skeleton = 'body26'):
        self._batch_size = batch_size
        self._det_model = det_model
        self._pose_model = pose_model
        self._device = device
        self._thr = thr
        self._skeleton = skeleton
        self.tracker = PoseTracker(det_model, pose_model, device)
        self.VISUALISATION_CFG = VISUALIZATION_CFG
        self.sigmas = VISUALIZATION_CFG[self._skeleton]['sigmas']
        self.states =  [self.tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=self.sigmas) for _ in range(batch_size)]

    def estimate(self, frames: List[np.ndarray])-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t0 = time.time()
        results = self.tracker.batch(self.states, frames, detects=[-1]*self._batch_size)
        print("time of inference :", time.time()-t0)
        # keypoints, bboxes, _ = results
        # keypoints = (keypoints[..., :2] ).astype(float)
        return results
    
    def visualize(self, 
                  frames: List[np.ndarray],
                  results : Tuple[np.ndarray, np.ndarray, np.ndarray],
                  resize=1280):
        
        skeleton = self.VISUALISATION_CFG[self._skeleton]['skeleton']
        palette = self.VISUALISATION_CFG[self._skeleton]['palette']
        link_color = self.VISUALISATION_CFG[self._skeleton]['link_color']
        point_color = self.VISUALISATION_CFG[self._skeleton]['point_color']

        for idx, (frame, result) in enumerate(zip(frames, results)):
            skeleton = self.VISUALISATION_CFG[self._skeleton]['skeleton']
            palette = self.VISUALISATION_CFG[self._skeleton]['palette']
            link_color = self.VISUALISATION_CFG[self._skeleton]['link_color']
            point_color = self.VISUALISATION_CFG[self._skeleton]['point_color']

            scale = resize / max(frame.shape[0], frame.shape[1])
            keypoints, bboxes, _ = result
            scores = keypoints[..., 2]
            keypoints = (keypoints[..., :2] * scale).astype(int)
            bboxes *= scale
            img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            for kpts, score, bbox in zip(keypoints, scores, bboxes):
                show = [1] * len(kpts)
                for (u, v), color in zip(skeleton, link_color):
                    if score[u] > self._thr and score[v] > self._thr:
                        cv2.line(img, tuple(kpts[u]), tuple(kpts[v]), palette[color], 1, cv2.LINE_AA)
                    else:
                        show[u] = show[v] = 0
                for kpt, show_flag, color in zip(kpts, show, point_color):
                    if show_flag:
                        cv2.circle(img, tuple(kpt), 1, palette[color], 2, cv2.LINE_AA)

            cv2.imshow(f'pose_tracker_{idx}', img)

        # If 'q' is pressed, exit visualization
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

        return True

class PoseTrackerProcess(Process):
    def __init__(self, 
                 DET_MODEL_PATH, 
                 POSE_MODEL_PATH, 
                 cam_id, 
                 camera_buffer, 
                 camera_lock, 
                 camera_frame_counter, 
                 timestamp_buffer,
                 result_queue, 
                 stop_event, 
                 frame_shape, 
                 ):
        super().__init__()
        self.cam_id = cam_id
        self.camera_buffer = camera_buffer
        self.camera_lock = camera_lock
        self.camera_frame_counter = camera_frame_counter
        self.timestamp_buffer = timestamp_buffer
        self.frame_shape = frame_shape  # (height, width, channels)
        self.result_queue = result_queue
        self.stop_event = stop_event

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tracker = PoseTrackerEstimator(DET_MODEL_PATH, POSE_MODEL_PATH, device=self.device)

        # State tracking
        self.last_processed = 0  # Last processed frame number

    def run(self):
        try:
            while not self.stop_event.is_set():
                with self.camera_lock:
                    arr = np.frombuffer(self.camera_buffer, dtype=np.uint8)
                    frame = arr.reshape(self.frame_shape).copy()
                    timestamp = self.timestamp_buffer[:26].decode('utf-8').strip('\0')

                    if self.camera_frame_counter.value > self.last_processed:
                        results = self.tracker.estimate(frame)
                        self.result_queue.put((timestamp, results))
                        self.last_processed = self.camera_frame_counter.value
        finally:
            pass
                
class DisplayPoseTracker(Process):
    def __init__(self, 
                 camera_buffers, 
                 camera_locks, 
                 camera_frame_counters,
                 result_queues,
                 timestamp_buffers,
                 stop_event, 
                 frame_shape, 
                 num_cameras):
        super().__init__()
        self.camera_buffers = camera_buffers
        self.camera_locks = camera_locks
        self.camera_frame_counters = camera_frame_counters
        self.result_queues = result_queues
        self.timestamp_buffers = timestamp_buffers
        self.frame_shape = frame_shape  # (height, width, channels)
        self.num_cameras = num_cameras
        self.stop_event = stop_event
        
    def run(self):
        window_names = [f'Camera {i}' for i in range(self.num_cameras)]
        
        # Optimization 1: Create a single window for all cameras
        combined_window = "Multi-Camera View"

        try:
            while not self.stop_event.is_set():
                frames = []
                
                # Collect frames from all cameras
                for i in range(self.num_cameras):
                    with self.camera_locks[i]:
                        arr = np.frombuffer(self.camera_buffers[i], dtype=np.uint8)
                        frame = arr.reshape(self.frame_shape).copy()
                        
                        # Optimization 2: Add timestamp overlay
                        ########################################
                        # Get current timestamp
                        timestamp = self.timestamp_buffers[i][:26].decode('utf-8').strip('\0')
                        
                        # Add text overlay (white text with black background)
                        cv2.putText(frame, timestamp, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                (0,0,0), 4, lineType=cv2.LINE_AA)
                        cv2.putText(frame, timestamp, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                (255,255,255), 2, lineType=cv2.LINE_AA)
                        ########################################
                        

                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:        
            cv2.destroyAllWindows()


