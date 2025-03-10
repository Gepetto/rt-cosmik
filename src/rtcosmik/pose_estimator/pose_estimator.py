from mmdeploy_runtime import PoseTracker
from .config import VISUALIZATION_CFG
import cv2
import numpy as np 
from typing import List, Tuple
import time
from multiprocessing import Process
from rtcosmik.utils.linear_algebra_utils import concat_frames
import torch
import queue

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
        results = self.tracker(self.state, frame, detect=-1)
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
        # If 'q' is pressed, exit visualization
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

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
        results = self.tracker.batch(self.states, frames, detects=[-1]*self._batch_size)
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
                 result_queue, 
                 barrier,
                 stop_event, 
                 frame_shape, 
                 ):
        super().__init__()
        self.DET_MODEL_PATH = DET_MODEL_PATH
        self.POSE_MODEL_PATH = POSE_MODEL_PATH
        self.cam_id = cam_id
        self.camera_buffer = camera_buffer
        self.camera_lock = camera_lock
        self.camera_frame_counter = camera_frame_counter
        self.frame_shape = frame_shape  # (height, width, channels)
        self.result_queue = result_queue
        self.barrier = barrier
        self.stop_event = stop_event

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # State tracking
        self.last_processed = 0  # Last processed frame number

    def run(self):
        self.barrier.wait()
        self.tracker = PoseTrackerEstimator(self.DET_MODEL_PATH, self.POSE_MODEL_PATH, device=self.device)

        while not self.stop_event.is_set():
            if self.camera_frame_counter.value > self.last_processed:
                with self.camera_lock:
                    # Read and copy shared data atomically
                    arr = np.frombuffer(self.camera_buffer, dtype=np.uint8)
                    frame = arr.reshape(self.frame_shape).copy()
                    current_counter = self.camera_frame_counter.value

                # Perform heavy processing without holding the lock
                results, infer_time = self.tracker.estimate(frame)
                print("Cam_id : ", self.cam_id, "inference time : ", infer_time)
                keypoints, bboxes, _ = results

                # Directly queue the results
                self.result_queue.put({
                    'frame_counter': current_counter,
                    'results': {
                        'keypoints': keypoints,  # full precision and structure preserved
                        'bboxes': bboxes
                    },
                    'inference_time': infer_time
                })

                # Update local counter; assuming last_processed is local and not shared
                self.last_processed = current_counter
            else:
                time.sleep(0.001) # prevent CPU hogging

class BatchPoseTrackerProcess(Process):
    def __init__(self, 
                 DET_MODEL_PATH, 
                 POSE_MODEL_PATH, 
                 camera_buffers, 
                 camera_timestamps,
                 camera_locks, 
                 stop_event, 
                 frame_shape, 
                 num_cameras,
                 ):
        super().__init__()
        self.DET_MODEL_PATH = DET_MODEL_PATH
        self.POSE_MODEL_PATH = POSE_MODEL_PATH
        self.camera_buffers = camera_buffers
        self.camera_timestamps = camera_timestamps
        self.camera_locks = camera_locks
        self.frame_shape = frame_shape  # (height, width, channels)
        self.stop_event = stop_event
        self.num_cameras = num_cameras

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run(self):
        self.tracker = BatchPoseTrackerEstimator(self.num_cameras,self.DET_MODEL_PATH, self.POSE_MODEL_PATH, device=self.device)
        # Warmup
        _ = self.tracker.estimate([np.zeros(self.frame_shape, dtype=np.uint8) for _ in range(self.num_cameras)])

        try:
            while not self.stop_event.is_set():
                    frames = []
                    for lock, buffer, cam_ts in zip(self.camera_locks, self.camera_buffers, self.camera_timestamps):
                        with lock:
                            # Read and copy shared data atomically
                            arr = np.frombuffer(buffer, dtype=np.uint8)
                            frame = arr.reshape(self.frame_shape).copy()
                            # Get current timestamp
                            timestamp = bytes(cam_ts[:]).decode().strip('\x00')
                            # Add timestamp overlay (white text with black outline for readability)
                            cv2.putText(frame, timestamp, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 0, 0), 4, lineType=cv2.LINE_AA)
                            cv2.putText(frame, timestamp, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 2, lineType=cv2.LINE_AA)
                            frames.append(frame)

                    # Perform heavy processing without holding the lock
                    results = self.tracker.estimate(frames)
                    if not self.tracker.visualize(frames, results):
                        break
        finally:        
            cv2.destroyAllWindows()
            print(f"Process for BatchPoseTracker terminated.")

                
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

        self.VISUALISATION_CFG = VISUALIZATION_CFG
        self.skeleton = 'body26'
        self.sigmas = VISUALIZATION_CFG[self.skeleton]['sigmas']
        self.thr = 0.1
        
    def run(self):
        combined_window = "Multi-Camera View"
        last_good_result = {}

        try:
            while not self.stop_event.is_set():
                frames = []
                # results_buffer = {}

                # For each camera, update the results buffer from the result queue
                for i in range(self.num_cameras):
                    try:
                        while True:
                            result = self.result_queues[i].get_nowait()
                            # If this is the first result or a newer one, update the last_good_result
                            if (i not in last_good_result) or (result['frame_counter'] >= last_good_result[i]['frame_counter']):
                                last_good_result[i] = result
                    except queue.Empty:
                        pass

                # Collect frames from all cameras
                for i in range(self.num_cameras):
                    with self.camera_locks[i]:
                        arr = np.frombuffer(self.camera_buffers[i], dtype=np.uint8)
                        frame = arr.reshape(self.frame_shape).copy()
                        current_frame_counter = self.camera_frame_counters[i].value
                        # Get current timestamp
                        timestamp = bytes(self.timestamp_buffers[i][:]).decode().strip('\x00')

                        # Check if a valid result exists for this camera
                    if i in last_good_result:
                        # Ensure the stored result is not from a future frame
                        if last_good_result[i]['frame_counter'] <= current_frame_counter:
                            res = last_good_result[i]
                            pose_results = res['results']
                            keypoints = pose_results['keypoints']
                            bboxes = pose_results['bboxes']
                            # Visualize pose estimation overlay on the frame
                            img = self._visualize(frame, keypoints, bboxes)
                        else:
                            img = frame
                    else:
                        img = frame

                    # Add timestamp overlay (white text with black outline for readability)
                    cv2.putText(img, timestamp, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 0), 4, lineType=cv2.LINE_AA)
                    cv2.putText(img, timestamp, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2, lineType=cv2.LINE_AA)

                    frames.append(img)

                # Combine frames from all cameras for a multi-camera display
                if frames:
                    if self.num_cameras > 1 and len(frames)==self.num_cameras:
                        combined_frame = concat_frames(frames)
                    else:
                        combined_frame = frames[0]
                    
                    cv2.imshow(combined_window, combined_frame)        

                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:        
            cv2.destroyAllWindows()
            print(f"Process for DisplayPoseTracker terminated.")

    def _visualize(self, 
                  frame,
                  keypoints,
                  bboxes,
                  resize=1280):
        
        skeleton = self.VISUALISATION_CFG[self.skeleton]['skeleton']
        palette = self.VISUALISATION_CFG[self.skeleton]['palette']
        link_color = self.VISUALISATION_CFG[self.skeleton]['link_color']
        point_color = self.VISUALISATION_CFG[self.skeleton]['point_color']

        scale = resize / max(frame.shape[0], frame.shape[1])
        scores = keypoints[..., 2]
        keypoints = (keypoints[..., :2] * scale).astype(int)
        bboxes *= scale
        img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        for kpts, score, bbox in zip(keypoints, scores, bboxes):
            show = [1] * len(kpts)

            for (u, v), color in zip(skeleton, link_color):
                if score[u] > self.thr and score[v] > self.thr:
                    cv2.line(img, kpts[u], tuple(kpts[v]), palette[color], 1,
                            cv2.LINE_AA)
                else:
                    show[u] = show[v] = 0

            for kpt, show, color in zip(kpts, show, point_color):
                if show:
                    cv2.circle(img, kpt, 1, palette[color], 2, cv2.LINE_AA)

        return img


