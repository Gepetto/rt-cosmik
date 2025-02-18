from mmdeploy_runtime import PoseTracker
from .config import VISUALIZATION_CFG
import cv2
import numpy as np 
from typing import List, Tuple
import time

class PoseTrackerEstimator:
    def __init__(self, det_model, pose_model, device='cuda', thr=0.5, skeleton = 'body26'):
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
    def __init__(self, batch_size: int, det_model, pose_model, device='cuda', thr=0.5, skeleton = 'body26'):
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
            print(idx)
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
