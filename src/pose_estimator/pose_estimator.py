from mmdeploy_runtime import PoseTracker
from .config import VISUALIZATION_CFG
import cv2
import numpy as np 
from typing import List, Tuple

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
        results = self.tracker(self.state, frame, detect=-1)
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
        results = self.tracker.batch(self.states, frames, detects=[-1]*self._batch_size)
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


class ConcatPoseTrackerEstimator:
    def __init__(self, det_model, pose_model, frame_width=640, device='cuda', thr=0.5, skeleton='body26'):
        self._det_model = det_model
        self._pose_model = pose_model
        self._device = device
        self._thr = thr
        self._skeleton = skeleton
        self.frame_width = frame_width
        self.tracker = PoseTracker(det_model, pose_model, device)
        self.VISUALIZATION_CFG = VISUALIZATION_CFG
        self.sigmas = VISUALIZATION_CFG[self._skeleton]['sigmas']
        self.state = self.tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=self.sigmas)

    def estimate(self, stacked_frame) -> Tuple[np.ndarray, np.ndarray]:
        results = self.tracker(self.state, stacked_frame, detect=-1)
        keypoints, bboxes, _ = results

        if keypoints is not None and len(keypoints) >= 2:
            mean_x_values = [kp[:, 0].mean() for kp in keypoints]
            left_idx = np.argmin(mean_x_values)
            right_idx = np.argmax(mean_x_values)
            
            left_skeleton = keypoints[[left_idx]]
            left_bboxes = bboxes[left_idx]
            
            right_skeleton = keypoints[[right_idx]]
            right_bboxes = bboxes[right_idx]
            
            right_skeleton[..., 0] -= self.frame_width
            left_result = left_skeleton, left_bboxes, _ 
            right_result = right_skeleton, right_bboxes, _

            
            return left_result,right_result
        
        return None, None
        

    def visualize(self,
              frame,
              results,
              idx,
              frame_id,
              thr=0.1,
              resize=1280,
              skeleton_type='body26'):
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
        # if output_dir:
        #     cv2.imwrite(f'{output_dir}/{str(frame_id).zfill(6)}.jpg', img)
        # else:
        cv2.imshow('pose_tracker'+str(idx), img)
        return cv2.waitKey(1) != 'q'
        return True
