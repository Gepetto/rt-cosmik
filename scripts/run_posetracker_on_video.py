import cv2
import sys
import os
import csv
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from src.rtcosmik.pose_estimator.pose_estimator import PoseTrackerEstimator

num_cam = sys.argv[1]
subject = sys.argv[2]
trial = sys.argv[3]
start_frame = int(sys.argv[4])  # new argument
end_frame = int(sys.argv[5])    # new argument

DET_MODEL_PATH = '/root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano'
POSE_MODEL_PATH = '/root/workspace/mmdeploy/rtmpose-trt/rtmpose-m'
VIDEO_PATH = f'/root/workspace/ros_ws/src/rt-cosmik/output/{subject}/{trial}/camera_{num_cam}.avi'
CSV_OUTPUT = f"/root/workspace/ros_ws/src/rt-cosmik/output/{subject}/{trial}/keypoints_cam{num_cam}.csv"

pose_estimator = PoseTrackerEstimator(det_model=DET_MODEL_PATH, pose_model=POSE_MODEL_PATH, device='cuda')

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if end_frame > total_frames:
    end_frame = total_frames

# Set the starting frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

with open(CSV_OUTPUT, mode='w', newline='') as f:
    writer = csv.writer(f)

    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_estimator.estimate(frame)
        pose_estimator.visualize(frame, results, 0)

        keypoints_scores, bboxes, _ = results
        keypoints = (keypoints_scores[..., :2]).astype(float)
        scores = (keypoints_scores[..., 2:]).astype(float)

        if keypoints is not None and scores is not None:
            keypoints_flat = keypoints.flatten().tolist()
            scores_flat = scores.flatten().tolist()
            scores_mean = [np.mean(scores_flat)]
            writer.writerow(scores_mean + keypoints_flat)

        current_frame += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



# while cap.isOpened():

#     ret, frame = cap.read()

#     if not ret:
#         break
    
#     results, t_inf = pose_estimator.estimate(frame)

#     if not pose_estimator.visualize(frame, results,0):
#         break
    
#     input()

# cap.release()
# cv2.destroyAllWindows()
