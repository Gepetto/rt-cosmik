from src.rtcosmik.pose_estimator.pose_estimator import PoseTrackerEstimator
import cv2
import sys

num_cam = sys.argv[1]
subject = sys.argv[2]
trial = sys.argv[3]

DET_MODEL_PATH = '/root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano'
POSE_MODEL_PATH = '/root/workspace/mmdeploy/rtmpose-trt/rtmpose-m'
VIDEO_PATH = f'/root/workspace/ros_ws/src/rt-cosmik/output/{subject}/{trial}/camera_{num_cam}.mp4'

pose_estimator = PoseTrackerEstimator(det_model=DET_MODEL_PATH, pose_model=POSE_MODEL_PATH, device='cuda')

cap = cv2.VideoCapture(VIDEO_PATH)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results, t_inf = pose_estimator.estimate(frame)
    if not pose_estimator.visualize(frame, results,0):
        break
    
    input()

cap.release()
cv2.destroyAllWindows()
