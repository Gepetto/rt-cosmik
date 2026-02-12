from rtcosmik.pose_estimator.pose_estimator import PoseTrackerEstimator
import cv2

DET_MODEL_PATH = '/root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano'
POSE_MODEL_PATH = '/root/workspace/mmdeploy/rtmpose-trt/rtmpose-m'
VIDEO_PATH = '/root/workspace/ros_ws/src/rt-cosmik/tests/videos/bedlam.mp4'

pose_estimator = PoseTrackerEstimator(det_model=DET_MODEL_PATH, pose_model=POSE_MODEL_PATH, device='cpu')

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
