#python3 -m unittests.test_pose_tracker_rtmlib
import cv2
from functools import partial
from rtmlib import PoseTracker, Wholebody, Custom, draw_skeleton, BodyWithFeet, RTMDet

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

cap = cv2.VideoCapture(0)

custom = partial(
            Custom,
            to_openpose=openpose_skeleton,
            det_class='RTMDet',
            det='/root/workspace/mmdeploy/rtmpose-ort/rtmdet-nano/end2end.onnx', # noqa
            det_input_size=(320, 320),
            pose_class='RTMPose',
            pose='/root/workspace/mmdeploy/rtmpose-ort/rtmpose-m/end2end.onnx', # noqa
            pose_input_size=(192, 256),
            backend=backend,
            device=device)
# then
pose_tracker = PoseTracker(custom,
                        det_frequency=10,
                        tracking = False,
                        to_openpose=openpose_skeleton,
                        backend=backend, device=device)

# pose_tracker = PoseTracker(BodyWithFeet,
#                         det_frequency=10,  # detect every 10 frames
#                         to_openpose=openpose_skeleton,
#                         backend=backend, device=device)

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = pose_tracker(frame)

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.43)

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)