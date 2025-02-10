import cv2
from functools import partial
from rtmlib import PoseTracker, Wholebody, Custom, draw_skeleton, BodyWithFeet, RTMPose, Body

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

cap = cv2.VideoCapture(0)

# pose_tracker = PoseTracker(BodyWithFeet,
#                         det_frequency=10,  # detect every 10 frames
#                         to_openpose=openpose_skeleton,
#                         backend=backend, device=device)

# pose_model = RTMPose(
#     onnx_model='/root/workspace/mmdeploy/rtmpose-trt/rtmpose-m/end2end.onnx',
#     model_input_size=(192, 256),
#     backend=backend,
#     device=device
# )

pose_model = Body(
                to_openpose=openpose_skeleton,
                mode='lightweight',  # balanced, performance, lightweight
                backend=backend,
                device=device)
                        
# # Initialized slightly differently for Custom solution:
# custom = partial(Custom,
#                 to_openpose=openpose_skeleton,
#                 pose_class='RTMO',
#                 pose='https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip', # noqa
#                 pose_input_size=(640,640),
#                 backend=backend,
#                 device=device)
# # or
# custom = partial(
#             Custom,
#             to_openpose=openpose_skeleton,
#             det_class='YOLOX',
#             det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip', # noqa
#             det_input_size=(640, 640),
#             pose_class='RTMPose',
#             pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip', # noqa
#             pose_input_size=(192, 256),
#             backend=backend,
#             device=device)
# # then
# pose_tracker = PoseTracker(custom,
#                         det_frequency=10,
#                         to_openpose=openpose_skeleton,
#                         backend=backend, device=device)


frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = pose_model(frame)

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.43)

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)