#python3 -m unittests.test_rtmlib_rgbcam
import cv2
from functools import partial
from rtmlib import PoseTracker, Wholebody, RTMO, Custom, draw_skeleton, BodyWithFeet, RTMPose, Body
from utils.calib_utils import list_cameras_with_v4l2
from utils.settings import Settings
import time
settings = Settings()
device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

# cap = cv2.VideoCapture(0)
camera_dict = list_cameras_with_v4l2()
captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in camera_dict.keys()]

for idx, cap in enumerate(captures):
    if not cap.isOpened():
        continue

    # Apply settings
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

pose_model = RTMO(
    onnx_model='/root/workspace/ros_ws/src/rt-cosmik/models/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211/end2end.onnx', 
    model_input_size=(640, 640),
    backend=backend,
    device=device
)

# custom = partial(
#             Custom,
#             to_openpose=openpose_skeleton,
#             det_class='RTMDet',
#             det='/root/workspace/mmdeploy/rtmpose-ort/rtmdet-nano/end2end.onnx', # noqa
#             det_input_size=(320, 320),
#             pose_class='RTMPose',
#             pose='/root/workspace/mmdeploy/rtmpose-ort/rtmpose-m/end2end.onnx', # noqa
#             pose_input_size=(192, 256),
#             backend=backend,
#             device=device)
# # then
# pose_model = PoseTracker(custom,
#                         det_frequency=10,
#                         tracking = False,
#                         to_openpose=openpose_skeleton,
#                         backend=backend, device=device)

# pose_model = Body( pose = 'rtmo',
#                 to_openpose=openpose_skeleton,
#                 mode='performance',  # balanced, performance, lightweight
#                 backend=backend,
#                 device=device)

# pose_model = PoseTracker(BodyWithFeet, #
#                         det_frequency=10,  # detect every 10 frames
#                         tracking = False,
#                         to_openpose=openpose_skeleton,
#                         backend=backend, device=device)



frame_idx = 0
while True:
    frames = [cap.read()[1] for cap in captures]

            
    frame_idx += 1  # Increment frame counter

    for idx, frame in enumerate(frames):
        t0 = time.time()
        keypoints, scores = pose_model(frame)
        t1 =time.time()
        print("Time of inference for one image",t1-t0)
        img_show = draw_skeleton(frame,
                                keypoints,
                                scores,
                                openpose_skeleton=openpose_skeleton,
                                kpt_thr=0.6)

        img_show = cv2.resize(img_show, (960, 540))
        cv2.imshow(f"Camera {idx}", img_show)
     # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break