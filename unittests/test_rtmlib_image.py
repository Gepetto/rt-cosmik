#python3 -m unittests.test_rtmlib_image
import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton, BodyWithFeet, RTMO, RTMPose, Body, PoseTracker

device = 'cpu'  # cpu, cuda
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

img = cv2.imread('unittests/demo.jpg')

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

#test rtmo, pas de body26
# pose_model = RTMO(
#     onnx_model='/root/workspace/ros_ws/src/rt-cosmik/models/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211/end2end.onnx',
#     model_input_size=(640, 640),
#     backend=backend,
#     device=device
# )

#test rtmpose without detector
# pose_model = RTMPose(
#     onnx_model='/root/workspace/ros_ws/src/rt-cosmik/models/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605/end2end.onnx',
#     model_input_size=(192, 256),
#     backend=backend,
#     device=device
# )
#test rtmpose with detector
pose_model = Body( #pose = 'rtmo',
                to_openpose=openpose_skeleton,
                mode='performance',  # balanced, performance, lightweight
                backend=backend,
                device=device)

# pose_model = PoseTracker(BodyWithFeet,
#                         det_frequency=10,  # detect every 10 frames
#                         to_openpose=openpose_skeleton,
#                         backend=backend, device=device)

# pose_model = BodyWithFeet(to_openpose=openpose_skeleton,
#                       mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
#                       backend=backend, device=device)

keypoints, scores = pose_model(img)
# bboxes = BodyWithFeet.det_model(img)


# visualize

# if you want to use black background instead of original image,
# img_show = np.zeros(img_show.shape, dtype=np.uint8)

img_show = draw_skeleton(img, keypoints, scores, kpt_thr=0.5)


cv2.imshow('img', img_show)
cv2.waitKey()