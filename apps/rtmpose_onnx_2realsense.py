import time
from typing import List, Tuple
import time
import cv2
import loguru
import numpy as np
import onnxruntime as ort
import pyrealsense2 as rs
import csv
import pinocchio as pin
import rospy
from sensor_msgs.msg import JointState
from datetime import datetime
from mmdeploy_runtime import PoseDetector

from utils.read_write_utils import formatting_keypoints, set_zero_data
from utils.model_utils import Robot, model_scaling
from utils.calib_utils import load_cam_params, load_cam_to_cam_params, load_cam_pose
from utils.triangulation_utils import triangulate_points
from utils.ik_utils import RT_Quadprog


logger = loguru.logger

# Define the paths and settings directly within the script
# onnx_file = './config/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504/'  # Replace with your ONNX model file path
onnx_file = "/home/gepetto/mmpose/mmdeploy/rtmpose-ort/rtmpose-m"
device = 'cpu'  # Choose 'cpu' or 'cuda' based on your setup
save_path = 'output.jpg'  # Path where the output image will be saved

def preprocess(
    img: np.ndarray, input_size: Tuple[int, int] = (192, 256)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_shape = img.shape[:2]
    bbox = np.array([0, 0, img_shape[1], img_shape[0]])

    center, scale = bbox_xyxy2cs(bbox, padding=1.25)
    resized_img, scale = top_down_affine(input_size, scale, center, img)

    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    resized_img = (resized_img - mean) / std

    return resized_img, center, scale


def build_session(onnx_file: str, device: str = 'cpu') -> ort.InferenceSession:
    providers = ['CPUExecutionProvider'
                 ] if device == 'cpu' else ['CUDAExecutionProvider']
    sess = ort.InferenceSession(path_or_bytes=onnx_file, providers=providers)
    return sess


def inference(sess: ort.InferenceSession, img: np.ndarray) -> np.ndarray:
    input = [img.transpose(2, 0, 1)]
    sess_input = {sess.get_inputs()[0].name: input}
    sess_output = [out.name for out in sess.get_outputs()]
    outputs = sess.run(sess_output, sess_input)
    return outputs


def postprocess(outputs: List[np.ndarray],
                model_input_size: Tuple[int, int],
                center: Tuple[int, int],
                scale: Tuple[int, int],
                simcc_split_ratio: float = 2.0
                ) -> Tuple[np.ndarray, np.ndarray]:
    simcc_x, simcc_y = outputs
    keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)
    keypoints = keypoints / model_input_size * scale + center - scale / 2
    return keypoints, scores

def decode(simcc_x: np.ndarray, simcc_y: np.ndarray,
           simcc_split_ratio) -> Tuple[np.ndarray, np.ndarray]:
    """Modulate simcc distribution with Gaussian.

    Args:
        simcc_x (np.ndarray[K, Wx]): model predicted simcc in x.
        simcc_y (np.ndarray[K, Wy]): model predicted simcc in y.
        simcc_split_ratio (int): The split ratio of simcc.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: keypoints in shape (K, 2) or (n, K, 2)
        - np.ndarray[float32]: scores in shape (K,) or (n, K)
    """
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio

    return keypoints, scores

def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # get maximum value across x and y axis
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals

def visualize(img: np.ndarray,
              keypoints: np.ndarray,
              scores: np.ndarray,
              filename: str = None,
              thr=0.3) -> np.ndarray:
    skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
    palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]]
    link_color = [1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
    point_color = [0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]

    for kpts, score in zip(keypoints, scores):
        keypoints_num = len(score)
        for kpt, color in zip(kpts, point_color):
            cv2.circle(img, tuple(kpt.astype(np.int32)), 1, palette[color], 1,
                       cv2.LINE_AA)
        for (u, v), color in zip(skeleton, link_color):
            if u < keypoints_num and v < keypoints_num and score[u] > thr and score[v] > thr:
                cv2.line(img, tuple(kpts[u].astype(np.int32)),
                         tuple(kpts[v].astype(np.int32)), palette[color], 2,
                         cv2.LINE_AA)
    
    if filename:
        cv2.imwrite(filename, img)

    return img

def bbox_xyxy2cs(bbox: np.ndarray,
                 padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    # get bbox center and scale
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale

def top_down_affine(input_size: Tuple[int, int], bbox_scale: np.ndarray, bbox_center: np.ndarray,
                    img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the bbox image as the model input by affine transform.

    Args:
        input_size (Tuple[int, int]): The input size of the model (width, height).
        bbox_scale (np.ndarray): The scale of the bounding box.
        bbox_center (np.ndarray): The center of the bounding box.
        img (np.ndarray): The original image.

    Returns:
        tuple: A tuple containing:
        - np.ndarray[float32]: The image after affine transformation.
        - np.ndarray[float32]: The scale of the bounding box after affine transformation.
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    # Reshape bbox to fixed aspect ratio
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

    # Get the affine matrix
    rot = 0
    warp_mat = get_warp_matrix(bbox_center, bbox_scale, rot, output_size=(w, h))

    # Apply the affine transform
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale

def get_warp_matrix(center: np.ndarray,
                    scale: np.ndarray,
                    rot: float,
                    output_size: Tuple[int, int],
                    shift: Tuple[float, float] = (0., 0.),
                    inv: bool = False) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # compute transformation matrix
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    # get four corners of the src rectangle in the original image
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    # get four corners of the dst rectangle in the input image
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat

def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt

def _fix_aspect_ratio(bbox_scale: np.ndarray,
                      aspect_ratio: float) -> np.ndarray:
    """Extend the scale to match the given aspect ratio.

    Args:
        scale (np.ndarray): The image scale (w, h) in shape (2, )
        aspect_ratio (float): The ratio of ``w/h``

    Returns:
        np.ndarray: The reshaped image scale in (2, )
    """
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale

def get_device_serial_numbers():
    """Get a list of serial numbers for connected RealSense devices."""
    ctx = rs.context()
    return [device.get_info(rs.camera_info.serial_number) for device in ctx.query_devices()]

def process_realsenses(sess: ort.InferenceSession, model_input_size: Tuple[int, int], show_interval=1):
    """Process frames from multiple Intel RealSense cameras and visualize predicted keypoints."""
    
    sn_list = []
    ctx = rs.context()
    if len(ctx.devices) > 0:
        for d in ctx.devices:

            print ('Found device: ', \

                    d.get_info(rs.camera_info.name), ' ', \

                    d.get_info(rs.camera_info.serial_number))
            sn_list.append(d.get_info(rs.camera_info.serial_number))

    else:
        print("No Intel Device connected")

    serial_numbers = get_device_serial_numbers()
    pipelines = []
    for serial in serial_numbers:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        pipelines.append(pipeline)

    # Loading human urdf
    human = Robot('models/human_urdf/urdf/human.urdf','models') 
    human_model = human.model

    # IK calculations 
    q = pin.neutral(human_model)
    dt = 1/30
    keys_to_track_list = ["Right Knee","Right Hip","Right Shoulder","Right Elbow","Right Wrist"]
    dict_dof_to_keypoints = dict(zip(keys_to_track_list,['knee_Z', 'lumbar_Z', 'shoulder_Z', 'elbow_Z', 'hand_fixed']))

    first_sample = True 
    frame_idx = 0
    detector = PoseDetector(
                model_path=onnx_file, device_name=device, device_id=0)
    
    try :
        while True:
            timestamp=datetime.now()
            formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f ")

            frames_list = [pipeline.wait_for_frames().get_color_frame() for pipeline in pipelines]
            if not all(frames_list):
                continue

            frame_idx += 1
            keypoints_list = []
            
            for idx, color_frame in enumerate(frames_list):
                time_init = time.time()
                frame = np.asanyarray(color_frame.get_data())
                
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # resized_img, center, scale = preprocess(frame_rgb, model_input_size)

                # outputs = inference(sess, resized_img)

                # keypoints, scores = postprocess(outputs, model_input_size, center, scale)

                # vis_result = visualize(frame, keypoints, scores)


                result = detector(frame_rgb)

                print(result)

                _, point_num, _ = result.shape
                points = result[:, :, :2].reshape(point_num, 2)
                for [x, y] in points.astype(int):
                    cv2.circle(frame_rgb, (x, y), 1, (0, 255, 0), 2)
                
                # Display the frame using OpenCV
                cv2.imshow(f'Visualization Result {idx}', frame_rgb)

            # Press 'q' to exit the loop, 's' to start/stop saving
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()


def main():
    logger.info('Start running model on RTMPose...')
    
    sess = build_session(onnx_file, device)
    h, w = sess.get_inputs()[0].shape[2:]
    model_input_size = (w, h)

    process_realsenses(sess, model_input_size, show_interval=1)

    logger.info('Done...')


if __name__ == '__main__':
    main()