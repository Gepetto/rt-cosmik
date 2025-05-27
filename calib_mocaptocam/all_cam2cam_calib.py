from utils import load_transformation, save_cam_to_cam_params, transform_to_local_frame
sys.path.append(os.path.abspath('../../'))
from cams_calibration.utils import list_cameras_with_v4l2

import numpy as np
import os  
import cv2 as cv
import pinocchio as pin 

cameras = list_cameras_with_v4l2()

def load_cam_params(path):
    """
    Loads camera parameters from a given file.
    Args:
        path (str): The path to the file containing the camera parameters.
    Returns:
        tuple: A tuple containing the camera matrix and distortion matrix.
            - camera_matrix (numpy.ndarray): The camera matrix.
            - dist_matrix (numpy.ndarray): The distortion matrix.
    """
    
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return camera_matrix, dist_matrix

soder_dir = f"/root/workspace/ros_ws/src/rt-cosmik/output/calib_mocap_2_cam"

for num, idx_cam in enumerate(cameras.keys()):
    if num == len(cameras.keys())-1:
        break
    K1, D1 = load_cam_params(os.path.join("/root/workspace/ros_ws/src/rt-cosmik/config/cam_params/", f"c{idx_cam}_params_color.yaml"))
    K2, D2 = load_cam_params(os.path.join("/root/workspace/ros_ws/src/rt-cosmik/config/cam_params/", f"c{str(int(idx_cam)+2)}_params_color.yaml"))


    cam2cam_dir = f"/root/workspace/ros_ws/src/rt-cosmik/config/cam_params/c{idx_cam}_to_c{str(int(idx_cam)+2)}_params_color.yaml"

    R_c1_in_mocap, d_c1_in_mocap, _, _ = load_transformation(soder_dir + f"{idx_cam}" + "/soder.txt")
    R_c2_in_mocap, d_c2_in_mocap, _, _ = load_transformation(soder_dir + f"{str(int(idx_cam)+2)}" + "/soder.txt")

    # c1_to_c2_in_mocap = d_c2_in_mocap - d_c1_in_mocap

    # R_c1_in_c2 = np.transpose(R_c1_in_mocap)@R_c2_in_mocap
    # d_c1_in_c2 = transform_to_local_frame(d_c2_in_mocap, d_c1_in_mocap, R_c1_in_mocap)

    R_c1_in_c2 = np.transpose(R_c2_in_mocap)@R_c1_in_mocap
    d_c1_in_c2 = transform_to_local_frame(d_c1_in_mocap, d_c2_in_mocap, R_c2_in_mocap)

    print(f"rot {idx_cam} in {str(int(idx_cam)+2)}", R_c1_in_c2)
    print(f"pos {idx_cam} in {str(int(idx_cam)+2)}", d_c1_in_c2)

    # M1 = pin.SE3(R_c1_in_mocap, d_c1_in_mocap)
    # M2 = pin.SE3(R_c2_in_mocap, d_c2_in_mocap)

    # M12 = M1.inverse()*M2
    # R_c1_in_c2 = M12.rotation
    # d_c1_in_c2 = M12.translation

    save_cam_to_cam_params(K1, 
                        D1, 
                        K2, 
                        D2, 
                        R_c1_in_c2, d_c1_in_c2, 0.0, 
                        cam2cam_dir)