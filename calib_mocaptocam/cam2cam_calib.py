from utils import load_transformation, save_cam_to_cam_params, transform_to_local_frame
import numpy as np


soder_dir = f"/root/workspace/ros_ws/src/rt-cosmik/output/calib_mocam_2_cam"
cam2cam_dir = f"/root/worskpace/ros_ws/src/rt-cosmik/config/cam_params/c1_to_c2_params_color.yaml"


R_c1_in_mocap, d_c1_in_mocap, _, _ = load_transformation(soder_dir + "0" + "/soder.txt")
R_c2_in_mocap, d_c2_in_mocap, _, _ = load_transformation(soder_dir + "1" + "/soder.txt")

c1_to_c2_in_mocap = d_c2_in_mocap - d_c1_in_mocap

R_c2_in_c1 = np.transpose(R_c1_in_mocap)*R_c2_in_mocap
d_c2_in_c1 = transform_to_local_frame(d_c2_in_mocap, d_c1_in_mocap, R_c1_in_mocap)

save_cam_to_cam_params(np.zeros((3,3)), 
                       np.zeros((5,1)), 
                       np.zeros((3,3)), 
                       np.zeros((5,1)), 
                       R_c2_in_c1, d_c2_in_c1, 0.0, 
                       cam2cam_dir)