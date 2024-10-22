# To run the code from RT-COSMIK root : python -m cams_calibration.calibrate_rs

import pyrealsense2 as rs
import numpy as np
import cv2
from utils.calib_utils import calibrate_camera, save_cam_params, load_cam_params, stereo_calibrate, save_cam_to_cam_params
import os
import sys

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
parent_directory = os.path.dirname(script_directory)

# Checking if at least two arguments are passed (including the script name)
if len(sys.argv) > 2:
    arg1 = sys.argv[1]  # First argument
    arg2 = sys.argv[2]  # Second argument

    # You can now use arg1 and arg2 in your script
    # Remember to convert them from strings if they represent other types
else:
    print("Not enough arguments provided. Usage: mycode.py <arg1> <arg2>")
    sys.exit(1)  # Exit the script

expe_no = str(arg1)
trial_no = str(arg2)

# Configure depth and color streams for both cameras
pipeline_1 = rs.pipeline()
pipeline_2 = rs.pipeline()
config_1 = rs.config()
config_2 = rs.config()

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

# Get device product line for setting a supporting resolution
pipeline_wrapper_1 = rs.pipeline_wrapper(pipeline_1)
pipeline_wrapper_2 = rs.pipeline_wrapper(pipeline_2)
pipeline_profile_1 = config_1.resolve(pipeline_wrapper_1)
pipeline_profile_2 = config_2.resolve(pipeline_wrapper_2)
device_1 = pipeline_profile_1.get_device()
device_2 = pipeline_profile_2.get_device()
device_product_line_1 = str(device_1.get_info(rs.camera_info.product_line))
device_product_line_2 = str(device_2.get_info(rs.camera_info.product_line))

found_rgb_1 = False
found_rgb_2 = False
for s in device_1.sensors:
    if s.get_info(rs.camera_info.name) == 'Stereo Module':
        found_rgb_1 = True
        break
for s in device_2.sensors:
    if s.get_info(rs.camera_info.name) == 'Stereo Module':
        found_rgb_2 = True
        break
if not found_rgb_1 or not found_rgb_2:
    print("The demo requires Depth cameras with Color sensors")
    exit(0)

print(sn_list)

# Replace these with your camera serial numbers
serial_number_1 = sn_list[0]
serial_number_2 = sn_list[1]

# Enable the devices with the serial numbers
config_1.enable_device(serial_number_1)
config_2.enable_device(serial_number_2)

config_1.enable_stream(rs.stream.infrared, 1280, 720, rs.format.y8, 30)
config_2.enable_stream(rs.stream.infrared, 1280, 720, rs.format.y8, 30)
config_1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config_2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)

# Use os.makedirs() to create your directory; exist_ok=True means it won't throw an error if the directory already exists
os.makedirs(os.path.join(parent_directory,"/cams_calibration/images_calib_cam_1/" + expe_no + "_" + trial_no + "/color"), exist_ok=True)
os.makedirs(os.path.join(parent_directory,"/cams_calibration/images_calib_cam_2/" + expe_no + "_" + trial_no + "/color"), exist_ok=True)
os.makedirs(os.path.join(parent_directory,"/cams_calibration/images_calib_cam_1/" + expe_no + "_" + trial_no + "/ir"), exist_ok=True)
os.makedirs(os.path.join(parent_directory,"/cams_calibration/images_calib_cam_2/" + expe_no + "_" + trial_no + "/ir"), exist_ok=True)

c1_ir_imgs_path = os.path.join(parent_directory,"/cams_calibration/images_calib_cam_1/" + expe_no + "_" + trial_no + "/ir/*")
c2_ir_imgs_path = os.path.join(parent_directory,"/cams_calibration/images_calib_cam_2/" + expe_no + "_" + trial_no + "/ir/*")
c1_ir_params_path = os.path.join(parent_directory,"/cams_calibration/cam_params/c1_params_ir_" + expe_no + "_" + trial_no + ".yml")
c2_ir_params_path = os.path.join(parent_directory,"/cams_calibration/cam_params/c2_params_ir_" + expe_no + "_" + trial_no + ".yml")
c1_to_c2_ir_params_path = os.path.join(parent_directory,"/cams_calibration/cam_params/c1_to_c2_params_ir_" + expe_no + "_" + trial_no + ".yml")

c1_color_imgs_path = os.path.join(parent_directory,"/cams_calibration/images_calib_cam_1/" + expe_no + "_" + trial_no + "/color/*")
c2_color_imgs_path = os.path.join(parent_directory,"/cams_calibration/images_calib_cam_2/" + expe_no + "_" + trial_no + "/color/*")
c1_color_params_path = os.path.join(parent_directory,"/cams_calibration/cam_params/c1_params_color_" + expe_no + "_" + trial_no + ".yml")
c2_color_params_path = os.path.join(parent_directory,"/cams_calibration/cam_params/c2_params_color_" + expe_no + "_" + trial_no + ".yml")
c1_to_c2_color_params_path = os.path.join(parent_directory,"/cams_calibration/cam_params/c1_to_c2_params_color_" + expe_no + "_" + trial_no + ".yml")


img_idx = 0
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        frames_2 = pipeline_2.wait_for_frames()
        color_frame_1 = frames_1.get_color_frame()
        color_frame_2 = frames_2.get_color_frame()

        ir_frame_1 = frames_1.get_infrared_frame()
        ir_frame_2 = frames_2.get_infrared_frame()

        if not color_frame_1 or not color_frame_2 or not ir_frame_1 or not ir_frame_2:
            continue

        # Convert images to numpy arrays
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())
        ir_image_1 = np.asanyarray(ir_frame_1.get_data())
        ir_image_2 = np.asanyarray(ir_frame_2.get_data())

        resized_color_image_1 = cv2.resize(color_image_1, (640, 480), interpolation = cv2.INTER_NEAREST)
        resized_color_image_2 = cv2.resize(color_image_2, (640, 480), interpolation = cv2.INTER_NEAREST) 
        resized_ir_image_1 = cv2.resize(ir_image_1, (640, 480), interpolation = cv2.INTER_NEAREST) 
        resized_ir_image_2 = cv2.resize(ir_image_2, (640, 480), interpolation = cv2.INTER_NEAREST)

        # If the IR image is grayscale, convert it to a 3-channel image
        if len(resized_ir_image_1.shape) == 2:
            resized_ir_image_1 = cv2.cvtColor(resized_ir_image_1, cv2.COLOR_GRAY2BGR)

        # If the depth image is grayscale, convert it to a 3-channel image
        if len(resized_ir_image_2.shape) == 2:
            resized_ir_image_2 = cv2.cvtColor(resized_ir_image_2, cv2.COLOR_GRAY2BGR)


        images_hstack_1 = np.hstack((resized_color_image_1, resized_color_image_2))
        images_hstack_2 = np.hstack((resized_ir_image_1, resized_ir_image_2))
        images_vstack = np.vstack((images_hstack_1, images_hstack_2))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images_vstack)
        c = cv2.waitKey(10)
        if c == ord('s'):
            print("image taken")
            cv2.imwrite(os.path.join(parent_directory,"/cams_calibration/images_calib_cam_1/" + expe_no + "_" + trial_no + "/color/img_" + str(img_idx) + ".png", color_image_1))
            cv2.imwrite(os.path.join(parent_directory,"/cams_calibration/images_calib_cam_2/" + expe_no + "_" + trial_no + "/color/img_" + str(img_idx) + ".png", color_image_2))
            cv2.imwrite(os.path.join(parent_directory,"/cams_calibration/images_calib_cam_1/" + expe_no + "_" + trial_no + "/ir/img_" + str(img_idx) + ".png", ir_image_1))
            cv2.imwrite(os.path.join(parent_directory,"/cams_calibration/images_calib_cam_2/" + expe_no + "_" + trial_no + "/ir/img_" + str(img_idx) + ".png", ir_image_2))
            img_idx = img_idx + 1
        if c == ord('q'):
            print("quit")
            break

finally:
    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()
    cv2.destroyAllWindows()

reproj1_color, mtx1_color, dist1_color = calibrate_camera(images_folder = c1_color_imgs_path)
save_cam_params(mtx1_color, dist1_color, reproj1_color, c1_color_params_path)
reproj2_color, mtx2_color, dist2_color = calibrate_camera(images_folder = c2_color_imgs_path)
save_cam_params(mtx2_color, dist2_color, reproj2_color, c2_color_params_path)
print("rmse cam1 RGB = ", reproj1_color)
print("rmse cam2 RGB = ", reproj2_color)
cv2.destroyAllWindows()

reproj1_ir, mtx1_ir, dist1_ir = calibrate_camera(images_folder = c1_ir_imgs_path)
save_cam_params(mtx1_ir, dist1_ir, reproj1_ir, c1_ir_params_path)
reproj2_ir, mtx2_ir, dist2_ir = calibrate_camera(images_folder = c2_ir_imgs_path)
save_cam_params(mtx2_ir, dist2_ir, reproj2_ir, c2_ir_params_path)
print("rmse cam1 IR = ", reproj1_ir)
print("rmse cam2 IR = ", reproj2_ir)
cv2.destroyAllWindows()

mtx_1_color, dist_1_color = load_cam_params(c1_color_params_path)
mtx_2_color, dist_2_color = load_cam_params(c2_color_params_path)
rmse_color, R_color, T_color = stereo_calibrate(mtx_1_color, dist_1_color, mtx_2_color, dist_2_color, c1_color_imgs_path, c2_color_imgs_path)
save_cam_to_cam_params(mtx_1_color, dist_1_color, mtx_2_color, dist_2_color, R_color, T_color, rmse_color, c1_to_c2_color_params_path)

mtx_1_ir, dist_1_ir = load_cam_params(c1_ir_params_path)
mtx_2_ir, dist_2_ir = load_cam_params(c2_ir_params_path)
rmse_ir, R_ir, T_ir = stereo_calibrate(mtx_1_ir, dist_1_ir, mtx_2_ir, dist_2_ir, c1_ir_imgs_path, c2_ir_imgs_path)
save_cam_to_cam_params(mtx_1_ir, dist_1_ir, mtx_2_ir, dist_2_ir, R_ir, T_ir, rmse_ir, c1_to_c2_ir_params_path)
cv2.destroyAllWindows()

print("computed translation RGB : ")
print(T_color)

print("computed translation IR : ")
print(T_ir)

print("rmse RGB: ", rmse_color)
print("rmse IR: ", rmse_ir)