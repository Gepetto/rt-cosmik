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

# Initialize the pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Enable the IR streams (IR1 and IR2)
config.enable_stream(rs.stream.infrared, 1)  # Enable IR1 (left)
config.enable_stream(rs.stream.infrared, 2)  # Enable IR2 (right)

# Start streaming
pipeline.start(config)

# Use os.makedirs() to create your directory; exist_ok=True means it won't throw an error if the directory already exists
os.makedirs(os.path.join(parent_directory,"cams_calibration/images_calib_cam_1/" + expe_no + "_" + trial_no + "/ir"), exist_ok=True)
os.makedirs(os.path.join(parent_directory,"cams_calibration/images_calib_cam_2/" + expe_no + "_" + trial_no + "/ir"), exist_ok=True)
os.makedirs(os.path.join(parent_directory,"cams_calibration/cam_params"), exist_ok=True)

c1_ir_imgs_path = os.path.join(parent_directory,"cams_calibration/images_calib_cam_1/" + expe_no + "_" + trial_no + "/ir/*")
c2_ir_imgs_path = os.path.join(parent_directory,"cams_calibration/images_calib_cam_2/" + expe_no + "_" + trial_no + "/ir/*")
c1_ir_params_path = os.path.join(parent_directory,"cams_calibration/cam_params/c1_params_ir_" + expe_no + "_" + trial_no + ".yml")
c2_ir_params_path = os.path.join(parent_directory,"cams_calibration/cam_params/c2_params_ir_" + expe_no + "_" + trial_no + ".yml")
c1_to_c2_ir_params_path = os.path.join(parent_directory,"cams_calibration/cam_params/c1_to_c2_params_ir_" + expe_no + "_" + trial_no + ".yml")

img_idx = 0
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        ir_frame_1 = frames.get_infrared_frame(1)
        ir_frame_2 = frames.get_infrared_frame(2)

        if not ir_frame_1 or not ir_frame_2:
            continue

        # Convert images to numpy arrays
        ir_image_1 = np.asanyarray(ir_frame_1.get_data())
        ir_image_2 = np.asanyarray(ir_frame_2.get_data())

        resized_ir_image_1 = cv2.resize(ir_image_1, (640, 480), interpolation = cv2.INTER_NEAREST) 
        resized_ir_image_2 = cv2.resize(ir_image_2, (640, 480), interpolation = cv2.INTER_NEAREST)

        # If the IR image is grayscale, convert it to a 3-channel image
        if len(resized_ir_image_1.shape) == 2:
            resized_ir_image_1 = cv2.cvtColor(resized_ir_image_1, cv2.COLOR_GRAY2BGR)

        # If the depth image is grayscale, convert it to a 3-channel image
        if len(resized_ir_image_2.shape) == 2:
            resized_ir_image_2 = cv2.cvtColor(resized_ir_image_2, cv2.COLOR_GRAY2BGR)


        images_hstack_2 = np.hstack((resized_ir_image_1, resized_ir_image_2))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images_hstack_2)
        c = cv2.waitKey(10)
        if c == ord('s'):
            print("s pressed")
            cv2.imwrite(os.path.join(parent_directory,"cams_calibration/images_calib_cam_1/" + expe_no + "_" + trial_no + "/ir/img_" + str(img_idx) + ".png"), ir_image_1)
            cv2.imwrite(os.path.join(parent_directory,"cams_calibration/images_calib_cam_2/" + expe_no + "_" + trial_no + "/ir/img_" + str(img_idx) + ".png"), ir_image_2)
            img_idx = img_idx + 1
        if c == ord('q'):
            print("q pressed")
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()


reproj1_ir, mtx1_ir, dist1_ir = calibrate_camera(images_folder = c1_ir_imgs_path)
save_cam_params(mtx1_ir, dist1_ir, reproj1_ir, c1_ir_params_path)
reproj2_ir, mtx2_ir, dist2_ir = calibrate_camera(images_folder = c2_ir_imgs_path)
save_cam_params(mtx2_ir, dist2_ir, reproj2_ir, c2_ir_params_path)
print("rmse cam1 IR = ", reproj1_ir)
print("rmse cam2 IR = ", reproj2_ir)
cv2.destroyAllWindows()

mtx_1_ir, dist_1_ir = load_cam_params(c1_ir_params_path)
mtx_2_ir, dist_2_ir = load_cam_params(c2_ir_params_path)
rmse_ir, R_ir, T_ir = stereo_calibrate(mtx_1_ir, dist_1_ir, mtx_2_ir, dist_2_ir, c1_ir_imgs_path, c2_ir_imgs_path)
save_cam_to_cam_params(mtx_1_ir, dist_1_ir, mtx_2_ir, dist_2_ir, R_ir, T_ir, rmse_ir, c1_to_c2_ir_params_path)
cv2.destroyAllWindows()

print("computed translation IR : ")
print(T_ir)

print("rmse IR: ", rmse_ir)