# To run the code from RT-COSMIK root : python -m cams_calibration.calibrate_cameras

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

width = 1080
height = 720

max_cameras = 10
available_cameras = []
for index in range(max_cameras):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        available_cameras.append(index)
        cap.release()  # Release the camera after checking

camera_indices = available_cameras
# print(camera_indices)
# if no webcam
# captures = [cv2.VideoCapture(idx) for idx in camera_indices]

# if webcam remove it 
captures = [cv2.VideoCapture(idx) for idx in camera_indices if idx !=2]

for cap in captures: 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # HD
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # HD
    cap.set(cv2.CAP_PROP_FPS, 40)  # Set frame rate to 40fps


# Check if cameras opened successfully
for i, cap in enumerate(captures):
    if not cap.isOpened():
        print(f"Error: Camera {i} not opened.")

# Use os.makedirs() to create your directory; exist_ok=True means it won't throw an error if the directory already exists
os.makedirs(os.path.join(parent_directory,"cams_calibration/images_calib_cam_1/" + expe_no + "_" + trial_no + "/color"), exist_ok=True)
os.makedirs(os.path.join(parent_directory,"cams_calibration/images_calib_cam_2/" + expe_no + "_" + trial_no + "/color"), exist_ok=True)

c1_color_imgs_path = os.path.join(parent_directory,"cams_calibration/images_calib_cam_1/" + expe_no + "_" + trial_no + "/color/*")
c2_color_imgs_path = os.path.join(parent_directory,"cams_calibration/images_calib_cam_2/" + expe_no + "_" + trial_no + "/color/*")
c1_color_params_path = os.path.join(parent_directory,"cams_calibration/cam_params/c1_params_color_" + expe_no + "_" + trial_no + ".yml")
c2_color_params_path = os.path.join(parent_directory,"cams_calibration/cam_params/c2_params_color_" + expe_no + "_" + trial_no + ".yml")
c1_to_c2_color_params_path = os.path.join(parent_directory,"cams_calibration/cam_params/c1_to_c2_params_color_" + expe_no + "_" + trial_no + ".yml")

img_idx = 0
try:
    while True:
        frames = [cap.read()[1] for cap in captures]
            
        if not all(frame is not None for frame in frames):
            continue

        # Convert images to numpy arrays
        color_image_1 = frames[0]
        color_image_2 = frames[1]

        resized_color_image_1 = cv2.resize(color_image_1, (640, 480), interpolation = cv2.INTER_NEAREST)
        resized_color_image_2 = cv2.resize(color_image_2, (640, 480), interpolation = cv2.INTER_NEAREST) 

        images_hstack_1 = np.hstack((resized_color_image_1, resized_color_image_2))
        
        # Show images
        cv2.namedWindow('RGB cams', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB cams', images_hstack_1)
        c = cv2.waitKey(10)
        if c == ord('s'):
            print("image taken")
            cv2.imwrite(os.path.join(parent_directory,"cams_calibration/images_calib_cam_1/" + expe_no + "_" + trial_no + "/color/img_" + str(img_idx) + ".png"), color_image_1)
            cv2.imwrite(os.path.join(parent_directory,"cams_calibration/images_calib_cam_2/" + expe_no + "_" + trial_no + "/color/img_" + str(img_idx) + ".png"), color_image_2)
            img_idx = img_idx + 1
        if c == ord('q'):
            print("quit")
            break

finally:
    # Release the camera captures
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()

reproj1_color, mtx1_color, dist1_color = calibrate_camera(images_folder = c1_color_imgs_path)
save_cam_params(mtx1_color, dist1_color, reproj1_color, c1_color_params_path)
reproj2_color, mtx2_color, dist2_color = calibrate_camera(images_folder = c2_color_imgs_path)
save_cam_params(mtx2_color, dist2_color, reproj2_color, c2_color_params_path)
print("rmse cam1 RGB = ", reproj1_color)
print("rmse cam2 RGB = ", reproj2_color)
cv2.destroyAllWindows()

mtx_1_color, dist_1_color = load_cam_params(c1_color_params_path)
mtx_2_color, dist_2_color = load_cam_params(c2_color_params_path)
rmse_color, R_color, T_color = stereo_calibrate(mtx_1_color, dist_1_color, mtx_2_color, dist_2_color, c1_color_imgs_path, c2_color_imgs_path)
save_cam_to_cam_params(mtx_1_color, dist_1_color, mtx_2_color, dist_2_color, R_color, T_color, rmse_color, c1_to_c2_color_params_path)
cv2.destroyAllWindows()

print("computed translation RGB : ")
print(T_color)

print("rmse RGB: ", rmse_color)