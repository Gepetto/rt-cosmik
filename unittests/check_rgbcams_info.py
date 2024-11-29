import cv2
from utils.calib_utils import list_cameras_with_v4l2

if __name__ == "__main__":
    cameras = list_cameras_with_v4l2()
    if cameras:
        print("Available cameras:", cameras)
    else:
        print("No cameras found.")
