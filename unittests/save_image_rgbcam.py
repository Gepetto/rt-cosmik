#python3 -m unittests.save_image_rgbcam
import cv2
from utils.calib_utils import list_cameras_with_v4l2
from utils.settings import Settings
# Open the webcam (0 for default camera)
settings = Settings()
camera_dict = list_cameras_with_v4l2()
captures = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in camera_dict.keys()]

# Apply settings
for idx, cap in enumerate(captures):
    if not cap.isOpened():
        continue

    # Apply settings
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)
    

# Check if the webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Capture a frame
for capture in captures:
    ret, frame = cap.read()
    cv2.imwrite(f"Camera {capture}.jpg", frame)


# Release the webcam
cap.release()
cv2.destroyAllWindows()
