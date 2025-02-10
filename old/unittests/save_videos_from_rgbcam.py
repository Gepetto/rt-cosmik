#python3 -m unittests.save_videos_from_rgbcam
import cv2
import os
from utils.calib_utils import list_cameras_with_v4l2
from utils.settings import Settings
import time
from datetime import datetime


def main():

    # Initialize camera streams
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

        fps_reported = cap.get(cv2.CAP_PROP_FPS)
        print(f"reports FPS: {fps_reported}")
    # Check if all cameras are opened successfully
    for i, capture in enumerate(captures):
        if not capture.isOpened():
            print(f"Error: Could not open camera {i}")
            return

    # Define codec and create VideoWriter objects for saving videos
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec if needed
    output_dir = './output'  # You can modify this as needed

    # Create directories for saving videos
    os.makedirs(output_dir, exist_ok=True)

    # Define output video filenames
    output_videos = [os.path.join(output_dir, f'camera{i}_output.avi') for i in range(len(captures))]

    # Create VideoWriter objects for each camera
    writers = [cv2.VideoWriter(output, fourcc, 40.0, (int(settings.width), int(settings.height))) for output in output_videos]

    frame_id = 0
    timestamp= datetime.now()

    while True:
        frames = []
        for capture in captures:
            timestamp2 = datetime.now()
            # fps_reported = capture.get(cv2.CAP_PROP_FPS)
            # print(fps_reported)
            ret, frame = capture.read()
            print( capture, " " , timestamp2 - timestamp)
            timestamp = timestamp2

            if not ret:
                print("Error: Failed to capture frame from one of the cameras")
                return
            # frames.append(frame)

        # Write the frames to the corresponding video files
        # for writer, frame in zip(writers, frames):
        #     writer.write(frame)

        # Optionally, display the frames from each camera
        # for i, frame in enumerate(frames):
        #     cv2.imshow(f"Camera {i}", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    # Release all the video captures and writers
    for capture in captures:
        capture.release()
    for writer in writers:
        writer.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
