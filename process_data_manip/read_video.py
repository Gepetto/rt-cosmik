import cv2

# Paths to your two video files
video_path1 = '/root/workspace/ros_ws/src/rt-cosmik/output/test/camera_0.mp4'
video_path2 = '/root/workspace/ros_ws/src/rt-cosmik/output/test/camera_2.mp4'

# Open the video files
cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)

# Check if both videos are opened successfully
if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Cannot open one of the video files.")
    exit()

print("Press any key to advance frame by frame, or 'q' to quit.")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("End of one of the videos.")
        break

    # Resize frames to the same height if needed
    if frame1.shape[0] != frame2.shape[0]:
        height = min(frame1.shape[0], frame2.shape[0])
        frame1 = cv2.resize(frame1, (int(frame1.shape[1] * height / frame1.shape[0]), height))
        frame2 = cv2.resize(frame2, (int(frame2.shape[1] * height / frame2.shape[0]), height))

    # Stack frames horizontally
    stacked = cv2.hconcat([frame1, frame2])

    # Display the stacked frames
    cv2.imshow('Frame-by-Frame Viewer', frame1)
    cv2.imshow('Frame-by-Frame Viewer_', frame2)

    key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
    if key == ord('q'):
        print("Quitting viewer.")
        break

# Release everything
cap1.release()
cap2.release()
cv2.destroyAllWindows()
