import cv2
import sys

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Total number of frames: {total_frames}")
    return total_frames

# Example usage
video_path = sys.argv[1]
count_frames(video_path)