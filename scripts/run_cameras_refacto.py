import sys
import os
import cv2
import time
import csv
from datetime import datetime
from multiprocessing import Process, Barrier, Queue, Event

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")) # src dir

from src.rtcosmik.camera.cam_utils import list_cameras
from src.rtcosmik.config_loader import settings
from src.rtcosmik.utils.linear_algebra_utils import concat_frames



def camera_process(queue, barrier, idx_cam):

    timestamps = []

    cap = cv2.VideoCapture(idx_cam, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: Could not open camera {idx_cam}")
        exit()
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(settings.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

    while True:

        barrier.wait()
        _, frame = cap.read()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        queue.put(frame)
        timestamps.append(timestamp)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            cap.release()

            with open(os.path.join(settings.SAVE_DIR, f"camera_{idx_cam}_timestamps.csv"), mode='w', newline='') as f:
                timestamps_writer = csv.writer(f)
                timestamps_writer.writerow(["frame_index", "timestamp"])
                for frame_idx, ts in enumerate(timestamps):
                    writer.writerow([str(frame_idx), ts])

            break


def display_process(queues, idxs_cams):

    while True:

        for queue, idx_cam in zip(queues, idxs_cams):
            list_of_frames = list(queue)
            globals()[f"last_frame_{idx_cam}"] = list_of_frames[-1]
        
        concatenated_frame = concat_frames([globals()[f"last_frame_{idx_cam}"] for idx_cam in idxs_cams])
        cv2.imshow('Streaming View', concatenated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def saver_process(queue, idx_cam, stop_event):

    fourcc = cv2.VideoWriter_fourcc(settings.fourcc)
    frame_size = (settings.height, settings.width)
    video_filename = os.path.join(settings.SAVE_DIR, f"camera_{idx_cam}.mp4")
    fps = settings.fs

    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

    while True:

        try:
            oldest_frame = queue.get()
            video_writer.write(oldest_frame)

        except queue.Empty:
            if stop_event.is_set():
                video_writer.release()
                print(f"Saver process for Camera {idx_cam} terminated.")
                break
            else: 
                print(f"Queue {idx_cam} is empty whilst a frame should have been put before by the camera process")




if __name__ == "__main__":

    cameras = list_cameras()

    barrier = Barrier(len(cameras))

    stop_event = Event()

    for idx_cam in cameras.keys():
        globals()[f"cam{idx_cam}_queue"] = Queue()

    cameras_processes = [
        Process(
            target=camera_process, 
            args=(globals()[f"cam{idx_cam}_queue"], barrier, idx_cam)
        ) 
        for idx_cam in cameras.keys()
    ]

    display_process = Process(
        target=display_process, 
        args=([globals()[f"cam{idx_cam}_queue"] for idx_cam in cameras.keys()], cameras.keys())
    )

    saver_processes = [
        Process(
            target=saver_process, 
            args=(globals()[f"cam{idx_cam}_queue"], idx_cam, stop_event)
        ) 
        for idx_cam in cameras.keys()
    ]

    all_processes = cameras_processes + [display_process] + saver_processes
    
    while True:

        if cv2.waitKey(1) & 0xFF == ord('s'):

            for process in all_processes:
                process.start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()

            while any([saver_process.is_alive() for saver_process in saver_processes]):
                print("Waiting for saver processes to finish writing ...")
                time.sleep(1)

            for process in all_processes:
                process.join()
            
            break
    
    print("All processes terminated.")
    
