import sys
import os
import cv2
import time
import csv
import numpy as np
import ctypes
from pynput import keyboard
from datetime import datetime
from multiprocessing import Process, Barrier, Queue, Event, Array
from threading import BrokenBarrierError
from queue import Empty

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")) # src dir

from src.rtcosmik.config_loader import settings
from src.rtcosmik.utils.linear_algebra_utils import concat_frames

subject = sys.argv[3]
trial = sys.argv[4]


def camera_process(queue, barrier, idx_cam, stop_event, current_frame):

    timestamps = []

    cap = cv2.VideoCapture(idx_cam)
    if not cap.isOpened():
        print(f"Error: Could not open camera {idx_cam}")
        exit()
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FPS, settings.fs)

    while not stop_event.is_set():

        try:
            barrier.wait()
            ret, frame = cap.read()
            if not ret:
                keyboard.Controller().press("q")
                raise Exception(f"Camera {idx_cam} has crashed, quitting the recording.")

            np.frombuffer(current_frame.get_obj(), dtype=np.uint8)[:] = frame.flatten()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            queue.put(frame)
            timestamps.append(timestamp)
        
        except BrokenBarrierError:
            break

    cap.release()

    with open(os.path.join(f"C:\\Users\\krauszm\\COSMIK\\rt-cosmik\\output\\{subject}\\{trial}", f"camera_{idx_cam}_timestamps.csv"), mode='w', newline='') as f:
        timestamps_writer = csv.writer(f)
        timestamps_writer.writerow(["frame_index", "timestamp"])
        for frame_idx, ts in enumerate(timestamps):
            timestamps_writer.writerow([str(frame_idx), ts])

    print(f"Camera {idx_cam} process terminated.")


def displays_process(current_frames, idx_cams):

    while True:
        
        for current_frame, idx_cam in zip(current_frames, idx_cams):
            globals()[f"displayed_frame_{idx_cam}"] = np.frombuffer(current_frame.get_obj(), dtype=np.uint8).reshape(settings.height, settings.width, 3)
            globals()[f"reduced_frame_{idx_cam}"] = cv2.resize(globals()[f"displayed_frame_{idx_cam}"], (640,400))

        concatenated_frame_resized = concat_frames([globals()[f"reduced_frame_{idx_cam}"] for idx_cam in idx_cams])
        cv2.imshow('Streaming View', concatenated_frame_resized)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
    
    print("Display process terminated.")


def saver_process(queue, idx_cam, stop_event):

    fourcc = cv2.VideoWriter_fourcc(*settings.fourcc)
    size = (settings.width, settings.height)
    video_filename = os.path.join(f"C:\\Users\\krauszm\\COSMIK\\rt-cosmik\\output\\{subject}\\{trial}", f"camera_{idx_cam}.avi")
    fps = settings.fs

    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, size)

    while True:

        try:
            oldest_frame = queue.get_nowait()
            video_writer.write(oldest_frame)

        except Empty:
            if stop_event.is_set():
                video_writer.release()
                break
            else: 
                pass
    
    print(f"Saver process for Camera {idx_cam} terminated.")


def on_press(key):
    try:
        if key.char == 's' and not start_event.is_set():
            start_event.set()

        elif key.char == 'q':
            stop_event.set()
            return False  # Arrête le listener
    except AttributeError:
        # touches spéciales (ex: ctrl, alt...) qu'on ignore ici
        pass



if __name__ == "__main__":

    try:
        os.makedirs(f"C:\\Users\\krauszm\\COSMIK\\rt-cosmik\\output\\{subject}\\{trial}", exist_ok=False)
    except FileExistsError:
        answer = input(f"Do you want to remove old {subject}/{trial} that already exists ? [yes/no]")
        if answer == "yes":
            os.makedirs(f"C:\\Users\\krauszm\\COSMIK\\rt-cosmik\\output\\{subject}\\{trial}", exist_ok=True)
        elif answer == "no":
            raise Exception("Then change the name of the subject and/or the trial and retry.")

    cameras = {int(sys.argv[1]) : "Intel(R) RealSense(TM) Depth Camera 455  RGB", int(sys.argv[2]) : "Intel(R) RealSense(TM) Depth Camera 455  RGB"}
    for idx in cameras.keys():
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            raise KeyError("Check if cameras indexes you typed are good.")

    print(cameras)

    barrier = Barrier(len(cameras))

    global start_event
    global stop_event
    start_event = Event()
    stop_event = Event()

    frame_size = settings.width*settings.height*3

    for idx_cam in cameras.keys():
        globals()[f"current_frame_{idx_cam}"] = Array(ctypes.c_ubyte, frame_size)
        globals()[f"cam{idx_cam}_queue"] = Queue()

    cameras_processes = [
        Process(
            target=camera_process, 
            args=(globals()[f"cam{idx_cam}_queue"], barrier, idx_cam, stop_event, globals()[f"current_frame_{idx_cam}"]),
            name=f"Process camera {idx_cam}"
        ) 
        for idx_cam in cameras.keys()
    ]

    display_process = Process(
        target=displays_process, 
        args=([globals()[f"current_frame_{idx_cam}"] for idx_cam in cameras.keys()], list(cameras.keys())),
        name="Display process"
    )

    saver_processes = [
        Process(
            target=saver_process, 
            args=(globals()[f"cam{idx_cam}_queue"], idx_cam, stop_event),
            name=f"Saver process {idx_cam}"
        ) 
        for idx_cam in cameras.keys()
    ]

    all_processes = cameras_processes + [display_process] + saver_processes

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    unrestarter = True
    print("Waiting for 's' to be pressed to start trial...")
    while True:

        if start_event.is_set() and unrestarter:
                
            print("\nStarting all processes...")
            for process in all_processes:
                process.start()

            unrestarter = False

            print("Press 'q' to stop recording.")

        if stop_event.is_set():

            barrier.abort()
                
            while any([saver_process.is_alive() for saver_process in saver_processes]):
                print("Waiting for saver processes to finish writing ...")
                time.sleep(2)
                
            break
    
    print("All saving processes terminated.")
