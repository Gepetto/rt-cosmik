import sys
import os
import cv2
import time
import csv
import numpy as np
import ctypes
from pynput import keyboard
from datetime import datetime
import subprocess
from multiprocessing import Process, Barrier, Queue, Event, Array
from threading import BrokenBarrierError
from queue import Empty

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")) # src dir

from src.rtcosmik.camera.cam_utils import list_cameras
from src.rtcosmik.config_loader import settings
from src.rtcosmik.utils.linear_algebra_utils import concat_frames



def camera_process(queue, barrier, idx_cam, stop_event, current_frame):

    timestamps = []

    cap = cv2.VideoCapture(idx_cam, cv2.CAP_V4L2)
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

    with open(os.path.join(settings.SAVE_DIR, f"camera_{idx_cam}_timestamps.csv"), mode='w', newline='') as f:
        timestamps_writer = csv.writer(f)
        timestamps_writer.writerow(["frame_index", "timestamp"])
        for frame_idx, ts in enumerate(timestamps):
            timestamps_writer.writerow([str(frame_idx), ts])

    print(f"Camera {idx_cam} process terminated.")


def displays_process(current_frames, idx_cams):

    while True:
        
        for current_frame, idx_cam in zip(current_frames, idx_cams):
            globals()[f"displayed_frame_{idx_cam}"] = np.frombuffer(current_frame.get_obj(), dtype=np.uint8).reshape(settings.height, settings.width, 3)

        concatenated_frame = concat_frames([globals()[f"displayed_frame_{idx_cam}"] for idx_cam in idx_cams])
        concatenated_frame_resized = cv2.resize(concatenated_frame, (1440, 900))
        cv2.imshow('Streaming View', concatenated_frame_resized)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
    
    print("Display process terminated.")


def saver_process(queue, idx_cam, stop_event):

    fourcc = cv2.VideoWriter_fourcc(*settings.fourcc)
    frame_size = (settings.width, settings.height)
    video_filename = os.path.join(settings.SAVE_DIR, f"camera_{idx_cam}.avi")
    fps = settings.fs

    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

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

    os.makedirs(settings.SAVE_DIR, exist_ok=True)

    cameras = list_cameras()

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
        args=([globals()[f"current_frame_{idx_cam}"] for idx_cam in cameras.keys()], cameras.keys()),
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

    # # Trouver tous les processus Python
    # ps_output = subprocess.check_output(['ps', 'aux'])
    # pids = []
    # for line in ps_output.decode('utf-8').split('\n'):
    #     if 'python scripts/run_cameras_refacto.py' in line:
    #         # Extraire le PID
    #         pid = int(line.split()[1])
    #         pids.append(pid)

    # # Tuer les processus
    # for pid in pids:
    #     try:
    #         # Envoyer le signal SIGTERM pour terminer le processus
    #         subprocess.call(['kill', str(pid)])
    #     except Exception as e:
    #         print(f"Erreur lors de la tentative de tuer le processus {pid}: {e}")
