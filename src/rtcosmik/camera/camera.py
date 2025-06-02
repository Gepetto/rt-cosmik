import cv2
import numpy as np
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Process, Array, Value, Lock, Barrier, Event, Queue
import logging
import select
import socket
import csv 
import time
import os
from src.rtcosmik.utils.linear_algebra_utils import concat_frames
class Camera(Process):
    def __init__(self, 
                 cam_id: int,
                 shared_buffer: Array,
                 timestamp_buffer: Array, # Character array for timestamp
                 lock: Lock,
                 frame_counter: Value,
                 barrier: Barrier,
                 stop_event: Event,
                 cam_event :Event,
                 save_dir: str,
                 frame_shape: tuple = (720, 1280, 3),
                 cam_fps: int = None,
                 cam_fourcc: str = "MJPG",
                 saving_flag = False):
        
        super().__init__()
        self.cam_id = cam_id
        self.shared_buffer = shared_buffer
        self.timestamp_buffer = timestamp_buffer  # For timestamp string
        self.lock = lock
        self.frame_counter = frame_counter
        self.barrier = barrier
        self.stop_event = stop_event
        self.cam_event = cam_event
        self.save_dir=save_dir
        self.saving_flag = saving_flag
        
        # Video capture parameters
        self.frame_shape = frame_shape  # (height, width, channels)
        self.cam_fps = cam_fps
        self.cam_fourcc = cam_fourcc

        # Validate timestamp buffer size (need 26 chars for format)
        if len(timestamp_buffer) != 26:
            raise ValueError("Timestamp buffer must be exactly 26 characters")

    def run(self):
        cap = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise Exception(f"Camera {self.cam_id} could not be opened.")
        
        # Set camera properties once if specified
        if self.cam_fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cam_fourcc))
        if self.frame_shape:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_shape[0])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_shape[1])
        if self.cam_fps:
            cap.set(cv2.CAP_PROP_FPS, self.cam_fps)

        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # reshape shared buffer once
        arr          = np.frombuffer(self.shared_buffer, dtype=np.uint8)
        frame_buffer = arr.reshape(self.frame_shape)

        # let everyone get to this point
        self.barrier.wait()
        self.saved_frames = []           
        self.saved_timestamps = []

        try:
            while not self.stop_event.is_set():
                # --- 1) all processes synchronize before grabbing next frame
                self.barrier.wait()

                # --- 2) tell the driver to queue the next frame
                cap.grab()
                if self.cam_id == 0:
                    self.cam_event.set()

                # --- 3) wait here until everyone has grabbed
                # self.barrier.wait()

                # --- 4) pull the actual image out of the buffer
                ret, frame = cap.retrieve()
                if not ret:
                    continue

                # --- 5) timestamp right away
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                # --- 6) resize/check, then write under lock
                resized = cv2.resize(frame, (self.frame_shape[1], self.frame_shape[0]))
                with self.lock:
                    self.barrier.wait()
                    np.copyto(frame_buffer, resized)
                    self.timestamp_buffer[:26] = now_str.ljust(26, "\0").encode("utf-8")
                    self.frame_counter.value += 1

                # optional: wait here if you need a post‑write barrier
                # self.barrier.wait()
                if self.saving_flag.value:
                    self.saved_frames.append(resized.copy())
                    self.saved_timestamps.append(now_str)

        finally:
            cap.release()
            print(f"Camera {self.cam_id} process exiting.")

            #  # --- Save all frames at the end ---
            print(os.path.join(self.save_dir, f"camera_{self.cam_id}.mp4"))
            out = cv2.VideoWriter(
                os.path.join(self.save_dir, f"camera_{self.cam_id}.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.cam_fps,
                (self.frame_shape[1], self.frame_shape[0])
            )

            for frame in self.saved_frames:
                out.write(frame)
            out.release()
            print(f"Camera {self.cam_id} video saved.")

            with open(os.path.join(self.save_dir, f"camera_{self.cam_id}_timestamps.csv"), mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["frame_index", "timestamp"])
                for idx, ts in enumerate(self.saved_timestamps):
                    writer.writerow([idx, ts])

class DisplayConsumer(Process):
    def __init__(self, 
                 frame_counters,
                 camera_buffers, 
                 camera_locks, 
                 timestamp_buffers, 
                 stop_event, 
                 frame_shape, 
                 num_cameras):
        super().__init__()
        self.camera_buffers = camera_buffers
        self.camera_locks = camera_locks
        self.timestamp_buffers = timestamp_buffers
        self.frame_shape = frame_shape  # (height, width, channels)
        self.num_cameras = num_cameras
        self.stop_event = stop_event

        self.last_frame_counters = [0] * self.num_cameras
        self.frame_counters = frame_counters
        
    def run(self):
        window_names = [f'Camera {i}' for i in range(self.num_cameras)]
        
        # Optimization 1: Create a single window for all cameras
        combined_window = "Multi-Camera View"
        
        try: 
            while not self.stop_event.is_set():
                frames = []
                keypoints_list = []
                new_counters = []
                for i, (lock, buffer, cam_ts, frame_counter) in enumerate(zip(self.camera_locks, self.camera_buffers, self.timestamp_buffers, self.frame_counters)):
                    with lock:
                        #  Only accept data if this camera has produced a new frame
                        if frame_counter.value > self.last_frame_counters[i]:
                            # Read and copy shared data atomically
                            arr = np.frombuffer(buffer, dtype=np.uint8)
                            frame = arr.reshape(self.frame_shape).copy()
                            # Get current timestamp
                            timestamp = bytes(cam_ts[:]).decode().strip('\x00')

                            if timestamp == '': # empty data
                                continue
                            else:
                                frames.append(frame)
                            new_counters.append(frame_counter.value)
                if len(frames)!=self.num_cameras:
                    continue
                self.last_frame_counters = new_counters.copy()
            
                # print(new_counters)
                # Optimization 1: Combine all frames into single view
                ########################################
                # Create a horizontal stack of frames
                # combined_frame = np.hstack(frames)
                combined_frame = concat_frames(frames)

                scale = 0.5
                resized_frame = cv2.resize(combined_frame, (0, 0), fx=scale, fy=scale)

                # Show resized view
                cv2.imshow(combined_window, resized_frame)
                ########################################
                
                # Original individual windows display (comment out when using combined view)
                # for i, frame in enumerate(frames):
                #     if frame.shape[2] == 3:
                #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #     cv2.imshow(window_names[i], frame)

                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:        
            cv2.destroyAllWindows()
            print("Display process terminated.")

