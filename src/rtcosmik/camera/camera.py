import time

import cv2

from rtcosmik.camera.sources import FFmpegSource
import numpy as np
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Process, Array, Value, Lock, Barrier, Event, Queue
import logging

LOGGER = logging.getLogger(__name__)

class Camera(Process):
    def __init__(self, 
                 cam_id: int,
                 shared_buffer: Array,
                 timestamp_buffer: Array, # Character array for timestamp
                 lock: Lock,
                 frame_counter: Value,
                 barrier: Barrier,
                 stop_event: Event,
                 frame_shape: tuple = (720, 1280, 3),
                 cam_fps: int = 40,
                 cam_fourcc: str = "MJPG",
                 source=None,
                 record_path=None,
                 realtime=False,
                 calibrated_event=None,
                 logger=None,
                 ):
        
        super().__init__()
        self.cam_id = cam_id
        self.shared_buffer = shared_buffer
        self.timestamp_buffer = timestamp_buffer  # For timestamp string
        self.lock = lock
        self.frame_counter = frame_counter
        self.barrier = barrier
        self.stop_event = stop_event
        
        # Video capture parameters
        self.frame_shape = frame_shape  # (height, width, channels)
        self.cam_fps = cam_fps
        self.cam_fourcc = cam_fourcc
        self.source = source
        self.record_path = record_path
        self.realtime = realtime
        self.calibrated_event = calibrated_event

        self.logger=logger or LOGGER

        # Validate timestamp buffer size (need 26 chars for format)
        if len(timestamp_buffer) != 26:
            raise ValueError("Timestamp buffer must be exactly 26 characters")

    def run(self):
        # ffmpeg rather than cv2.VideoCapture: it honours the low-latency flags
        # OpenCV ignores, and it can copy the camera's own stream to disk with no
        # re-encode. The same class replays a recording as a fake camera, which
        # is how the online path is tested without a rig.
        source = self.source if self.source is not None else f"/dev/video{self.cam_id}"
        cap = FFmpegSource(
            source,
            width=self.frame_shape[1], height=self.frame_shape[0],
            fps=self.cam_fps, input_format=("mjpeg" if self.cam_fourcc == "MJPG"
                                            else "yuyv422"),
            realtime=self.realtime, record_path=self.record_path,
            logger=self.logger).open()
        if not cap.isOpened():
            raise Exception(f"Camera {self.cam_id} could not be opened ({source}).")

        # reshape shared buffer once
        arr          = np.frombuffer(self.shared_buffer, dtype=np.uint8)
        frame_buffer = arr.reshape(self.frame_shape)

        # let everyone get to this point
        self.logger.info(f"[INFO] Camera {self.cam_id} is ready to acquire images ...")
        self.barrier.wait()

        held = None
        try:
            while not self.stop_event.is_set():
                # --- 1) all processes synchronize before grabbing next frame
                self.barrier.wait()

                # A replay must not run ahead while the model calibrates: in a
                # real session the subject stands still and waits, so the
                # recording is held on its first frame until the pipeline says
                # it is calibrated. Without this the first second of the trial
                # is consumed before tracking even starts.
                advance = (held is None or self.calibrated_event is None
                           or self.calibrated_event.is_set())

                # --- 2) tell the driver to queue the next frame
                if advance:
                    cap.grab()

                # --- 3) wait here until everyone has grabbed
                self.barrier.wait()

                # --- 4) pull the actual image out of the buffer
                if advance:
                    ret, frame = cap.retrieve()
                    if not ret:
                        continue
                    held = frame
                else:
                    # Pace the hold at the camera's own rate. Without this the
                    # loop spins as fast as the barrier allows -- ffmpeg is not
                    # being read, so nothing throttles it -- and the frame
                    # counter races through tens of thousands of duplicates
                    # before calibration finishes.
                    frame = held
                    time.sleep(1.0 / max(self.cam_fps, 1))

                # --- 5) timestamp right away
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                # --- 6) resize/check, then write under lock. ffmpeg already
                # scales to frame_shape, so this is a no-op unless something
                # upstream changed the geometry.
                resized = (frame if frame.shape == tuple(self.frame_shape)
                           else cv2.resize(frame,
                                           (self.frame_shape[1], self.frame_shape[0])))
                with self.lock:
                    np.copyto(frame_buffer, resized)
                    self.timestamp_buffer[:26] = now_str.ljust(26, "\0").encode("utf-8")
                    self.frame_counter.value += 1

        finally:
            cap.release()
            self.logger.info(f"[INFO] Camera process for camera {self.cam_id} terminated...")

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
            
                print(new_counters)
                # Optimization 1: Combine all frames into single view
                ########################################
                # Create a horizontal stack of frames
                combined_frame = np.hstack(frames)
                
                # Show combined view
                cv2.imshow(combined_window,  combined_frame)
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

