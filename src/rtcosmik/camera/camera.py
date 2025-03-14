import cv2
import numpy as np
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Process, Array, Value, Lock, Barrier, Event

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
                 cam_fps: int = None,
                 cam_fourcc: str = "MJPG"):
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

        # Validate timestamp buffer size (need 26 chars for format)
        if len(timestamp_buffer) != 26:
            raise ValueError("Timestamp buffer must be exactly 26 characters")

    def run(self):
        # Initialize camera once at start
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


        # Correct frame buffer reshaping
        arr = np.frombuffer(self.shared_buffer, dtype=np.uint8)
        try:
            frame_buffer = arr.reshape(self.frame_shape)
        except ValueError:
            actual_size = arr.size
            expected_size = np.prod(self.frame_shape)
            raise RuntimeError(
                f"Buffer size mismatch. Expected {expected_size} elements, "
                f"got {actual_size}. Check frame_shape: {self.frame_shape}"
            )
        
        self.barrier.wait()

        # Main capture loop
        try: 
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue  # Exit on failure

                # Validate frame before processing
                if frame.size != np.prod(self.frame_shape):
                    print(f"Frame size mismatch: {frame.shape} vs {self.frame_shape}")
                    continue

                # Generate timestamp
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                
                # Resize and ensure 3 channels
                resized = cv2.resize(frame, (self.frame_shape[1], self.frame_shape[0]))
                
                # Update shared memory
                with self.lock:
                    np.copyto(frame_buffer, resized)
                    self.timestamp_buffer[:26] = timestamp_str.ljust(26, '\0').encode('utf-8')
                    # print(self.frame_counter.value)
                    self.frame_counter.value += 1

        finally:
            cap.release()
            print(f"Process for Camera {self.cam_id} terminated.")

class DisplayConsumer(Process):
    def __init__(self, 
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
        
    def run(self):
        window_names = [f'Camera {i}' for i in range(self.num_cameras)]
        
        # Optimization 1: Create a single window for all cameras
        combined_window = "Multi-Camera View"
        
        try: 
            while not self.stop_event.is_set():
                frames = []
                
                # Collect frames from all cameras
                for i in range(self.num_cameras):
                    with self.camera_locks[i]:
                        arr = np.frombuffer(self.camera_buffers[i], dtype=np.uint8)
                        frame = arr.reshape(self.frame_shape).copy()
                        # Get current timestamp
                        timestamp = bytes(self.timestamp_buffers[i][:]).decode().strip('\x00')

                    # Optimization 2: Add timestamp overlay
                    ########################################  
                    # Add text overlay (white text with black background)
                    cv2.putText(frame, timestamp, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                            (0,0,0), 4, lineType=cv2.LINE_AA)
                    cv2.putText(frame, timestamp, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                            (255,255,255), 2, lineType=cv2.LINE_AA)
                    ########################################
                    
                    frames.append(frame)

                # Optimization 1: Combine all frames into single view
                ########################################
                # Create a horizontal stack of frames
                combined_frame = np.hstack(frames)
                
                # Show combined view
                cv2.imshow(combined_window, combined_frame)
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