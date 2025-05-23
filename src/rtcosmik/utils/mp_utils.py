import numpy as np
import multiprocessing as mp
import ctypes
from pynput import keyboard
import os
def create_shared_buffer(shape, dtype):
    """Create shared memory buffer for camera frames"""
    # Convert numpy dtype to ctype
    ctype = np.ctypeslib.as_ctypes_type(dtype)  # Fix typo: as_ctypes_type
    size = int(np.prod(shape))
    return mp.Array(ctype, size, lock=False)

def create_camera_shared_ressources(num_cameras, frame_shape):
    """Create shared resources for camera processes"""
    camera_buffers = []
    camera_timestamps = []
    camera_locks = []
    frame_counters = [] 

    barrier = mp.Barrier(num_cameras)
    stop_event = mp.Event()
    
    for _ in range(num_cameras):
        # Create frame buffer
        camera_buffers.append(create_shared_buffer(frame_shape, np.uint8))
        # Create timestamp buffer
        camera_timestamps.append(mp.Array('c', 26))  # 26-character buffer
        camera_locks.append(mp.Lock())
        frame_counters.append(mp.Value('L', 0))  # Unsigned long counter
    
    return camera_buffers, camera_timestamps, camera_locks, frame_counters, barrier, stop_event

def create_pose_estimator_shared_ressources(num_cameras):
    # Result queues (one per camera)
    queues = [mp.Queue(maxsize=30) for _ in range(num_cameras)]
    barrier = mp.Barrier(num_cameras)
    return queues, barrier

def create_pipeline_shared_ressources():
    return [mp.Queue(maxsize=30) for _ in range(3)]

def create_pipeline_shared_resources_with_buffers():
    # Create locks for safe access (optional but recommended)
    locks = [mp.Lock() for _ in range(4)]

    # Shared counters: 2 integers
    shared_counters = mp.Array(ctypes.c_int, 2)

    # Shared keypoints: 26 * 3 floats
    shared_kp = mp.Array(ctypes.c_float, 26 * 3)

    shared_mks = mp.Array(ctypes.c_float, 43 * 3)

    # Shared q output: 32 floats
    shared_q = mp.Array(ctypes.c_float, 32)

    buffers = {
        'counters': shared_counters,
        'keypoints': shared_kp,
        'markers': shared_mks,
        'q': shared_q,
        'locks': locks
    }

    return buffers

def create_udp_buffer(mks_names):
    valid_event = mp.Event()
    cam_event = mp.Event()
    shared_ts     = mp.Array('c', 26, lock=False)
    shared_values = mp.Array('f', len(mks_names)*3, lock=False)
    lock          = mp.Lock()
    return shared_ts,shared_values,lock,cam_event,valid_event


def keyboard_listener(saving_flag):
    def on_press(key):
        try:
            if key.char == 's':
                print("[Main] Start saving data")
                with saving_flag.get_lock():
                    saving_flag.value = True
            elif key.char == 'q':
                print("[Main] Stop saving data")
                with saving_flag.get_lock():
                    saving_flag.value = False
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener

def ensure_directory_exists(path):
    """Ensure that the directory at `path` exists. Create it if it doesn't."""
    os.makedirs(path, exist_ok=True)