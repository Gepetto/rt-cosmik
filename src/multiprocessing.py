import numpy as np 
import multiprocessing as mp

def create_shared_buffer(shape, dtype):
    size = int(np.prod(shape))
    return mp.Array(np.ctypeslib.as_ctype_type(dtype), size, lock=False)

def create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE):
    # Shared resources per camera
    camera_buffers = []
    camera_timestamps = []
    camera_locks = []

    for _ in range(NUM_CAMERAS):
        camera_buffers.append(create_shared_buffer(FRAME_SHAPE, np.uint8))
        camera_timestamps.append(ts_buffer = mp.Array('c', 26))  # Fixed size for "%Y-%m-%d %H:%M:%S.%f"
        camera_locks.append(mp.Lock())
    
    return camera_buffers, camera_timestamps, camera_locks