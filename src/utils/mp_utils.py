import numpy as np
import multiprocessing as mp

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
    
    for _ in range(num_cameras):
        # Create frame buffer
        camera_buffers.append(create_shared_buffer(frame_shape, np.uint8))
        # Create timestamp buffer
        camera_timestamps.append(mp.Array('c', 26))  # 26-character buffer
        camera_locks.append(mp.Lock())
    
    return camera_buffers, camera_timestamps, camera_locks
