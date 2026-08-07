import subprocess
import numpy as np
import os  
import cv2 as cv
import yaml

def list_cameras():
    """
    Use v4l2-ctl to list all connected cameras and their device paths.
    Return:
        cameras: dictionary of camera indices and associated device names.
    """
    cameras = {}
    try:
        output = subprocess.check_output("v4l2-ctl --list-devices", shell=True).decode("utf-8")
        devices = output.split("\n\n")
        for device in devices:
            lines = device.split("\n")
            if len(lines) > 1:
                device_name = lines[0].strip()
                video_path = lines[1].strip()
                if "/dev/video" in video_path:
                    index = int(video_path.split("video")[-1])
                    cap = cv.VideoCapture(index, cv.CAP_V4L2)
                    if cap.isOpened():
                        cameras[index] = device_name
                    cap.release()
    except Exception as e:
        print("Error using v4l2-ctl:", e)
    return cameras

def rt_to_homogeneous(R, translation_matrix):
    """
    Convert (R, translation_matrix) to a 4x4 homogeneous transformation matrix.
    Parameters: 
        R (numpy.ndarray): rotation matrix (3x3)
        translation_matrix (numpy.ndarray): translation matrix (3x1)
    Returns:
        T (numpy.ndarray): homogeneous translation matrix (4x4)
    """
    
    translation_matrix = translation_matrix.reshape(3,)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation_matrix
    return T

def invert_homogeneous(T):
    """
    Invert a 4x4 homogeneous transformation matrix.
    Parameter: 
        T (numpy.ndarray): homogeneous translation matrix (4x4)
    Returns:
        T_inv (numpy.ndarray): the inverse matrix of T
    """
    
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def decompose_homogeneous(T):
    """
    Extract (R, translation_matrix) from T a 4x4 homogeneous matrix.
    Parameter:
        T (numpy.ndarray): a 4x4 homogeneous matrix
    Returns:
        R (numpy.ndarray): Rotation matrix (3x3) from transformation matrix T
        translation_matrix (numpy.ndarray): translation_matrix (3x1) from transformation matrix T
    """
    
    R = T[:3, :3]
    translation_matrix = T[:3, 3]
    return R, translation_matrix


def load_cam_params(path):
    """
    Loads camera parameters from a given file.
    Parameter:
        path (str): The path to the file containing the camera parameters.
    Returns:
        tuple: A tuple containing the camera matrix and distortion matrix.
            - camera_matrix (numpy.ndarray): The camera matrix.
            - distortion_matrix (numpy.ndarray): The distortion matrix.
    """
    
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve otherwise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    distortion_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return camera_matrix, distortion_matrix


def load_cam_to_cam_params(path):
    """
    Loads camera-to-camera calibration parameters from a given file.
    This function reads the rotation matrix R (3x3) and translation matrix (3x1) from a 
    specified file using OpenCV's FileStorage. The file should contain these parameters 
    stored under the keys 'R' and 'T'.
    Parameter:
        path (str): The file path to the calibration parameters.
    Returns:
        tuple: A tuple containing:
            - R (numpy.ndarray): The rotation matrix.
            - translation_matrix (numpy.ndarray): The translation matrix.
    """
    
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve otherwise we only get a
    # FileNode object back instead of a matrix
    R = cv_file.getNode('R').mat()
    translation_matrix = cv_file.getNode('T').mat()

    cv_file.release()
    return R, translation_matrix

def load_global_cam_params(path, cam_index):
    """
    Loads the global camera transformation parameters for a specified camera
    from a YAML file. This function reads the rotation matrix R (3x3) and translation
    matrix (3x1) stored under the keys 'camera_{cam_index}_R' and 'camera_{cam_index}_T'.
    
    Parameters:
        path (str): The file path to the YAML file.
        cam_index (int): The camera index to load.
        
    Returns:
        tuple: A tuple containing:
            - R (numpy.ndarray): The rotation matrix.
            - translation_matrix (numpy.ndarray): The translation vector.
    """
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    R = cv_file.getNode(f'camera_{cam_index}_R').mat()
    translation_matrix = cv_file.getNode(f'camera_{cam_index}_T').mat()
    cv_file.release()
    return R, translation_matrix


def load_cam_pose(filename):
    """
        Load the rotation matrix (3x3) and translation matrix (3x1) from a YAML file.
        Parameters:
            filename (str): The path to the YAML file.
        Returns:
            rotation_matrix (np.ndarray): The 3x3 rotation matrix.
            translation_matrix (np.ndarray): The 3x1 translation matrix.
    """

    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    rotation_matrix = np.array(data['rotation_matrix']['data']).reshape((3, 3))
    translation_matrix = np.array(data['translation_vector']['data']).reshape((3, 1))
    
    return rotation_matrix, translation_matrix

def load_camera_parameters(config_path, num_cameras=2):
    """
    Loads camera parameters for 2, or 4 cameras.
    
    Parameters:
        config_path (str): Path to the configuration directory.
        num_cameras (int): Total number of physical cameras (must be even and >= 2).

    Returns:
        mtx_list (list of np.ndarray): Camera matrices, each with shape (3, 3)
        dist_list (list of np.ndarray): Camera distortion coefficients, each with shape (1, 5)
        projection_list (list of np.ndarray): Projection matrices [R | T], each with shape (3, 4).
        rotation_list (list of np.ndarray): Camera rotation matrices, each with shape (3, 3).
        translation_list (list of np.ndarray): Camera translation matrices, each with shape (3, 1).
    """

    
    mtx_list = []
    dist_list = []
    rotation_list = []
    translation_list = []
    projection_list = []

    for i in range(num_cameras):
        cam_nb = i * 2
        
        K, D = load_cam_params(os.path.join(config_path, f"c{cam_nb}_params_color.yaml"))
        mtx_list.append(np.array(K))
        dist_list.append(D)

        if i == 0:
            # The first camera (c0) acts as the world origin
            R = np.eye(3)
            translation = np.zeros((3, 1))
        else:
            # All subsequent cameras read their file relative to c0
            extrinsic_file = os.path.join(config_path, f"c0_to_c{cam_nb}_params_color.yaml")
            R, translation = load_cam_to_cam_params(extrinsic_file)
            
            R = np.array(R)
            translation = np.array(translation).reshape(3, 1) # Force standard 3x1 vertical vector layout

        rotation_list.append(R)
        translation_list.append(translation)

        projection = np.concatenate([R, translation], axis=-1)
        projection_list.append(projection)

    return mtx_list, dist_list, projection_list, rotation_list, translation_list

def load_world_transformation(config_path):
    """
    Load world transformation matrix from a file.
    Parameters:
        config_path (str): The path to the configuration file.
    Returns:
        world_R1_cam (np.ndarray): The rotation matrix from camera 0 to the world.
        world_translation1_cam (np.ndarray): The translation matrix from camera 0 to the world.
    """
    world_R1_cam, world_translation1_cam = load_cam_pose(os.path.join(config_path, "camera0_pose.yaml"))
    return world_R1_cam, world_translation1_cam.reshape((3,))