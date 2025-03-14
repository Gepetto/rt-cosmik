import subprocess
import numpy as np
import os  
import cv2 as cv
import yaml

def list_cameras():
    """
    Use v4l2-ctl to list all connected cameras and their device paths.
    Returns a dictionary of camera indices and associated device names.
    """
    cameras = {}
    try:
        # Get list of video devices
        output = subprocess.check_output("v4l2-ctl --list-devices", shell=True).decode("utf-8")
        devices = output.split("\n\n")  # Separate different devices
        for device in devices:
            lines = device.split("\n")
            if len(lines) > 1:
                device_name = lines[0].strip()
                video_path = lines[1].strip()
                if "/dev/video" in video_path:
                    index = int(video_path.split("video")[-1])
                    cameras[index] = device_name
    except Exception as e:
        print("Error using v4l2-ctl:", e)
    return cameras

def get_cameras_params(K1, D1, K2, D2, R, T):
    dict_cam = {
        "cam1": {
            "mtx":np.array(K1),
            "dist":D1,
            "rotation":np.eye(3),
            "translation":[
                0.,
                0.,
                0.,
            ],
        },
        "cam2": {
            "mtx":np.array(K2),
            "dist":D2,
            "rotation":R,
            "translation":T,
        },
    }

    rotations=[]
    translations=[]
    dists=[]
    mtxs=[]
    projections=[]

    for cam in dict_cam :
        rotation=np.array(dict_cam[cam]["rotation"])
        rotations.append(rotation)
        translation=np.array([dict_cam[cam]["translation"]]).reshape(3,1)
        translations.append(translation)
        projection = np.concatenate([rotation, translation], axis=-1)
        projections.append(projection)
        dict_cam[cam]["projection"] = projection
        dists.append(dict_cam[cam]["dist"])
        mtxs.append(dict_cam[cam]["mtx"])
    return mtxs, dists, projections, rotations, translations

def load_cam_params(path):
    """
    Loads camera parameters from a given file.
    Args:
        path (str): The path to the file containing the camera parameters.
    Returns:
        tuple: A tuple containing the camera matrix and distortion matrix.
            - camera_matrix (numpy.ndarray): The camera matrix.
            - dist_matrix (numpy.ndarray): The distortion matrix.
    """
    
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return camera_matrix, dist_matrix


def load_cam_to_cam_params(path):
    """
    Loads camera-to-camera calibration parameters from a given file.
    This function reads the rotation matrix (R) and translation vector (T) from a 
    specified file using OpenCV's FileStorage. The file should contain these parameters 
    stored under the keys 'R' and 'T'.
    Args:
        path (str): The file path to the calibration parameters.
    Returns:
        tuple: A tuple containing:
            - R (numpy.ndarray): The rotation matrix.
            - T (numpy.ndarray): The translation vector.
    """
    
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    R = cv_file.getNode('R').mat()
    T = cv_file.getNode('T').mat()

    cv_file.release()
    return R, T


def load_cam_pose(filename):
    """
        Load the rotation matrix and translation vector from a YAML file.
        Args:
            filename (str): The path to the YAML file.
        Returns:
            rotation_matrix (np.ndarray): The 3x3 rotation matrix.
            translation_vector (np.ndarray): The 3x1 translation vector.
    """

    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    rotation_matrix = np.array(data['rotation_matrix']['data']).reshape((3, 3))
    translation_vector = np.array(data['translation_vector']['data']).reshape((3, 1))
    
    return rotation_matrix, translation_vector

def load_cam_pose_rpy(filename):
    """
        Load the euler angles and translation vector from a YAML file.
        Args:
            filename (str): The path to the YAML file.
        Returns:
            euler (np.ndarray): The 3x1 euler sequence.
            translation_vector (np.ndarray): The 3x1 translation vector.
    """

    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    euler = np.array(data['rotation_rpy']['data']).reshape((3, 1))
    translation_vector = np.array(data['translation_vector']['data']).reshape((3, 1))
    
    return euler, translation_vector


def load_camera_parameters(config_path):
    """Load intrinsic and extrinsic camera parameters."""
    K1, D1 = load_cam_params(os.path.join(config_path, "c1_params_color.yaml"))
    K2, D2 = load_cam_params(os.path.join(config_path, "c2_params_color.yaml"))
    R, T = load_cam_to_cam_params(os.path.join(config_path, "c1_to_c2_params_color.yaml"))
    return get_cameras_params(K1, D1, K2, D2, R, T)

def load_world_transformation(config_path):
    """Load world transformation matrix."""
    cam_R1_world, cam_T1_world = load_cam_pose(os.path.join(config_path, "camera1_pose.yaml"))
    world_R1_cam = cam_R1_world.T
    world_T1_cam = -world_R1_cam @ cam_T1_world
    return world_R1_cam, world_T1_cam.reshape((3,))
