from pydoc import doc
import cv2 as cv
import yaml
import glob
import numpy as np
from scipy.spatial.transform import Rotation
import subprocess

def calibrate_camera(images_folder):
    """
    Calibrates the camera using images of a checkerboard pattern.
    Args:
        images_folder (str): Path to the folder containing checkerboard images.
    Returns:
        tuple: A tuple containing the following elements:
            - ret (float): The overall RMS re-projection error.
            - mtx (numpy.ndarray): The camera matrix.
            - dist (numpy.ndarray): The distortion coefficients.
    """

    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)
 
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # # LITTLE CHECKERBOARD
    # rows = 7 #number of checkerboard rows.
    # columns = 10 #number of checkerboard columns.
    # world_scaling = 0.025 #change this to the real world square size. Or not.

    # # BIGGER CHECKERBOARD AT LAAS
    # rows = 6 #number of checkerboard rows.
    # columns = 7 #number of checkerboard columns.
    # world_scaling = 0.108 #change this to the real world square size.

    # BIGGER CHECKERBOARD AT NUS RLS
    rows = 5 #number of checkerboard rows.
    columns = 7 #number of checkerboard columns.
    world_scaling = 0.107 #change this to the real world square size.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
 
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
 
        if ret == True:
 
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.imshow('img', frame)
            k = cv.waitKey(50)
 
            objpoints.append(objp)
            imgpoints.append(corners)
 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    # print('rmse:', ret)
    # print('camera matrix:\n', mtx)
    # print('distortion coeffs:', dist)
    cv.destroyAllWindows()
    # print('Rs:\n', rvecs)
    # print('Ts:\n', tvecs)
 
    return ret, mtx, dist

def is_order_consistent(corners1, corners2):
    """
    Checks if the corner order is consistent between two sets of detected corners.
    Args:
        corners1 (numpy.ndarray): Detected corners in the first camera image.
        corners2 (numpy.ndarray): Detected corners in the second camera image.
    Returns:
        bool: True if the order is consistent, False otherwise.
    """
    # Get the relative position of the first and last corners in each image
    top_left_1, bottom_right_1 = corners1[0][0], corners1[-1][0]
    top_left_2, bottom_right_2 = corners2[0][0], corners2[-1][0]
    
    # Compute the direction vectors
    vector_1 = bottom_right_1 - top_left_1
    vector_2 = bottom_right_2 - top_left_2
    
    # Check if the vectors have the same orientation
    angle_diff = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    return angle_diff > 0.9  # Adjust threshold as needed to ensure similar orientation

def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder_1, frames_folder_2):
    """
    Perform stereo calibration using images from two cameras.
    Args:
        mtx1 (numpy.ndarray): Camera matrix for the first camera.
        dist1 (numpy.ndarray): Distortion coefficients for the first camera.
        mtx2 (numpy.ndarray): Camera matrix for the second camera.
        dist2 (numpy.ndarray): Distortion coefficients for the second camera.
        frames_folder_1 (str): Path to the folder containing images from the first camera.
        frames_folder_2 (str): Path to the folder containing images from the second camera.
    Returns:
        tuple: A tuple containing:
            - ret (float): The overall RMS re-projection error.
            - R (numpy.ndarray): The rotation matrix between the coordinate systems of the first and second cameras.
            - T (numpy.ndarray): The translation vector between the coordinate systems of the first and second cameras.
    """

    #read the synched frames
    c1_images_names = sorted(glob.glob(frames_folder_1))
    c2_images_names = sorted(glob.glob(frames_folder_2))

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    
    # # LITTLE CHECKERBOARD
    # rows = 7 #number of checkerboard rows.
    # columns = 10 #number of checkerboard columns.
    # world_scaling = 0.025 #change this to the real world square size. Or not.

    # # BIGGER CHECKERBOARD AT LAAS
    # rows = 6 #number of checkerboard rows.
    # columns = 7 #number of checkerboard columns.
    # world_scaling = 0.108 #change this to the real world square size.

    # BIGGER CHECKERBOARD AT NUS RLS
    rows = 5 #number of checkerboard rows.
    columns = 7 #number of checkerboard columns.
    world_scaling = 0.107 #change this to the real world square size.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            if is_order_consistent(corners1, corners2):
 
                cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
                cv.imshow('img', frame1)

                cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
                cv.imshow('img2', frame2)
                k = cv.waitKey(0)

                objpoints.append(objp)
                imgpoints_left.append(corners1)
                imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
    cv.destroyAllWindows()
    return ret, R, T

def save_cam_params(mtx, dist, reproj, path):
    """
    Save camera parameters to a file.
    Args:
        mtx (numpy.ndarray): Camera matrix.
        dist (numpy.ndarray): Distortion coefficients.
        reproj (numpy.ndarray): Reprojection error.
        path (str): Path to the file where parameters will be saved.
    Returns:
        None
    """
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    cv_file.write('reproj', reproj)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

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

def save_cam_to_cam_params(mtx1, dist1, mtx2, dist2, R, T, rmse, path):
    """
    Save stereo camera calibration parameters to a file.
    Args:
        mtx1 (numpy.ndarray): Camera matrix for the first camera.
        dist1 (numpy.ndarray): Distortion coefficients for the first camera.
        mtx2 (numpy.ndarray): Camera matrix for the second camera.
        dist2 (numpy.ndarray): Distortion coefficients for the second camera.
        R (numpy.ndarray): Rotation matrix between the two cameras.
        T (numpy.ndarray): Translation vector between the two cameras.
        rmse (float): Root Mean Square Error of the calibration.
        path (str): Path to the file where the parameters will be saved.
    Returns:
        None
    """
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    cv_file.write('K1', mtx1)
    cv_file.write('D1', dist1)
    cv_file.write('K2', mtx2)
    cv_file.write('D2', dist2)
    cv_file.write('R', R)
    cv_file.write('T', T)
    cv_file.write('rmse', rmse)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

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

# Function to detect the ArUco marker and estimate the camera pose
def get_aruco_pose(frame, camera_matrix, dist_coeffs, detector, marker_size):
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    
    # Detect the markers in the image
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None and len(corners) > 0:
        # Extract the corners of the first detected marker for pose estimation
        # Reshape the first marker's corners for solvePnP
        corners_for_solvePnP = corners[0].reshape(-1, 2)
        
        # Estimate the pose of each marker
        _, R, t = cv.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv.SOLVEPNP_IPPE_SQUARE)
        
        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv.Rodrigues(R)
        
        # Now we can form the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = t.flatten()
        
        return transformation_matrix, corners[0], R, t
    else:
        return None, None, None, None
    
def get_relative_pose_robot_in_cam(images_folder,camera_matrix,dist_coeffs, detector, marker_size):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    assert len(images)==4, "number of images to get robot base must be 4"
    
    wand_local = np.array([[-0.00004],[0.262865],[-0.000009]])

    wand_pos_cam_frame = []
    for ii, frame in enumerate(images):
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        
        # Detect the markers in the image
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(corners) > 0:
            # Extract the corners of the first detected marker for pose estimation
            # Reshape the first marker's corners for solvePnP
            corners_for_solvePnP = corners[0].reshape(-1, 2)
            
            # Estimate the pose of each marker
            _, R, t = cv.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv.SOLVEPNP_IPPE_SQUARE)
            
            # Convert the rotation vector to a rotation matrix
            rotation_matrix, _ = cv.Rodrigues(R)
            
            # Now we can form the transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = t.flatten()
        
            wand_pos_cam_frame.append((t+rotation_matrix@wand_local).flatten())

    P1 = cam_center_robot = (wand_pos_cam_frame[0]+wand_pos_cam_frame[1])/2
    P2 = wand_pos_cam_frame[3]
    P3 = wand_pos_cam_frame[0]

    P1P2 = P2-P1
    P1P3 = P3-P1

    Vz = np.cross(P1P2, P1P3)
    Vy = np.cross(Vz,P1P2)
    Vx = P1P2

    x_axis = Vx/np.linalg.norm(Vx)
    y_axis = Vy/np.linalg.norm(Vy)
    z_axis = Vz/np.linalg.norm(Vz)

    # 1. Construct the rotation matrix
    cam_R_robot = np.column_stack((x_axis, y_axis, z_axis))

    return cam_center_robot, cam_R_robot

def get_relative_pose_human_in_cam(images_folder,camera_matrix,dist_coeffs, detector, marker_size):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    assert len(images)==3, "number of images to get robot base must be 4"
    
    wand_local = np.array([[-0.00004],[0.262865],[-0.000009]])

    wand_pos_cam_frame = []
    for ii, frame in enumerate(images):
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        
        # Detect the markers in the image
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(corners) > 0:
            # Extract the corners of the first detected marker for pose estimation
            # Reshape the first marker's corners for solvePnP
            corners_for_solvePnP = corners[0].reshape(-1, 2)
            
            # Estimate the pose of each marker
            _, R, t = cv.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv.SOLVEPNP_IPPE_SQUARE)
            
            # Convert the rotation vector to a rotation matrix
            rotation_matrix, _ = cv.Rodrigues(R)
            
            # Now we can form the transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = t.flatten()
        
            wand_pos_cam_frame.append((t+rotation_matrix@wand_local).flatten())

    P1 = cam_center_human = wand_pos_cam_frame[0]
    P2 = wand_pos_cam_frame[1]
    P3 = wand_pos_cam_frame[2]

    P1P2 = P2-P1
    P1P3 = P3-P1

    Vy = np.cross(P1P2,P1P3)
    Vz = np.cross(P1P2,Vy)
    Vx = P1P2

    x_axis = Vx/np.linalg.norm(Vx)
    y_axis = Vy/np.linalg.norm(Vy)
    z_axis = Vz/np.linalg.norm(Vz)

    # 1. Construct the rotation matrix
    cam_R_human = np.column_stack((x_axis, y_axis, z_axis))

    return cam_center_human, cam_R_human

def get_relative_pose_world_in_cam(images_folder,camera_matrix,dist_coeffs, detector, marker_size):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    translation_vectors = []
    quaternions = []

    for ii, frame in enumerate(images):
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        
        # Detect the markers in the image
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(corners) > 0:
            # Extract the corners of the first detected marker for pose estimation
            # Reshape the first marker's corners for solvePnP
            corners_for_solvePnP = corners[0].reshape(-1, 2)
            
            # Estimate the pose of each marker
            _, R, t = cv.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv.SOLVEPNP_IPPE_SQUARE)
            
            rotation_matrix = cv.Rodrigues(R)[0]

            translation_vectors.append(t)
            quaternions.append(Rotation.from_matrix(rotation_matrix).as_quat())

    mean_translation = np.mean(np.array(translation_vectors),axis=0)

    for i in range(1, len(quaternions)):
        if np.dot(quaternions[0], quaternions[i]) < 0:
            quaternions[i] = -quaternions[i]
    
    # Normalize the quaternions (if not already normalized)
    quaternions = np.array(quaternions)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)

    # Compute the mean quaternion
    mean_quaternion = np.mean(quaternions, axis=0)
    mean_quaternion /= np.linalg.norm(mean_quaternion)  # Normalize the result

    # Step 4: Convert the averaged quaternion back to a rotation matrix
    mean_rotation_matrix = Rotation.from_quat(mean_quaternion).as_matrix()

    return mean_translation, mean_rotation_matrix
    
# Function to save the translation vector to a YAML file
def save_pose_rpy_to_yaml(translation_vector, rotation_sequence, filename):

    # Ensure inputs are 1D or column vectors of correct shape
    assert translation_vector.shape in [(3,), (3, 1)], "Translation vector must have shape (3,) or (3, 1)"
    assert rotation_sequence.shape in [(3,), (3, 1)], "Rotation sequence must have shape (3,) or (3, 1)"
    
    # Prepare the data to be saved in YAML format
    data = {
        'translation_vector': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': translation_vector.flatten().tolist()
        },
        'rotation_rpy': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': rotation_sequence.flatten().tolist()
        }
    }
    
    # Write to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)  # Use block style for readability

# Function to detect the ArUco marker and estimate the camera pose
def get_camera_pose(frame, camera_matrix, dist_coeffs, detector, marker_size):
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    
    # Detect the markers in the image
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None and len(corners) > 0:
        # Extract the corners of the first detected marker for pose estimation
        # Reshape the first marker's corners for solvePnP
        corners_for_solvePnP = corners[0].reshape(-1, 2)
        
        # Estimate the pose of each marker
        _, R, t = cv.solvePnP(marker_points, corners_for_solvePnP, camera_matrix, dist_coeffs, False, cv.SOLVEPNP_IPPE_SQUARE)

        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv.Rodrigues(R)
        
        # Now we can form the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = t.flatten()
        
        return transformation_matrix, corners[0], R, t
    else:
        return None, None, None, None

# Function to save the rotation matrix and translation vector to a YAML file
def save_pose_matrix_to_yaml(rotation_matrix, translation_vector, filename):
    # Prepare the data to be saved in YAML format
    data = {
        'rotation_matrix': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': rotation_matrix.flatten().tolist()
        },
        'translation_vector': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': translation_vector.flatten().tolist()
        }
    }
    
    # Write to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file)

def list_available_cameras(max_cameras=10):
    available_cameras = []
    for index in range(max_cameras):
        cap = cv.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()  # Release the camera after checking
    return available_cameras

def list_cameras_with_v4l2():
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