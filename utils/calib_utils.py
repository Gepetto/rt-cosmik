from pydoc import doc
import cv2 as cv
import yaml
import glob
import numpy as np


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
    rows = 4 #number of checkerboard rows.
    columns = 6 #number of checkerboard columns.
    world_scaling = 0.106 #change this to the real world square size.
 
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
    rows = 4 #number of checkerboard rows.
    columns = 6 #number of checkerboard columns.
    world_scaling = 0.106 #change this to the real world square size.
 
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
 
            cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
            cv.imshow('img', frame1)
 
            cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
            cv.imshow('img2', frame2)
            k = cv.waitKey(1)
 
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