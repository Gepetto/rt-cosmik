import cv2 as cv
import yaml
import glob
import numpy as np


def calibrate_camera(images_folder):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)
 
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    rows = 6 #number of checkerboard rows.
    columns = 10 #number of checkerboard columns.
    # columns = 7 #number of checkerboard columns.
    world_scaling = 0.025 #change this to the real world square size.
    # world_scaling = 0.108 #change this to the real world square size.
 
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
 
    # rows = 7 #number of checkerboard rows.
    # columns = 10 #number of checkerboard columns.
    # world_scaling = 0.025 #change this to the real world square size. Or not.

    rows = 6 #number of checkerboard rows.
    # columns = 10 #number of checkerboard columns.
    columns = 7 #number of checkerboard columns.
    # world_scaling = 0.025 #change this to the real world square size.
    world_scaling = 0.108 #change this to the real world square size.
 
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
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    cv_file.write('reproj', reproj)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_cam_pose(filename):
    """
        Load the rotation matrix and translation vector from a YAML file.
        
        Parameters:
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
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return camera_matrix, dist_matrix

def save_cam_to_cam_params(mtx1, dist1, mtx2, dist2, R, T, rmse, path):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
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
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    R = cv_file.getNode('R').mat()
    T = cv_file.getNode('T').mat()

    cv_file.release()
    return R, T



# task_name = "right_front"
# expe_no = "8"
# trial_no = "0"

# # # calibrate_<camera_aruco_markers("../Sensors/UDPClient/Expe_2Vimus_patients/expe_0/trial_1/anat_calib/raw/cam_2/img_*")
# c1_imgs_path = "./images_calib_cam_1/" + expe_no + "_" + trial_no + "/*"
# c2_imgs_path = "./images_calib_cam_2/" + expe_no + "_" + trial_no + "/*"
# c1_params_path = "./cam_params/c1_params_" + expe_no + "_" + trial_no + ".yml"
# c2_params_path = "./cam_params/c2_params_" + expe_no + "_" + trial_no + ".yml"
# c1_to_c2_params_path = "./cam_params/c1_to_c2_params" + expe_no + "_" + trial_no + ".yml"

# reproj1, mtx1, dist1 = calibrate_camera(images_folder = c1_imgs_path)
# save_cam_params(mtx1, dist1, reproj1, c1_params_path)
# reproj2, mtx2, dist2 = calibrate_camera(images_folder = c2_imgs_path)
# save_cam_params(mtx2, dist2, reproj2, c2_params_path)

# mtx_1, dist_1 = load_cam_params(c1_params_path)
# mtx_2, dist_2 = load_cam_params(c2_params_path)
# rmse, R, T = stereo_calibrate(mtx_1, dist_1, mtx_2, dist_2, c1_imgs_path, c2_imgs_path)

# print("computed R :")
# print(R)

# print("computed T : ")
# print(T)

# print("rmse : ")
# print(rmse)

# save_cam_to_cam_params(mtx_1, dist_1, mtx_2, dist_2, R, T, rmse, c1_to_c2_params_path)
# R_loaded, T_loaded = load_cam_to_cam_params(c1_to_c2_params_path)

# # print("loaded R :")
# # print(R_loaded)
# # print("loaded T : ")
# # print(T_loaded)

# # print("rmse : ")
# # print(rmse)