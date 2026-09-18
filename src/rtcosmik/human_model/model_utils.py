import pinocchio as pin
import numpy as np 
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Dict
from rtcosmik.utils.linear_algebra_utils import col_vector_3D
import logging 
LOGGER = logging.getLogger(__name__)

SGTS_JOINTS_CALIB_MAPPING = {
    "pelvis": ["root_joint"],
    "right_upperleg": ["right_hip_Z"],
    "right_lowerleg": ["right_knee_Z"],
    "right_foot": ["right_ankle_Z"],
    "left_upperleg": ["left_hip_Z"],
    "left_lowerleg": ["left_knee_Z"],
    "left_foot": ["left_ankle_Z"],
    "thorax": ["middle_thoracic_Z"],
    "torso": ["right_clavicle_joint_X", "left_clavicle_joint_X"],
    "head": ["middle_cervical_Z"],
    "right_upperarm": ["right_shoulder_Z"],
    "right_lowerarm": ["right_elbow_Z"],
    "right_hand": ["right_wrist_Z"],
    "left_upperarm": ["left_shoulder_Z"],
    "left_lowerarm": ["left_elbow_Z"],
    "left_hand": ["left_wrist_Z"],
}

SGTS_MKS_MAPPING = {
     "head":      ["Nose", "Head", "REar", "LEar", "REye", "LEye"],     
     "pelvis":    ['RPSI','LPSI','RASI','LASI','T11'],    
     "torso":     ['RSHO', 'LSHO'],
     "thorax":     ['C7', 'T6'], 
     "right_upperarm": ['RMELB','RELB'],    
     "right_lowerarm": ['RMWRI','RWRI'],    
     "right_hand":     ['RTHU', 'RMID', 'RPIN'],    
     "left_upperarm": ['LMELB','LELB'],     
     "left_lowerarm": ['LMWRI','LWRI'],
     "left_hand":     ['LTHU', 'LMID', 'LPIN'],          
     "right_upperleg":    ['RKNE','RMKNE'],                
     "right_lowerleg":    ['RMANK','RANK'],             
     "right_foot":     ['RTOE','R5MHD','RHEE'],     
     "left_upperleg":    ['LKNE','LMKNE'],                
     "left_lowerleg":    ['LMANK','LANK'],              
     "left_foot":     ['LTOE','L5MHD','LHEE'],       
    }

def check_orthogonality(matrix: np.ndarray):
    # Vecteurs colonnes
    X = matrix[:3, 0]
    Y = matrix[:3, 1]
    Z = matrix[:3, 2]
    
    # Calcul des produits scalaires
    dot_XY = np.dot(X, Y)
    dot_XZ = np.dot(X, Z)
    dot_YZ = np.dot(Y, Z)
    
    # Tolérance pour les erreurs numériques
    tolerance = 1e-6
    
    print(f"Dot product X.Y: {dot_XY}")
    print(f"Dot product X.Z: {dot_XZ}")
    print(f"Dot product Y.Z: {dot_YZ}")
    
    assert np.abs(dot_XY) < tolerance, "Vectors X and Y are not orthogonal"
    assert np.abs(dot_XZ) < tolerance, "Vectors X and Z are not orthogonal"
    assert np.abs(dot_YZ) < tolerance, "Vectors Y and Z are not orthogonal"


#Build inertia matrix from 6 inertia components
def make_inertia_matrix(ixx:float, ixy:float, ixz:float, iyy:float, iyz:float, izz:float)->np.ndarray:
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

#Function that takes as input a matrix and orthogonalizes it
#Its mainly used to orthogonalize rotation matrices constructed by hand
def orthogonalize_matrix(matrix:np.ndarray)->np.ndarray:
    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(matrix)
    # Reconstruct the orthogonal matrix
    orthogonal_matrix = U @ Vt
    # Ensure the determinant is 1
    if np.linalg.det(orthogonal_matrix) < 0:
        U[:, -1] *= -1
        orthogonal_matrix = U @ Vt
    return orthogonal_matrix


def get_left_upperarm_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pose of the left upper arm based on  marker positions.
    This function computes the transformation matrix representing the pose of the left upper arm.
    The pose is calculated using the positions of specific markers on the body, such as the shoulder
    and elbow markers. The resulting pose matrix is a 4x4 homogeneous transformation matrix.
    Args:
        mks_positions (dict): A dictionary containing the positions of  markers.
            The keys are marker names (e.g., 'LSHO', 'RSHO', 'LMELB', 'LELB'),
            and the values are numpy arrays of shape (3,) representing the 3D coordinates of the markers.
    Returns:
        numpy.ndarray: A 4x4 homogeneous transformation matrix representing the pose of the left upper arm.
    """

    pose = np.eye(4,4)
    X, Y, Z, shoulder_center = [], [], [], []
    torso_pose = get_torso_pose(mks_positions)
    bi_acromial_dist = np.linalg.norm(mks_positions['LSHO'] - mks_positions['RSHO'])
    shoulder_center = mks_positions['LSHO'].reshape(3,1) + (torso_pose[:3, :3].reshape(3,3) @ col_vector_3D(0.0, -0.17*bi_acromial_dist, 0.0)).reshape(3,1)
    elbow_center = (mks_positions['LMELB'] + mks_positions['LELB']).reshape(3,1)/2.0
    
    Y = shoulder_center - elbow_center
    Y = Y/np.linalg.norm(Y)

    Z = (mks_positions['LMELB'] - mks_positions['LELB']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)

    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)


    pose[:3, 0] = X.flatten()
    pose[:3, 1] = Y.flatten()
    pose[:3, 2] = Z.flatten()
    pose[:3, 3] = shoulder_center.flatten()
    pose[:3, :3] = orthogonalize_matrix(pose[:3, :3])

    # print("Upperarm Left Pose:\n", pose)  # Impression pour débogage
    # check_orthogonality(pose)  # Ajoutez cette ligne pour vérifier l'orthogonalité

    return pose

#construct thigh frames and get their poses
def get_right_upperleg_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pose of the right thigh based on  marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of  markers. 
                                Expected keys include 'RASI', 'LASI', 'RKNE', 
                                'RMKNE'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right thigh. The matrix 
                   includes rotation and translation components.
    """
    if gender == 'm':
        ratio_x = 0.3
        ratio_y = 0.37
        ratio_z = 0.361
    else : 
        ratio_x = 0.3
        ratio_y = 0.336
        ratio_z = 0.372

    pose = np.eye(4,4)
    X, Y, Z = [], [], []
    hip_center = np.zeros((3,1))

    dist_rPL_lPL = np.linalg.norm(mks_positions["RASI"]-mks_positions["LASI"])
    virtual_pelvis_pose = get_virtual_pelvis_pose(mks_positions)
    hip_center = virtual_pelvis_pose[:3, 3].reshape(3,1)

    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(-ratio_x*dist_rPL_lPL, 0.0, 0.0)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, -ratio_y*dist_rPL_lPL, 0.0)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, 0.0, ratio_z*dist_rPL_lPL)

    knee_center = (mks_positions['RKNE'] + mks_positions['RMKNE']).reshape(3,1)/2.0
    Y = hip_center - knee_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['RKNE'] - mks_positions['RMKNE']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = hip_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose


def get_left_upperleg_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pose of the left thigh based on  marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of  markers. 
                                Expected keys are 'LASI', 'RASI', 'LKNE', 'LMKNE'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the left thigh. The matrix includes
                   rotation and translation components.
    """
    if gender == 'm':
        ratio_x = 0.3
        ratio_y = 0.37
        ratio_z = 0.361
    else : 
        ratio_x = 0.3
        ratio_y = 0.336
        ratio_z = 0.372

    pose = np.eye(4,4)
    X, Y, Z = [], [], []
    hip_center = np.zeros((3,1))

    dist_rPL_lPL = np.linalg.norm(mks_positions["LASI"]-mks_positions["RASI"])
    virtual_pelvis_pose = get_virtual_pelvis_pose(mks_positions)
    hip_center = virtual_pelvis_pose[:3, 3].reshape(3,1)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(-ratio_x*dist_rPL_lPL, 0.0, 0.0)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, -ratio_y*dist_rPL_lPL, 0.0)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, 0.0, -ratio_z*dist_rPL_lPL)

    knee_center = (mks_positions['LKNE'] + mks_positions['LMKNE']).reshape(3,1)/2.0
    Y = hip_center - knee_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['LMKNE'] - mks_positions['LKNE']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = hip_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

#construct shank frames and get their poses
def get_right_lowerleg_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pose of the right shank based on  marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of  markers. 
                                The keys should include 'RKNE', 'RMKNE', 
                                'RMANK', 'RANK'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right shank. The matrix 
                   includes rotation (in the top-left 3x3 submatrix) and translation (in the top-right 
                   3x1 subvector).
    """

    pose = np.eye(4,4)
    X, Y, Z, knee_center, ankle_center = [], [], [], [], []

    knee_center = (mks_positions['RKNE'] + mks_positions['RMKNE']).reshape(3,1)/2.0
    ankle_center = (mks_positions['RMANK'] + mks_positions['RANK']).reshape(3,1)/2.0
    Y = knee_center - ankle_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['RKNE'] - mks_positions['RMKNE']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = knee_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

def get_left_lowerleg_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pose of the left shank based on  marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of  markers. 
                                The keys should include 'LKNE', 'LMKNE', 
                                'LMANK', 'LANK'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the left shank. The matrix 
                   includes the rotation (3x3) and translation (3x1) components.
    """

    pose = np.eye(4,4)
    X, Y, Z, knee_center, ankle_center = [], [], [], [], []

    knee_center = (mks_positions['LKNE'] + mks_positions['LMKNE']).reshape(3,1)/2.0
    ankle_center = (mks_positions['LMANK'] + mks_positions['LANK']).reshape(3,1)/2.0
    Y = knee_center - ankle_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['LMKNE'] - mks_positions['LKNE']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = knee_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

#construct foot frames and get their poses
def get_right_foot_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pose of the right foot based on  marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of  markers. 
                                The keys include 'RMANK', 'RANK', 'RTOE', 
                                'RHEE', 'R5MHD'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right foot. The matrix 
                   includes the orientation (rotation) and position (translation) of the foot.
    """

    pose = np.eye(4,4)
    X, Y, Z, ankle_center = [], [], [], []

    ankle_center = (mks_positions['RMANK'] + mks_positions['RANK']).reshape(3,1)/2.0
    toe_pos = (mks_positions['RTOE'] + mks_positions['R5MHD'])/2.0
    
    X = (toe_pos - mks_positions['RHEE']).reshape(3,1)  
    X = X/np.linalg.norm(X)
    Z = (mks_positions['RANK'] - mks_positions['RMANK']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ankle_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

def get_left_foot_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pose of the left foot based on  marker positions.
    This function computes the transformation matrix (pose) of the left foot using
    the positions of various markers from  data. The pose is represented
    as a 4x4 homogeneous transformation matrix.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of  markers.
                                The keys are marker names and the values are their respective
                                3D coordinates (numpy arrays).
    Returns:
    numpy.ndarray: A 4x4 homogeneous transformation matrix representing the pose of the left foot.
    Notes:
    - The function checks for the presence of specific markers ('LMANK', 'LANK',
      'LTOE', 'LHEE') to compute the pose.
    - The resulting pose matrix includes the orientation (rotation) and position (translation)
      of the left foot.
    - The orientation matrix is orthogonalized to ensure it is a valid rotation matrix.
    """

    pose = np.eye(4,4)
    X, Y, Z, ankle_center = [], [], [], []

    ankle_center = (mks_positions['LMANK'] + mks_positions['LANK']).reshape(3,1)/2.0
    toe_pos = (mks_positions['LTOE'] + mks_positions['L5MHD'])/2.0

    X = (toe_pos - mks_positions['LHEE']).reshape(3,1)
    X = X/np.linalg.norm(X)
    Z = (mks_positions['LMANK'] - mks_positions['LANK']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ankle_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

#get_virtual_pelvis_pose, used to get thigh pose
def get_virtual_pelvis_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pelvis pose matrix from  marker positions.
    The function computes the pelvis pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the pelvis coordinate system. Finally, it constructs the 
    pose matrix and ensures it is orthogonal.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of the  markers.
                                The keys include 'RPSI', 'LPSI', 'RASI', 
                                'LASI'.
    Returns:
    numpy.ndarray: A 4x4 pose matrix representing the pelvis pose.
    """

    pose = np.eye(4,4)
    X, Y, Z = [], [], []
    center_PSIS = []
    center_ASIS = []

    center_PSIS = (mks_positions['RPSI'] + mks_positions['LPSI']).reshape(3,1)/2.0
    center_ASIS = (mks_positions['RASI'] + mks_positions['LASI']).reshape(3,1)/2.0

    X = center_ASIS - center_PSIS
    X = X/np.linalg.norm(X)
    Z = mks_positions['RASI'] - mks_positions['LASI']
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = center_ASIS.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose


def get_pelvis_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pelvis pose matrix from  marker positions.
    The function computes the pelvis pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the pelvis coordinate system. Finally, it constructs the 
    pose matrix and ensures it is orthogonal.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of the  markers.
                                The keys include 'RPSI', 'LPSI', 'RASI', 
                                'LASI'.
    Returns:
    numpy.ndarray: A 4x4 pose matrix representing the pelvis pose.
    """

    if gender == 'm':
        ratio_x = 0.335
        ratio_y = -0.032
        ratio_z = 0.0
    else : 
        ratio_x = 0.34
        ratio_y = 0.049
        ratio_z = 0.0

    pose = np.eye(4,4)
    center_PSIS = []
    center_ASIS = []
    center_right_ASIS_PSIS = []
    center_left_ASIS_PSIS = []
    LJC=np.zeros((3,1))

    dist_rPL_lPL = np.linalg.norm(mks_positions["RASI"]-mks_positions["LASI"])
    virtual_pelvis_pose = get_virtual_pelvis_pose(mks_positions)
    LJC = virtual_pelvis_pose[:3, 3].reshape(3,1)


    center_PSIS = (mks_positions['RPSI'] + mks_positions['LPSI']).reshape(3,1)/2.0
    center_ASIS = (mks_positions['RASI'] + mks_positions['LASI']).reshape(3,1)/2.0
    
    center_right_ASIS_PSIS = (mks_positions['RPSI'] + mks_positions['RASI']).reshape(3,1)/2.0
    center_left_ASIS_PSIS = (mks_positions['LPSI'] + mks_positions['LASI']).reshape(3,1)/2.0
    
    offset_local = col_vector_3D(
                                -ratio_x * dist_rPL_lPL,
                                +ratio_y * dist_rPL_lPL,
                                ratio_z * dist_rPL_lPL
                                )
    LJC = LJC + virtual_pelvis_pose[:3, :3] @ offset_local
 
    X = center_ASIS - center_PSIS
    X = X/np.linalg.norm(X)
    Z = center_right_ASIS_PSIS - center_left_ASIS_PSIS
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)


    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ((center_right_ASIS_PSIS + center_left_ASIS_PSIS)/2.0).reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose

def get_left_lowerarm_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pose of the left lower arm based on  marker positions.
    This function computes the transformation matrix representing the pose of the left lower arm.
    It uses the positions of specific markers to determine the orientation and position of the arm.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of  markers.
                                The keys should include 'LMELB', 'LELB', 
                                'LMWRI', 'LWRI'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the left lower arm.
    """

    pose = np.eye(4,4)
    X, Y, Z, elbow_center = [], [], [], []
    elbow_center = (mks_positions['LMELB'] + mks_positions['LELB']).reshape(3,1)/2.0
    wrist_center = (mks_positions['LMWRI'] + mks_positions['LWRI']).reshape(3,1)/2.0
    
    Y = elbow_center - wrist_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['LMWRI'] - mks_positions['LWRI']).reshape(3,1)
    Z = Z.reshape(3, 1) / np.linalg.norm(Z)

    X = np.cross(Y, Z, axis=0)
    X = X.reshape(3, 1) / np.linalg.norm(X)

    Z = np.cross(X.flatten(), Y.flatten())
    Z = Z.reshape(3, 1) / np.linalg.norm(Z)

    pose[:3, 0] = X.flatten()
    pose[:3, 1] = Y.flatten()
    pose[:3, 2] = Z.flatten()
    pose[:3, 3] = elbow_center.flatten()
    pose[:3, :3] = orthogonalize_matrix(pose[:3, :3])

    # print("Lowerarm Left Pose:\n", pose)  # Impression pour débogage
    # check_orthogonality(pose)  # Ajoutez cette ligne pour vérifier l'orthogonalité

    return pose


#construct upperarm frames and get their poses
def get_right_upperarm_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pose of the right upper arm based on  marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of  markers. 
                                Expected keys include 'LSHO', 'RSHO', 'RMELB', 
                                'RELB'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right upper arm. 
                   The matrix includes rotation (3x3) and translation (3x1) components.
    """
    
    pose = np.eye(4,4)
    X, Y, Z, shoulder_center = [], [], [], []

    torso_pose = get_torso_pose(mks_positions)
    bi_acromial_dist = np.linalg.norm(mks_positions['LSHO'] - mks_positions['RSHO'])
    shoulder_center = mks_positions['RSHO'].reshape(3,1) + torso_pose[:3, :3] @ col_vector_3D(0.0, -0.17*bi_acromial_dist, 0.0)
    elbow_center = (mks_positions['RMELB'] + mks_positions['RELB']).reshape(3,1)/2.0
    
    Y = shoulder_center - elbow_center
    Y = Y/np.linalg.norm(Y)

    Z = (mks_positions['RELB'] - mks_positions['RMELB']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)

    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

        
    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = shoulder_center.reshape(3,)

    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose

#construct abdomen frame and get its pose (middle thoracic joint in urdf)
def get_thorax_pose(mks_positions, gender='m', subject_height= 1.80):
    #pelvis + distance selon y
    """
    Calculate the abdomen pose matrix from  marker positions.
    The function computes the abdomen pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the abdomen coordinate system. Finally, it constructs the 
    pose matrix and ensures it is orthogonal.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of the  markers.
                                The keys include 'RPSI', 'LPSI', 'RASI', 
                                'LASI'.
    Returns:
    numpy.ndarray: A 4x4 pose matrix representing the abdomen pose.
    """
    if gender == 'm' : 
        abdomen_ratio = 0.0839
    else : 
        abdomen_ratio = 0.0776
    
    pelvis_pose =(get_pelvis_pose(mks_positions,gender)[:3,3]).reshape(3,1)
    torso_pose = (get_torso_pose(mks_positions)[:3,3]).reshape(3,1)
    direction = torso_pose - pelvis_pose                     
    direction = direction / np.linalg.norm(direction)  

    p_local=col_vector_3D(0.0, subject_height * abdomen_ratio,0.0)

    p_global = (get_pelvis_pose(mks_positions,gender)[:3,:3].reshape(3,3) @ p_local).reshape(3,1)
    
    pose = np.eye(4,4)
    X, Y, Z = [], [], []
    center_PSIS = []
    center_ASIS = []

    center_PSIS = (mks_positions['RPSI'] + mks_positions['LPSI']).reshape(3,1)/2.0
    center_ASIS = (mks_positions['RASI'] + mks_positions['LASI']).reshape(3,1)/2.0

    center_right_ASIS_PSIS = (mks_positions['RPSI'] + mks_positions['RASI']).reshape(3,1)/2.0
    center_left_ASIS_PSIS = (mks_positions['LPSI'] + mks_positions['LASI']).reshape(3,1)/2.0
    
    X = center_ASIS - center_PSIS
    X = X/np.linalg.norm(X)
    Z = center_right_ASIS_PSIS - center_left_ASIS_PSIS
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ((get_pelvis_pose(mks_positions,gender)[:3,3]).reshape(3,1)+ (direction*p_global)).reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose
 

def get_head_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pose of the head based on  marker positions.
    The function computes a 4x4 transformation matrix representing the pose of the head.
    The matrix includes rotation and translation components derived from the positions
    of specific markers.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of  markers.
                                Expected keys are either 'C7', 'Head', 'RSHO', 'LSHO', 'REar', 'LEar' or 'RSHO', 'LSHO', 'FHD', 'BHD', 'RHD', 'LHD'. Each key should map to a 
                                numpy array of shape (3,).
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the head pose.
    """

    pose = np.eye(4)

    # default origin (your current behavior)
    shoulder_center = (mks_positions['RSHO'] + mks_positions['LSHO']) / 2.0
    head_center = shoulder_center

    X = Y = Z = None

    if 'Head' in mks_positions:
        top_head = mks_positions['Head']

        Y = (top_head - shoulder_center).reshape(3, 1)
        Y = Y / np.linalg.norm(Y)

        Z = (mks_positions['REar'] - mks_positions['LEar']).reshape(3, 1)
        Z = Z / np.linalg.norm(Z)

        X = np.cross(Y, Z, axis=0)        # X in WORLD coords
        X = X / np.linalg.norm(X)
        Z = np.cross(X, Y, axis=0)        # re-orthogonalize
        Z = Z / np.linalg.norm(Z)

        # special origin: ONLY if both Head and C7 exist
        if 'C7' in mks_positions:
            ear_width = np.linalg.norm(mks_positions['REar'] - mks_positions['LEar'])
            dx = 0.70*ear_width     # meters if markers are in meters
            head_center = mks_positions['C7'] + dx * X.reshape(3,)

    else:
        # unchanged fallback branch
        X = (mks_positions['FHD'] - mks_positions['BHD'])
        X = X / np.linalg.norm(X)
        Z = (mks_positions['RHD'] - mks_positions['LHD'])
        Z = Z / np.linalg.norm(Z)
        Y = np.cross(Z, X, axis=0)
        Z = np.cross(X, Y, axis=0)

    pose[:3, 0] = X.reshape(3,)
    pose[:3, 1] = Y.reshape(3,)
    pose[:3, 2] = Z.reshape(3,)
    pose[:3, 3] = head_center.reshape(3,)
    pose[:3, :3] = orthogonalize_matrix(pose[:3, :3])
    return pose


#construct torso frame and get its pose from a dictionnary of mks positions and names
def get_torso_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the torso pose matrix from  marker positions.
    The function computes a 4x4 transformation matrix representing the pose of the torso.
    The matrix includes rotation and translation components derived from the positions
    of specific markers.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of  markers.
                                Expected keys are 'C7', 'RSHO', 'LSHO', 'RASI', 'LASI', 'RPSI', 'LPSI'. Each key should map to a 
                                numpy array of shape (3,).
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the torso pose.
    """

    pose = np.eye(4,4)
    X, Y, Z, trunk_center = [], [], [], []

    trunk_center = (mks_positions['RSHO'] + mks_positions['LSHO'])/2.0 
    midhip = (mks_positions['RASI'] +
                mks_positions['LASI'] +
                mks_positions['RPSI'] +
                mks_positions['LPSI'] )/4.0

    Y = (trunk_center - midhip).reshape(3,1)
    Y = Y/np.linalg.norm(Y)
    X = (trunk_center - mks_positions['C7']).reshape(3,1)
    X = X/np.linalg.norm(X)
   
    Z = np.cross(X, Y, axis=0)
    X = np.cross(Y, Z, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = trunk_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

def get_right_lowerarm_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pose of the right lower arm based on  marker positions.
    The function computes the transformation matrix (pose) of the right lower arm using the positions of specific markers.
    It first checks for the presence of 'RMELB' in the marker positions to determine which set of markers to use.
    The pose is represented as a 4x4 homogeneous transformation matrix.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of  markers. The keys are marker names,
                                and the values are their corresponding 3D positions (numpy arrays). Expected keys include 'RMELB', 'RELB', 'RWRI', 'RMWRI'
    Returns:
    numpy.ndarray: A 4x4 homogeneous transformation matrix representing the pose of the right lower arm.
    """

    pose = np.eye(4,4)
    X, Y, Z, elbow_center = [], [], [], []
    elbow_center = (mks_positions['RMELB'] + mks_positions['RELB']).reshape(3,1)/2.0
    wrist_center = (mks_positions['RWRI'] + mks_positions['RMWRI']).reshape(3,1)/2.0
    
    Y = elbow_center - wrist_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['RWRI'] - mks_positions['RMWRI']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = elbow_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

#construct shank frames and get their poses
def get_right_lowerleg_pose(mks_positions, gender='m', subject_height= 1.80):
    """
    Calculate the pose of the right shank based on  marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of  markers. 
                                The keys should include  'RKNE', 'RMKNE', 
                                'RMANK', 'RANK'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right shank. The matrix 
                   includes rotation (in the top-left 3x3 submatrix) and translation (in the top-right 
                   3x1 subvector).
    """

    pose = np.eye(4,4)
    X, Y, Z, knee_center, ankle_center = [], [], [], [], []

    knee_center = (mks_positions['RKNE'] + mks_positions['RMKNE']).reshape(3,1)/2.0
    ankle_center = (mks_positions['RMANK'] + mks_positions['RANK']).reshape(3,1)/2.0
    Y = knee_center - ankle_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['RKNE'] - mks_positions['RMKNE']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)


    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = knee_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

def get_right_hand_pose(mks_positions, gender='m', subject_height=1.80):
    pose = np.eye(4, 4)

    wrist_center = ((mks_positions['RWRI'] + mks_positions['RMWRI']) / 2.0).reshape(3, 1)

    Y = (wrist_center - mks_positions['RMID'].reshape(3, 1))
    Y = Y / np.linalg.norm(Y)

    Z = (mks_positions['RWRI'].reshape(3, 1) - mks_positions['RMWRI'].reshape(3, 1))
    Z = Z / np.linalg.norm(Z)

    X = np.cross(Y, Z, axis=0)
    X = X / np.linalg.norm(X)

    Z = np.cross(X, Y, axis=0)
    Z = Z / np.linalg.norm(Z)

    pose[:3, 0] = X.reshape(3,)
    pose[:3, 1] = Y.reshape(3,)
    pose[:3, 2] = Z.reshape(3,)
    pose[:3, 3] = wrist_center.reshape(3,)

    pose[:3, :3] = orthogonalize_matrix(pose[:3, :3])
    return pose


def get_left_hand_pose(mks_positions, gender='m', subject_height=1.80):
    pose = np.eye(4, 4)

    wrist_center = ((mks_positions['LWRI'] + mks_positions['LMWRI']) / 2.0).reshape(3, 1)

    Y = (wrist_center - mks_positions['LMID'].reshape(3, 1))
    Y = Y / np.linalg.norm(Y)

    Z = (mks_positions['LMWRI'].reshape(3, 1) - mks_positions['LWRI'].reshape(3, 1))
    Z = Z / np.linalg.norm(Z)

    X = np.cross(Y, Z, axis=0)
    X = X / np.linalg.norm(X)

    Z = np.cross(X, Y, axis=0)
    Z = Z / np.linalg.norm(Z)

    pose[:3, 0] = X.reshape(3,)
    pose[:3, 1] = Y.reshape(3,)
    pose[:3, 2] = Z.reshape(3,)
    pose[:3, 3] = wrist_center.reshape(3,)

    pose[:3, :3] = orthogonalize_matrix(pose[:3, :3])
    return pose


def get_local_mks_positions(sgts_poses: Dict, mks_positions: Dict, sgts_mks_dict: Dict)-> Dict:
    """_Get the local 3D position of the markers_

    Args:
        sgts_poses (Dict): _sgts_poses corresponds to a dictionnary to segments poses and names, constructed from global mks positions_
        mks_positions (Dict): _mks_positions is a dictionnary of lstm mks names and 3x1 global positions_
        sgts_mks_dict (Dict): _sgts_mks_dict a dictionnary containing the segments names, and the corresponding list of lstm mks names attached to the segment_

    Returns:
        Dict: _returns a dictionnary of lstm mks names and their 3x1 local positions_
    """
    mks_local_positions = {}

    for segment, markers in sgts_mks_dict.items():
        # Get the segment's transformation matrix
                 
        if segment in sgts_poses:
            segment_pose = sgts_poses[segment]
       
            # Compute the inverse of the segment's transformation matrix
            segment_pose_inv = np.eye(4,4)
            segment_pose_inv[:3,:3] = np.transpose(segment_pose[:3,:3])
            segment_pose_inv[:3,3] = -np.transpose(segment_pose[:3,:3]) @ segment_pose[:3,3]
            for marker in markers:
                if marker in mks_positions:
                    # Get the marker's global position
                    marker_global_pos = np.append(mks_positions[marker], 1)  # Convert to homogeneous coordinates
                    marker_local_pos_hom = segment_pose_inv @ marker_global_pos  # Transform to local coordinates
                    marker_local_pos = marker_local_pos_hom[:3]  # Convert back to 3x1 coordinates
                     
                    if marker not in mks_local_positions:
                        # Store the local position in the dictionary
                        mks_local_positions[marker] = marker_local_pos

    return mks_local_positions

def get_local_segments_positions(sgts_poses: Dict)->Dict:
    """_Get the local positions of the segments_

    Args:
        sgts_poses (Dict): _a dictionnary of segment poses_

    Returns:
        Dict: _returns a dictionnary of local positions for each segment except pelvis_
    """
    # Initialize the dictionary to store local positions
    local_positions = {}

    # Pelvis is the base, so it does not have a local position
    if "pelvis" in sgts_poses:
        pelvis_pose = sgts_poses["pelvis"]
    
    # Compute local positions for each segment
    # Thorax with respect to pelvis
    if "thorax" in sgts_poses:
        thorax_global = sgts_poses["thorax"]
        local_positions["thorax"] = (np.linalg.inv(pelvis_pose) @ thorax_global @ np.array([0, 0, 0, 1]))[:3]
    
    # Torso with respect to thorax
    if "torso" in sgts_poses:
        torso_global = sgts_poses["torso"]
        thorax_global = sgts_poses["thorax"]
        local_positions["torso"] = (np.linalg.inv(thorax_global) @ torso_global @ np.array([0, 0, 0, 1]))[:3]
        #need to adjust torso frame to aligned it with thorax and pelvis frames.

    # Head with respect to thorax
    if "head" in sgts_poses:
        head_global = sgts_poses["head"]
        thorax_global = sgts_poses["thorax"]
        local_positions["head"] = (np.linalg.inv(thorax_global) @ head_global @ np.array([0, 0, 0, 1]))[:3]

    # Upperarm with respect to torso
    if "right_upperarm" in sgts_poses:
        upperarm_global = sgts_poses["right_upperarm"]
        torso_global = sgts_poses["torso"]
        local_positions["right_upperarm"] = (np.linalg.inv(torso_global) @ upperarm_global @ np.array([0, 0, 0, 1]))[:3]

    if "left_upperarm" in sgts_poses:
        upperarm_global = sgts_poses["left_upperarm"]
        torso_global = sgts_poses["torso"]
        local_positions["left_upperarm"] = (np.linalg.inv(torso_global) @ upperarm_global @ np.array([0, 0, 0, 1]))[:3]

    # Lowerarm with respect to upperarm
    if "right_lowerarm" in sgts_poses:
        lowerarm_global = sgts_poses["right_lowerarm"]
        upperarm_global = sgts_poses["right_upperarm"]
        local_positions["right_lowerarm"] = (np.linalg.inv(upperarm_global) @ lowerarm_global @ np.array([0, 0, 0, 1]))[:3]

    if "left_lowerarm" in sgts_poses:
        lowerarm_global = sgts_poses["left_lowerarm"]
        upperarm_global = sgts_poses["left_upperarm"]
        local_positions["left_lowerarm"] = (np.linalg.inv(upperarm_global) @ lowerarm_global @ np.array([0, 0, 0, 1]))[:3]

    # Hand with respect to lowerarm
    if "right_hand" in sgts_poses:
        hand_global = sgts_poses["right_hand"]
        lowerarm_global = sgts_poses["right_lowerarm"]
        local_positions["right_hand"] = (np.linalg.inv(lowerarm_global) @ hand_global @ np.array([0, 0, 0, 1]))[:3]

    if "left_hand" in sgts_poses:
        hand_global = sgts_poses["left_hand"]
        lowerarm_global = sgts_poses["left_lowerarm"]
        local_positions["left_hand"] = (np.linalg.inv(lowerarm_global) @ hand_global @ np.array([0, 0, 0, 1]))[:3]
            
    # Thigh with respect to pelvis
    if "right_upperleg" in sgts_poses:
        thigh_global = sgts_poses["right_upperleg"]
        local_positions["right_upperleg"] = (np.linalg.inv(pelvis_pose) @ thigh_global @ np.array([0, 0, 0, 1]))[:3]

    if "left_upperleg" in sgts_poses:
        thigh_global = sgts_poses["left_upperleg"]
        local_positions["left_upperleg"] = (np.linalg.inv(pelvis_pose) @ thigh_global @ np.array([0, 0, 0, 1]))[:3]

    # Shank with respect to thigh
    if "right_lowerleg" in sgts_poses:
        shank_global = sgts_poses["right_lowerleg"]
        thigh_global = sgts_poses["right_upperleg"]
        local_positions["right_lowerleg"] = (np.linalg.inv(thigh_global) @ shank_global @ np.array([0, 0, 0, 1]))[:3]

    if "left_lowerleg" in sgts_poses:
        shank_global = sgts_poses["left_lowerleg"]
        thigh_global = sgts_poses["left_upperleg"]
        local_positions["left_lowerleg"] = (np.linalg.inv(thigh_global) @ shank_global @ np.array([0, 0, 0, 1]))[:3]

    # Foot with respect to shank
    if "right_foot" in sgts_poses:
        foot_global = sgts_poses["right_foot"]
        shank_global = sgts_poses["right_lowerleg"]
        local_positions["right_foot"] = (np.linalg.inv(shank_global) @ foot_global @ np.array([0, 0, 0, 1]))[:3]
    
    if "left_foot" in sgts_poses:
        foot_global = sgts_poses["left_foot"]
        shank_global = sgts_poses["left_lowerleg"]
        local_positions["left_foot"] = (np.linalg.inv(shank_global) @ foot_global @ np.array([0, 0, 0, 1]))[:3]
    return local_positions

def construct_segments_frames(mks_positions, gender='m', subject_height=1.80): 
    """
    Constructs a dictionary of segment poses from  marker positions.
    Args:
        mocap_mks_positions (dict): A dictionary containing the positions of  markers.
    Returns:
        dict: A dictionary where keys are segment names (e.g., 'torso', 'upperarmR') and values are the corresponding poses.
    """
    
    # Check if all required markers are in the dataset for each segment
    sgts_poses = {}
    
    def maybe_add_pose(segment_name, marker_list, compute_func):
        if all(m in mks_positions for m in marker_list):
            sgts_poses[segment_name] = compute_func(mks_positions, gender, subject_height)

    maybe_add_pose("head",      ['RSHO', 'LSHO', 'C7', 'Head', 'REar', 'LEar'],          get_head_pose)

    maybe_add_pose("torso",     ['RSHO', 'LSHO', 'RASI', 'LASI', 'RPSI', 'LPSI', 'C7'], get_torso_pose)
    maybe_add_pose("thorax",     ['RSHO', 'LSHO', 'RASI', 'LASI', 'RPSI', 'LPSI', 'C7'], get_thorax_pose) # same as torso as it calls torso

    maybe_add_pose("right_upperarm", ['LSHO','RSHO','RMELB','RELB'],      get_right_upperarm_pose)
    maybe_add_pose("right_lowerarm", ['RMELB','RELB','RMWRI','RWRI'],          get_right_lowerarm_pose)
    maybe_add_pose("right_hand",     ['RMWRI','RWRI','RMID'],      get_right_hand_pose)
    
    maybe_add_pose("left_upperarm", ['LSHO','RSHO','LMELB','LELB'],      get_left_upperarm_pose)
    maybe_add_pose("left_lowerarm", ['LMELB','LELB','LMWRI','LWRI'],          get_left_lowerarm_pose)
    maybe_add_pose("left_hand",     ['LMWRI','LWRI','LMID'],      get_left_hand_pose)
    
    maybe_add_pose("pelvis",    ['RPSI','LPSI','RASI','LASI'],                  get_pelvis_pose)
    
    maybe_add_pose("right_upperleg",    ['RASI','LASI','RKNE','RMKNE'],                 get_right_upperleg_pose)
    maybe_add_pose("right_lowerleg",    ['RKNE','RMKNE','RMANK','RANK'],              get_right_lowerleg_pose)
    maybe_add_pose("right_foot",     ['RMANK','RANK','RTOE','R5MHD','RHEE'],      get_right_foot_pose)
    
    maybe_add_pose("left_upperleg",    ['LASI','RASI','LKNE','LMKNE'],                 get_left_upperleg_pose)
    maybe_add_pose("left_lowerleg",    ['LKNE','LMKNE','LMANK','LANK'],              get_left_lowerleg_pose)
    maybe_add_pose("left_foot",     ['LMANK','LANK','LTOE','L5MHD','LHEE'],      get_left_foot_pose)
    return sgts_poses


#: Which joint each marker frame hangs off. Module level so the OCP
#: generator can register the same frames without needing a capture.
MKS_COSMIK_2_JOINTS = {
    "RASI":  "root_joint",
    "LASI":  "root_joint",
    "RPSI":  "root_joint",
    "LPSI":  "root_joint",

    "C7":    "middle_thoracic_Y",
    "T11":   "middle_lumbar_X",
    "T6":    "middle_thoracic_Y",
    "RSHO":  "right_clavicle_joint_X",
    "LSHO":  "left_clavicle_joint_X",

    "RELB":  "right_shoulder_Y",
    "LELB":  "left_shoulder_Y",
    "RMELB": "right_shoulder_Y",
    "LMELB": "left_shoulder_Y",

    "RWRI":  "right_elbow_Y",
    "LWRI":  "left_elbow_Y",
    "RMWRI": "right_elbow_Y",
    "LMWRI": "left_elbow_Y",

    "RTHU": "right_wrist_X",
    "LTHU": "left_wrist_X",
    "RMID": "right_wrist_X",
    "LMID": "left_wrist_X",
    "RPIN": "right_wrist_X",
    "LPIN": "left_wrist_X",

    "RKNE":   "right_hip_Y",
    "LKNE":   "left_hip_Y",
    "RMKNE":  "right_hip_Y",
    "LMKNE":  "left_hip_Y",

    "RANK":   "right_knee_Z",
    "LANK":   "left_knee_Z",
    "RMANK":  "right_knee_Z",
    "LMANK":  "left_knee_Z",

    "R5MHD":  "right_ankle_X",
    "L5MHD":  "left_ankle_X",
    "RTOE":   "right_ankle_X",
    "LTOE":   "left_ankle_X",
    "RHEE":   "right_ankle_X",
    "LHEE":   "left_ankle_X",

    "Nose": "middle_cervical_Y",
    "Head": "middle_cervical_Y",
    "REar": "middle_cervical_Y",
    "LEar": "middle_cervical_Y",
    "REye": "middle_cervical_Y",
    "LEye": "middle_cervical_Y",
}


def scale_human_model(model, mks_dict, gender='m', subject_height=1.80):
    """
    Scales the segment lenghts a human Pinocchio model only.

    Parameters
    ----------
    model : pin.Model
        The Pinocchio kinematic model.
    
    mks_dict : dict
        Dictionary of 3D markers expressed in the world frame.

    gender: str
        Gender of the person, either m for male or f for female
    
    subject_height: float
        Height of the person in meters


    Returns
    -------
    model : pin.Model
        The updated model with scaled joint placements.

    """
    local_segments_positions = get_local_segments_positions(construct_segments_frames(mks_dict, gender, subject_height))

    q = pin.neutral(model)
    data = pin.Data(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    for segment_name, joint_names in SGTS_JOINTS_CALIB_MAPPING.items():
        if segment_name not in local_segments_positions:
            continue

        for joint_name in joint_names:
            joint_id = model.getJointId(joint_name)
            if joint_id == 0:
                LOGGER.info(f"[WARN] Joint '{joint_name}' not found in model.")
                continue

            if joint_name=="right_hip_Z" or joint_name=="left_hip_Z" or joint_name=="right_shoulder_Z" or joint_name=="left_shoulder_Z": # only 3D scaling of the model for this joints
                model.jointPlacements[joint_id].translation = local_segments_positions[segment_name]
            else:
                if model.jointPlacements[joint_id].translation[1]<0:
                    model.jointPlacements[joint_id].translation[1] = - np.linalg.norm(local_segments_positions[segment_name])
                else:
                    model.jointPlacements[joint_id].translation[1] = np.linalg.norm(local_segments_positions[segment_name])
            LOGGER.info(f"[INFO] Updated joint '{joint_name}' (ID {joint_id}) using segment '{segment_name}'")

    return model

def mks_registration(model, mks_dict, gender='m', subject_height=1.80):
    """
    Registers marker frames to a Pinocchio model using a hardcoded marker->joint mapping.

    Parameters
    ----------
    model : pin.Model
        The Pinocchio kinematic model.
    
    mks_dict : dict
        Dictionary of 3D markers expressed in the world frame.

    gender: str
        Gender of the person, either m for male or f for female
    
    subject_height: float
        Height of the person in meters

    Returns
    -------
    model : pin.Model
        The updated Pinocchio model with additional frames corresponding to the markers.
    """

    sgts_mks_dict=SGTS_MKS_MAPPING

    mks_local_positions=get_local_mks_positions(construct_segments_frames(mks_dict, gender, subject_height), mks_dict, sgts_mks_dict)
    

    # Hardcoded mapping: marker_name -> joint_name_in_model
    # (hoisted to module scope: MKS_COSMIK_2_JOINTS)

    inertia = pin.Inertia.Zero()

    for segment, marker_names in sgts_mks_dict.items():
        for marker_name in marker_names:
            # Get joint name from hardcoded mapping
            joint_name = MKS_COSMIK_2_JOINTS[marker_name]

            # Joint and its attached frame
            joint_id = model.getJointId(joint_name)
            parent_frame_id = model.joints[joint_id].id

            # Local position in that joint frame
            trans = np.array(mks_local_positions[marker_name]).reshape(3)
            frame_placement = pin.SE3(np.eye(3), trans)

            frame = pin.Frame(
                marker_name,          # frame name
                joint_id,             # parent joint id
                parent_frame_id,      # parent frame id
                frame_placement,      # SE3 (local in parent joint frame)
                pin.FrameType.OP_FRAME,
                inertia
            )

            model.addFrame(frame, False)

    return model

def recalibrate_marker_frames_in_joint_space(model, q_ref: np.ndarray, mks_dict: Dict[str, np.ndarray], marker_names: List[str]):
    """
    After a first IK gave a plausible configuration q_ref, recompute each marker local offset
    in its parent joint frame so that the marker frame matches the measured marker position at q_ref.

    This is the key to get rid of the "bootstrap" bias coming from segment-frame construction.

    Parameters
    ----------
    model : pin.Model
    q_ref : np.ndarray
        Reference configuration (output of a first IK).
    mks_world_dict : dict[str, np.ndarray]
        Marker positions in world, same frame as forward kinematics.
    marker_names : list[str]
        Markers to recalibrate (typically settings.marker_names).

    Returns
    -------
    pin.Model
    """
    data = pin.Data(model)
    pin.forwardKinematics(model, data, q_ref)
    pin.updateFramePlacements(model, data)

    for name in marker_names:
        if name not in mks_dict:
            continue
        try:
            fid = model.getFrameId(name)
        except Exception:
            continue
        if fid < 0 or fid >= model.nframes or model.frames[fid].name != name:
            continue

        parent_joint = model.frames[fid].parentJoint
        oMj = data.oMi[parent_joint]
        p_world = np.asarray(mks_dict[name], dtype=float).reshape(3)

        # p_local = R^T (p_world - t)
        p_local = oMj.rotation.T @ (p_world - oMj.translation)
        model.frames[fid].placement.translation = p_local

    return model
