from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
import numpy as np 
import hppfcl as fcl
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from typing import List, Tuple, Dict
from utils.linear_algebra_utils import col_vector_3D

class Robot(RobotWrapper):
    """_Class to load a given urdf_

    Args:
        RobotWrapper (_type_): _description_
    """
    def __init__(self,robot_urdf,package_dirs,isFext=False,freeflyer_ori =None,):
        """_Init of the robot class. User can choose between floating base or not and to set the transformation matrix for this floating base._

        Args:
            robot_urdf (_str_): _path to the robot urdf_
            package_dirs (_str_): _path to the meshes_
            isFext (bool, optional): _Adds a floating base if set to True_. Defaults to False.
            freeflyer_ori (_array_, optional): _Orientation of the floating base, given as a rotation matrix_. Defaults to None.
        """

        # intrinsic dynamic parameter names
        self.params_name = (
            "Ixx",
            "Ixy",
            "Ixz",
            "Iyy",
            "Iyz",
            "Izz",
            "mx",
            "my",
            "mz",
            "m",
        )

        # defining conditions
        self.isFext = isFext

        # folder location
        self.robot_urdf = robot_urdf

        # initializing robot's models
        if not isFext:
            self.initFromURDF(robot_urdf, package_dirs=package_dirs)
        else:
            self.initFromURDF(robot_urdf, package_dirs=package_dirs,
                              root_joint=pin.JointModelFreeFlyer())
            
        if freeflyer_ori is not None and isFext == True : 
            self.model.jointPlacements[self.model.getJointId('root_joint')].rotation = freeflyer_ori
            ub = self.model.upperPositionLimit
            ub[:7] = 1
            self.model.upperPositionLimit = ub
            lb = self.model.lowerPositionLimit
            lb[:7] = -1
            self.model.lowerPositionLimit = lb
            self.data = self.model.createData()
        else:
            self.model.upperPositionLimit = np.array([np.pi,np.pi,np.pi/8,np.pi/8,8*np.pi/9])
            self.model.lowerPositionLimit = np.array([0,0,-np.pi,-10*np.pi/9,0])

        ## \todo test that this is equivalent to reloading the model
        self.geom_model = self.collision_model
    
def model_scaling_df(model, keypoints_df):
    """
    Scales the model based on the distances between keypoints provided in the keypoints DataFrame.
    Parameters:
    model (object): The model object that contains joint and frame information.
    keypoints_df (DataFrame): A pandas DataFrame containing keypoints with columns 'Keypoint', 'X', 'Y', and 'Z'.
    Returns:
    tuple: A tuple containing the scaled model and the created data object.
    The function calculates the Euclidean distances between specific keypoints to determine the lengths of various body segments:
    - Lower leg length (Right Ankle to Right Knee)
    - Upper leg length (Right Knee to Right Hip)
    - Trunk length (Right Hip to Right Shoulder)
    - Upper arm length (Right Shoulder to Right Elbow)
    - Lower arm length (Right Elbow to Right Wrist)
    These lengths are then used to update the translations of the corresponding joints and frames in the model.
    """

    lowerleg_l = np.linalg.norm(np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Ankle')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Ankle')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Ankle')]['Z']])-np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Knee')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Knee')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Knee')]['Z']]))
    upperleg_l = np.linalg.norm(np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Knee')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Knee')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Knee')]['Z']])-np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Hip')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Hip')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Hip')]['Z']]))
    trunk_l = np.linalg.norm(np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Hip')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Hip')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Hip')]['Z']])-np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Shoulder')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Shoulder')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Shoulder')]['Z']]))
    upperarm_l = np.linalg.norm(np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Shoulder')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Shoulder')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Shoulder')]['Z']])-np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Elbow')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Elbow')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Elbow')]['Z']]))
    lowerarm_l = np.linalg.norm(np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Elbow')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Elbow')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Elbow')]['Z']])-np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Wrist')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Wrist')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Wrist')]['Z']]))

    model.jointPlacements[model.getJointId('knee_Z')].translation=np.array([lowerleg_l,0,0])
    model.jointPlacements[model.getJointId('lumbar_Z')].translation=np.array([upperleg_l,0,0])
    model.jointPlacements[model.getJointId('shoulder_Z')].translation=np.array([trunk_l,0,0])
    model.jointPlacements[model.getJointId('elbow_Z')].translation=np.array([upperarm_l,0,0])
    model.frames[model.getFrameId('hand_fixed')].translation=np.array([lowerarm_l,0,0])
    model.frames[model.getFrameId('hand')].translation=np.array([lowerarm_l,0,0])

    data=model.createData()

    return model, data

def model_scaling(model, keypoints):
    """
    Scales the given model based on the provided keypoints.
    Parameters:
    model (object): The model object that contains joint and frame information.
    keypoints (numpy.ndarray): An array of keypoints representing body parts. 
                               The keypoints should be in the order of:
                               ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    Returns:
    tuple: A tuple containing the scaled model and the created data object.
    """


    keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]


    mapping = dict(zip(keypoint_names,[i for i in range(len(keypoint_names))]))

    lowerleg_l = np.linalg.norm(keypoints[mapping['Right Ankle']][:]-keypoints[mapping['Right Knee']][:])
    upperleg_l = np.linalg.norm(keypoints[mapping['Right Knee']][:]-keypoints[mapping['Right Hip']][:])    
    trunk_l = np.linalg.norm(keypoints[mapping['Right Hip']][:]-keypoints[mapping['Right Shoulder']][:])
    upperarm_l = np.linalg.norm(keypoints[mapping['Right Shoulder']][:]-keypoints[mapping['Right Elbow']][:])
    lowerarm_l = np.linalg.norm(keypoints[mapping['Right Elbow']][:]-keypoints[mapping['Right Wrist']][:])

    model.jointPlacements[model.getJointId('knee_Z')].translation=np.array([lowerleg_l,0,0])
    model.jointPlacements[model.getJointId('lumbar_Z')].translation=np.array([upperleg_l,0,0])
    model.jointPlacements[model.getJointId('shoulder_Z')].translation=np.array([trunk_l,0,0])
    model.jointPlacements[model.getJointId('elbow_Z')].translation=np.array([upperarm_l,0,0])
    model.frames[model.getFrameId('hand_fixed')].translation=np.array([lowerarm_l,0,0])
    model.frames[model.getFrameId('hand')].translation=np.array([lowerarm_l,0,0])

    data=model.createData()

    return model, data

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


#construct torso frame and get its pose from a dictionnary of mks positions and names
def get_torso_pose(mocap_mks_positions):
    """
    Calculate the torso pose matrix from motion capture marker positions.
    The function computes a 4x4 transformation matrix representing the pose of the torso.
    The matrix includes rotation and translation components derived from the positions
    of specific markers.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of motion capture markers.
                                Expected keys are 'Neck', 'midHip', 'C7_study', 'CV7', 'SJN', 
                                'HeadR', 'HeadL', 'RSAT', and 'LSAT'. Each key should map to a 
                                numpy array of shape (3,).
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the torso pose.
    """

    pose = np.eye(4,4)
    X, Y, Z, trunk_center = [], [], [], []
    if 'Neck' in mocap_mks_positions:
        trunk_center = mocap_mks_positions['Neck']
        Y = (mocap_mks_positions['Neck'] - mocap_mks_positions['midHip']).reshape(3,1)
        Y = Y/np.linalg.norm(Y)
        X = (mocap_mks_positions['Neck'] - mocap_mks_positions['C7_study']).reshape(3,1)
        X = X/np.linalg.norm(X)
        Z = np.cross(X, Y, axis=0)
        X = np.cross(Y, Z, axis=0)
    else:
        trunk_center = ((mocap_mks_positions['CV7'] + mocap_mks_positions['SJN'])/2.0).reshape(3,1)
        Y = ((mocap_mks_positions['HeadR'] + mocap_mks_positions['HeadL'])/2.0 - mocap_mks_positions['SJN']).reshape(3,1)
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['RSAT'] - mocap_mks_positions['LSAT']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Y = np.cross(Z, X, axis=0)


    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = trunk_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

#construct upperarm frames and get their poses
def get_upperarmR_pose(mocap_mks_positions):
    """
    Calculate the pose of the right upper arm based on motion capture marker positions.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                Expected keys include 'RShoulder', 'r_melbow_study', 'r_lelbow_study', 
                                'RHLE', 'RHME', 'RSAT', and 'LSAT'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right upper arm. 
                   The matrix includes rotation (3x3) and translation (3x1) components.
    """

    pose = np.eye(4,4)
    X, Y, Z, shoulder_center = [], [], [], []
    if 'RShoulder' in mocap_mks_positions:
        shoulder_center = mocap_mks_positions['RShoulder'].reshape(3,1)
        elbow_center = (mocap_mks_positions['r_melbow_study'] + mocap_mks_positions['r_lelbow_study']).reshape(3,1)/2.0
        
        Y = shoulder_center - elbow_center
        Y = Y/np.linalg.norm(Y)

        Z = (mocap_mks_positions['r_lelbow_study'] - mocap_mks_positions['r_melbow_study']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)

        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
    else:
        elbow_center = (mocap_mks_positions['RHLE'] + mocap_mks_positions['RHME']).reshape(3,1)/2.0

        torso_pose = get_torso_pose(mocap_mks_positions)
        bi_acromial_dist = np.linalg.norm(mocap_mks_positions['RSAT'] - mocap_mks_positions['LSAT'])

        shoulder_center = mocap_mks_positions['RSAT'] + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0)

        Y = shoulder_center - elbow_center
        Y = Y/np.linalg.norm(Y)

        Z = (mocap_mks_positions['RHLE'] - mocap_mks_positions['RHME']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)

        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
        
    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = shoulder_center.reshape(3,)

    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose


def get_upperarmL_pose(mocap_mks_positions):
    """
    Calculate the pose of the left upper arm based on motion capture marker positions.
    This function computes the transformation matrix representing the pose of the left upper arm.
    The pose is calculated using the positions of specific markers on the body, such as the shoulder
    and elbow markers. The resulting pose matrix is a 4x4 homogeneous transformation matrix.
    Args:
        mocap_mks_positions (dict): A dictionary containing the positions of motion capture markers.
            The keys are marker names (e.g., 'LShoulder', 'L_melbow_study', 'L_lelbow_study', 'LHLE', 'LHME', 'LSAT', 'RSAT'),
            and the values are numpy arrays of shape (3,) representing the 3D coordinates of the markers.
    Returns:
        numpy.ndarray: A 4x4 homogeneous transformation matrix representing the pose of the left upper arm.
    """

    pose = np.eye(4,4)
    X, Y, Z, shoulder_center = [], [], [], []
    if 'LShoulder' in mocap_mks_positions:
        shoulder_center = mocap_mks_positions['LShoulder'].reshape(3,1)
        elbow_center = (mocap_mks_positions['L_melbow_study'] + mocap_mks_positions['L_lelbow_study']).reshape(3,1)/2.0
        
        Y = shoulder_center - elbow_center
        Y = Y/np.linalg.norm(Y)

        Z = (mocap_mks_positions['L_lelbow_study'] - mocap_mks_positions['L_melbow_study']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)

        X = np.cross(Y.flatten(), Z.flatten())
        # X = np.cross(Y, Z, axis=0)
        X = X.reshape(3, 1) / np.linalg.norm(X)

        # Z = np.cross(X, Y, axis=0)
        Z = np.cross(X.flatten(), Y.flatten())
        Z = Z.reshape(3, 1) / np.linalg.norm(Z)
    else:
        elbow_center = (mocap_mks_positions['LHLE'] + mocap_mks_positions['LHME']).reshape(3,1)/2.0
        torso_pose = get_torso_pose(mocap_mks_positions)
        bi_acromial_dist = np.linalg.norm(mocap_mks_positions['LSAT'] - mocap_mks_positions['RSAT'])
        shoulder_center = mocap_mks_positions['LSAT'] + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0)
        Y = shoulder_center - elbow_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['LHLE'] - mocap_mks_positions['LHME']).reshape(3,1)
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


#construct lowerarm frames and get their poses
def get_lowerarmR_pose(mocap_mks_positions):
    """
    Calculate the pose of the right lower arm based on motion capture marker positions.
    The function computes the transformation matrix (pose) of the right lower arm using the positions of specific markers.
    It first checks for the presence of 'r_melbow_study' in the marker positions to determine which set of markers to use.
    The pose is represented as a 4x4 homogeneous transformation matrix.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of motion capture markers. The keys are marker names,
                                and the values are their corresponding 3D positions (numpy arrays).
    Returns:
    numpy.ndarray: A 4x4 homogeneous transformation matrix representing the pose of the right lower arm.
    """

    pose = np.eye(4,4)
    X, Y, Z, elbow_center = [], [], [], []
    if 'r_melbow_study' in mocap_mks_positions:
        elbow_center = (mocap_mks_positions['r_melbow_study'] + mocap_mks_positions['r_lelbow_study']).reshape(3,1)/2.0
        wrist_center = (mocap_mks_positions['r_mwrist_study'] + mocap_mks_positions['r_lwrist_study']).reshape(3,1)/2.0
        
        Y = elbow_center - wrist_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['r_lwrist_study'] - mocap_mks_positions['r_mwrist_study']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
    else:
        elbow_center = (mocap_mks_positions['RHLE'] + mocap_mks_positions['RHME']).reshape(3,1)/2.0
        wrist_center = (mocap_mks_positions['RRSP'] + mocap_mks_positions['RUSP']).reshape(3,1)/2.0
        
        Y = elbow_center - wrist_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['RRSP'] - mocap_mks_positions['RUSP']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = elbow_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose


def get_lowerarmL_pose(mocap_mks_positions):
    """
    Calculate the pose of the left lower arm based on motion capture marker positions.
    This function computes the transformation matrix representing the pose of the left lower arm.
    It uses the positions of specific markers to determine the orientation and position of the arm.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of motion capture markers.
                                The keys should include either 'L_melbow_study', 'L_lelbow_study', 
                                'L_mwrist_study', 'L_lwrist_study' or 'LHLE', 'LHME', 'LRSP', 'LUSP'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the left lower arm.
    """

    pose = np.eye(4,4)
    X, Y, Z, elbow_center = [], [], [], []
    if 'L_melbow_study' in mocap_mks_positions:
        elbow_center = (mocap_mks_positions['L_melbow_study'] + mocap_mks_positions['L_lelbow_study']).reshape(3,1)/2.0
        wrist_center = (mocap_mks_positions['L_mwrist_study'] + mocap_mks_positions['L_lwrist_study']).reshape(3,1)/2.0
        
        Y = elbow_center - wrist_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['L_mwrist_study'] - mocap_mks_positions['L_lwrist_study']).reshape(3,1)
        # Z = Z/np.linalg.norm(Z)
        Z = Z.reshape(3, 1) / np.linalg.norm(Z)

        X = np.cross(Y, Z, axis=0)
        X = X.reshape(3, 1) / np.linalg.norm(X)

        Z = np.cross(X.flatten(), Y.flatten())
        Z = Z.reshape(3, 1) / np.linalg.norm(Z)
        # Z = np.cross(X, Y, axis=0)
    else:
        elbow_center = (mocap_mks_positions['LHLE'] + mocap_mks_positions['LHME']).reshape(3,1)/2.0
        wrist_center = (mocap_mks_positions['LRSP'] + mocap_mks_positions['LUSP']).reshape(3,1)/2.0
        
        Y = elbow_center - wrist_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['LRSP'] - mocap_mks_positions['LUSP']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)

    pose[:3, 0] = X.flatten()
    pose[:3, 1] = Y.flatten()
    pose[:3, 2] = Z.flatten()
    pose[:3, 3] = elbow_center.flatten()
    pose[:3, :3] = orthogonalize_matrix(pose[:3, :3])


    # print("Lowerarm Left Pose:\n", pose)  # Impression pour débogage
    # check_orthogonality(pose)  # Ajoutez cette ligne pour vérifier l'orthogonalité

    return pose

#construct pelvis frame and get its pose
def get_pelvis_pose(mocap_mks_positions):
    """
    Calculate the pelvis pose matrix from motion capture marker positions.
    The function computes the pelvis pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the pelvis coordinate system. Finally, it constructs the 
    pose matrix and ensures it is orthogonal.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of the motion capture markers.
                                The keys can be either 'r.PSIS_study', 'L.PSIS_study', 'r.ASIS_study', 
                                'L.ASIS_study', or 'RIPS', 'LIPS', 'RIAS', 'LIAS'.
    Returns:
    numpy.ndarray: A 4x4 pose matrix representing the pelvis pose.
    """

    pose = np.eye(4,4)
    center_PSIS = []
    center_ASIS = []
    center_right_ASIS_PSIS = []
    center_left_ASIS_PSIS = []
    if "r.PSIS_study" in mocap_mks_positions:
        center_PSIS = (mocap_mks_positions['r.PSIS_study'] + mocap_mks_positions['L.PSIS_study']).reshape(3,1)/2.0
        center_ASIS = (mocap_mks_positions['r.ASIS_study'] + mocap_mks_positions['L.ASIS_study']).reshape(3,1)/2.0
        center_right_ASIS_PSIS = (mocap_mks_positions['r.PSIS_study'] + mocap_mks_positions['r.ASIS_study']).reshape(3,1)/2.0
        center_left_ASIS_PSIS = (mocap_mks_positions['L.PSIS_study'] + mocap_mks_positions['L.ASIS_study']).reshape(3,1)/2.0
    else:
        center_PSIS = (mocap_mks_positions['RIPS'] + mocap_mks_positions['LIPS']).reshape(3,1)/2.0
        center_ASIS = (mocap_mks_positions['RIAS'] + mocap_mks_positions['LIAS']).reshape(3,1)/2.0
        center_right_ASIS_PSIS = (mocap_mks_positions['RIPS'] + mocap_mks_positions['RIAS']).reshape(3,1)/2.0
        center_left_ASIS_PSIS = (mocap_mks_positions['LIPS'] + mocap_mks_positions['LIAS']).reshape(3,1)/2.0

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

#construct thigh frames and get their poses
def get_thighR_pose(mocap_mks_positions):
    """
    Calculate the pose of the right thigh based on motion capture marker positions.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                Expected keys include 'RHip', 'r_knee_study', 'r_mknee_study', 
                                'RIAS', 'LIAS', 'RFLE', and 'RFME'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right thigh. The matrix 
                   includes rotation and translation components.
    """

    pose = np.eye(4,4)
    X, Y, Z = [], [], []
    hip_center = np.zeros((3,1))
    if "RHip" in mocap_mks_positions:
        hip_center = mocap_mks_positions['RHip'].reshape(3,1)
        knee_center = (mocap_mks_positions['r_knee_study'] + mocap_mks_positions['r_mknee_study']).reshape(3,1)/2.0
        Y = hip_center - knee_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['r_knee_study'] - mocap_mks_positions['r_mknee_study']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
    else:
        dist_rPL_lPL = np.linalg.norm(mocap_mks_positions["RIAS"]-mocap_mks_positions["LIAS"])
        pelvis_pose = get_pelvis_pose(mocap_mks_positions)
        hip_center = pelvis_pose[:3, 3].reshape(3,1)
        hip_center = hip_center + pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(-0.14*dist_rPL_lPL, 0.0, 0.0)
        hip_center = hip_center + pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, -0.3*dist_rPL_lPL, 0.0)
        hip_center = hip_center + pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, 0.0, 0.22*dist_rPL_lPL)
        knee_center = (mocap_mks_positions['RFLE'] + mocap_mks_positions['RFME']).reshape(3,1)/2.0
        Y = hip_center - knee_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['RFLE'] - mocap_mks_positions['RFME']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = hip_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose


def get_thighL_pose(mocap_mks_positions):
    """
    Calculate the pose of the left thigh based on motion capture marker positions.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                Expected keys are 'LHip', 'L_knee_study', 'L_mknee_study', 'LIAS', 'RIAS', 'LFLE', and 'LFME'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the left thigh. The matrix includes
                   rotation and translation components.
    """

    pose = np.eye(4,4)
    X, Y, Z = [], [], []
    hip_center = np.zeros((3,1))
    if "LHip" in mocap_mks_positions:
        hip_center = mocap_mks_positions['LHip'].reshape(3,1)
        knee_center = (mocap_mks_positions['L_knee_study'] + mocap_mks_positions['L_mknee_study']).reshape(3,1)/2.0
        Y = hip_center - knee_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['L_mknee_study'] - mocap_mks_positions['L_knee_study']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
    else:
        dist_rPL_lPL = np.linalg.norm(mocap_mks_positions["LIAS"]-mocap_mks_positions["RIAS"])
        pelvis_pose = get_pelvis_pose(mocap_mks_positions)
        hip_center = pelvis_pose[:3, 3].reshape(3,1)
        hip_center = hip_center + pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(-0.14*dist_rPL_lPL, 0.0, 0.0)
        hip_center = hip_center + pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, -0.3*dist_rPL_lPL, 0.0)
        hip_center = hip_center + pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, 0.0, 0.22*dist_rPL_lPL)
        knee_center = (mocap_mks_positions['LFLE'] + mocap_mks_positions['LFME']).reshape(3,1)/2.0
        Y = hip_center - knee_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['LFLE'] - mocap_mks_positions['LFME']).reshape(3,1)
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
def get_shankR_pose(mocap_mks_positions):
    """
    Calculate the pose of the right shank based on motion capture marker positions.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                The keys should include either 'r_knee_study', 'r_mknee_study', 
                                'r_mankle_study', 'r_ankle_study' or 'RFLE', 'RFME', 'RTAM', 'RFAL'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right shank. The matrix 
                   includes rotation (in the top-left 3x3 submatrix) and translation (in the top-right 
                   3x1 subvector).
    """

    pose = np.eye(4,4)
    X, Y, Z, knee_center, ankle_center = [], [], [], [], []
    if "r_knee_study" in mocap_mks_positions:
        knee_center = (mocap_mks_positions['r_knee_study'] + mocap_mks_positions['r_mknee_study']).reshape(3,1)/2.0
        ankle_center = (mocap_mks_positions['r_mankle_study'] + mocap_mks_positions['r_ankle_study']).reshape(3,1)/2.0
        Y = knee_center - ankle_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['r_knee_study'] - mocap_mks_positions['r_mknee_study']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
    else:
        knee_center = (mocap_mks_positions['RFLE'] + mocap_mks_positions['RFME']).reshape(3,1)/2.0
        ankle_center = (mocap_mks_positions['RTAM'] + mocap_mks_positions['RFAL']).reshape(3,1)/2.0
        Y = knee_center - ankle_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['RFLE'] - mocap_mks_positions['RFME']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = knee_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose


def get_shankL_pose(mocap_mks_positions):
    """
    Calculate the pose of the left shank based on motion capture marker positions.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                The keys should include either 'L_knee_study', 'L_mknee_study', 
                                'L_mankle_study', 'L_ankle_study' or 'LFLE', 'LFME', 'LTAM', 'LFAL'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the left shank. The matrix 
                   includes the rotation (3x3) and translation (3x1) components.
    """

    pose = np.eye(4,4)
    X, Y, Z, knee_center, ankle_center = [], [], [], [], []
    if "L_knee_study" in mocap_mks_positions:
        knee_center = (mocap_mks_positions['L_knee_study'] + mocap_mks_positions['L_mknee_study']).reshape(3,1)/2.0
        ankle_center = (mocap_mks_positions['L_mankle_study'] + mocap_mks_positions['L_ankle_study']).reshape(3,1)/2.0
        Y = knee_center - ankle_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['L_mknee_study'] - mocap_mks_positions['L_knee_study']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
    else:
        knee_center = (mocap_mks_positions['LFLE'] + mocap_mks_positions['LFME']).reshape(3,1)/2.0
        ankle_center = (mocap_mks_positions['LTAM'] + mocap_mks_positions['LFAL']).reshape(3,1)/2.0
        Y = knee_center - ankle_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['LFLE'] - mocap_mks_positions['LFME']).reshape(3,1)
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
def get_footR_pose(mocap_mks_positions):
    """
    Calculate the pose of the right foot based on motion capture marker positions.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                The keys can be either 'r_mankle_study', 'r_ankle_study', 'r_toe_study', 
                                'r_calc_study' or 'RTAM', 'RFAL', 'RFM5', 'RFM1', 'RFCC'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right foot. The matrix 
                   includes the orientation (rotation) and position (translation) of the foot.
    """

    pose = np.eye(4,4)
    X, Y, Z, ankle_center = [], [], [], []
    if "r_mankle_study" in mocap_mks_positions:
        ankle_center = (mocap_mks_positions['r_mankle_study'] + mocap_mks_positions['r_ankle_study']).reshape(3,1)/2.0
        X = (mocap_mks_positions['r_toe_study'] - mocap_mks_positions['r_calc_study']).reshape(3,1)
        X = X/np.linalg.norm(X)
        Z = (mocap_mks_positions['r_ankle_study'] - mocap_mks_positions['r_mankle_study']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        Y = np.cross(Z, X, axis=0)
        Z = np.cross(X, Y, axis=0)
    else:
        ankle_center = (mocap_mks_positions['RTAM'] + mocap_mks_positions['RFAL']).reshape(3,1)/2.0
        toe_pos = (mocap_mks_positions['RFM5']+mocap_mks_positions['RFM1'])/2.0
        print("toe pos:", toe_pos)
        print("heel pos:", mocap_mks_positions['RFCC'])
        X = (toe_pos- mocap_mks_positions['RFCC']).reshape(3,1)
        X = X/np.linalg.norm(X)
        Z = (mocap_mks_positions['RFAL'] - mocap_mks_positions['RTAM']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        Y = np.cross(Z, X, axis=0)
        Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ankle_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

def get_footL_pose(mocap_mks_positions):
    """
    Calculate the pose of the left foot based on motion capture marker positions.
    This function computes the transformation matrix (pose) of the left foot using
    the positions of various markers from motion capture data. The pose is represented
    as a 4x4 homogeneous transformation matrix.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of motion capture markers.
                                The keys are marker names and the values are their respective
                                3D coordinates (numpy arrays).
    Returns:
    numpy.ndarray: A 4x4 homogeneous transformation matrix representing the pose of the left foot.
    Notes:
    - The function checks for the presence of specific markers ('L_mankle_study', 'L_ankle_study',
      'L_toe_study', 'L_calc_study') to compute the pose. If these markers are not present, it
      uses alternative markers ('LTAM', 'LFAL', 'LFM5', 'LFM1', 'LFCC').
    - The resulting pose matrix includes the orientation (rotation) and position (translation)
      of the left foot.
    - The orientation matrix is orthogonalized to ensure it is a valid rotation matrix.
    """

    pose = np.eye(4,4)
    X, Y, Z, ankle_center = [], [], [], []
    if "L_mankle_study" in mocap_mks_positions:
        ankle_center = (mocap_mks_positions['L_mankle_study'] + mocap_mks_positions['L_ankle_study']).reshape(3,1)/2.0
        X = (mocap_mks_positions['L_toe_study'] - mocap_mks_positions['L_calc_study']).reshape(3,1)
        X = X/np.linalg.norm(X)
        Z = (mocap_mks_positions['L_mankle_study'] - mocap_mks_positions['L_ankle_study']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        Y = np.cross(Z, X, axis=0)
        Z = np.cross(X, Y, axis=0)
    else:
        ankle_center = (mocap_mks_positions['LTAM'] + mocap_mks_positions['LFAL']).reshape(3,1)/2.0
        toe_pos = (mocap_mks_positions['LFM5']+mocap_mks_positions['LFM1'])/2.0
        print("toe pos:", toe_pos)
        print("heel pos:", mocap_mks_positions['LFCC'])
        X = (toe_pos- mocap_mks_positions['LFCC']).reshape(3,1)
        X = X/np.linalg.norm(X)
        Z = (mocap_mks_positions['LFAL'] - mocap_mks_positions['LTAM']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        Y = np.cross(Z, X, axis=0)
        Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ankle_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

#Construct challenge segments frames from mocap mks
# - mocap_mks_positions is a dictionnary of mocap mks names and 3x1 global positions
# - returns sgts_poses which correspond to a dictionnary to segments poses and names, constructed from mks global positions
def construct_segments_frames_challenge(mocap_mks_positions): 
    """
    Constructs a dictionary of segment poses from motion capture marker positions.
    Args:
        mocap_mks_positions (dict): A dictionary containing the positions of motion capture markers.
    Returns:
        dict: A dictionary where keys are segment names (e.g., 'torso', 'upperarmR') and values are the corresponding poses.
    """

    torso_pose = get_torso_pose(mocap_mks_positions)
    upperarmR_pose = get_upperarmR_pose(mocap_mks_positions)
    lowerarmR_pose = get_lowerarmR_pose(mocap_mks_positions)
    upperarmL_pose = get_upperarmL_pose(mocap_mks_positions)
    lowerarmL_pose = get_lowerarmL_pose(mocap_mks_positions)
    pelvis_pose = get_pelvis_pose(mocap_mks_positions)
    thighR_pose = get_thighR_pose(mocap_mks_positions)
    shankR_pose = get_shankR_pose(mocap_mks_positions)
    footR_pose = get_footR_pose(mocap_mks_positions)
    thighL_pose = get_thighL_pose(mocap_mks_positions)
    shankL_pose = get_shankL_pose(mocap_mks_positions)
    footL_pose = get_footL_pose(mocap_mks_positions)
    
    # Constructing the dictionary to store segment poses
    sgts_poses = {
        "torso": torso_pose,
        "upperarmR": upperarmR_pose,
        "lowerarmR": lowerarmR_pose,
        "upperarmL": upperarmL_pose,
        "lowerarmL": lowerarmL_pose,
        "pelvis": pelvis_pose,
        "thighR": thighR_pose,
        "shankR": shankR_pose,
        "footR": footR_pose,
        "thighL": thighL_pose,
        "shankL": shankL_pose,
        "footL": footL_pose
    }
    # for name, pose in sgts_poses.items():
    #     print(name, " rot det : ", np.linalg.det(pose[:3,:3]))
    return sgts_poses

def compare_offsets(mocap_mks_positions, lstm_mks_positions): 
    mocap_sgts_poses = construct_segments_frames_challenge(mocap_mks_positions)
    lstm_sgts_poses = construct_segments_frames_challenge(lstm_mks_positions)
    sgts_lenghts_lstm = {
        "upperarm": np.linalg.norm(lstm_sgts_poses["upperarm"][:3,3]-lstm_sgts_poses["lowerarm"][:3,3]),
        "lowerarm": np.linalg.norm(lstm_sgts_poses["lowerarm"][:3,3]-(lstm_mks_positions['r_lwrist_study'] + lstm_mks_positions['r_mwrist_study']).reshape(3,)/2.0),
        "thigh": np.linalg.norm(lstm_sgts_poses["thigh"][:3,3]-lstm_sgts_poses["shank"][:3,3]),
        "shank": np.linalg.norm(lstm_sgts_poses["shank"][:3,3]-lstm_sgts_poses["foot"][:3,3]),
    }

    sgts_lenghts_mocap = {
        "upperarm": np.linalg.norm(mocap_sgts_poses["upperarm"][:3,3]-mocap_sgts_poses["lowerarm"][:3,3]),
        "lowerarm": np.linalg.norm(mocap_sgts_poses["lowerarm"][:3,3]-(mocap_mks_positions['RRSP'] + mocap_mks_positions['RUSP']).reshape(3,)/2.0),
        "thigh": np.linalg.norm(mocap_sgts_poses["thigh"][:3,3]-mocap_sgts_poses["shank"][:3,3]),
        "shank": np.linalg.norm(mocap_sgts_poses["shank"][:3,3]-mocap_sgts_poses["foot"][:3,3]),
    }
    offset_rots = {}
    for key, value in mocap_sgts_poses.items():
        offset_rots[key] = mocap_sgts_poses[key][:3,:3].T @ lstm_sgts_poses[key][:3,:3]
    
    print("------ segments lengths -------")
    for key, value in sgts_lenghts_lstm.items():
        print(key, " lstm: ", sgts_lenghts_lstm[key], " m")
        print(key, " mocap: ", sgts_lenghts_mocap[key], " m")
    print("------ segments lengths error ------")
    for key, value in sgts_lenghts_lstm.items():
        print(key, sgts_lenghts_lstm[key] - sgts_lenghts_mocap[key], " m")

    print("------ rotation offset ------")
    for key, value in offset_rots.items():
        print(key, R.from_matrix(value).as_euler('ZYX', degrees=True), " deg")

def get_segments_lstm_mks_dict_challenge()->Dict:
    #This fuction returns a dictionnary containing the segments names, and the corresponding list of lstm
    #mks names attached to the segment
    # Constructing the dictionary to store segment poses
    sgts_mks_dict = {
        "torso": ['RShoulder', 'r_shoulder_study', 'L_shoulder_study', 'LShoulder', 'Neck', 'C7_study'],
        "upperarmR": ['r_melbow_study', 'r_lelbow_study', 'RElbow'],
        "lowerarmR": ['r_lwrist_study', 'r_mwrist_study', 'RWrist',],
        "upperarmL" : ['L_melbow_study', 'L_lelbow_study', 'LElbow'],
        "lowerarmL": ['L_lwrist_study', 'L_mwrist_study', 'LWrist',],
        "pelvis": ['r.PSIS_study', 'L.PSIS_study', 'r.ASIS_study', 'L.ASIS_study', 'RHip', 'LHip', 'LHJC_study', 'RHJC_study', 'midHip'],
        "thighR": ['r_knee_study', 'r_mknee_study', 'RKnee', 'r_thigh2_study', 'r_thigh3_study', 'r_thigh1_study'],
        "thighL": ['L_knee_study', 'L_mknee_study', 'LKnee', 'L_thigh2_study', 'L_thigh3_study', 'L_thigh1_study'],
        "shankR": ['r_sh3_study', 'r_sh2_study', 'r_sh1_study'],
        "shankL": ['L_sh3_study', 'L_sh2_study', 'L_sh1_study'],
        "footR": ['r_ankle_study', 'r_mankle_study', 'RAnkle', 'r_calc_study', 'RHeel', 'r_5meta_study', 'RSmallToe', 'r_toe_study', 'RBigToe'],
        "footL": ['L_ankle_study', 'L_mankle_study', 'LAnkle', 'L_calc_study', 'LHeel', 'L_5meta_study', 'LSmallToe', 'L_toe_study', 'LBigToe']
    }
    return sgts_mks_dict

def get_subset_challenge_mks_names()->List:
    """_This function returns the subset of markers used to track the right body side kinematics with pinocchio_

    Returns:
        List: _the subset of markers used to track the right body side kinematics with pinocchio_
    """
    mks_names = ['RShoulder', 'r_shoulder_study', 'L_shoulder_study', 'LShoulder', 'Neck', 'C7_study', 'r_melbow_study', 'r_lelbow_study', 'RElbow',
                 'r_lwrist_study', 'r_mwrist_study', 'RWrist','r.PSIS_study', 'L.PSIS_study', 'r.ASIS_study', 'L.ASIS_study', 'RHip', 'LHip', 'LHJC_study', 'RHJC_study', 'midHip',
                 'r_knee_study', 'r_mknee_study', 'RKnee', 'r_thigh2_study', 'r_thigh3_study', 'r_thigh1_study','r_sh3_study', 'r_sh2_study', 'r_sh1_study',
                 'r_ankle_study', 'r_mankle_study', 'RAnkle', 'r_calc_study', 'RHeel', 'r_5meta_study', 'RSmallToe', 'r_toe_study', 'RBigToe','L_melbow_study', 'L_lelbow_study', 
                 'LElbow', 'L_lwrist_study', 'L_mwrist_study', 'LWrist', 'L_knee_study', 'L_mknee_study', 'LKnee', 'L_thigh2_study', 
                 'L_thigh3_study', 'L_thigh1_study', 'L_sh3_study', 'L_sh2_study', 'L_sh1_study', 'L_ankle_study', 'L_mankle_study', 'LAnkle', 'L_calc_study', 'LHeel', 
                 'L_5meta_study', 'LSmallToe', 'L_toe_study', 'LBigToe',
                 
                 ]
    return mks_names

def get_local_lstm_mks_positions(sgts_poses: Dict, lstm_mks_positions: Dict, sgts_mks_dict: Dict)-> Dict:
    """_Get the local 3D position of the lstms markers_

    Args:
        sgts_poses (Dict): _sgts_poses corresponds to a dictionnary to segments poses and names, constructed from global mks positions_
        lstm_mks_positions (Dict): _lstm_mks_positions is a dictionnary of lstm mks names and 3x1 global positions_
        sgts_mks_dict (Dict): _sgts_mks_dict a dictionnary containing the segments names, and the corresponding list of lstm mks names attached to the segment_

    Returns:
        Dict: _returns a dictionnary of lstm mks names and their 3x1 local positions_
    """
    lstm_mks_local_positions = {}

    for segment, markers in sgts_mks_dict.items():
        # Get the segment's transformation matrix
        segment_pose = sgts_poses[segment]
        
        # Compute the inverse of the segment's transformation matrix
        segment_pose_inv = np.eye(4,4)
        segment_pose_inv[:3,:3] = np.transpose(segment_pose[:3,:3])
        segment_pose_inv[:3,3] = -np.transpose(segment_pose[:3,:3]) @ segment_pose[:3,3]
        for marker in markers:
            if marker in lstm_mks_positions:
                # Get the marker's global position
                marker_global_pos = np.append(lstm_mks_positions[marker], 1)  # Convert to homogeneous coordinates

                marker_local_pos_hom = segment_pose_inv @ marker_global_pos  # Transform to local coordinates
                marker_local_pos = marker_local_pos_hom[:3]  # Convert back to 3x1 coordinates
                # Store the local position in the dictionary
                lstm_mks_local_positions[marker] = marker_local_pos

    return lstm_mks_local_positions

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
    pelvis_pose = sgts_poses["pelvis"]

    # Compute local positions for each segment
    # Torso with respect to pelvis
    if "torso" in sgts_poses:
        torso_global = sgts_poses["torso"]
        local_positions["torso"] = (np.linalg.inv(pelvis_pose) @ torso_global @ np.array([0, 0, 0, 1]))[:3]

    # Upperarm with respect to torso
    if "upperarmR" in sgts_poses:
        upperarm_global = sgts_poses["upperarmR"]
        torso_global = sgts_poses["torso"]
        local_positions["upperarmR"] = (np.linalg.inv(torso_global) @ upperarm_global @ np.array([0, 0, 0, 1]))[:3]

    if "upperarmL" in sgts_poses:
        upperarm_global = sgts_poses["upperarmL"]
        torso_global = sgts_poses["torso"]
        local_positions["upperarmL"] = (np.linalg.inv(torso_global) @ upperarm_global @ np.array([0, 0, 0, 1]))[:3]

    # Lowerarm with respect to upperarm
    if "lowerarmR" in sgts_poses:
        lowerarm_global = sgts_poses["lowerarmR"]
        upperarm_global = sgts_poses["upperarmR"]
        local_positions["lowerarmR"] = (np.linalg.inv(upperarm_global) @ lowerarm_global @ np.array([0, 0, 0, 1]))[:3]

    if "lowerarmL" in sgts_poses:
        lowerarm_global = sgts_poses["lowerarmL"]
        upperarm_global = sgts_poses["upperarmL"]
        local_positions["lowerarmL"] = (np.linalg.inv(upperarm_global) @ lowerarm_global @ np.array([0, 0, 0, 1]))[:3]

    # Thigh with respect to pelvis
    if "thighR" in sgts_poses:
        thigh_global = sgts_poses["thighR"]
        local_positions["thighR"] = (np.linalg.inv(pelvis_pose) @ thigh_global @ np.array([0, 0, 0, 1]))[:3]

    if "thighL" in sgts_poses:
        thigh_global = sgts_poses["thighL"]
        local_positions["thighL"] = (np.linalg.inv(pelvis_pose) @ thigh_global @ np.array([0, 0, 0, 1]))[:3]

    # Shank with respect to thigh
    if "shankR" in sgts_poses:
        shank_global = sgts_poses["shankR"]
        thigh_global = sgts_poses["thighR"]
        local_positions["shankR"] = (np.linalg.inv(thigh_global) @ shank_global @ np.array([0, 0, 0, 1]))[:3]

    if "shankL" in sgts_poses:
        shank_global = sgts_poses["shankL"]
        thigh_global = sgts_poses["thighL"]
        local_positions["shankL"] = (np.linalg.inv(thigh_global) @ shank_global @ np.array([0, 0, 0, 1]))[:3]

    # Foot with respect to shank
    if "footR" in sgts_poses:
        foot_global = sgts_poses["footR"]
        shank_global = sgts_poses["shankR"]
        local_positions["footR"] = (np.linalg.inv(shank_global) @ foot_global @ np.array([0, 0, 0, 1]))[:3]
    
    if "footL" in sgts_poses:
        foot_global = sgts_poses["footL"]
        shank_global = sgts_poses["shankL"]
        local_positions["footL"] = (np.linalg.inv(shank_global) @ foot_global @ np.array([0, 0, 0, 1]))[:3]
    return local_positions


def build_model_challenge(mocap_mks_positions: Dict, lstm_mks_positions: Dict, meshes_folder_path: str)->Tuple[pin.Model,pin.Model, Dict]:
    """_Build the biomechanical model associated to one exercise for one subject_

    Args:
        mocap_mks_positions (Dict): _mocap_mks_positions is a dictionnary of mocap mks names and 3x1 global positions_
        lstm_mks_positions (Dict): _lstm_mks_positions is a dictionnary of lstm mks names and 3x1 global positions_
        meshes_folder_path (str): _meshes_folder_path is the path to the folder containing the meshes_

    Returns:
        Tuple[pin.Model,pin.GeomModel, Dict]: _returns the pinocchio model, geometry model, and a dictionnary with visuals._
    """

    body_color = np.array([0.005, 0.005, 0.005, 0.6])

    # TODO: Check that this model match the one in the urdf human.urdf and add abdomen joints ??
    sgts_poses = construct_segments_frames_challenge(mocap_mks_positions)
    sgts_mks_dict = get_segments_lstm_mks_dict_challenge()
    lstm_mks_local_positions = get_local_lstm_mks_positions(sgts_poses, lstm_mks_positions, sgts_mks_dict)
    local_segments_positions = get_local_segments_positions(sgts_poses)
    visuals_dict = {}

    # Meshes rotations
    rtorso = R.from_rotvec(np.pi/2 * np.array([0, 1, 0]))
    rupperarm = R.from_rotvec(np.pi/2 * np.array([0, 1, 0]))
    rlowerarm = R.from_rotvec(np.pi/2 * np.array([0, 1, 0]))
    rhand = R.from_rotvec(np.pi * np.array([0, 1, 0]))

    # Mesh loader
    mesh_loader = fcl.MeshLoader()

    # MODEL GENERATION 
    inertia = pin.Inertia.Zero()
    model= pin.Model() #Modèle géométrique
    geom_model = pin.GeometryModel() #Modèle pour l'affichage

    # pelvis with Freeflyer
    IDX_PELV_JF = model.addJoint(0,pin.JointModelFreeFlyer(),pin.SE3(np.array([[1,0,0],[0,0,-1],[0,1,0]]), np.matrix([0,0,0]).T),'root_joint')
    pelvis = pin.Frame('pelvis',IDX_PELV_JF,0,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_PELV_SF = model.addFrame(pelvis,False)
    # Add markers data
    idx_frame = IDX_PELV_SF
    for i in sgts_mks_dict["pelvis"]:
        frame = pin.Frame(i,IDX_PELV_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    pelvis_visual = pin.GeometryObject('pelvis', IDX_PELV_SF, IDX_PELV_JF, mesh_loader.load(meshes_folder_path+'/pelvis_mesh.STL'), pin.SE3(rtorso.as_matrix(), np.matrix([-0.15, -0.17, 0.16]).T), meshes_folder_path+'/pelvis_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), False, body_color)
    geom_model.addGeometryObject(pelvis_visual)
    visuals_dict["pelvis"] = pelvis_visual

    # Lumbar L5-S1 flexion/extension
    IDX_L5S1_JF = model.addJoint(IDX_PELV_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T),'middle_lumbar_Z') 
    torso = pin.Frame('torso_z',IDX_L5S1_JF,idx_frame,pin.SE3(np.eye(3),np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_TORSO_SF = model.addFrame(torso,False)
    idx_frame = IDX_TORSO_SF

    IDX_L5S1_R_EXT_INT_JF = model.addJoint(IDX_L5S1_JF,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T),'middle_lumbar_Y') 
    torso = pin.Frame('torso',IDX_L5S1_R_EXT_INT_JF,idx_frame,pin.SE3(np.eye(3), np.matrix(local_segments_positions['torso']).T),pin.FrameType.OP_FRAME, inertia)
    IDX_TORSO_SF = model.addFrame(torso,False)
    idx_frame = IDX_TORSO_SF

    for i in sgts_mks_dict["torso"]:
        frame = pin.Frame(i,IDX_L5S1_R_EXT_INT_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]+ local_segments_positions['torso']).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    torso_visual = pin.GeometryObject('torso', IDX_TORSO_SF, IDX_L5S1_JF, mesh_loader.load(meshes_folder_path+'/torso_mesh.STL'), pin.SE3(rtorso.as_matrix(), np.matrix([-0.15, 0.17, 0.15]).T), meshes_folder_path+'/torso_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), False, body_color)
    geom_model.addGeometryObject(torso_visual)
    visuals_dict["torso"] = torso_visual

    abdomen_visual = pin.GeometryObject('abdomen', IDX_TORSO_SF, IDX_L5S1_JF, mesh_loader.load(meshes_folder_path+'/abdomen_mesh.STL'), pin.SE3(rtorso.as_matrix(), np.matrix([-0.12, 0.05, 0.12]).T), meshes_folder_path+'/abdomen_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), False, body_color)
    geom_model.addGeometryObject(abdomen_visual)
    visuals_dict["abdomen"] = abdomen_visual

    # Right Shoulder ZXY
    IDX_SH_Z_JF_R = model.addJoint(IDX_L5S1_R_EXT_INT_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['upperarmR'] + local_segments_positions['torso']).T),'right_shoulder_Z') 
    upperarmR = pin.Frame('upperarm_z_R',IDX_SH_Z_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    idx_frame = IDX_UPA_SF_R

    shoulder_visual_R = pin.GeometryObject('shoulder_R', IDX_UPA_SF_R, IDX_SH_Z_JF_R, mesh_loader.load(meshes_folder_path+'/shoulder_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.16, -0.045, -0.045]).T), meshes_folder_path+'/shoulder_mesh.STL',np.array([0.0055, 0.0055, 0.0055]), False , body_color)
    geom_model.addGeometryObject(shoulder_visual_R)
    visuals_dict["shoulder_R"] = shoulder_visual_R

    IDX_SH_X_JF_R = model.addJoint(IDX_SH_Z_JF_R,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'right_shoulder_X') 
    upperarmR = pin.Frame('upperarm_x_R',IDX_SH_X_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    idx_frame = IDX_UPA_SF_R

    IDX_SH_Y_JF_R = model.addJoint(IDX_SH_X_JF_R,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'right_shoulder_Y') 
    upperarmR = pin.Frame('upperarmR',IDX_SH_Y_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    idx_frame = IDX_UPA_SF_R
    for i in sgts_mks_dict["upperarmR"]:
        frame = pin.Frame(i,IDX_SH_Y_JF_R,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    upperarm_visual_R = pin.GeometryObject('upperarm_R', IDX_UPA_SF_R, IDX_SH_Y_JF_R, mesh_loader.load(meshes_folder_path+'/upperarm_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.07, -0.29, 0.18]).T), meshes_folder_path+'/upperarm_mesh.STL',np.array([0.0063, 0.0060, 0.007]), False , body_color)
    geom_model.addGeometryObject(upperarm_visual_R)
    visuals_dict["upperarm_R"] = upperarm_visual_R

    # Right Elbow ZY 
    IDX_EL_Z_JF_R = model.addJoint(IDX_SH_Y_JF_R,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['lowerarmR']).T),'right_elbow_Z') 
    lowerarmR = pin.Frame('lowerarm_z',IDX_EL_Z_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF = model.addFrame(lowerarmR,False)
    idx_frame = IDX_LOA_SF

    elbow_visual = pin.GeometryObject('elbow', IDX_LOA_SF, IDX_EL_Z_JF_R, mesh_loader.load(meshes_folder_path+'/elbow_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.15, -0.02, -0.035]).T), meshes_folder_path+'/elbow_mesh.STL',np.array([0.0055, 0.0055, 0.0055]), False , body_color)
    geom_model.addGeometryObject(elbow_visual)
    visuals_dict["elbow"] = elbow_visual

    IDX_EL_Y_JF = model.addJoint(IDX_EL_Z_JF_R,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'right_elbow_Y') 
    lowerarmR = pin.Frame('lowerarmR',IDX_EL_Y_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF = model.addFrame(lowerarmR,False)
    idx_frame = IDX_LOA_SF

    for i in sgts_mks_dict["lowerarmR"]:
        frame = pin.Frame(i,IDX_EL_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    lowerarm_visual_R = pin.GeometryObject('lowerarm',IDX_LOA_SF, IDX_EL_Y_JF, mesh_loader.load(meshes_folder_path+'/lowerarm_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.05, -0.25, 0.17]).T), meshes_folder_path+'/lowerarm_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , body_color)
    geom_model.addGeometryObject(lowerarm_visual_R)
    visuals_dict["lowerarm_R"] = lowerarm_visual_R

    # Left shoulder ZXY
    IDX_SH_Z_JF_L = model.addJoint(IDX_L5S1_R_EXT_INT_JF, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix(local_segments_positions['upperarmL'] + local_segments_positions['torso']).T), 'left_shoulder_Z') 
    upperarmL = pin.Frame('upperarm_z_L', IDX_SH_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    idx_frame = IDX_UPA_SF_L

    shoulder_visual_L = pin.GeometryObject('shoulder_L', IDX_UPA_SF_L, IDX_SH_Z_JF_L, mesh_loader.load(meshes_folder_path+'/shoulder_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.16, -0.045, -0.045]).T), meshes_folder_path+'/shoulder_mesh.STL', np.array([0.0055, 0.0055, 0.0055]), False, body_color)
    geom_model.addGeometryObject(shoulder_visual_L)
    visuals_dict["shoulder_L"] = shoulder_visual_L

    IDX_SH_X_JF_L = model.addJoint(IDX_SH_Z_JF_L, pin.JointModelRX(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'left_shoulder_X') 
    upperarmL = pin.Frame('upperarm_x_L', IDX_SH_X_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    idx_frame = IDX_UPA_SF_L

    IDX_SH_Y_JF_L = model.addJoint(IDX_SH_X_JF_L, pin.JointModelRY(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'left_shoulder_Y') 
    upperarmL = pin.Frame('upperarmL', IDX_SH_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    idx_frame = IDX_UPA_SF_L

    for i in sgts_mks_dict["upperarmL"]:
        frame = pin.Frame(i, IDX_SH_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix(lstm_mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    upperarm_visual_L = pin.GeometryObject('upperarm_L', IDX_UPA_SF_L, IDX_SH_Y_JF_L, mesh_loader.load(meshes_folder_path+'/upperarm_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.07, -0.29, 0.18]).T), meshes_folder_path+'/upperarm_mesh.STL', np.array([0.0063, 0.0060, 0.007]), False, body_color)
    geom_model.addGeometryObject(upperarm_visual_L)
    visuals_dict["upperarm_L"] = upperarm_visual_L

    # Left Elbow ZY
    IDX_EL_Z_JF_L = model.addJoint(IDX_SH_Y_JF_L, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix(local_segments_positions['lowerarmL']).T), 'left_elbow_Z')
    lowerarmL = pin.Frame('lowerarm_z_L', IDX_EL_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF_L = model.addFrame(lowerarmL, False)
    idx_frame = IDX_LOA_SF_L

    elbow_visual_L = pin.GeometryObject('elbow_L', IDX_LOA_SF_L, IDX_EL_Z_JF_L, mesh_loader.load(meshes_folder_path+'/elbow_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.15, -0.02, -0.035]).T), meshes_folder_path+'/elbow_mesh.STL', np.array([0.0055, 0.0055, 0.0055]), False, body_color)
    geom_model.addGeometryObject(elbow_visual_L)
    visuals_dict["elbow_L"] = elbow_visual_L

    IDX_EL_Y_JF_L = model.addJoint(IDX_EL_Z_JF_L, pin.JointModelRY(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'left_elbow_Y') 
    lowerarmL = pin.Frame('lowerarmL', IDX_EL_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF_L = model.addFrame(lowerarmL, False)
    idx_frame = IDX_LOA_SF_L

    for i in sgts_mks_dict["lowerarmL"]:
        frame = pin.Frame(i, IDX_EL_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix(lstm_mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    lowerarm_visual_L = pin.GeometryObject('lowerarm_L', IDX_LOA_SF_L, IDX_EL_Y_JF_L, mesh_loader.load(meshes_folder_path+'/lowerarm_mesh.STL'), pin.SE3(rlowerarm.as_matrix(), np.matrix([-0.05, -0.25, 0.17]).T), meshes_folder_path+'/lowerarm_mesh.STL', np.array([0.0060, 0.0060, 0.0060]), False, body_color)
    geom_model.addGeometryObject(lowerarm_visual_L)
    visuals_dict["lowerarm_L"] = lowerarm_visual_L

    # Right Hip ZXY
    IDX_HIP_Z_JF = model.addJoint(IDX_PELV_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['thighR']).T),'right_hip_Z') 
    thighR = pin.Frame('thigh_z',IDX_HIP_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_THIGH_SF = model.addFrame(thighR,False)
    idx_frame = IDX_THIGH_SF

    IDX_HIP_X_JF = model.addJoint(IDX_HIP_Z_JF,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'right_hip_X') 
    thighR = pin.Frame('thigh_x',IDX_HIP_X_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_THIGH_SF = model.addFrame(thighR,False)
    idx_frame = IDX_THIGH_SF

    IDX_HIP_Y_JF = model.addJoint(IDX_HIP_X_JF,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'right_hip_Y') 
    thighR = pin.Frame('thighR',IDX_HIP_Y_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_THIGH_SF = model.addFrame(thighR,False)
    idx_frame = IDX_THIGH_SF

    for i in sgts_mks_dict["thighR"]:
        frame = pin.Frame(i,IDX_HIP_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    upperleg_visual_R = pin.GeometryObject('upperleg_R',IDX_THIGH_SF, IDX_HIP_Y_JF, mesh_loader.load(meshes_folder_path+'/upperleg_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.13, -0.37, 0.1]).T), meshes_folder_path+'/upperleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , body_color)
    geom_model.addGeometryObject(upperleg_visual_R)
    visuals_dict["upperleg_R"] = upperleg_visual_R


    # Right Knee Z
    IDX_KNEE_Z_JF = model.addJoint(IDX_HIP_X_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['shankR']).T),'right_knee_Z') 
    shankR = pin.Frame('shankR',IDX_KNEE_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SHANK_SF = model.addFrame(shankR,False)
    idx_frame = IDX_SHANK_SF

    for i in sgts_mks_dict["shankR"]:
        frame = pin.Frame(i,IDX_KNEE_Z_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    knee_visual = pin.GeometryObject('knee_R',IDX_SHANK_SF, IDX_KNEE_Z_JF, mesh_loader.load(meshes_folder_path+'/knee_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.13, 0, -0.015]).T), meshes_folder_path+'/knee_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , body_color)
    geom_model.addGeometryObject(knee_visual)
    visuals_dict["knee_R"] = knee_visual

    lowerleg_visual_R = pin.GeometryObject('lowerleg_R',IDX_SHANK_SF, IDX_KNEE_Z_JF, mesh_loader.load(meshes_folder_path+'/lowerleg_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.11, -0.40, 0.1]).T), meshes_folder_path+'/lowerleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , body_color)
    geom_model.addGeometryObject(lowerleg_visual_R)
    visuals_dict["lowerleg_R"] = lowerleg_visual_R


    # Right Ankle Z
    IDX_ANKLE_Z_JF = model.addJoint(IDX_KNEE_Z_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['footR']).T),'right_ankle_Z') 
    footR = pin.Frame('footR',IDX_ANKLE_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SFOOT_SF = model.addFrame(footR,False)
    idx_frame = IDX_SFOOT_SF

    for i in sgts_mks_dict["footR"]:
        frame = pin.Frame(i,IDX_ANKLE_Z_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    foot_visual_R = pin.GeometryObject('foot_R',IDX_SFOOT_SF, IDX_ANKLE_Z_JF, mesh_loader.load(meshes_folder_path+'/foot_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.11, -0.07, 0.09]).T), meshes_folder_path+'/foot_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , body_color)
    geom_model.addGeometryObject(foot_visual_R)
    visuals_dict["foot_R"] = foot_visual_R


    # Left Hip ZXY
    IDX_HIP_Z_JF_L = model.addJoint(IDX_PELV_JF, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix(local_segments_positions['thighL']).T), 'left_hip_Z') 
    thighL = pin.Frame('thigh_z_L', IDX_HIP_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_TGH_SF_L = model.addFrame(thighL, False)
    idx_frame = IDX_TGH_SF_L

    IDX_HIP_X_JF_L = model.addJoint(IDX_HIP_Z_JF_L,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'left_hip_X') 
    thighL = pin.Frame('thigh_x_L',IDX_HIP_X_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_THIGH_SF = model.addFrame(thighL,False)
    idx_frame = IDX_TGH_SF_L

    IDX_HIP_Y_JF_L = model.addJoint(IDX_HIP_X_JF_L, pin.JointModelRY(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'left_hip_Y') 
    thighL = pin.Frame('thighL', IDX_HIP_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_TGH_SF_L = model.addFrame(thighL, False)
    idx_frame = IDX_TGH_SF_L

    for i in sgts_mks_dict["thighL"]:
        frame = pin.Frame(i, IDX_HIP_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix(lstm_mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    thigh_visual_L = pin.GeometryObject('upperleg_L', IDX_TGH_SF_L, IDX_HIP_Y_JF_L, mesh_loader.load(meshes_folder_path+'/upperleg_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.13, -0.37, -0.075]).T), meshes_folder_path+'/upperleg_mesh.STL', np.array([0.0060, 0.0060, 0.0060]), False, body_color)
    geom_model.addGeometryObject(thigh_visual_L)
    visuals_dict["upperleg_L"] = thigh_visual_L

    # Left Knee Z
    IDX_KNEE_Z_JF_L = model.addJoint(IDX_HIP_Y_JF_L,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['shankL']).T),'left_knee_Z') 
    shankR = pin.Frame('shankL',IDX_KNEE_Z_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SHANK_SF_L = model.addFrame(shankR,False)
    idx_frame = IDX_SHANK_SF_L

    for i in sgts_mks_dict["shankL"]:
        frame = pin.Frame(i,IDX_KNEE_Z_JF_L,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    knee_visual = pin.GeometryObject('knee_L',IDX_SHANK_SF_L, IDX_KNEE_Z_JF_L, mesh_loader.load(meshes_folder_path+'/knee_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.13, 0, -0.04]).T), meshes_folder_path+'/knee_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , body_color)
    geom_model.addGeometryObject(knee_visual)
    visuals_dict["knee_L"] = knee_visual

    lowerleg_visual_L = pin.GeometryObject('lowerleg_L',IDX_SHANK_SF_L, IDX_KNEE_Z_JF_L, mesh_loader.load(meshes_folder_path+'/lowerleg_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.11, -0.40, 0.08]).T), meshes_folder_path+'/lowerleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , body_color)
    geom_model.addGeometryObject(lowerleg_visual_L)
    visuals_dict["lowerleg_L"] = lowerleg_visual_L

    # Left Ankle Z
    IDX_ANKLE_Z_JF_L = model.addJoint(IDX_KNEE_Z_JF_L,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['footL']).T),'left_ankle_Z') 
    footL = pin.Frame('footL',IDX_ANKLE_Z_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SFOOT_SF_L = model.addFrame(footL,False)
    idx_frame = IDX_SFOOT_SF_L

    for i in sgts_mks_dict["footL"]:
        frame = pin.Frame(i,IDX_ANKLE_Z_JF_L,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    foot_visual_L = pin.GeometryObject('foot_L',IDX_SFOOT_SF_L, IDX_ANKLE_Z_JF_L, mesh_loader.load(meshes_folder_path+'/foot_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.11, -0.07, 0.07]).T), meshes_folder_path+'/foot_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , body_color)
    geom_model.addGeometryObject(foot_visual_L)
    visuals_dict["foot_L"] = foot_visual_L

    model.upperPositionLimit[7:] = np.array([np.deg2rad(25),     #L5S1_FE + 
                                            np.deg2rad(45),      #L5S1_R_EXT_INT +
                                            np.deg2rad(124),     #Shoulder_Z_R +
                                            np.deg2rad(91),      #Shoulder_X_R +
                                            np.deg2rad(76),      #Shoulder_Y_R +
                                            np.deg2rad(145),     #Elbow_Z_R +
                                            np.deg2rad(73),      #Elbow_Y_R + 
                                            np.deg2rad(124),     #Shoulder_Z_L +
                                            np.deg2rad(126),     #Shoulder_X_L +
                                            np.deg2rad(76),      #Shoulder_Y_L +
                                            np.deg2rad(145),     #Elbow_Z_L +
                                            np.deg2rad(90),      #Elbow_Y_L +
                                            np.deg2rad(134),     #Hip_Z_R +
                                            np.deg2rad(38),      #Hip_X_R +
                                            np.deg2rad(40) ,     #Hip_Y_R +
                                            np.deg2rad(5),       #Knee_Z_R +
                                            np.deg2rad(21),      #Ankle_Z_R +
                                            np.deg2rad(134),     #Hip_Z_L +
                                            np.deg2rad(45),      #Hip_X_L +
                                            np.deg2rad(40),      #Hip_Y_L +
                                            np.deg2rad(5),       #Knee_Z_L +
                                            np.deg2rad(21),      #Ankle_Z_L +
                                            ]) 
    
    model.lowerPositionLimit[7:] = np.array([np.deg2rad(-66),    #L5S1_FE -
                                            np.deg2rad(-45),     #L5S1_R_EXT_INT -
                                            np.deg2rad(-55),     #Shoulder_Z_R -
                                            np.deg2rad(-126),    #Shoulder_X_R -
                                            np.deg2rad(-117),    #Shoulder_Y_R -
                                            np.deg2rad(0),       #Elbow_Z_R -
                                            np.deg2rad(-90),     #Elbow_Y_R -
                                            np.deg2rad(-55),     #Shoulder_Z_L -
                                            np.deg2rad(-91),     #Shoulder_X_L -
                                            np.deg2rad(-117),    #Shoulder_Y_L -
                                            np.deg2rad(0),       #Elbow_Z_L -
                                            np.deg2rad(-73),     #Elbow_Y_L -
                                            np.deg2rad(-27),     #Hip_Z_R -
                                            np.deg2rad(-45),     #Hip_X_R -
                                            np.deg2rad(-40),     #Hip_Y_R -
                                            np.deg2rad(-142),    #Knee_Z_R -
                                            np.deg2rad(-47),     #Ankle_Z_R -
                                            np.deg2rad(-27),     #Hip_Z_L -
                                            np.deg2rad(-38),     #Hip_X_L -
                                            np.deg2rad(-40),     #Hip_Y_L -
                                            np.deg2rad(-142),    #Knee_Z_L -
                                            np.deg2rad(-47),     #Ankle_Z_L -
                                            ])
    
    return model, geom_model, visuals_dict

#   Joint 2 L5S1_FE: parent=1
#   Joint 3 L5S1_R_EXT_INT: parent=2
#   Joint 4 Shoulder_Z_R: parent=3     7.0*np.pi/6    -np.pi/2.0
#   Joint 5 Shoulder_X_R: parent=4     0.6        -np.pi,
#   Joint 6 Shoulder_Y_R: parent=5     np.pi/2.0 + 0.5     -np.pi/3
#   Joint 7 Elbow_Z_R: parent=6      np.pi            0.0
#   Joint 8 Elbow_Y_R: parent=7      3*np.pi/4        -np.pi/6
# #   Joint 9 Shoulder_Z_L: parent=3   
# #   Joint 10 Shoulder_X_L: parent=9
# #   Joint 11 Shoulder_Y_L: parent=10
# #   Joint 12 Elbow_Z_L: parent=11        
# #   Joint 13 Elbow_Y_L: parent=12
#   Joint 14 Hip_Z_R: parent=1
#   Joint 15 Hip_X_R: parent=14
#   Joint 16 Hip_Y_R: parent=15
#   Joint 17 Knee_Z_R: parent=15
#   Joint 18 Ankle_Z_R: parent=17
# #   Joint 19 Hip_Z_L: parent=1
# #   Joint 20 Hip_X_L: parent=19
# #   Joint 21 Hip_Y_L: parent=20
# #   Joint 22 Knee_Z_L: parent=21
# #   Joint 23 Ankle_Z_L: parent=22