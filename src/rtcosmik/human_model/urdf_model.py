from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
import numpy as np 
from typing import List, Tuple, Dict
from human_model.model_utils import get_torso_pose
from utils.linear_algebra_utils import col_vector_3D

class Robot(RobotWrapper):
    """_Class to load a given urdf_

    Args:
        RobotWrapper (_type_): _description_
    """
    def __init__(self,
                 robot_urdf,
                 package_dirs,
                 isFext=False,
                 freeflyer_ori = None):
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
            # self.model.upperPositionLimit = np.array([np.pi,np.pi,np.pi/8,np.pi/8,8*np.pi/9])
            # self.model.lowerPositionLimit = np.array([0,0,-np.pi,-10*np.pi/9,0])
            self.model.upperPositionLimit = np.array([np.pi/2,np.pi])
            self.model.lowerPositionLimit = np.array([-np.pi/2,-1.57])

        ## \todo test that this is equivalent to reloading the model
        self.geom_model = self.collision_model

# scale model of 5dofs
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

def model_scaling_from_dict(model, dict):

    lowerleg_l = dict['Knee']
    upperleg_l = dict['Hip']   
    trunk_l = dict['Shoulder']
    upperarm_l = dict['Elbow']
    lowerarm_l = dict['Wrist']

    model.jointPlacements[model.getJointId('knee_Z')].translation=np.array([lowerleg_l,0,0])
    model.jointPlacements[model.getJointId('lumbar_Z')].translation=np.array([upperleg_l,0,0])
    model.jointPlacements[model.getJointId('shoulder_Z')].translation=np.array([trunk_l,0,0])
    model.jointPlacements[model.getJointId('elbow_Z')].translation=np.array([upperarm_l,0,0])
    model.frames[model.getFrameId('hand_fixed')].translation=np.array([lowerarm_l,0,0])
    model.frames[model.getFrameId('hand')].translation=np.array([lowerarm_l,0,0])

    return model

def get_jcp_global_pos(mocap_mks_positions, pos_ankle_calib, side_to_track):
    names = ['Ankle', 'Knee', 'midHip', 'Shoulder', 'Elbow', 'Wrist']

    if side_to_track == "bilateral":
        ankle_center = ((mocap_mks_positions['L_mankle_study'] + mocap_mks_positions['L_ankle_study']).reshape(3,1)/2.0 + (mocap_mks_positions['r_mankle_study'] + mocap_mks_positions['r_ankle_study']).reshape(3,1)/2.0)/2
        ankle_offset = ankle_center - pos_ankle_calib.reshape(3,1)
        knee_center = ((mocap_mks_positions['L_knee_study'] + mocap_mks_positions['L_mknee_study']).reshape(3,1)/2.0 + (mocap_mks_positions['r_knee_study'] + mocap_mks_positions['r_mknee_study']).reshape(3,1)/2.0)/2
        midhip = (mocap_mks_positions['r.ASIS_study'] + mocap_mks_positions['L.ASIS_study'] + mocap_mks_positions['r.PSIS_study'] + mocap_mks_positions['L.PSIS_study']).reshape(3,1)/4.0

        trunk_center = (mocap_mks_positions['r_shoulder_study'] + mocap_mks_positions['L_shoulder_study']).reshape(3,1)/2.0 

        Y = (trunk_center - midhip).reshape(3,1)
        Y = Y/np.linalg.norm(Y)
        X = (trunk_center - mocap_mks_positions['C7_study'].reshape(3,1)).reshape(3,1)
        X = X/np.linalg.norm(X)
        Z = np.cross(X, Y, axis=0)
        X = np.cross(Y, Z, axis=0)

        pose = np.eye(4,4)
        pose[:3,0] = X.reshape(3,)
        pose[:3,1] = Y.reshape(3,)
        pose[:3,2] = Z.reshape(3,)
        pose[:3,3] = trunk_center.reshape(3,)
        pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

        torso_pose = pose

        bi_acromial_dist = np.linalg.norm(mocap_mks_positions['L_shoulder_study'].reshape(3,1) - mocap_mks_positions['r_shoulder_study'].reshape(3,1))
        Rshoulder_center = mocap_mks_positions['r_shoulder_study'].reshape(3,1) + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0).reshape(3,1)
        Lshoulder_center = mocap_mks_positions['L_shoulder_study'].reshape(3,1) + (torso_pose[:3, :3].reshape(3,3) @ col_vector_3D(0., -0.17*bi_acromial_dist, 0)).reshape(3,1)
        shoulder_center = (Rshoulder_center.reshape(3,1) + Lshoulder_center.reshape(3,1))/2
        elbow_center = ((mocap_mks_positions['L_melbow_study'] + mocap_mks_positions['L_lelbow_study']).reshape(3,1)/2.0 +  (mocap_mks_positions['r_melbow_study']+ mocap_mks_positions['r_lelbow_study']).reshape(3,1)/2.0)/2
        wrist_center = ((mocap_mks_positions['L_mwrist_study'] + mocap_mks_positions['L_lwrist_study']).reshape(3,1)/2.0 + (mocap_mks_positions['r_mwrist_study'] + mocap_mks_positions['r_lwrist_study']).reshape(3,1)/2.0)/2
    
    elif side_to_track == "right":
        ankle_center = (mocap_mks_positions['r_mankle_study'] + mocap_mks_positions['r_ankle_study']).reshape(3,1)/2.0
        ankle_offset = ankle_center - pos_ankle_calib.reshape(3,1)
        knee_center = (mocap_mks_positions['r_knee_study'] + mocap_mks_positions['r_mknee_study']).reshape(3,1)/2.0
        midhip = mocap_mks_positions['RHJC_study'].reshape(3,1)

        trunk_center = (mocap_mks_positions['r_shoulder_study'] + mocap_mks_positions['L_shoulder_study']).reshape(3,1)/2.0 

        Y = (trunk_center - midhip).reshape(3,1)
        Y = Y/np.linalg.norm(Y)
        X = (trunk_center - mocap_mks_positions['C7_study'].reshape(3,1)).reshape(3,1)
        X = X/np.linalg.norm(X)
        Z = np.cross(X, Y, axis=0)
        X = np.cross(Y, Z, axis=0)

        pose = np.eye(4,4)
        pose[:3,0] = X.reshape(3,)
        pose[:3,1] = Y.reshape(3,)
        pose[:3,2] = Z.reshape(3,)
        pose[:3,3] = trunk_center.reshape(3,)
        pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

        torso_pose = pose

        bi_acromial_dist = np.linalg.norm(mocap_mks_positions['L_shoulder_study'].reshape(3,1) - mocap_mks_positions['r_shoulder_study'].reshape(3,1))
        Rshoulder_center = mocap_mks_positions['r_shoulder_study'].reshape(3,1) + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0).reshape(3,1)
        shoulder_center = Rshoulder_center.reshape(3,1)

        elbow_center =  (mocap_mks_positions['r_melbow_study']+ mocap_mks_positions['r_lelbow_study']).reshape(3,1)/2.0
        wrist_center =  (mocap_mks_positions['r_mwrist_study'] + mocap_mks_positions['r_lwrist_study']).reshape(3,1)/2.0
    
    elif side_to_track == "left":
        ankle_center = (mocap_mks_positions['L_mankle_study'] + mocap_mks_positions['L_ankle_study']).reshape(3,1)/2.0
        ankle_offset = ankle_center - pos_ankle_calib.reshape(3,1)
        knee_center = (mocap_mks_positions['L_knee_study'] + mocap_mks_positions['L_mknee_study']).reshape(3,1)/2.0
        midhip = mocap_mks_positions['LHJC_study'].reshape(3,1)

        trunk_center = (mocap_mks_positions['r_shoulder_study'] + mocap_mks_positions['L_shoulder_study']).reshape(3,1)/2.0 

        Y = (trunk_center - midhip).reshape(3,1)
        Y = Y/np.linalg.norm(Y)
        X = (trunk_center - mocap_mks_positions['C7_study'].reshape(3,1)).reshape(3,1)
        X = X/np.linalg.norm(X)
        Z = np.cross(X, Y, axis=0)
        X = np.cross(Y, Z, axis=0)

        pose = np.eye(4,4)
        pose[:3,0] = X.reshape(3,)
        pose[:3,1] = Y.reshape(3,)
        pose[:3,2] = Z.reshape(3,)
        pose[:3,3] = trunk_center.reshape(3,)
        pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

        torso_pose = pose

        bi_acromial_dist = np.linalg.norm(mocap_mks_positions['L_shoulder_study'].reshape(3,1) - mocap_mks_positions['r_shoulder_study'].reshape(3,1))
        Lshoulder_center = mocap_mks_positions['L_shoulder_study'].reshape(3,1) + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0).reshape(3,1)
        shoulder_center = Lshoulder_center.reshape(3,1)

        elbow_center =  (mocap_mks_positions['L_melbow_study']+ mocap_mks_positions['L_lelbow_study']).reshape(3,1)/2.0
        wrist_center =  (mocap_mks_positions['L_mwrist_study'] + mocap_mks_positions['L_lwrist_study']).reshape(3,1)/2.0

    # Set the ankle joint center to zero in global frame
    ankle_center -= ankle_offset
    knee_center -= ankle_offset
    midhip -= ankle_offset
    shoulder_center -= ankle_offset
    elbow_center -= ankle_offset
    wrist_center -= ankle_offset

    jcp = [ankle_center, knee_center, midhip, shoulder_center, elbow_center, wrist_center]

    return dict(zip(names,jcp))

def calculate_segment_lengths_from_dict(dict):
    lowerleg_l = np.linalg.norm(dict['Knee']-dict['Ankle'])
    upperleg_l = np.linalg.norm(dict['midHip']-dict['Knee'])
    trunk_l = np.linalg.norm(dict['Shoulder']-dict['midHip'])
    upperarm_l = np.linalg.norm(dict['Elbow']-dict['Shoulder'])
    lowerarm_l = np.linalg.norm(dict['Wrist']-dict['Elbow'])

    return np.array([lowerleg_l, upperleg_l, trunk_l, upperarm_l, lowerarm_l])


####2dofs

def model_scaling_from_dict_2dof(model, dict):

    upperarm_l = dict['Elbow']
    lowerarm_l = dict['Wrist']
    
    model.jointPlacements[model.getJointId('elbow')].translation=np.array([upperarm_l,0,0])
    model.frames[model.getFrameId('hand_fixed')].translation=np.array([lowerarm_l,0,0])
    model.frames[model.getFrameId('hand')].translation=np.array([lowerarm_l,0,0])

    return model

def calculate_segment_lengths_from_dict_2dof(dict):
    upperarm_l = np.linalg.norm(dict['Elbow']-dict['Shoulder'])
    lowerarm_l = np.linalg.norm(dict['Wrist']-dict['Elbow'])

    return np.array([upperarm_l, lowerarm_l])


def get_jcp_global_pos_2dof(mocap_mks_positions, side_to_track):
    names = ['Shoulder', 'Elbow', 'Wrist']
    torso_pose = []

    if side_to_track == "right":
        torso_pose = get_torso_pose(mocap_mks_positions)
        bi_acromial_dist = np.linalg.norm(mocap_mks_positions['L_shoulder_study'].reshape(3,1) - mocap_mks_positions['r_shoulder_study'].reshape(3,1))
        Rshoulder_center = mocap_mks_positions['r_shoulder_study'].reshape(3,1) + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0).reshape(3,1)
        shoulder_center = Rshoulder_center.reshape(3,1)

        elbow_center =  (mocap_mks_positions['r_melbow_study']+ mocap_mks_positions['r_lelbow_study']).reshape(3,1)/2.0
        wrist_center =  (mocap_mks_positions['r_mwrist_study'] + mocap_mks_positions['r_lwrist_study']).reshape(3,1)/2.0
    
    elif side_to_track == "left":
        torso_pose = get_torso_pose(mocap_mks_positions)
        bi_acromial_dist = np.linalg.norm(mocap_mks_positions['L_shoulder_study'].reshape(3,1) - mocap_mks_positions['r_shoulder_study'].reshape(3,1))
        Lshoulder_center = mocap_mks_positions['L_shoulder_study'].reshape(3,1) + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0).reshape(3,1)
        shoulder_center = Lshoulder_center.reshape(3,1)

        elbow_center =  (mocap_mks_positions['L_melbow_study']+ mocap_mks_positions['L_lelbow_study']).reshape(3,1)/2.0
        wrist_center =  (mocap_mks_positions['L_mwrist_study'] + mocap_mks_positions['L_lwrist_study']).reshape(3,1)/2.0


    jcp = [shoulder_center, elbow_center, wrist_center]

    return dict(zip(names,jcp))
