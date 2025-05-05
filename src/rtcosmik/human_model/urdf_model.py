from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
import numpy as np 
from typing import List, Tuple, Dict
from rtcosmik.human_model.model_utils import get_torso_pose
from rtcosmik.utils.linear_algebra_utils import col_vector_3D
from .model_utils import orthogonalize_matrix, construct_segments_frames, get_segments_mks_dict, get_local_mks_positions, get_local_segments_positions

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
        # else:
        #     # print(self.model.upperPositionLimit)
        #     self.model.upperPositionLimit = np.array([np.pi,np.pi,np.pi/8,np.pi/8,8*np.pi/9])
        #     self.model.lowerPositionLimit = np.array([0,0,-np.pi,-10*np.pi/9,0])
            # self.model.upperPositionLimit = np.array([np.pi/2,np.pi])
            # self.model.lowerPositionLimit = np.array([-np.pi/2,-1.57])

        ## \todo test that this is equivalent to reloading the model
        self.geom_model = self.collision_model

def scale_human_model(model, mks_positions, with_hand=True,gender='male',subject_height=1.80):

    sgts_poses = construct_segments_frames(mks_positions, with_hand=with_hand, gender='male',subject_height=1.8)
    local_segments_positions = get_local_segments_positions(sgts_poses)

    model.jointPlacements[model.getJointId('left_hip_Z')].translation=local_segments_positions['thighL']
    model.jointPlacements[model.getJointId('left_knee')].translation=local_segments_positions['shankL']
    model.jointPlacements[model.getJointId('left_ankle_Z')].translation=local_segments_positions['footL']

    model.jointPlacements[model.getJointId('middle_lumbar_Z')].translation=np.array([0,0,0])
    model.jointPlacements[model.getJointId('middle_thoracic_Z')].translation=local_segments_positions['thorax']

    model.jointPlacements[model.getJointId('middle_cervical_Z')].translation=local_segments_positions['torso']
    model.jointPlacements[model.getJointId('left_clavicle_joint_X')].translation=local_segments_positions['torso']
    model.jointPlacements[model.getJointId('right_clavicle_joint_X')].translation=local_segments_positions['torso']

    model.jointPlacements[model.getJointId('left_shoulder_Z')].translation=local_segments_positions['upperarmL']
    model.jointPlacements[model.getJointId('left_elbow_Z')].translation=local_segments_positions['lowerarmL']

    model.jointPlacements[model.getJointId('right_shoulder_Z')].translation=local_segments_positions['upperarmR']
    model.jointPlacements[model.getJointId('right_elbow_Z')].translation=local_segments_positions['lowerarmR']
    model.jointPlacements[model.getJointId('right_hip_Z')].translation=local_segments_positions['thighR']
    model.jointPlacements[model.getJointId('right_knee')].translation=local_segments_positions['shankR']
    model.jointPlacements[model.getJointId('right_ankle_Z')].translation=local_segments_positions['footR']

    if with_hand:
        model.jointPlacements[model.getJointId('left_wrist_Z')].translation=local_segments_positions['handL']
        model.jointPlacements[model.getJointId('right_wrist_Z')].translation=local_segments_positions['handR']
    return model

def mks_registration(model,mks_positions, with_hand=True):
    sgts_poses = construct_segments_frames(mks_positions, with_hand=with_hand, gender='male',subject_height=1.8)
    sgts_mks_dict = get_segments_mks_dict(mks_positions)
    mks_local_positions = get_local_mks_positions(sgts_poses, mks_positions, sgts_mks_dict)

    inertia = pin.Inertia.Zero()

    idx_frame = model.getFrameId('middle_pelvis')
    joint = model.getJointId('root_joint')
    for i in sgts_mks_dict["pelvis"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        model.addFrame(frame,False)

    idx_frame = model.getFrameId('middle_thorax')
    joint = model.getJointId('middle_thoracic_Z')
    for i in sgts_mks_dict["thorax"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        model.addFrame(frame,False)
    
    idx_frame = model.getFrameId('middle_head')
    joint = model.getJointId('middle_cervical_Y')
    for i in sgts_mks_dict["head"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    idx_frame = model.getFrameId('right_clavicle')
    joint = model.getJointId('right_clavicle_joint_X')
    for i in sgts_mks_dict["right_clavicle"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    idx_frame = model.getFrameId('left_clavicle')
    joint = model.getJointId('left_clavicle_joint_X')
    for i in sgts_mks_dict["left_clavicle"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    idx_frame = model.getFrameId('right_upperarm')
    joint = model.getJointId('right_shoulder_X')
    for i in sgts_mks_dict["upperarmR"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    idx_frame = model.getFrameId('left_upperarm')
    joint = model.getJointId('left_shoulder_X')
    for i in sgts_mks_dict["upperarmL"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    idx_frame = model.getFrameId('right_lowerarm')
    joint = model.getJointId('right_elbow_Y')
    for i in sgts_mks_dict["lowerarmR"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    idx_frame = model.getFrameId('left_lowerarm')
    joint = model.getJointId('left_elbow_Y')
    for i in sgts_mks_dict["lowerarmL"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    idx_frame = model.getFrameId('right_upperleg')
    joint = model.getJointId('right_hip_Y')
    for i in sgts_mks_dict["thighR"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    idx_frame = model.getFrameId('left_upperleg')
    joint = model.getJointId('left_hip_Y')
    for i in sgts_mks_dict["thighL"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    idx_frame = model.getFrameId('right_lowerleg')
    joint = model.getJointId('right_knee')
    for i in sgts_mks_dict["shankR"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    idx_frame = model.getFrameId('left_lowerleg')
    joint = model.getJointId('left_knee')
    for i in sgts_mks_dict["shankL"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    idx_frame = model.getFrameId('right_foot')
    joint = model.getJointId('right_ankle_X')
    for i in sgts_mks_dict["footR"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    idx_frame = model.getFrameId('left_foot')
    joint = model.getJointId('left_ankle_X')
    for i in sgts_mks_dict["footL"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)



    return model







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

def get_jcp_global_pos(mks_positions, pos_ankle_calib, side_to_track):
    names = ['Ankle', 'Knee', 'midHip', 'Shoulder', 'Elbow', 'Wrist']

    if side_to_track == "bilateral":
        ankle_center = ((mks_positions['L_mankle_study'] + mks_positions['L_ankle_study']).reshape(3,1)/2.0 + (mks_positions['r_mankle_study'] + mks_positions['r_ankle_study']).reshape(3,1)/2.0)/2
        ankle_offset = ankle_center - pos_ankle_calib.reshape(3,1)
        knee_center = ((mks_positions['L_knee_study'] + mks_positions['L_mknee_study']).reshape(3,1)/2.0 + (mks_positions['r_knee_study'] + mks_positions['r_mknee_study']).reshape(3,1)/2.0)/2
        midhip = (mks_positions['r.ASIS_study'] + mks_positions['L.ASIS_study'] + mks_positions['r.PSIS_study'] + mks_positions['L.PSIS_study']).reshape(3,1)/4.0

        trunk_center = (mks_positions['r_shoulder_study'] + mks_positions['L_shoulder_study']).reshape(3,1)/2.0 

        Y = (trunk_center - midhip).reshape(3,1)
        Y = Y/np.linalg.norm(Y)
        X = (trunk_center - mks_positions['C7_study'].reshape(3,1)).reshape(3,1)
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

        bi_acromial_dist = np.linalg.norm(mks_positions['L_shoulder_study'].reshape(3,1) - mks_positions['r_shoulder_study'].reshape(3,1))
        Rshoulder_center = mks_positions['r_shoulder_study'].reshape(3,1) + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0).reshape(3,1)
        Lshoulder_center = mks_positions['L_shoulder_study'].reshape(3,1) + (torso_pose[:3, :3].reshape(3,3) @ col_vector_3D(0., -0.17*bi_acromial_dist, 0)).reshape(3,1)
        shoulder_center = (Rshoulder_center.reshape(3,1) + Lshoulder_center.reshape(3,1))/2
        elbow_center = ((mks_positions['L_melbow_study'] + mks_positions['L_lelbow_study']).reshape(3,1)/2.0 +  (mks_positions['r_melbow_study']+ mks_positions['r_lelbow_study']).reshape(3,1)/2.0)/2
        wrist_center = ((mks_positions['L_mwrist_study'] + mks_positions['L_lwrist_study']).reshape(3,1)/2.0 + (mks_positions['r_mwrist_study'] + mks_positions['r_lwrist_study']).reshape(3,1)/2.0)/2
    
    elif side_to_track == "right":
        ankle_center = (mks_positions['r_mankle_study'] + mks_positions['r_ankle_study']).reshape(3,1)/2.0
        ankle_offset = ankle_center - pos_ankle_calib.reshape(3,1)
        knee_center = (mks_positions['r_knee_study'] + mks_positions['r_mknee_study']).reshape(3,1)/2.0
        midhip = mks_positions['RHJC_study'].reshape(3,1)

        trunk_center = (mks_positions['r_shoulder_study'] + mks_positions['L_shoulder_study']).reshape(3,1)/2.0 

        Y = (trunk_center - midhip).reshape(3,1)
        Y = Y/np.linalg.norm(Y)
        X = (trunk_center - mks_positions['C7_study'].reshape(3,1)).reshape(3,1)
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

        bi_acromial_dist = np.linalg.norm(mks_positions['L_shoulder_study'].reshape(3,1) - mks_positions['r_shoulder_study'].reshape(3,1))
        Rshoulder_center = mks_positions['r_shoulder_study'].reshape(3,1) + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0).reshape(3,1)
        shoulder_center = Rshoulder_center.reshape(3,1)

        elbow_center =  (mks_positions['r_melbow_study']+ mks_positions['r_lelbow_study']).reshape(3,1)/2.0
        wrist_center =  (mks_positions['r_mwrist_study'] + mks_positions['r_lwrist_study']).reshape(3,1)/2.0
    
    elif side_to_track == "left":
        ankle_center = (mks_positions['L_mankle_study'] + mks_positions['L_ankle_study']).reshape(3,1)/2.0
        ankle_offset = ankle_center - pos_ankle_calib.reshape(3,1)
        knee_center = (mks_positions['L_knee_study'] + mks_positions['L_mknee_study']).reshape(3,1)/2.0
        midhip = mks_positions['LHJC_study'].reshape(3,1)

        trunk_center = (mks_positions['r_shoulder_study'] + mks_positions['L_shoulder_study']).reshape(3,1)/2.0 

        Y = (trunk_center - midhip).reshape(3,1)
        Y = Y/np.linalg.norm(Y)
        X = (trunk_center - mks_positions['C7_study'].reshape(3,1)).reshape(3,1)
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

        bi_acromial_dist = np.linalg.norm(mks_positions['L_shoulder_study'].reshape(3,1) - mks_positions['r_shoulder_study'].reshape(3,1))
        Lshoulder_center = mks_positions['L_shoulder_study'].reshape(3,1) + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0).reshape(3,1)
        shoulder_center = Lshoulder_center.reshape(3,1)

        elbow_center =  (mks_positions['L_melbow_study']+ mks_positions['L_lelbow_study']).reshape(3,1)/2.0
        wrist_center =  (mks_positions['L_mwrist_study'] + mks_positions['L_lwrist_study']).reshape(3,1)/2.0

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


def get_jcp_global_pos_2dof(mks_positions, side_to_track):
    names = ['Shoulder', 'Elbow', 'Wrist']
    torso_pose = []

    if side_to_track == "right":
        torso_pose = get_torso_pose(mks_positions)
        bi_acromial_dist = np.linalg.norm(mks_positions['L_shoulder_study'].reshape(3,1) - mks_positions['r_shoulder_study'].reshape(3,1))
        Rshoulder_center = mks_positions['r_shoulder_study'].reshape(3,1) + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0).reshape(3,1)
        shoulder_center = Rshoulder_center.reshape(3,1)

        elbow_center =  (mks_positions['r_melbow_study']+ mks_positions['r_lelbow_study']).reshape(3,1)/2.0
        wrist_center =  (mks_positions['r_mwrist_study'] + mks_positions['r_lwrist_study']).reshape(3,1)/2.0
    
    elif side_to_track == "left":
        torso_pose = get_torso_pose(mks_positions)
        bi_acromial_dist = np.linalg.norm(mks_positions['L_shoulder_study'].reshape(3,1) - mks_positions['r_shoulder_study'].reshape(3,1))
        Lshoulder_center = mks_positions['L_shoulder_study'].reshape(3,1) + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0).reshape(3,1)
        shoulder_center = Lshoulder_center.reshape(3,1)

        elbow_center =  (mks_positions['L_melbow_study']+ mks_positions['L_lelbow_study']).reshape(3,1)/2.0
        wrist_center =  (mks_positions['L_mwrist_study'] + mks_positions['L_lwrist_study']).reshape(3,1)/2.0


    jcp = [shoulder_center, elbow_center, wrist_center]

    return dict(zip(names,jcp))
