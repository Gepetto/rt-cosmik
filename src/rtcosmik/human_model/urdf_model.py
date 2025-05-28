from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
import numpy as np 
from typing import List, Tuple, Dict
from src.rtcosmik.human_model.model_utils import get_torso_pose
from src.rtcosmik.utils.linear_algebra_utils import col_vector_3D
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
        #     self.model.upperPositionLimit = np.full(43, np.pi)
        #     self.model.lowerPositionLimit = np.full(43, -np.pi)
            
        ## \todo test that this is equivalent to reloading the model
        self.geom_model = self.collision_model

def scale_human_model(model, mks_positions, with_hand=True,gender='male',subject_height=1.80):

    sgts_poses = construct_segments_frames(mks_positions, with_hand=with_hand, gender=gender,subject_height=subject_height)
    local_segments_positions = get_local_segments_positions(sgts_poses,with_hand=True)

    model.jointPlacements[model.getJointId('left_hip_Z')].translation=local_segments_positions['thighL']
    model.jointPlacements[model.getJointId('left_knee')].translation=col_vector_3D(0, -np.linalg.norm(local_segments_positions['shankL']),0)
    model.jointPlacements[model.getJointId('left_ankle_Z')].translation=col_vector_3D(0, -np.linalg.norm(local_segments_positions['footL']),0)

    model.jointPlacements[model.getJointId('middle_lumbar_Z')].translation=np.array([0,0,0])
    model.jointPlacements[model.getJointId('middle_thoracic_Z')].translation=col_vector_3D(0, np.linalg.norm(local_segments_positions['thorax']),0)

    model.jointPlacements[model.getJointId('left_clavicle_joint_X')].translation=col_vector_3D(0, np.linalg.norm(local_segments_positions['torso']),0)
    model.jointPlacements[model.getJointId('right_clavicle_joint_X')].translation=col_vector_3D(0, np.linalg.norm(local_segments_positions['torso']),0)
    model.jointPlacements[model.getJointId('middle_cervical_Z')].translation=col_vector_3D(0, np.linalg.norm(local_segments_positions['torso']),0)
   
    model.jointPlacements[model.getJointId('left_shoulder_Z')].translation=local_segments_positions['upperarmL']
    model.jointPlacements[model.getJointId('left_elbow_Z')].translation=col_vector_3D(0, -np.linalg.norm(local_segments_positions['lowerarmL']),0)

    model.jointPlacements[model.getJointId('right_shoulder_Z')].translation=local_segments_positions['upperarmR']
    model.jointPlacements[model.getJointId('right_elbow_Z')].translation=col_vector_3D(0, -np.linalg.norm(local_segments_positions['lowerarmR']),0)
    model.jointPlacements[model.getJointId('right_hip_Z')].translation=local_segments_positions['thighR']
    model.jointPlacements[model.getJointId('right_knee')].translation=col_vector_3D(0, -np.linalg.norm(local_segments_positions['shankR']),0)
    model.jointPlacements[model.getJointId('right_ankle_Z')].translation=col_vector_3D(0, -np.linalg.norm(local_segments_positions['footR']),0)

    if with_hand:
        model.jointPlacements[model.getJointId('left_wrist_Z')].translation=col_vector_3D(0, -np.linalg.norm(local_segments_positions['handL']),0)
        model.jointPlacements[model.getJointId('right_wrist_Z')].translation=col_vector_3D(0, -np.linalg.norm(local_segments_positions['handR']),0)
    return model

def mks_registration(model,mks_positions, with_hand=True, gender='male',subject_height=1.8):
    #attach mks to segment and joint
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
    joint = model.getJointId('middle_thoracic_Y')
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
    joint = model.getJointId('right_shoulder_Y')
    for i in sgts_mks_dict["upperarmR"]:
        frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    idx_frame = model.getFrameId('left_upperarm')
    joint = model.getJointId('left_shoulder_Y')
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

    if with_hand:
        idx_frame = model.getFrameId('right_hand')
        joint = model.getJointId('right_wrist_X')
        for i in sgts_mks_dict["handR"]:
            frame = pin.Frame(i,joint,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
            idx_frame = model.addFrame(frame,False)

        idx_frame = model.getFrameId('left_hand')
        joint = model.getJointId('left_wrist_X')
        for i in sgts_mks_dict["handL"]:
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