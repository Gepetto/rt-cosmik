import pinocchio as pin
import numpy as np 
import hppfcl as fcl
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Dict
from rtcosmik.utils.linear_algebra_utils import col_vector_3D
from .model_utils import construct_segments_frames, get_segments_mks_dict, get_local_mks_positions, get_local_segments_positions

#### MODEL DESCRIPTION ####
#  Joint 0 universe  
#  Joint 1 root_joint: parent=0
# #   Joint 2 L5S1_FE: parent=1
# #   Joint 3 L5S1_R_EXT_INT: parent=2
# #   Joint 4 Neck_FE: parent=3
# #   Joint 5 Neck_LAT_BEND: parent=4
# #   Joint 6 Neck_R_EXT_INT: parent=5
#   Joint 7 Shoulder_Z_R: parent=3     7.0*np.pi/6    -np.pi/2.0
#   Joint 8 Shoulder_X_R: parent=7     0.6        -np.pi,
#   Joint 9 Shoulder_Y_R: parent=8     np.pi/2.0 + 0.5     -np.pi/3
#   Joint 10 Elbow_Z_R: parent=9      np.pi            0.0
#   Joint 11 Elbow_Y_R: parent=10      3*np.pi/4        -np.pi/6
# #   Joint 12 Shoulder_Z_L: parent=3   
# #   Joint 13 Shoulder_X_L: parent=12
# #   Joint 14 Shoulder_Y_L: parent=13
# #   Joint 15 Elbow_Z_L: parent=14        
# #   Joint 16 Elbow_Y_L: parent=15
#   Joint 17 Hip_Z_R: parent=1
#   Joint 18 Hip_X_R: parent=17
#   Joint 19 Hip_Y_R: parent=18
#   Joint 20 Knee_Z_R: parent=19
#   Joint 21 Ankle_Z_R: parent=20
# #   Joint 22 Hip_Z_L: parent=1
# #   Joint 23 Hip_X_L: parent=22
# #   Joint 24 Hip_Y_L: parent=23
# #   Joint 25 Knee_Z_L: parent=24
# #   Joint 26 Ankle_Z_L: parent=25

def build_model_no_visuals(mocap_mks_positions: Dict)->pin.Model:
    """_Build the biomechanical model associated to one exercise for one subject_

    Args:
        mocap_mks_positions (Dict): _mocap_mks_positions is a dictionnary of mocap mks names and 3x1 global positions_
        mks_positions (Dict): _mks_positions is a dictionnary of lstm mks names and 3x1 global positions_
        meshes_folder_path (str): _meshes_folder_path is the path to the folder containing the meshes_

    Returns:
        Tuple[pin.Model,pin.GeomModel, Dict]: _returns the pinocchio model, geometry model, and a dictionnary with visuals._
    """

    sgts_poses = construct_segments_frames(mocap_mks_positions)
    sgts_mks_dict = get_segments_mks_dict()
    mks_local_positions = get_local_mks_positions(sgts_poses, mocap_mks_positions, sgts_mks_dict)
    local_segments_positions = get_local_segments_positions(sgts_poses)

    # MODEL GENERATION 
    inertia = pin.Inertia.Zero()
    model= pin.Model() # pin model

    # pelvis with Freeflyer
    IDX_PELV_JF = model.addJoint(0,pin.JointModelFreeFlyer(),pin.SE3(np.array([[1,0,0],[0,0,-1],[0,1,0]]), np.matrix([0,0,0]).T),'root_joint')
    pelvis = pin.Frame('pelvis',IDX_PELV_JF,0,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_PELV_SF = model.addFrame(pelvis,False)
    # Add markers data
    idx_frame = IDX_PELV_SF
    for i in sgts_mks_dict["pelvis"]:
        frame = pin.Frame(i,IDX_PELV_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

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
        frame = pin.Frame(i,IDX_L5S1_R_EXT_INT_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]+ local_segments_positions['torso']).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Cervical ZXY
    IDX_NECK_Z_JF = model.addJoint(IDX_L5S1_R_EXT_INT_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['head'] + local_segments_positions['torso']).T),'cervical_Z')
    head = pin.Frame('head_z',IDX_NECK_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_HEAD_SF = model.addFrame(head,False)
    idx_frame = IDX_HEAD_SF

    IDX_NECK_X_JF = model.addJoint(IDX_NECK_Z_JF,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'cervical_X')
    head = pin.Frame('head_x',IDX_NECK_X_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_HEAD_SF = model.addFrame(head,False)
    idx_frame = IDX_HEAD_SF

    IDX_NECK_Y_JF = model.addJoint(IDX_NECK_X_JF,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'cervical_Y')
    head = pin.Frame('head',IDX_NECK_Y_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_HEAD_SF = model.addFrame(head,False)
    idx_frame = IDX_HEAD_SF

    for i in sgts_mks_dict["head"]:
        frame = pin.Frame(i,IDX_NECK_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Right Shoulder ZXY
    IDX_SH_Z_JF_R = model.addJoint(IDX_L5S1_R_EXT_INT_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['upperarmR'] + local_segments_positions['torso']).T),'right_shoulder_Z') 
    upperarmR = pin.Frame('upperarm_z_R',IDX_SH_Z_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    idx_frame = IDX_UPA_SF_R

    IDX_SH_X_JF_R = model.addJoint(IDX_SH_Z_JF_R,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'right_shoulder_X') 
    upperarmR = pin.Frame('upperarm_x_R',IDX_SH_X_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    idx_frame = IDX_UPA_SF_R

    IDX_SH_Y_JF_R = model.addJoint(IDX_SH_X_JF_R,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'right_shoulder_Y') 
    upperarmR = pin.Frame('upperarmR',IDX_SH_Y_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    idx_frame = IDX_UPA_SF_R
    for i in sgts_mks_dict["upperarmR"]:
        frame = pin.Frame(i,IDX_SH_Y_JF_R,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Right Elbow ZY 
    IDX_EL_Z_JF_R = model.addJoint(IDX_SH_Y_JF_R,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['lowerarmR']).T),'right_elbow_Z') 
    lowerarmR = pin.Frame('lowerarm_z',IDX_EL_Z_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF = model.addFrame(lowerarmR,False)
    idx_frame = IDX_LOA_SF

    IDX_EL_Y_JF = model.addJoint(IDX_EL_Z_JF_R,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'right_elbow_Y') 
    lowerarmR = pin.Frame('lowerarmR',IDX_EL_Y_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF = model.addFrame(lowerarmR,False)
    idx_frame = IDX_LOA_SF

    for i in sgts_mks_dict["lowerarmR"]:
        frame = pin.Frame(i,IDX_EL_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Left shoulder ZXY
    IDX_SH_Z_JF_L = model.addJoint(IDX_L5S1_R_EXT_INT_JF, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix(local_segments_positions['upperarmL'] + local_segments_positions['torso']).T), 'left_shoulder_Z') 
    upperarmL = pin.Frame('upperarm_z_L', IDX_SH_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    idx_frame = IDX_UPA_SF_L

    IDX_SH_X_JF_L = model.addJoint(IDX_SH_Z_JF_L, pin.JointModelRX(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'left_shoulder_X') 
    upperarmL = pin.Frame('upperarm_x_L', IDX_SH_X_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    idx_frame = IDX_UPA_SF_L

    IDX_SH_Y_JF_L = model.addJoint(IDX_SH_X_JF_L, pin.JointModelRY(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'left_shoulder_Y') 
    upperarmL = pin.Frame('upperarmL', IDX_SH_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    idx_frame = IDX_UPA_SF_L

    for i in sgts_mks_dict["upperarmL"]:
        frame = pin.Frame(i, IDX_SH_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix(mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    # Left Elbow ZY
    IDX_EL_Z_JF_L = model.addJoint(IDX_SH_Y_JF_L, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix(local_segments_positions['lowerarmL']).T), 'left_elbow_Z')
    lowerarmL = pin.Frame('lowerarm_z_L', IDX_EL_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF_L = model.addFrame(lowerarmL, False)
    idx_frame = IDX_LOA_SF_L

    IDX_EL_Y_JF_L = model.addJoint(IDX_EL_Z_JF_L, pin.JointModelRY(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'left_elbow_Y') 
    lowerarmL = pin.Frame('lowerarmL', IDX_EL_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF_L = model.addFrame(lowerarmL, False)
    idx_frame = IDX_LOA_SF_L

    for i in sgts_mks_dict["lowerarmL"]:
        frame = pin.Frame(i, IDX_EL_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix(mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

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
        frame = pin.Frame(i,IDX_HIP_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Right Knee Z
    IDX_KNEE_Z_JF = model.addJoint(IDX_HIP_X_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['shankR']).T),'right_knee_Z') 
    shankR = pin.Frame('shankR',IDX_KNEE_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SHANK_SF = model.addFrame(shankR,False)
    idx_frame = IDX_SHANK_SF

    for i in sgts_mks_dict["shankR"]:
        frame = pin.Frame(i,IDX_KNEE_Z_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Right Ankle Z
    IDX_ANKLE_Z_JF = model.addJoint(IDX_KNEE_Z_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['footR']).T),'right_ankle_Z') 
    footR = pin.Frame('footR',IDX_ANKLE_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SFOOT_SF = model.addFrame(footR,False)
    idx_frame = IDX_SFOOT_SF

    for i in sgts_mks_dict["footR"]:
        frame = pin.Frame(i,IDX_ANKLE_Z_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
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
        frame = pin.Frame(i, IDX_HIP_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix(mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    # Left Knee Z
    IDX_KNEE_Z_JF_L = model.addJoint(IDX_HIP_Y_JF_L,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['shankL']).T),'left_knee_Z') 
    shankR = pin.Frame('shankL',IDX_KNEE_Z_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SHANK_SF_L = model.addFrame(shankR,False)
    idx_frame = IDX_SHANK_SF_L

    for i in sgts_mks_dict["shankL"]:
        frame = pin.Frame(i,IDX_KNEE_Z_JF_L,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Left Ankle Z
    IDX_ANKLE_Z_JF_L = model.addJoint(IDX_KNEE_Z_JF_L,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['footL']).T),'left_ankle_Z') 
    footL = pin.Frame('footL',IDX_ANKLE_Z_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SFOOT_SF_L = model.addFrame(footL,False)
    idx_frame = IDX_SFOOT_SF_L

    for i in sgts_mks_dict["footL"]:
        frame = pin.Frame(i,IDX_ANKLE_Z_JF_L,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    model.upperPositionLimit[7:] = np.array([5*np.pi/36,       #L5S1_FE + 
                                          np.pi/3,             #L5S1_R_EXT_INT +
                                          np.pi/2,             # Neck_Z +
                                          np.pi/2,             # Neck_X +
                                          np.pi/2,             # Neck_Y +
                                          np.pi,               #Shoulder_Z_R +
                                          np.pi/3,             #Shoulder_X_R +
                                          np.pi/2,             #Shoulder_Y_R +
                                          5*np.pi/6,           #Elbow_Z_R +
                                          np.pi,               #Elbow_Y_R + 
                                          np.pi,               #Shoulder_Z_L +
                                          np.pi,               #Shoulder_X_L +
                                          np.pi/2,             #Shoulder_Y_L +
                                          5*np.pi/6,           #Elbow_Z_L +
                                          0.2,                 #Elbow_Y_L +
                                          np.pi/2,             #Hip_Z_R +
                                          np.pi/3,             #Hip_X_R +
                                          np.pi/3,             #Hip_Y_R +
                                          0,                   #Knee_Z_R +
                                          np.pi/4,             #Ankle_Z_R +
                                          np.pi/2,             #Hip_Z_L +
                                          np.pi/2,             #Hip_X_L +
                                          np.pi/2,             #Hip_Y_L +
                                          0,                   #Knee_Z_L +
                                          np.pi/4,             #Ankle_Z_L +
                                          ]) 
    
    model.lowerPositionLimit[7:] = np.array([-np.pi/2,           #L5S1_FE -
                                            -np.pi/3,            #L5S1_R_EXT_INT -
                                            -np.pi/2,            # Neck_Z -
                                            -np.pi/2,            # Neck_X -
                                            -np.pi/2,            # Neck_Y -
                                            -np.pi,              #Shoulder_Z_R -
                                            -np.pi,              #Shoulder_X_R -
                                            -np.pi/2,            #Shoulder_Y_R -
                                            0,                   #Elbow_Z_R -
                                            -0.2,                #Elbow_Y_R -
                                            -np.pi,              #Shoulder_Z_L -
                                            -np.pi/3,            #Shoulder_X_L -
                                            -np.pi/2,            #Shoulder_Y_L -
                                            0,                   #Elbow_Z_L -
                                            -np.pi,              #Elbow_Y_L -
                                            -np.pi/3,            #Hip_Z_R -
                                            -np.pi/2,            #Hip_X_R -
                                            -np.pi/2,            #Hip_Y_R -
                                            -5*np.pi/6,          #Knee_Z_R -
                                            -np.pi/2,            #Ankle_Z_R -
                                            -np.pi/3,            #Hip_Z_L -
                                            -np.pi/3,            #Hip_X_L -
                                            -np.pi/3,            #Hip_Y_L -
                                            -5*np.pi/6,          #Knee_Z_L -
                                            -np.pi/2,            #Ankle_Z_L -
                                            ])
    
    return model

def build_model(mocap_mks_positions: Dict, meshes_folder_path: str)->Tuple[pin.Model,pin.Model, Dict]:
    """_Build the biomechanical model associated to one exercise for one subject_

    Args:
        mocap_mks_positions (Dict): _mocap_mks_positions is a dictionnary of mocap mks names and 3x1 global positions_
        mks_positions (Dict): _mks_positions is a dictionnary of lstm mks names and 3x1 global positions_
        meshes_folder_path (str): _meshes_folder_path is the path to the folder containing the meshes_

    Returns:
        Tuple[pin.Model,pin.GeomModel, Dict]: _returns the pinocchio model, geometry model, and a dictionnary with visuals._
    """

    body_color = np.array([0,0,0,0.5])

    # TODO: Check that this model match the one in the urdf human.urdf and add abdomen joints ??
    sgts_poses = construct_segments_frames(mocap_mks_positions)
    sgts_mks_dict = get_segments_mks_dict()
    mks_local_positions = get_local_mks_positions(sgts_poses, mocap_mks_positions, sgts_mks_dict)
    local_segments_positions = get_local_segments_positions(sgts_poses)
    visuals_dict = {}

    # Mesh loader
    mesh_loader = fcl.MeshLoader()

    # MODEL GENERATION 
    inertia = pin.Inertia.Zero()
    model= pin.Model() # pin model
    geom_model = pin.GeometryModel() # geometry model

    # pelvis with Freeflyer
    IDX_PELV_JF = model.addJoint(0,pin.JointModelFreeFlyer(),pin.SE3(np.array([[1,0,0],[0,0,-1],[0,1,0]]), np.matrix([0,0,0]).T),'root_joint')
    pelvis = pin.Frame('pelvis',IDX_PELV_JF,0,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_PELV_SF = model.addFrame(pelvis,False)
    # Add markers data
    idx_frame = IDX_PELV_SF
    for i in sgts_mks_dict["pelvis"]:
        frame = pin.Frame(i,IDX_PELV_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    pelvis_visual = pin.GeometryObject('pelvis', IDX_PELV_SF, IDX_PELV_JF, mesh_loader.load(meshes_folder_path+'/pelvis_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/pelvis_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), True, body_color)
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
        frame = pin.Frame(i,IDX_L5S1_R_EXT_INT_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]+ local_segments_positions['torso']).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    torso_visual = pin.GeometryObject('torso', IDX_TORSO_SF, IDX_L5S1_JF, mesh_loader.load(meshes_folder_path+'/torso_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/torso_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), True, body_color)
    geom_model.addGeometryObject(torso_visual)
    visuals_dict["torso"] = torso_visual

    abdomen_visual = pin.GeometryObject('abdomen', IDX_TORSO_SF, IDX_L5S1_JF, mesh_loader.load(meshes_folder_path+'/abdomen_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/abdomen_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), True, body_color)
    geom_model.addGeometryObject(abdomen_visual)
    visuals_dict["abdomen"] = abdomen_visual

    # Cervical ZXY
    IDX_NECK_Z_JF = model.addJoint(IDX_L5S1_R_EXT_INT_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['head'] + local_segments_positions['torso']).T),'cervical_Z')
    head = pin.Frame('head_z',IDX_NECK_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_HEAD_SF = model.addFrame(head,False)
    idx_frame = IDX_HEAD_SF

    IDX_NECK_X_JF = model.addJoint(IDX_NECK_Z_JF,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'cervical_X')
    head = pin.Frame('head_x',IDX_NECK_X_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_HEAD_SF = model.addFrame(head,False)
    idx_frame = IDX_HEAD_SF

    IDX_NECK_Y_JF = model.addJoint(IDX_NECK_X_JF,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'cervical_Y')
    head = pin.Frame('head',IDX_NECK_Y_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_HEAD_SF = model.addFrame(head,False)
    idx_frame = IDX_HEAD_SF

    for i in sgts_mks_dict["head"]:
        frame = pin.Frame(i,IDX_NECK_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    head_visual = pin.GeometryObject('head', IDX_HEAD_SF, IDX_NECK_Y_JF, mesh_loader.load(meshes_folder_path+'/head_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/head_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), True, body_color)
    geom_model.addGeometryObject(head_visual)
    visuals_dict["head"] = head_visual

    neck_visual = pin.GeometryObject('neck', IDX_HEAD_SF, IDX_NECK_Y_JF, mesh_loader.load(meshes_folder_path+'/neck_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/neck_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), True, body_color)
    geom_model.addGeometryObject(neck_visual)
    visuals_dict["neck"] = neck_visual

    # Right Shoulder ZXY
    IDX_SH_Z_JF_R = model.addJoint(IDX_L5S1_R_EXT_INT_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['upperarmR'] + local_segments_positions['torso']).T),'right_shoulder_Z') 
    upperarmR = pin.Frame('upperarm_z_R',IDX_SH_Z_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    idx_frame = IDX_UPA_SF_R

    shoulder_visual_R = pin.GeometryObject('shoulder_R', IDX_UPA_SF_R, IDX_SH_Z_JF_R, mesh_loader.load(meshes_folder_path+'/shoulder_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/shoulder_mesh.STL',np.array([0.0055, 0.0055, 0.0055]), True , body_color)
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
        frame = pin.Frame(i,IDX_SH_Y_JF_R,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    upperarm_visual_R = pin.GeometryObject('upperarm_R', IDX_UPA_SF_R, IDX_SH_Y_JF_R, mesh_loader.load(meshes_folder_path+'/upperarm_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/upperarm_mesh.STL',np.array([0.0063, 0.0060, 0.007]), True , body_color)
    geom_model.addGeometryObject(upperarm_visual_R)
    visuals_dict["upperarm_R"] = upperarm_visual_R

    # Right Elbow ZY 
    IDX_EL_Z_JF_R = model.addJoint(IDX_SH_Y_JF_R,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['lowerarmR']).T),'right_elbow_Z') 
    lowerarmR = pin.Frame('lowerarm_z',IDX_EL_Z_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF = model.addFrame(lowerarmR,False)
    idx_frame = IDX_LOA_SF

    elbow_visual = pin.GeometryObject('elbow', IDX_LOA_SF, IDX_EL_Z_JF_R, mesh_loader.load(meshes_folder_path+'/elbow_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/elbow_mesh.STL',np.array([0.0055, 0.0055, 0.0055]), True , body_color)
    geom_model.addGeometryObject(elbow_visual)
    visuals_dict["elbow"] = elbow_visual

    IDX_EL_Y_JF = model.addJoint(IDX_EL_Z_JF_R,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'right_elbow_Y') 
    lowerarmR = pin.Frame('lowerarmR',IDX_EL_Y_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF = model.addFrame(lowerarmR,False)
    idx_frame = IDX_LOA_SF

    for i in sgts_mks_dict["lowerarmR"]:
        frame = pin.Frame(i,IDX_EL_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    lowerarm_visual_R = pin.GeometryObject('lowerarm',IDX_LOA_SF, IDX_EL_Y_JF, mesh_loader.load(meshes_folder_path+'/lowerarm_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/lowerarm_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(lowerarm_visual_R)
    visuals_dict["lowerarm_R"] = lowerarm_visual_R

    # Left shoulder ZXY
    IDX_SH_Z_JF_L = model.addJoint(IDX_L5S1_R_EXT_INT_JF, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix(local_segments_positions['upperarmL'] + local_segments_positions['torso']).T), 'left_shoulder_Z') 
    upperarmL = pin.Frame('upperarm_z_L', IDX_SH_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    idx_frame = IDX_UPA_SF_L

    shoulder_visual_L = pin.GeometryObject('shoulder_L', IDX_UPA_SF_L, IDX_SH_Z_JF_L, mesh_loader.load(meshes_folder_path+'/shoulder_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/shoulder_mesh.STL', np.array([0.0055, 0.0055, 0.0055]), True, body_color)
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
        frame = pin.Frame(i, IDX_SH_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix(mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    upperarm_visual_L = pin.GeometryObject('upperarm_L', IDX_UPA_SF_L, IDX_SH_Y_JF_L, mesh_loader.load(meshes_folder_path+'/upperarm_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/upperarm_mesh.STL', np.array([0.0063, 0.0060, 0.007]), True, body_color)
    geom_model.addGeometryObject(upperarm_visual_L)
    visuals_dict["upperarm_L"] = upperarm_visual_L

    # Left Elbow ZY
    IDX_EL_Z_JF_L = model.addJoint(IDX_SH_Y_JF_L, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix(local_segments_positions['lowerarmL']).T), 'left_elbow_Z')
    lowerarmL = pin.Frame('lowerarm_z_L', IDX_EL_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF_L = model.addFrame(lowerarmL, False)
    idx_frame = IDX_LOA_SF_L

    elbow_visual_L = pin.GeometryObject('elbow_L', IDX_LOA_SF_L, IDX_EL_Z_JF_L, mesh_loader.load(meshes_folder_path+'/elbow_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/elbow_mesh.STL', np.array([0.0055, 0.0055, 0.0055]), True, body_color)
    geom_model.addGeometryObject(elbow_visual_L)
    visuals_dict["elbow_L"] = elbow_visual_L

    IDX_EL_Y_JF_L = model.addJoint(IDX_EL_Z_JF_L, pin.JointModelRY(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'left_elbow_Y') 
    lowerarmL = pin.Frame('lowerarmL', IDX_EL_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF_L = model.addFrame(lowerarmL, False)
    idx_frame = IDX_LOA_SF_L

    for i in sgts_mks_dict["lowerarmL"]:
        frame = pin.Frame(i, IDX_EL_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix(mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    lowerarm_visual_L = pin.GeometryObject('lowerarm_L', IDX_LOA_SF_L, IDX_EL_Y_JF_L, mesh_loader.load(meshes_folder_path+'/lowerarm_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/lowerarm_mesh.STL', np.array([0.0060, 0.0060, 0.0060]), True, body_color)
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
        frame = pin.Frame(i,IDX_HIP_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    upperleg_visual_R = pin.GeometryObject('upperleg_R',IDX_THIGH_SF, IDX_HIP_Y_JF, mesh_loader.load(meshes_folder_path+'/upperleg_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/upperleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(upperleg_visual_R)
    visuals_dict["upperleg_R"] = upperleg_visual_R


    # Right Knee Z
    IDX_KNEE_Z_JF = model.addJoint(IDX_HIP_X_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['shankR']).T),'right_knee_Z') 
    shankR = pin.Frame('shankR',IDX_KNEE_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SHANK_SF = model.addFrame(shankR,False)
    idx_frame = IDX_SHANK_SF

    for i in sgts_mks_dict["shankR"]:
        frame = pin.Frame(i,IDX_KNEE_Z_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    knee_visual = pin.GeometryObject('knee_R',IDX_SHANK_SF, IDX_KNEE_Z_JF, mesh_loader.load(meshes_folder_path+'/knee_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/knee_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(knee_visual)
    visuals_dict["knee_R"] = knee_visual

    lowerleg_visual_R = pin.GeometryObject('lowerleg_R',IDX_SHANK_SF, IDX_KNEE_Z_JF, mesh_loader.load(meshes_folder_path+'/lowerleg_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/lowerleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(lowerleg_visual_R)
    visuals_dict["lowerleg_R"] = lowerleg_visual_R


    # Right Ankle Z
    IDX_ANKLE_Z_JF = model.addJoint(IDX_KNEE_Z_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['footR']).T),'right_ankle_Z') 
    footR = pin.Frame('footR',IDX_ANKLE_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SFOOT_SF = model.addFrame(footR,False)
    idx_frame = IDX_SFOOT_SF

    for i in sgts_mks_dict["footR"]:
        frame = pin.Frame(i,IDX_ANKLE_Z_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    foot_visual_R = pin.GeometryObject('foot_R',IDX_SFOOT_SF, IDX_ANKLE_Z_JF, mesh_loader.load(meshes_folder_path+'/foot_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/foot_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
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
        frame = pin.Frame(i, IDX_HIP_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix(mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    thigh_visual_L = pin.GeometryObject('upperleg_L', IDX_TGH_SF_L, IDX_HIP_Y_JF_L, mesh_loader.load(meshes_folder_path+'/upperleg_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/upperleg_mesh.STL', np.array([0.0060, 0.0060, 0.0060]), True, body_color)
    geom_model.addGeometryObject(thigh_visual_L)
    visuals_dict["upperleg_L"] = thigh_visual_L

    # Left Knee Z
    IDX_KNEE_Z_JF_L = model.addJoint(IDX_HIP_Y_JF_L,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['shankL']).T),'left_knee_Z') 
    shankR = pin.Frame('shankL',IDX_KNEE_Z_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SHANK_SF_L = model.addFrame(shankR,False)
    idx_frame = IDX_SHANK_SF_L

    for i in sgts_mks_dict["shankL"]:
        frame = pin.Frame(i,IDX_KNEE_Z_JF_L,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    knee_visual = pin.GeometryObject('knee_L',IDX_SHANK_SF_L, IDX_KNEE_Z_JF_L, mesh_loader.load(meshes_folder_path+'/knee_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0, 0.]).T), meshes_folder_path+'/knee_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(knee_visual)
    visuals_dict["knee_L"] = knee_visual

    lowerleg_visual_L = pin.GeometryObject('lowerleg_L',IDX_SHANK_SF_L, IDX_KNEE_Z_JF_L, mesh_loader.load(meshes_folder_path+'/lowerleg_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/lowerleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(lowerleg_visual_L)
    visuals_dict["lowerleg_L"] = lowerleg_visual_L

    # Left Ankle Z
    IDX_ANKLE_Z_JF_L = model.addJoint(IDX_KNEE_Z_JF_L,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['footL']).T),'left_ankle_Z') 
    footL = pin.Frame('footL',IDX_ANKLE_Z_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SFOOT_SF_L = model.addFrame(footL,False)
    idx_frame = IDX_SFOOT_SF_L

    for i in sgts_mks_dict["footL"]:
        frame = pin.Frame(i,IDX_ANKLE_Z_JF_L,idx_frame,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    foot_visual_L = pin.GeometryObject('foot_L',IDX_SFOOT_SF_L, IDX_ANKLE_Z_JF_L, mesh_loader.load(meshes_folder_path+'/foot_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/foot_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(foot_visual_L)
    visuals_dict["foot_L"] = foot_visual_L

    model.upperPositionLimit[7:] = np.array([5*np.pi/36,       #L5S1_FE + 
                                          np.pi/3,             #L5S1_R_EXT_INT +
                                          np.pi/2,             # Neck_Z +
                                          np.pi/2,             # Neck_X +
                                          np.pi/2,             # Neck_Y +
                                          np.pi,               #Shoulder_Z_R +
                                          np.pi/3,             #Shoulder_X_R +
                                          np.pi/2,             #Shoulder_Y_R +
                                          5*np.pi/6,           #Elbow_Z_R +
                                          np.pi,               #Elbow_Y_R + 
                                          np.pi,               #Shoulder_Z_L +
                                          np.pi,               #Shoulder_X_L +
                                          np.pi/2,             #Shoulder_Y_L +
                                          5*np.pi/6,           #Elbow_Z_L +
                                          0.2,                 #Elbow_Y_L +
                                          np.pi/2,             #Hip_Z_R +
                                          np.pi/3,             #Hip_X_R +
                                          np.pi/3,             #Hip_Y_R +
                                          0,                   #Knee_Z_R +
                                          np.pi/4,             #Ankle_Z_R +
                                          np.pi/2,             #Hip_Z_L +
                                          np.pi/2,             #Hip_X_L +
                                          np.pi/2,             #Hip_Y_L +
                                          0,                   #Knee_Z_L +
                                          np.pi/4,             #Ankle_Z_L +
                                          ]) 
    
    model.lowerPositionLimit[7:] = np.array([-np.pi/2,           #L5S1_FE -
                                            -np.pi/3,            #L5S1_R_EXT_INT -
                                            -np.pi/2,            # Neck_Z -
                                            -np.pi/2,            # Neck_X -
                                            -np.pi/2,            # Neck_Y -
                                            -np.pi,              #Shoulder_Z_R -
                                            -np.pi,              #Shoulder_X_R -
                                            -np.pi/2,            #Shoulder_Y_R -
                                            0,                   #Elbow_Z_R -
                                            -0.2,                #Elbow_Y_R -
                                            -np.pi,              #Shoulder_Z_L -
                                            -np.pi/3,            #Shoulder_X_L -
                                            -np.pi/2,            #Shoulder_Y_L -
                                            0,                   #Elbow_Z_L -
                                            -np.pi,              #Elbow_Y_L -
                                            -np.pi/3,            #Hip_Z_R -
                                            -np.pi/2,            #Hip_X_R -
                                            -np.pi/2,            #Hip_Y_R -
                                            -5*np.pi/6,          #Knee_Z_R -
                                            -np.pi/2,            #Ankle_Z_R -
                                            -np.pi/3,            #Hip_Z_L -
                                            -np.pi/3,            #Hip_X_L -
                                            -np.pi/3,            #Hip_Y_L -
                                            -5*np.pi/6,          #Knee_Z_L -
                                            -np.pi/2,            #Ankle_Z_L -
                                            ])
    
    return model, geom_model, visuals_dict

def build_dummy_model(meshes_folder_path: str)->Tuple[pin.Model,pin.Model, Dict]:
    """_Build the biomechanical model associated to one exercise for one subject_

    Args:
        meshes_folder_path (str): _meshes_folder_path is the path to the folder containing the meshes_

    Returns:
        Tuple[pin.Model,pin.GeomModel, Dict]: _returns the pinocchio model, geometry model, and a dictionnary with visuals._
    """

    body_color = np.array([0,0,0,0.5])
    visuals_dict = {}

    # Mesh loader
    mesh_loader = fcl.MeshLoader()

    # MODEL GENERATION 
    inertia = pin.Inertia.Zero()
    model= pin.Model() # pin model
    geom_model = pin.GeometryModel() # geometry model

    # pelvis with Freeflyer
    IDX_PELV_JF = model.addJoint(0,pin.JointModelFreeFlyer(),pin.SE3(np.array([[1,0,0],[0,0,-1],[0,1,0]]), np.matrix([0,0,0]).T),'root_joint')
    pelvis = pin.Frame('pelvis',IDX_PELV_JF,0,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_PELV_SF = model.addFrame(pelvis,False)
    idx_frame = IDX_PELV_SF
    
    pelvis_visual = pin.GeometryObject('pelvis', IDX_PELV_SF, IDX_PELV_JF, mesh_loader.load(meshes_folder_path+'/pelvis_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/pelvis_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), True, body_color)
    geom_model.addGeometryObject(pelvis_visual)
    visuals_dict["pelvis"] = pelvis_visual

    # Lumbar L5-S1 flexion/extension
    IDX_L5S1_JF = model.addJoint(IDX_PELV_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T),'middle_lumbar_Z') 
    torso = pin.Frame('torso_z',IDX_L5S1_JF,idx_frame,pin.SE3(np.eye(3),np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_TORSO_SF = model.addFrame(torso,False)
    idx_frame = IDX_TORSO_SF

    IDX_L5S1_R_EXT_INT_JF = model.addJoint(IDX_L5S1_JF,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T),'middle_lumbar_Y') 
    torso = pin.Frame('torso',IDX_L5S1_R_EXT_INT_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_TORSO_SF = model.addFrame(torso,False)
    idx_frame = IDX_TORSO_SF

    torso_visual = pin.GeometryObject('torso', IDX_TORSO_SF, IDX_L5S1_JF, mesh_loader.load(meshes_folder_path+'/torso_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/torso_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), True, body_color)
    geom_model.addGeometryObject(torso_visual)
    visuals_dict["torso"] = torso_visual

    abdomen_visual = pin.GeometryObject('abdomen', IDX_TORSO_SF, IDX_L5S1_JF, mesh_loader.load(meshes_folder_path+'/abdomen_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/abdomen_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), True, body_color)
    geom_model.addGeometryObject(abdomen_visual)
    visuals_dict["abdomen"] = abdomen_visual

    # Cervical ZXY
    IDX_NECK_Z_JF = model.addJoint(IDX_L5S1_R_EXT_INT_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, 0.5, 0]).T),'cervical_Z')
    head = pin.Frame('head_z',IDX_NECK_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_HEAD_SF = model.addFrame(head,False)
    idx_frame = IDX_HEAD_SF

    IDX_NECK_X_JF = model.addJoint(IDX_NECK_Z_JF,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'cervical_X')
    head = pin.Frame('head_x',IDX_NECK_X_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_HEAD_SF = model.addFrame(head,False)
    idx_frame = IDX_HEAD_SF

    IDX_NECK_Y_JF = model.addJoint(IDX_NECK_X_JF,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'cervical_Y')
    head = pin.Frame('head',IDX_NECK_Y_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_HEAD_SF = model.addFrame(head,False)
    idx_frame = IDX_HEAD_SF

    head_visual = pin.GeometryObject('head', IDX_HEAD_SF, IDX_NECK_Y_JF, mesh_loader.load(meshes_folder_path+'/head_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/head_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), True, body_color)
    geom_model.addGeometryObject(head_visual)
    visuals_dict["head"] = head_visual

    neck_visual = pin.GeometryObject('neck', IDX_HEAD_SF, IDX_NECK_Y_JF, mesh_loader.load(meshes_folder_path+'/neck_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/neck_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), True, body_color)
    geom_model.addGeometryObject(neck_visual)
    visuals_dict["neck"] = neck_visual

    # Right Shoulder ZXY
    IDX_SH_Z_JF_R = model.addJoint(IDX_L5S1_R_EXT_INT_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0.00805, 0.4067, 0.2037]).T),'right_shoulder_Z') 
    upperarmR = pin.Frame('upperarm_z_R',IDX_SH_Z_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    idx_frame = IDX_UPA_SF_R

    shoulder_visual_R = pin.GeometryObject('shoulder_R', IDX_UPA_SF_R, IDX_SH_Z_JF_R, mesh_loader.load(meshes_folder_path+'/shoulder_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/shoulder_mesh.STL',np.array([0.0055, 0.0055, 0.0055]), True , body_color)
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

    upperarm_visual_R = pin.GeometryObject('upperarm_R', IDX_UPA_SF_R, IDX_SH_Y_JF_R, mesh_loader.load(meshes_folder_path+'/upperarm_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/upperarm_mesh.STL',np.array([0.0063, 0.0060, 0.007]), True , body_color)
    geom_model.addGeometryObject(upperarm_visual_R)
    visuals_dict["upperarm_R"] = upperarm_visual_R

    # Right Elbow ZY 
    IDX_EL_Z_JF_R = model.addJoint(IDX_SH_Y_JF_R,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, -0.2737, 0]).T),'right_elbow_Z') 
    lowerarmR = pin.Frame('lowerarm_z',IDX_EL_Z_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF = model.addFrame(lowerarmR,False)
    idx_frame = IDX_LOA_SF

    elbow_visual = pin.GeometryObject('elbow', IDX_LOA_SF, IDX_EL_Z_JF_R, mesh_loader.load(meshes_folder_path+'/elbow_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/elbow_mesh.STL',np.array([0.0055, 0.0055, 0.0055]), True , body_color)
    geom_model.addGeometryObject(elbow_visual)
    visuals_dict["elbow"] = elbow_visual

    IDX_EL_Y_JF = model.addJoint(IDX_EL_Z_JF_R,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'right_elbow_Y') 
    lowerarmR = pin.Frame('lowerarmR',IDX_EL_Y_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF = model.addFrame(lowerarmR,False)
    idx_frame = IDX_LOA_SF

    lowerarm_visual_R = pin.GeometryObject('lowerarm',IDX_LOA_SF, IDX_EL_Y_JF, mesh_loader.load(meshes_folder_path+'/lowerarm_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/lowerarm_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(lowerarm_visual_R)
    visuals_dict["lowerarm_R"] = lowerarm_visual_R

    # Left shoulder ZXY
    IDX_SH_Z_JF_L = model.addJoint(IDX_L5S1_R_EXT_INT_JF, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix([0.00805, 0.4067, -0.2037]).T), 'left_shoulder_Z') 
    upperarmL = pin.Frame('upperarm_z_L', IDX_SH_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    idx_frame = IDX_UPA_SF_L

    shoulder_visual_L = pin.GeometryObject('shoulder_L', IDX_UPA_SF_L, IDX_SH_Z_JF_L, mesh_loader.load(meshes_folder_path+'/shoulder_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/shoulder_mesh.STL', np.array([0.0055, 0.0055, 0.0055]), True, body_color)
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

    upperarm_visual_L = pin.GeometryObject('upperarm_L', IDX_UPA_SF_L, IDX_SH_Y_JF_L, mesh_loader.load(meshes_folder_path+'/upperarm_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/upperarm_mesh.STL', np.array([0.0063, 0.0060, 0.007]), True, body_color)
    geom_model.addGeometryObject(upperarm_visual_L)
    visuals_dict["upperarm_L"] = upperarm_visual_L

    # Left Elbow ZY
    IDX_EL_Z_JF_L = model.addJoint(IDX_SH_Y_JF_L, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix([0, -0.2737, 0]).T), 'left_elbow_Z')
    lowerarmL = pin.Frame('lowerarm_z_L', IDX_EL_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF_L = model.addFrame(lowerarmL, False)
    idx_frame = IDX_LOA_SF_L

    elbow_visual_L = pin.GeometryObject('elbow_L', IDX_LOA_SF_L, IDX_EL_Z_JF_L, mesh_loader.load(meshes_folder_path+'/elbow_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/elbow_mesh.STL', np.array([0.0055, 0.0055, 0.0055]), True, body_color)
    geom_model.addGeometryObject(elbow_visual_L)
    visuals_dict["elbow_L"] = elbow_visual_L

    IDX_EL_Y_JF_L = model.addJoint(IDX_EL_Z_JF_L, pin.JointModelRY(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'left_elbow_Y') 
    lowerarmL = pin.Frame('lowerarmL', IDX_EL_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF_L = model.addFrame(lowerarmL, False)
    idx_frame = IDX_LOA_SF_L

    lowerarm_visual_L = pin.GeometryObject('lowerarm_L', IDX_LOA_SF_L, IDX_EL_Y_JF_L, mesh_loader.load(meshes_folder_path+'/lowerarm_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/lowerarm_mesh.STL', np.array([0.0060, 0.0060, 0.0060]), True, body_color)
    geom_model.addGeometryObject(lowerarm_visual_L)
    visuals_dict["lowerarm_L"] = lowerarm_visual_L

    # Right Hip ZXY
    IDX_HIP_Z_JF = model.addJoint(IDX_PELV_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0.053375, -0.0749, 0.079975]).T),'right_hip_Z') 
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

    upperleg_visual_R = pin.GeometryObject('upperleg_R',IDX_THIGH_SF, IDX_HIP_Y_JF, mesh_loader.load(meshes_folder_path+'/upperleg_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/upperleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(upperleg_visual_R)
    visuals_dict["upperleg_R"] = upperleg_visual_R

    # Right Knee Z
    IDX_KNEE_Z_JF = model.addJoint(IDX_HIP_X_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, -0.427, 0]).T),'right_knee_Z') 
    shankR = pin.Frame('shankR',IDX_KNEE_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SHANK_SF = model.addFrame(shankR,False)
    idx_frame = IDX_SHANK_SF
    
    knee_visual = pin.GeometryObject('knee_R',IDX_SHANK_SF, IDX_KNEE_Z_JF, mesh_loader.load(meshes_folder_path+'/knee_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/knee_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(knee_visual)
    visuals_dict["knee_R"] = knee_visual

    lowerleg_visual_R = pin.GeometryObject('lowerleg_R',IDX_SHANK_SF, IDX_KNEE_Z_JF, mesh_loader.load(meshes_folder_path+'/lowerleg_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/lowerleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(lowerleg_visual_R)
    visuals_dict["lowerleg_R"] = lowerleg_visual_R


    # Right Ankle Z
    IDX_ANKLE_Z_JF = model.addJoint(IDX_KNEE_Z_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, -0.42805, 0]).T),'right_ankle_Z') 
    footR = pin.Frame('footR',IDX_ANKLE_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SFOOT_SF = model.addFrame(footR,False)
    idx_frame = IDX_SFOOT_SF
    
    foot_visual_R = pin.GeometryObject('foot_R',IDX_SFOOT_SF, IDX_ANKLE_Z_JF, mesh_loader.load(meshes_folder_path+'/foot_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/foot_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(foot_visual_R)
    visuals_dict["foot_R"] = foot_visual_R


    # Left Hip ZXY
    IDX_HIP_Z_JF_L = model.addJoint(IDX_PELV_JF, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix([0.053375, -0.0749, -0.079975]).T), 'left_hip_Z') 
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

    thigh_visual_L = pin.GeometryObject('upperleg_L', IDX_TGH_SF_L, IDX_HIP_Y_JF_L, mesh_loader.load(meshes_folder_path+'/upperleg_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/upperleg_mesh.STL', np.array([0.0060, 0.0060, 0.0060]), True, body_color)
    geom_model.addGeometryObject(thigh_visual_L)
    visuals_dict["upperleg_L"] = thigh_visual_L

    # Left Knee Z
    IDX_KNEE_Z_JF_L = model.addJoint(IDX_HIP_Y_JF_L,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, -0.427, 0]).T),'left_knee_Z') 
    shankR = pin.Frame('shankL',IDX_KNEE_Z_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SHANK_SF_L = model.addFrame(shankR,False)
    idx_frame = IDX_SHANK_SF_L
    
    knee_visual = pin.GeometryObject('knee_L',IDX_SHANK_SF_L, IDX_KNEE_Z_JF_L, mesh_loader.load(meshes_folder_path+'/knee_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0, 0.]).T), meshes_folder_path+'/knee_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(knee_visual)
    visuals_dict["knee_L"] = knee_visual

    lowerleg_visual_L = pin.GeometryObject('lowerleg_L',IDX_SHANK_SF_L, IDX_KNEE_Z_JF_L, mesh_loader.load(meshes_folder_path+'/lowerleg_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/lowerleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(lowerleg_visual_L)
    visuals_dict["lowerleg_L"] = lowerleg_visual_L

    # Left Ankle Z
    IDX_ANKLE_Z_JF_L = model.addJoint(IDX_KNEE_Z_JF_L,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, -0.42805, 0]).T),'left_ankle_Z') 
    footL = pin.Frame('footL',IDX_ANKLE_Z_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SFOOT_SF_L = model.addFrame(footL,False)
    idx_frame = IDX_SFOOT_SF_L
    
    foot_visual_L = pin.GeometryObject('foot_L',IDX_SFOOT_SF_L, IDX_ANKLE_Z_JF_L, mesh_loader.load(meshes_folder_path+'/foot_mesh.STL'), pin.SE3(np.eye(3), np.matrix([0., 0., 0.]).T), meshes_folder_path+'/foot_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), True , body_color)
    geom_model.addGeometryObject(foot_visual_L)
    visuals_dict["foot_L"] = foot_visual_L
    
    return model, geom_model, visuals_dict

def build_dummy_model_no_visuals()->pin.Model:
    # MODEL GENERATION 
    inertia = pin.Inertia.Zero()
    model= pin.Model() # pin model

    # pelvis with Freeflyer
    IDX_PELV_JF = model.addJoint(0,pin.JointModelFreeFlyer(),pin.SE3(np.array([[1,0,0],[0,0,-1],[0,1,0]]), np.matrix([0,0,0]).T),'root_joint')
    pelvis = pin.Frame('pelvis',IDX_PELV_JF,0,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_PELV_SF = model.addFrame(pelvis,False)
    # Add markers data
    idx_frame = IDX_PELV_SF
    for i in ['r.PSIS_study', 'L.PSIS_study', 'r.ASIS_study', 'L.ASIS_study']:
        frame = pin.Frame(i,IDX_PELV_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Lumbar L5-S1 flexion/extension
    IDX_L5S1_JF = model.addJoint(IDX_PELV_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T),'middle_lumbar_Z') 
    torso = pin.Frame('torso_z',IDX_L5S1_JF,idx_frame,pin.SE3(np.eye(3),np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_TORSO_SF = model.addFrame(torso,False)
    idx_frame = IDX_TORSO_SF

    # Lumbar L5-S1 external/internal rotation
    IDX_L5S1_R_EXT_INT_JF = model.addJoint(IDX_L5S1_JF,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T),'middle_lumbar_Y') 
    torso = pin.Frame('torso',IDX_L5S1_R_EXT_INT_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_TORSO_SF = model.addFrame(torso,False)
    idx_frame = IDX_TORSO_SF

    for i in ['r_shoulder_study', 'L_shoulder_study', 'C7_study']:
        frame = pin.Frame(i,IDX_L5S1_R_EXT_INT_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix([0, 0, 0]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Right Shoulder ZXY
    IDX_SH_Z_JF_R = model.addJoint(IDX_L5S1_R_EXT_INT_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0.00805, 0.4067, 0.2037]).T),'right_shoulder_Z') # Hardcoded dummy values
    upperarmR = pin.Frame('upperarm_z_R',IDX_SH_Z_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    idx_frame = IDX_UPA_SF_R

    IDX_SH_X_JF_R = model.addJoint(IDX_SH_Z_JF_R,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'right_shoulder_X') 
    upperarmR = pin.Frame('upperarm_x_R',IDX_SH_X_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    idx_frame = IDX_UPA_SF_R

    IDX_SH_Y_JF_R = model.addJoint(IDX_SH_X_JF_R,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'right_shoulder_Y') 
    upperarmR = pin.Frame('upperarmR',IDX_SH_Y_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    idx_frame = IDX_UPA_SF_R

    for i in ['r_melbow_study', 'r_lelbow_study']:
        frame = pin.Frame(i,IDX_SH_Y_JF_R,idx_frame,pin.SE3(np.eye(3,3), np.matrix([0, 0, 0]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Right Elbow ZY 
    IDX_EL_Z_JF_R = model.addJoint(IDX_SH_Y_JF_R,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, -0.2737, 0]).T),'right_elbow_Z') 
    lowerarmR = pin.Frame('lowerarm_z',IDX_EL_Z_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF = model.addFrame(lowerarmR,False)
    idx_frame = IDX_LOA_SF

    IDX_EL_Y_JF = model.addJoint(IDX_EL_Z_JF_R,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'right_elbow_Y') 
    lowerarmR = pin.Frame('lowerarmR',IDX_EL_Y_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF = model.addFrame(lowerarmR,False)
    idx_frame = IDX_LOA_SF

    for i in ['r_lwrist_study', 'r_mwrist_study']:
        frame = pin.Frame(i,IDX_EL_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix([0, 0, 0]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Left shoulder ZXY
    IDX_SH_Z_JF_L = model.addJoint(IDX_L5S1_R_EXT_INT_JF, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix([0.00805, 0.4067, -0.2037]).T), 'left_shoulder_Z') 
    upperarmL = pin.Frame('upperarm_z_L', IDX_SH_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    idx_frame = IDX_UPA_SF_L

    IDX_SH_X_JF_L = model.addJoint(IDX_SH_Z_JF_L, pin.JointModelRX(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'left_shoulder_X') 
    upperarmL = pin.Frame('upperarm_x_L', IDX_SH_X_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    idx_frame = IDX_UPA_SF_L

    IDX_SH_Y_JF_L = model.addJoint(IDX_SH_X_JF_L, pin.JointModelRY(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'left_shoulder_Y') 
    upperarmL = pin.Frame('upperarmL', IDX_SH_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    idx_frame = IDX_UPA_SF_L

    for i in ['L_melbow_study', 'L_lelbow_study']:
        frame = pin.Frame(i, IDX_SH_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    # Left Elbow ZY
    IDX_EL_Z_JF_L = model.addJoint(IDX_SH_Y_JF_L, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix([0, -0.2737, 0]).T), 'left_elbow_Z')
    lowerarmL = pin.Frame('lowerarm_z_L', IDX_EL_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF_L = model.addFrame(lowerarmL, False)
    idx_frame = IDX_LOA_SF_L

    IDX_EL_Y_JF_L = model.addJoint(IDX_EL_Z_JF_L, pin.JointModelRY(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'left_elbow_Y') 
    lowerarmL = pin.Frame('lowerarmL', IDX_EL_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF_L = model.addFrame(lowerarmL, False)
    idx_frame = IDX_LOA_SF_L

    for i in ['L_lwrist_study', 'L_mwrist_study']:
        frame = pin.Frame(i, IDX_EL_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    # Right Hip ZXY
    IDX_HIP_Z_JF = model.addJoint(IDX_PELV_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0.053375, -0.0749, 0.079975]).T),'right_hip_Z') 
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

    for i in ['r_knee_study', 'r_mknee_study','r_thigh2_study', 'r_thigh3_study', 'r_thigh1_study']:
        frame = pin.Frame(i,IDX_HIP_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix([0, 0, 0]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Right Knee Z
    IDX_KNEE_Z_JF = model.addJoint(IDX_HIP_X_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, -0.427, 0]).T),'right_knee_Z') 
    shankR = pin.Frame('shankR',IDX_KNEE_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SHANK_SF = model.addFrame(shankR,False)
    idx_frame = IDX_SHANK_SF

    for i in ['r_ankle_study', 'r_mankle_study','r_sh3_study', 'r_sh2_study', 'r_sh1_study']:
        frame = pin.Frame(i,IDX_KNEE_Z_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix([0, 0, 0]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Right Ankle Z
    IDX_ANKLE_Z_JF = model.addJoint(IDX_KNEE_Z_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, -0.42805, 0]).T),'right_ankle_Z') 
    footR = pin.Frame('footR',IDX_ANKLE_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SFOOT_SF = model.addFrame(footR,False)
    idx_frame = IDX_SFOOT_SF

    for i in ['r_calc_study' ,'r_5meta_study','r_toe_study']:
        frame = pin.Frame(i,IDX_ANKLE_Z_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix([0, 0, 0]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Left Hip ZXY
    IDX_HIP_Z_JF_L = model.addJoint(IDX_PELV_JF, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix([0.053375, -0.0749, -0.079975]).T), 'left_hip_Z') 
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

    for i in ['L_knee_study', 'L_mknee_study','L_thigh2_study', 'L_thigh3_study', 'L_thigh1_study']:
        frame = pin.Frame(i, IDX_HIP_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    # Left Knee Z
    IDX_KNEE_Z_JF_L = model.addJoint(IDX_HIP_Y_JF_L,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, -0.427, 0]).T),'left_knee_Z') 
    shankR = pin.Frame('shankL',IDX_KNEE_Z_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SHANK_SF_L = model.addFrame(shankR,False)
    idx_frame = IDX_SHANK_SF_L

    for i in ['L_ankle_study', 'L_mankle_study','L_sh3_study', 'L_sh2_study', 'L_sh1_study']:
        frame = pin.Frame(i,IDX_KNEE_Z_JF_L,idx_frame,pin.SE3(np.eye(3,3), np.matrix([0, 0, 0]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    # Left Ankle Z
    IDX_ANKLE_Z_JF_L = model.addJoint(IDX_KNEE_Z_JF_L,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, -0.42805, 0]).T),'left_ankle_Z') 
    footL = pin.Frame('footL',IDX_ANKLE_Z_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SFOOT_SF_L = model.addFrame(footL,False)
    idx_frame = IDX_SFOOT_SF_L

    for i in ['L_calc_study', 'L_5meta_study', 'L_toe_study']:
        frame = pin.Frame(i,IDX_ANKLE_Z_JF_L,idx_frame,pin.SE3(np.eye(3,3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    model.upperPositionLimit[7:] = np.array([5*np.pi/36,         #L5S1_FE + 
                                          np.pi/3,             #L5S1_R_EXT_INT +
                                          np.pi,               #Shoulder_Z_R +
                                          np.pi/3,             #Shoulder_X_R +
                                          np.pi/2,             #Shoulder_Y_R +
                                          5*np.pi/6,           #Elbow_Z_R +
                                          np.pi,               #Elbow_Y_R + 
                                          np.pi,               #Shoulder_Z_L +
                                          np.pi,               #Shoulder_X_L +
                                          np.pi/2,             #Shoulder_Y_L +
                                          5*np.pi/6,           #Elbow_Z_L +
                                          0.2,               #Elbow_Y_L +
                                          np.pi/2,             #Hip_Z_R +
                                          np.pi/3,             #Hip_X_R +
                                          np.pi/3,             #Hip_Y_R +
                                          0,                   #Knee_Z_R +
                                          np.pi/4,             #Ankle_Z_R +
                                          np.pi/2,           #Hip_Z_L +
                                          np.pi/2,             #Hip_X_L +
                                          np.pi/2,             #Hip_Y_L +
                                          0,                   #Knee_Z_L +
                                          np.pi/4,             #Ankle_Z_L +
                                          ]) 
    
    model.lowerPositionLimit[7:] = np.array([-np.pi/2,           #L5S1_FE -
                                            -np.pi/3,            #L5S1_R_EXT_INT -
                                            -np.pi,              #Shoulder_Z_R -
                                            -np.pi,              #Shoulder_X_R -
                                            -np.pi/2,            #Shoulder_Y_R -
                                            0,                   #Elbow_Z_R -
                                            -0.2,                #Elbow_Y_R -
                                            -np.pi,              #Shoulder_Z_L -
                                            -np.pi/3,            #Shoulder_X_L -
                                            -np.pi/2,            #Shoulder_Y_L -
                                            0,                   #Elbow_Z_L -
                                            -np.pi,          #Elbow_Y_L -
                                            -np.pi/3,            #Hip_Z_R -
                                            -np.pi/2,            #Hip_X_R -
                                            -np.pi/2,            #Hip_Y_R -
                                            -5*np.pi/6,          #Knee_Z_R -
                                            -np.pi/2,          #Ankle_Z_R -
                                            -np.pi/3,            #Hip_Z_L -
                                            -np.pi/3,            #Hip_X_L -
                                            -np.pi/3,             #Hip_Y_L -
                                            -5*np.pi/6,          #Knee_Z_L -
                                            -np.pi/2,          #Ankle_Z_L -
                                            ])
        
    return model

def rescale_human_model(model: pin.Model, mks_dict: Dict)->pin.Model:
    inertia = pin.Inertia.Zero()

    sgts_poses = construct_segments_frames(mks_dict)
    sgts_mks_dict = get_segments_mks_dict()
    mks_local_positions = get_local_mks_positions(sgts_poses, mks_dict, sgts_mks_dict)
    local_segments_positions = get_local_segments_positions(sgts_poses)

    # Add markers data
    IDX_PELVF = model.getFrameId('pelvis')
    IDX_PELV_JF = model.getJointId('root_joint')
    for i in sgts_mks_dict["pelvis"]:
        frame = pin.Frame(i,IDX_PELV_JF,IDX_PELVF,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    IDX_TORSOF = model.getFrameId('torso')
    IDX_L5S1_R_EXT_INT_JF = model.getJointId('middle_lumbar_Y')
    for i in sgts_mks_dict["torso"]:
        frame = pin.Frame(i,IDX_L5S1_R_EXT_INT_JF,IDX_TORSOF,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]+ local_segments_positions['torso']).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    IDX_UPARF = model.getFrameId('upperarmR')
    IDX_SH_Y_JF_R = model.getJointId('right_shoulder_Y')
    for i in sgts_mks_dict["upperarmR"]:
        frame = pin.Frame(i,IDX_SH_Y_JF_R,IDX_UPARF,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    IDX_LOARF = model.getFrameId('lowerarmR')
    IDX_EL_Y_JF = model.getJointId('right_elbow_Y')
    for i in sgts_mks_dict["lowerarmR"]:
        frame = pin.Frame(i,IDX_EL_Y_JF,IDX_LOARF,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    IDX_UPALF = model.getFrameId('upperarmL')
    IDX_SH_Y_JF_L = model.getJointId('left_shoulder_Y')
    for i in sgts_mks_dict["upperarmL"]:
        frame = pin.Frame(i, IDX_SH_Y_JF_L, IDX_UPALF, pin.SE3(np.eye(3), np.matrix(mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    IDX_LOALF = model.getFrameId('lowerarmL')
    IDX_EL_Y_JF_L = model.getJointId('left_elbow_Y')
    for i in sgts_mks_dict["lowerarmL"]:
        frame = pin.Frame(i, IDX_EL_Y_JF_L, IDX_LOALF, pin.SE3(np.eye(3), np.matrix(mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    IDX_THIGHRF = model.getFrameId('thighR')
    IDX_HIP_Y_JF = model.getJointId('right_hip_Y')
    for i in sgts_mks_dict["thighR"]:
        frame = pin.Frame(i,IDX_HIP_Y_JF,IDX_THIGHRF,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    IDX_SHANKRF = model.getFrameId('shankR')
    IDX_KNEE_Z_JF = model.getJointId('right_knee_Z')
    for i in sgts_mks_dict["shankR"]:
        frame = pin.Frame(i,IDX_KNEE_Z_JF,IDX_SHANKRF,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    IDX_FOOTRF = model.getFrameId('footR')
    IDX_ANKLE_Z_JF = model.getJointId('right_ankle_Z')
    for i in sgts_mks_dict["footR"]:
        frame = pin.Frame(i,IDX_ANKLE_Z_JF,IDX_FOOTRF,pin.SE3(np.eye(3,3), np.matrix(mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    IDX_THIGHLF = model.getFrameId('thighL')
    IDX_HIP_Y_JF_L = model.getJointId('left_hip_Y')
    for i in sgts_mks_dict["thighL"]:
        frame = pin.Frame(i, IDX_HIP_Y_JF_L, IDX_THIGHLF, pin.SE3(np.eye(3), np.matrix(mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    IDX_SHANKLF = model.getFrameId('shankL')
    IDX_KNEE_Z_JF_L = model.getJointId('left_knee_Z')
    for i in sgts_mks_dict["shankL"]:
        frame = pin.Frame(i, IDX_KNEE_Z_JF_L, IDX_SHANKLF, pin.SE3(np.eye(3), np.matrix(mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)
    
    IDX_FOOTLF = model.getFrameId('footL')
    IDX_ANKLE_Z_JF_L = model.getJointId('left_ankle_Z')
    for i in sgts_mks_dict["footL"]:
        frame = pin.Frame(i, IDX_ANKLE_Z_JF_L, IDX_FOOTLF, pin.SE3(np.eye(3), np.matrix(mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    # Segment lengths scaling
    IDX_SH_Z_JF_R = model.getJointId('right_shoulder_Z')
    model.jointPlacements[IDX_SH_Z_JF_R].translation[:] = np.array(local_segments_positions['upperarmR'] + local_segments_positions['torso'])
    
    IDX_EL_Z_JF_R = model.getJointId('right_elbow_Z')
    model.jointPlacements[IDX_EL_Z_JF_R].translation[:] = np.array(local_segments_positions['lowerarmR'])

    IDX_SH_Z_JF_L = model.getJointId('left_shoulder_Z')
    model.jointPlacements[IDX_SH_Z_JF_L].translation[:] = np.array(local_segments_positions['upperarmL'] + local_segments_positions['torso'])

    IDX_EL_Z_JF_L = model.getJointId('left_elbow_Z')
    model.jointPlacements[IDX_EL_Z_JF_L].translation[:] = np.array(local_segments_positions['lowerarmL'])

    IDX_HIP_Z_JF_R = model.getJointId('right_hip_Z')
    model.jointPlacements[IDX_HIP_Z_JF_R].translation[:] = np.array(local_segments_positions['thighR'])

    IDX_KNEE_Z_JF_R = model.getJointId('right_knee_Z')
    model.jointPlacements[IDX_KNEE_Z_JF_R].translation[:] = np.array(local_segments_positions['shankR'])

    IDX_ANKLE_Z_JF_R = model.getJointId('right_ankle_Z')
    model.jointPlacements[IDX_ANKLE_Z_JF_R].translation[:] = np.array(local_segments_positions['footR'])

    IDX_HIP_Z_JF_L = model.getJointId('left_hip_Z')
    model.jointPlacements[IDX_HIP_Z_JF_L].translation[:] = np.array(local_segments_positions['thighL'])

    IDX_KNEE_Z_JF_L = model.getJointId('left_knee_Z')
    model.jointPlacements[IDX_KNEE_Z_JF_L].translation[:] = np.array(local_segments_positions['shankL'])

    IDX_ANKLE_Z_JF_L = model.getJointId('left_ankle_Z')
    model.jointPlacements[IDX_ANKLE_Z_JF_L].translation[:] = np.array(local_segments_positions['footL'])

    return model