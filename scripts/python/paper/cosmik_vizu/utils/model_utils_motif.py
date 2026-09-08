import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import hppfcl
import meshcat

import os
import sys 
  
from typing import Dict
import numpy as np

from utils.src_rtcosmik.rtcosmik.utils.linear_algebra_utils import col_vector_3D

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from collections import defaultdict


# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
parent_directory = os.path.dirname(script_directory)

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


def create_mks_positions_dict(mks_labels_all, mks_positions, MKS_SET_USER_MAPPING,frame_idx):
    
    marker_names = MKS_SET_USER_MAPPING.keys()
    # marker_names = set()
    # for label in labels:
    #     if label != 'time' and label.endswith(('_X', '_Y', '_Z')):
    #         marker_names.add(label[:-2])
    # marker_names = sorted(list(marker_names))

    
    def get_marker_columns(labels, marker):
        try:
            return [
                np.where(labels == f"{marker}_X")[0][0],
                np.where(labels == f"{marker}_Y")[0][0],
                np.where(labels == f"{marker}_Z")[0][0]
            ]
        except IndexError:
            print(f"Marker {marker} not found in labels")
            return None

    marker_columns = {marker: get_marker_columns(mks_labels_all, marker) for marker in marker_names}
    marker_columns = {k: v for k, v in marker_columns.items() if v is not None}

    
    frame_data = mks_positions[frame_idx, :]
    mks_positions_dict = {}
    for marker, col_indices in marker_columns.items():
        pos = frame_data[col_indices].astype(float)
        mks_positions_dict[marker] = pos

    mks_positions_dict_full = []
    nb_frames = mks_positions.shape[0] - 2
    for i in range(2, nb_frames + 2):
        frame_data = mks_positions[i, :]
        frame_dict = {}
        for marker, col_indices in marker_columns.items():
            pos = frame_data[col_indices].astype(float)
            frame_dict[marker] = pos
        mks_positions_dict_full.append(frame_dict)

    return mks_positions_dict, mks_positions_dict_full



def build_biomechanical_model(robot, param, old_robot=None):
    # This function is used to fix the dof of the biomechanical model from the original cosmik whole-body model
    model=robot.model
    visual_model = robot.visual_model
    collision_model = robot.collision_model


    # Retrieve all frame names
    all_joint_names = [model.names[j] for j in range(model.njoints)]

    #print( all_joint_names )
    # for jn in model.names:
    #     joint_id = model.getJointId(jn)
    #     print(f"{jn:<20} | {joint_id}")

    # Get joint IDs of joints not to lock
 
 
    jointsNotToLock = [jn for jn in all_joint_names if jn in param.active_joints]
    
    jointsNotToLockIDs = []
  
    for jn in jointsNotToLock:
         
        if model.existJointName(jn):
            if jn != "universe" :
                      
                    jointsNotToLockIDs.append(model.getJointId(jn)-1) #NOTE -1 because of universe joint
        else:
            print('Warning: joint ' + str(jn) + ' does not belong to the model!')


   
    if param.free_flyer==True:#remove the free flyer joint as it does not have torque value
        jointsNotToLockIDs=jointsNotToLockIDs[1:]
       
   
    if len(param.n_tau) != len(jointsNotToLockIDs):
        param.n_tau = [param.n_tau[i] for i in jointsNotToLockIDs]
        param.n_dq = [param.n_dq[i] for i in jointsNotToLockIDs]
        param.n_ddq = [param.n_ddq[i] for i in jointsNotToLockIDs]
        param.qdi = [param.qdi[i] for i in jointsNotToLockIDs]
   
        param.q_min = [param.q_min[i] for i in jointsNotToLockIDs]
        param.q_max = [param.q_max[i] for i in jointsNotToLockIDs]
    
    # # param["q_min"] = [-3.14 for i in jointsNotToLockIDs]
    # # param["q_max"] = [3.14 for i in jointsNotToLockIDs]
    
    # param["dq_lim"] = [param["dq_lim"][i] for i in jointsNotToLockIDs]


         
    # Identify joints to lock (joints NOT in active_joints)
    jointsToLock = [jn for jn in all_joint_names if jn not in param.active_joints]


    # Get joint IDs of joints to lock
    jointsToLockIDs = []
    for jn in jointsToLock:
        if model.existJointName(jn):
            jointsToLockIDs.append(model.getJointId(jn))
        else:
            print('Warning: joint ' + str(jn) + ' does not belong to the model!')


    
    # build reduced mechanical model and update meshes
    initialJointConfig = pin.neutral(model)
    # geom_models = [collision_model,visual_model]
    # model, geom_models = pin.buildReducedModel(model,list_of_geom_models=geom_models,list_of_joints_to_lock=jointsToLockIDs,reference_configuration=initialJointConfig)
    
    # collision_model = geom_models[0] 
    # visual_model = geom_models[1]
    
   
  
  # Option 3: Build the reduced model including multiple geometric models (for example:
# visuals, collision).
    geom_models = [visual_model, collision_model]
    model, geometric_models_reduced = pin.buildReducedModel(
    model,
    list_of_geom_models=geom_models,
    list_of_joints_to_lock=jointsToLockIDs,
    reference_configuration=initialJointConfig)

    visual_model, collision_model = (geometric_models_reduced[0], geometric_models_reduced[1],)
  
    
  


    # #now that the model is reduced we can save limits from urdf
    if param.free_flyer==True:
        param.q_max=model.upperPositionLimit[7:]
        param.q_min=model.lowerPositionLimit[7:]
    else:
        param.q_max=model.upperPositionLimit  
        param.q_min=model.lowerPositionLimit
         
    # for jn in model.names:
    #     joint_id = model.getJointId(jn)
    #     print(f"{jn:<20} | {joint_id}") 

 
 


    return model, collision_model,  visual_model, param






def create_mapped_mk_labels(mks_positions,mks_positions_full, MKS_SET_MAPPING):
    
    mks_positions_renamed = {
    MKS_SET_MAPPING.get(marker, marker): position
    for marker, position in mks_positions.items()}

    mks_position_full_renamed = []

    for frame_dict in mks_positions_full:
        renamed = {
            MKS_SET_MAPPING.get(k, k): v for k, v in frame_dict.items()}
        
        mks_position_full_renamed.append(renamed)
    return mks_positions_renamed, mks_position_full_renamed




def save_scaled_urdf(new_model_name, new_model_path, scaled_model, visual_model=None, collision_model=None, data=None):
    """
    Saves a scaled Pinocchio model as a URDF file, preserving joint types and properties.

    Args:
        new_model_name (str): Name of the robot model in the URDF.
        new_model_path (str): File path where the URDF will be saved.
        scaled_model (pin.Model): The scaled Pinocchio model.
        visual_model (pin.GeometryModel, optional): Visual geometry model.
        collision_model (pin.GeometryModel, optional): Collision geometry model.
        data (pin.Data, optional): Precomputed data for the model (not used here).
    """
    urdf = ET.Element("robot", name=new_model_name)

    # Define materials
    materials = {
        "body_color": "0.2 0.05 0.8 0.3",
        "body_color_R": "0.8 0.05 0.2 0.6",
        "body_color_L": "0.05 0.8 0.2 0.6",
        "Black": "0 0 0 1",
        "marker_color": "1 0 0 1"
    }
    for mat_name, rgba in materials.items():
        material = ET.SubElement(urdf, "material", name=mat_name)
        ET.SubElement(material, "color", rgba=rgba)
        ET.SubElement(material, "texture")

    # Create a mapping from joint IDs to BODY frame names (first frame per joint)
    joint_id_to_link_name = {}
    for frame in scaled_model.frames:
        if frame.type == pin.FrameType.BODY:
            joint_id = frame.parentJoint
            if joint_id not in joint_id_to_link_name:
                joint_id_to_link_name[joint_id] = frame.name


    # Add links for all BODY frames
    body_frames = [frame for frame in scaled_model.frames if frame.type == pin.FrameType.BODY]

    body_frames_by_joint = defaultdict(list)
    unique_masses = {}
    unique_coms = {}
    unique_inertia_matrices = {}

    for frame in scaled_model.frames:
        if frame.type == pin.BODY:
            parent_joint_idx = frame.parentJoint
            parent_joint_name = scaled_model.names[parent_joint_idx]
            mass = scaled_model.inertias[parent_joint_idx].mass
            com = scaled_model.inertias[parent_joint_idx].lever
            inertia_matrix = scaled_model.inertias[parent_joint_idx].inertia
            body_frames_by_joint[parent_joint_name].append(frame.name)
            if parent_joint_name not in unique_masses:
                unique_masses[parent_joint_name] = mass
                unique_coms[parent_joint_name] = com
                unique_inertia_matrices[parent_joint_name] = inertia_matrix

    for frame in body_frames:
        link_name = frame.name
        for i, sublist in enumerate(list(body_frames_by_joint.values())):
            if link_name in sublist:
                # Find the index of the first non-virtual name in this sublist
                non_virtual_indices = [idx for idx, name in enumerate(sublist) if 'virtual' not in name]
                if non_virtual_indices:
                    first_non_virtual_index = min(non_virtual_indices)
                    j = sublist.index(link_name)
                    if j == first_non_virtual_index:
                        mass = list(unique_masses.values())[i]
                        com = list(unique_coms.values())[i]
                        inertia_matrix = list(unique_inertia_matrices.values())[i]
                    else:
                        mass = 0.0
                        com = np.zeros(3)
                        inertia_matrix = np.zeros((3, 3))
                else:
                    mass = 0.0
                    com = np.zeros(3)
                    inertia_matrix = np.zeros((3, 3))

        link_elem = ET.SubElement(urdf, "link", name=link_name)
        inertial_elem = ET.SubElement(link_elem, "inertial")
        ET.SubElement(inertial_elem, "mass", value=str(mass))
        ET.SubElement(inertial_elem, "origin", xyz=f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}", rpy="0 0 0")
        ET.SubElement(inertial_elem, "inertia",
                      ixx=str(inertia_matrix[0, 0]), ixy=str(inertia_matrix[0, 1]), ixz=str(inertia_matrix[0, 2]),
                      iyy=str(inertia_matrix[1, 1]), iyz=str(inertia_matrix[1, 2]), izz=str(inertia_matrix[2, 2]))

        # Add visual geometries
        if visual_model:
            frame_id = scaled_model.getFrameId(link_name)
            for geom in visual_model.geometryObjects:
                if geom.parentFrame == frame_id:
                    visual_elem = ET.SubElement(link_elem, "visual")
                    placement = geom.placement
                    xyz = placement.translation
                    rpy = pin.rpy.matrixToRpy(placement.rotation)
                    ET.SubElement(visual_elem, "origin", xyz=f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
                                  rpy=f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}")
                    geometry_elem = ET.SubElement(visual_elem, "geometry")
                    mesh_path = geom.meshPath
                    scale = geom.meshScale
                    ET.SubElement(geometry_elem, "mesh", filename=mesh_path,
                                  scale=f"{scale[0]:.6f} {scale[1]:.6f} {scale[2]:.6f}")
                    material_name = "body_color_L" if "left_" in link_name else "body_color_R" if "right_" in link_name else "body_color"
                    ET.SubElement(visual_elem, "material", name=material_name)

        # Add collision geometries (optional)
        if collision_model:
            frame_id = scaled_model.getFrameId(link_name)
            for geom in collision_model.geometryObjects:
                if geom.parentFrame == frame_id:
                    collision_elem = ET.SubElement(link_elem, "collision")
                    placement = geom.placement
                    xyz = placement.translation
                    rpy = pin.rpy.matrixToRpy(placement.rotation)
                    ET.SubElement(collision_elem, "origin", xyz=f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
                                  rpy=f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}")
                    geometry_elem = ET.SubElement(collision_elem, "geometry")
                    mesh_path = geom.meshPath
                    # if not mesh_path.startswith("package://"):
                    #     mesh_path = f"package://human_urdf/meshes/{os.path.basename(mesh_path)}"
                    scale = geom.meshScale
                    ET.SubElement(geometry_elem, "mesh", filename=mesh_path,
                                  scale=f"{scale[0]:.6f} {scale[1]:.6f} {scale[2]:.6f}")

    # Define joint type mapping
    joint_type_map = {
        "FF": "floating",
        "RX": "revolute",
        "RY": "revolute",
        "RZ": "revolute",
        "RevoluteUnaligned": "revolute",
        "PR": "prismatic",
        "SP": "spherical",
        "Fixed": "fixed"
    }

    # Add active joints
    for i in range(2, scaled_model.njoints):  # Start from 1 to skip universe
        joint = scaled_model.joints[i]
        joint_name = scaled_model.names[i]
        parent_idx = scaled_model.parents[i]
        parent_link = joint_id_to_link_name.get(parent_idx, "middle_pelvis")
        child_link = joint_id_to_link_name.get(i)
        if not child_link:
            print(f"Warning: No BODY frame for joint {joint_name}, skipping")
            continue

        # Get joint type
        joint_shortname = joint.shortname() if hasattr(joint, 'shortname') else "Unknown"
        actual_shortname = joint_shortname.split("JointModel")[
            1] if "JointModel" in joint_shortname else joint_shortname
        joint_type = joint_type_map.get(actual_shortname, "fixed")

        # Create joint element
        joint_elem = ET.SubElement(urdf, "joint", name=joint_name, type=joint_type)
        ET.SubElement(joint_elem, "parent", link=parent_link)
        ET.SubElement(joint_elem, "child", link=child_link)

        # Add origin
        placement = scaled_model.jointPlacements[i]
        xyz = placement.translation
        rpy = pin.rpy.matrixToRpy(placement.rotation)
        ET.SubElement(joint_elem, "origin", xyz=f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
                      rpy=f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}")

        # Add axis and limits for revolute joints
        if joint_type == "revolute":
            joint_data = joint.createData()
            joint.calc(joint_data, np.zeros(joint.nq))
            axis = joint_data.S[3:6]  # Extract rotation axis
            ET.SubElement(joint_elem, "axis", xyz=f"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f}")

            idx_q = joint.idx_q
            idx_v = joint.idx_v
            lower = scaled_model.lowerPositionLimit[idx_q]
            upper = scaled_model.upperPositionLimit[idx_q]
            effort = scaled_model.effortLimit[idx_v]
            velocity = scaled_model.velocityLimit[idx_v]
            if lower is not None and upper is not None:
                ET.SubElement(joint_elem, "limit", effort=str(effort), velocity=str(velocity),
                              lower=str(lower), upper=str(upper))

    # Add fixed joints for multiple BODY frames sharing the same parentJoint
    frames_by_joint = defaultdict(list)
    for frame in body_frames:
        frames_by_joint[frame.parentJoint].append(frame)
    for joint_id, frames in frames_by_joint.items():
        if len(frames) > 1:  # Multiple BODY frames for this joint
            main_link = joint_id_to_link_name.get(joint_id, frames[0].name)
            main_frame = next(f for f in frames if f.name == main_link)
            for other_frame in frames:
                if other_frame.name != main_link:
                    other_link = other_frame.name
                    joint_name = f"fixed_{main_link}_to_{other_link}"
                    joint_elem = ET.SubElement(urdf, "joint", name=joint_name, type="fixed")
                    ET.SubElement(joint_elem, "parent", link=main_link)
                    ET.SubElement(joint_elem, "child", link=other_link)
                    # rel_placement = main_frame.placement.inverse() * other_frame.placement
                    rel_placement = main_frame.placement
                    xyz = rel_placement.translation
                    rpy = pin.rpy.matrixToRpy(rel_placement.rotation)
                    ET.SubElement(joint_elem, "origin", xyz=f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
                                  rpy=f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}")

    # Add registered markers
    for frame in scaled_model.frames:
        if frame.type == pin.FrameType.OP_FRAME:
            marker_name = frame.name
            print(marker_name)
            parent_joint_id = frame.parentJoint
            # if marker_name in ["RKNE", "RMKNE"]:
            #     parent_link = "right_upperleg"
            # elif marker_name in ["LKNE", "LMKNE"]:
            #     parent_link = "left_upperleg"
            # else:
            #     parent_link = joint_id_to_link_name.get(parent_joint_id, "middle_pelvis")
            parent_link = joint_id_to_link_name.get(parent_joint_id, "middle_pelvis")
            placement = frame.placement
            marker_link_elem = ET.SubElement(urdf, "link", name=marker_name)
            visual_elem = ET.SubElement(marker_link_elem, "visual")
            ET.SubElement(visual_elem, "origin", xyz="0 0 0", rpy="0 0 0")
            geometry_elem = ET.SubElement(visual_elem, "geometry")
            ET.SubElement(geometry_elem, "sphere", radius="0.01")
            ET.SubElement(visual_elem, "material", name="marker_color")
            joint_name = f"joint_{marker_name}"
            joint_elem = ET.SubElement(urdf, "joint", name=joint_name, type="fixed")
            ET.SubElement(joint_elem, "parent", link=parent_link)
            ET.SubElement(joint_elem, "child", link=marker_name)
            xyz = placement.translation
            rpy = pin.rpy.matrixToRpy(placement.rotation)
            ET.SubElement(joint_elem, "origin", xyz=f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
                          rpy=f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}")

    # Save the URDF file
    ET.indent(ET.ElementTree(urdf), space="  ")
    os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
    ET.ElementTree(urdf).write(new_model_path, encoding="utf-8", xml_declaration=True)
    print(f"Scaled URDF saved to: {new_model_path}")















################################################################################################################################################################################################################
################################################################################################################################################################################################################
################################################################################################################################################################################################################
###################                       Functions below were adapted from COSMIK on 25th of June 2025, hence the _motif at the end of the functions                                                                                      ###################        
################################################################################################################################################################################################################
################################################################################################################################################################################################################
################################################################################################################################################################################################################




  
  
  
#Construct challenge segments frames from mocap mks
# - mocap_mks_positions is a dictionnary of mocap mks names and 3x1 global positions
# - returns sgts_poses which correspond to a dictionnary to segments poses and names, constructed from mks global positions
def construct_segments_frames_motif(mocap_mks_positions): 
    """
    Constructs a dictionary of segment poses from motion capture marker positions.
    Args:
        mocap_mks_positions (dict): A dictionary containing the positions of motion capture markers.
    Returns:
        dict: A dictionary where keys are segment names (e.g., 'torso', 'upperarmR') and values are the corresponding poses.
    """
    
    # Check if all required markers are in the dataset for each segment
    sgts_poses = {}
    
    def maybe_add_pose(segment_name, marker_list, compute_func):
        if all(m in mocap_mks_positions for m in marker_list):
            sgts_poses[segment_name] = compute_func(mocap_mks_positions)

    maybe_add_pose("head",      ['RSHO', 'LSHO', 'Head'],                   get_head_pose)

    maybe_add_pose("torso",     ['RSHO', 'LSHO', 'RASI', 'LASI', 'RPSI', 'LPSI', 'C7'], get_torso_pose)
    maybe_add_pose("thorax",     ['RSHO', 'LSHO', 'RASI', 'LASI', 'RPSI', 'LPSI', 'C7'], get_thorax_pose) # same as torso as it calls torso

    maybe_add_pose("right_upperarm", ['LSHO','RSHO','RMELB','RELB'],      get_right_upperarm_pose)
    maybe_add_pose("right_lowerarm", ['RMELB','RELB','RMWRI','RWRI'],          get_right_lowerarm_pose)
    
    maybe_add_pose("left_upperarm", ['LSHO','RSHO','LMELB','LELB'],      get_left_upperarm_pose)
    maybe_add_pose("left_lowerarm", ['LMELB','LELB','LMWRI','LWRI'],          get_left_lowerarm_pose)
    
    maybe_add_pose("pelvis",    ['RPSI','LPSI','RASI','LASI'],                  get_pelvis_pose)
    
    maybe_add_pose("right_upperleg",    ['RASI','LASI','RKNE','RMKNE'],                 get_right_upperleg_pose)
    maybe_add_pose("right_lowerleg",    ['RKNE','RMKNE','RMANK','RANK'],              get_right_lowerleg_pose)
    maybe_add_pose("right_foot",     ['RMANK','RANK','RTOE','R5MHD','RHEE'],      get_right_foot_pose)
    
    maybe_add_pose("left_upperleg",    ['LASI','RASI','LKNE','LMKNE'],                 get_left_upperleg_pose)
    maybe_add_pose("left_lowerleg",    ['LKNE','LMKNE','LMANK','LANK'],              get_left_lowerleg_pose)
    maybe_add_pose("left_foot",     ['LMANK','LANK','LTOE','L5MHD','LHEE'],      get_left_foot_pose)
     
    
    return sgts_poses


def get_left_upperarm_pose(mks_positions):
    """
    Calculate the pose of the left upper arm based on motion capture marker positions.
    This function computes the transformation matrix representing the pose of the left upper arm.
    The pose is calculated using the positions of specific markers on the body, such as the shoulder
    and elbow markers. The resulting pose matrix is a 4x4 homogeneous transformation matrix.
    Args:
        mks_positions (dict): A dictionary containing the positions of motion capture markers.
            The keys are marker names (e.g., 'LShoulder', 'LMELB', 'LELB', 'LHLE', 'LHME', 'LSAT', 'RSAT'),
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
def get_right_upperleg_pose(mks_positions, gender='male'):
    """
    Calculate the pose of the right thigh based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                Expected keys include 'RHip', 'RKNE', 'RMKNE', 
                                'RIAS', 'LIAS', 'RFLE', and 'RFME'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right thigh. The matrix 
                   includes rotation and translation components.
    """
    if gender == 'male':
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


def get_left_upperleg_pose(mks_positions, gender='male'):
    """
    Calculate the pose of the left thigh based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                Expected keys are 'LHip', 'LKNE', 'LMKNE', 'LIAS', 'RIAS', 'LFLE', and 'LFME'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the left thigh. The matrix includes
                   rotation and translation components.
    """
    if gender == 'male':
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
def get_right_lowerleg_pose(mks_positions):
    """
    Calculate the pose of the right shank based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                The keys should include either 'RKNE', 'RMKNE', 
                                'RMANK', 'RANK' or 'RFLE', 'RFME', 'RTAM', 'RFAL'.
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

def get_left_lowerleg_pose(mks_positions):
    """
    Calculate the pose of the left shank based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                The keys should include either 'LKNE', 'LMKNE', 
                                'LMANK', 'LANK' or 'LFLE', 'LFME', 'LTAM', 'LFAL'.
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
def get_right_foot_pose(mks_positions):
    """
    Calculate the pose of the right foot based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                The keys can be either 'RMANK', 'RANK', 'RTOE', 
                                'RHEE' or 'RTAM', 'RFAL', 'RFM5', 'RFM1', 'RFCC'.
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

def get_left_foot_pose(mks_positions):
    """
    Calculate the pose of the left foot based on motion capture marker positions.
    This function computes the transformation matrix (pose) of the left foot using
    the positions of various markers from motion capture data. The pose is represented
    as a 4x4 homogeneous transformation matrix.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers.
                                The keys are marker names and the values are their respective
                                3D coordinates (numpy arrays).
    Returns:
    numpy.ndarray: A 4x4 homogeneous transformation matrix representing the pose of the left foot.
    Notes:
    - The function checks for the presence of specific markers ('LMANK', 'LANK',
      'LTOE', 'LHEE') to compute the pose. If these markers are not present, it
      uses alternative markers ('LTAM', 'LFAL', 'LFM5', 'LFM1', 'LFCC').
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
def get_virtual_pelvis_pose(mks_positions):
    """
    Calculate the pelvis pose matrix from motion capture marker positions.
    The function computes the pelvis pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the pelvis coordinate system. Finally, it constructs the 
    pose matrix and ensures it is orthogonal.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of the motion capture markers.
                                The keys can be either 'RPSI', 'LPSI', 'RASI', 
                                'LASI', or 'RIPS', 'LIPS', 'RIAS', 'LIAS'.
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


def get_pelvis_pose(mks_positions, gender = 'male'):
    """
    Calculate the pelvis pose matrix from motion capture marker positions.
    The function computes the pelvis pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the pelvis coordinate system. Finally, it constructs the 
    pose matrix and ensures it is orthogonal.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of the motion capture markers.
                                The keys can be either 'RPSI', 'LPSI', 'RASI', 
                                'LASI', or 'RIPS', 'LIPS', 'RIAS', 'LIAS'.
    Returns:
    numpy.ndarray: A 4x4 pose matrix representing the pelvis pose.
    """

    if gender == 'male':
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
    # Z = mks_positions['RASI'] - mks_positions['LASI']
    Z = center_right_ASIS_PSIS - center_left_ASIS_PSIS
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)


    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ((center_right_ASIS_PSIS + center_left_ASIS_PSIS)/2.0).reshape(3,)
    # pose[:3,3] = LJC.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose

def get_left_lowerarm_pose(mks_positions):
    """
    Calculate the pose of the left lower arm based on motion capture marker positions.
    This function computes the transformation matrix representing the pose of the left lower arm.
    It uses the positions of specific markers to determine the orientation and position of the arm.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers.
                                The keys should include either 'LMELB', 'LELB', 
                                'LMWRI', 'LWRI' or 'LHLE', 'LHME', 'LRSP', 'LUSP'.
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
    # Z = Z/np.linalg.norm(Z)
    Z = Z.reshape(3, 1) / np.linalg.norm(Z)

    X = np.cross(Y, Z, axis=0)
    X = X.reshape(3, 1) / np.linalg.norm(X)

    Z = np.cross(X.flatten(), Y.flatten())
    Z = Z.reshape(3, 1) / np.linalg.norm(Z)
    # Z = np.cross(X, Y, axis=0)


    pose[:3, 0] = X.flatten()
    pose[:3, 1] = Y.flatten()
    pose[:3, 2] = Z.flatten()
    pose[:3, 3] = elbow_center.flatten()
    pose[:3, :3] = orthogonalize_matrix(pose[:3, :3])


    # print("Lowerarm Left Pose:\n", pose)  # Impression pour débogage
    # check_orthogonality(pose)  # Ajoutez cette ligne pour vérifier l'orthogonalité

    return pose


#construct upperarm frames and get their poses
def get_right_upperarm_pose(mks_positions):
    """
    Calculate the pose of the right upper arm based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                Expected keys include 'RShoulder', 'RMELB', 'RELB', 
                                'RHLE', 'RHME', 'RSAT', and 'LSAT'.
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
def get_thorax_pose(mks_positions,gender='male',subject_height= 1.80):
    #pelvis + distance selon y
    """
    Calculate the abdomen pose matrix from motion capture marker positions.
    The function computes the abdomen pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the abdomen coordinate system. Finally, it constructs the 
    pose matrix and ensures it is orthogonal.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of the motion capture markers.
                                The keys can be either 'RPSI', 'LPSI', 'RASI', 
                                'LASI', or 'RIPS', 'LIPS', 'RIAS', 'LIAS'.
    Returns:
    numpy.ndarray: A 4x4 pose matrix representing the abdomen pose.
    """
    if gender == 'male' : 
        abdomen_ratio = 0.0839
    else : 
        abdomen_ratio = 0.0776
    
    pelvis_pose =(get_pelvis_pose(mks_positions,gender)[:3,3]).reshape(3,1)
    torso_pose = (get_torso_pose(mks_positions)[:3,3]).reshape(3,1)
    direction = torso_pose - pelvis_pose                     
    direction = direction / np.linalg.norm(direction)  
    # pos_torso_in_pelvis = (np.linalg.inv(get_virtual_pelvis_pose(mks_positions)) @ torso_pose)[:3,3]

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
    # Z = mks_positions['RASI'] - mks_positions['LASI']
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

def get_head_pose(mks_positions):
    """
    Calculate the pose of the head based on motion capture marker positions.
    The function computes a 4x4 transformation matrix representing the pose of the head.
    The matrix includes rotation and translation components derived from the positions
    of specific markers.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers.
                                Expected keys are 'Neck', 'midHip', 'C7', 'CV7', 'SJN', 
                                'HeadR', 'HeadL', 'RSAT', and 'LSAT'. Each key should map to a 
                                numpy array of shape (3,).
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the head pose.
    """

    pose = np.eye(4,4)
    X, Y, Z, head_center = [], [], [], []
    if 'Head' in mks_positions:
        head_center = (mks_positions['RSHO'] + mks_positions['LSHO'])/2.0 
        top_head = mks_positions['Head']
        Y = (top_head - head_center).reshape(3,1)
        Y = Y/np.linalg.norm(Y)

        Z = (mks_positions['REar'] - mks_positions['LEar']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
    else: 
        head_center = (mks_positions['RSHO'] + mks_positions['LSHO'])/2.0 
        X = mks_positions['FHD'] - mks_positions['BHD']
        X = X/np.linalg.norm(X)
        Z = mks_positions['RHD'] - mks_positions['LHD']
        Z = Z/np.linalg.norm(Z)
        Y = np.cross(Z, X, axis=0)
        Z = np.cross(X, Y, axis=0)


    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = head_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose



#construct torso frame and get its pose from a dictionnary of mks positions and names
def get_torso_pose(mks_positions):
    """
    Calculate the torso pose matrix from motion capture marker positions.
    The function computes a 4x4 transformation matrix representing the pose of the torso.
    The matrix includes rotation and translation components derived from the positions
    of specific markers.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers.
                                Expected keys are 'Neck', 'midHip', 'C7', 'CV7', 'SJN', 
                                'HeadR', 'HeadL', 'RSAT', and 'LSAT'. Each key should map to a 
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

#construct upperarm frames and get their poses
def get_right_upperarm_pose(mks_positions):
    """
    Calculate the pose of the right upper arm based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                Expected keys include 'RShoulder', 'RMELB', 'RELB', 
                                'RHLE', 'RHME', 'RSAT', and 'LSAT'.
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



def get_local_mks_positions_motif(sgts_poses: Dict, mks_positions: Dict, sgts_mks_dict: Dict)-> Dict:
    """_Get the local 3D position of the lstms markers_

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



def scale_human_model_motif(model,visual_model, local_segments_positions,SGTS_JOINTS_CALIB_MAPPING):
    """
    Scales a human Pinocchio model and updates associated visual geometry placements.

    Parameters
    ----------
    model : pin.Model
        The Pinocchio kinematic model.

    local_segments_positions : dict
        Dictionary mapping segment names to their new local translation vectors.

    SGTS_JOINTS_CALIB_MAPPING : dict
        Maps segment names to joint names.

    visual_model : pin.GeometryModel
        Visual geometry model (e.g., from pin.buildGeomFromUrdf(...).visual_model)

    Returns
    -------
    model : pin.Model
        The updated model with scaled joint placements.

    visual_model : pin.GeometryModel
        The updated visual model with modified geometry placements.
    """
    SGTS_VISUAL_MAPPING = {
    "pelvis" :'root_joint',
    "right_upperleg":'right_hip_X', # parent root_joint
    "right_lowerleg":'right_knee',# parent right_hip_Y
    "right_foot":'right_ankle_X',
    
    "left_upperleg":'left_hip_Y',
    "left_lowerleg":'left_knee',
    'left_foot':'left_ankle_X',
    
    #"abdomen":'middle_lumbar_X', 
    "torso":'middle_thoracic_Y',
    
    #"head":'middle_cervical_Y',
  
    
    "right_upperarm":'right_shoulder_Y',
    "right_lowerarm":'right_elbow_Y',
    "right_hand":'right_wrist_X',
 
    "left_upperarm":'left_shoulder_Y',
    "left_lowerarm":'left_elbow_Y',
    "left_hand":'left_wrist_X'
    }
    
    q = pin.neutral(model)
    data = pin.Data(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    original_length={}
    for segment_name, joint_name in SGTS_JOINTS_CALIB_MAPPING.items():
    
    
        if segment_name not in local_segments_positions:
            continue  # Skip if segment not in provided positions
        
    
        joint_id = model.getJointId(joint_name)
        if joint_id == 0:
            print(f"⚠️ Joint '{joint_name}' not found in model.")
            continue
            
        #print(model.names[joint_id-1] + " to " + model.names[joint_id])
        original_length[segment_name]=np.linalg.norm(model.jointPlacements[joint_id-1].translation - model.jointPlacements[joint_id].translation)
        
    #print(original_length)
    for segment_name, joint_name in SGTS_JOINTS_CALIB_MAPPING.items():
        
        
        if segment_name not in local_segments_positions:
            continue  # Skip if segment not in provided positions
        
      
        joint_id = model.getJointId(joint_name)
        if joint_id == 0:
            print(f"⚠️ Joint '{joint_name}' not found in model.")
            continue
 
        
        # Update only the translation of the joint placement
        model.jointPlacements[joint_id].translation = local_segments_positions[segment_name]
        print(f"✅ Updated joint '{joint_name}' (ID {joint_id}) with new translation for segment '{segment_name}'")
        parent = model.parents[joint_id]
        print(segment_name)
        print(parent) 
        print(model.names[parent] + " to " + model.names[joint_id])
        new_length=np.linalg.norm(model.jointPlacements[parent].translation - model.jointPlacements[joint_id].translation)
         

  
        # Compute uniform scale factor
        scale_factor =  1*new_length / original_length[segment_name]
        if segment_name=="left_foot" or  segment_name=="right_foot":
            scale_factor=1 # no need to modify the foot lengths
        # print("new_length")
        # print(segment_name)
        # print(new_length)
        # print(original_length[segment_name])
        # print(scale_factor)
        
        scale_vector = np.array([0.9*scale_factor,1*scale_factor,0.9*scale_factor])#[1,1,1]##
        if segment_name=="torso" :
            scale_vector = np.array([0.8*scale_factor,0.9*scale_factor,0.8*scale_factor])#[1,1,1]##

        
        visual_joint_name = SGTS_VISUAL_MAPPING.get(segment_name)
        print(visual_joint_name)
        visual_joint_id = model.getJointId(visual_joint_name)#-1
        # Update visuals linked to this joint
        for visual in visual_model.geometryObjects:
            
            if visual.parentJoint == visual_joint_id:
                
                
                visual.meshScale *= scale_vector 
         

    return model,visual_model


def get_local_segments_positions_motif(sgts_poses: Dict, with_hand=True)->Dict:
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
    
    if "thorax" in sgts_poses:
        thorax_global = sgts_poses["thorax"]
        local_positions["thorax"] = (np.linalg.inv(pelvis_pose) @ thorax_global @ np.array([0, 0, 0, 1]))[:3]
    
     # Torso with respect to pelvis
    if "torso" in sgts_poses:
        torso_global = sgts_poses["torso"]
        thorax_global = sgts_poses["thorax"]
        local_positions["torso"] = (np.linalg.inv(thorax_global) @ torso_global @ np.array([0, 0, 0, 1]))[:3]
        #need to adjust torso frame to aligned it with thorax and pelvis frames.

    # Head with respect to torso
    if "head" in sgts_poses:
        head_global = sgts_poses["head"]
        torso_global = sgts_poses["torso"]
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

def get_right_lowerarm_pose(mks_positions):
    """
    Calculate the pose of the right lower arm based on motion capture marker positions.
    The function computes the transformation matrix (pose) of the right lower arm using the positions of specific markers.
    It first checks for the presence of 'RMELB' in the marker positions to determine which set of markers to use.
    The pose is represented as a 4x4 homogeneous transformation matrix.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. The keys are marker names,
                                and the values are their corresponding 3D positions (numpy arrays).
    Returns:
    numpy.ndarray: A 4x4 homogeneous transformation matrix representing the pose of the right lower arm.
    """

    pose = np.eye(4,4)
    X, Y, Z, elbow_center = [], [], [], []
    elbow_center = (mks_positions['RMELB'] + mks_positions['RELB']).reshape(3,1)/2.0
    wrist_center = (mks_positions['r_mwrist'] + mks_positions['r_lwrist']).reshape(3,1)/2.0
    
    Y = elbow_center - wrist_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['r_lwrist'] - mks_positions['r_mwrist']).reshape(3,1)
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
def get_right_lowerleg_pose(mks_positions):
    """
    Calculate the pose of the right shank based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                The keys should include either 'RKNE', 'RMKNE', 
                                'RMANK', 'RANK' or 'RFLE', 'RFME', 'RTAM', 'RFAL'.
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

def mks_registration_motif(model, mks_local_positions, sgts_mks_dict, SGTS_JOINTS_CALIB_MAPPING):
    """
    Registers marker frames to a Pinocchio model using provided segment-to-joint mappings.
    ####### WARNING this function was modifed from COSMIK repo to handle special atatchements of knees, shoulder and feet atatched markers by using the following:
    
     flag = -1 if any(x in marker_name for x in ["KNE",  "HEE", "TOE", "5MHD"]) else 0
          
    
    Parameters
    ----------
    model : pin.Model
        The Pinocchio model to which the marker frames will be added.

    mks_local_positions : dict
        Dictionary mapping marker names to their local 3D position in the segment frame.

    sgts_mks_dict : dict
        Dictionary mapping segment names to a list of marker names (filtered to only include available markers).

    SGTS_JOINTS_CALIB_MAPPING : dict
        Dictionary mapping segment names to the name of the associated joint in the model.

    Returns
    -------
    model : pin.Model
        The updated Pinocchio model with additional frames corresponding to the markers.
    """

    inertia = pin.Inertia.Zero()

    for segment, marker_names in sgts_mks_dict.items():
        if segment not in SGTS_JOINTS_CALIB_MAPPING:
            print(f"[WARNING] No joint mapping for segment '{segment}'. Skipping.")
            continue
        
        joint_name = SGTS_JOINTS_CALIB_MAPPING[segment]

      
        
        try:
            joint_id = model.getJointId(joint_name)
        except:
            print(f"[WARNING] Joint '{joint_name}' not found in model. Skipping segment '{segment}'.")
            continue

        try:
            parent_frame_id = model.getFrameId(joint_name)
        except:
            parent_frame_id = 0  # fallback to world if frame not found
      
        for marker_name in marker_names:
            
            if marker_name not in mks_local_positions:
                print(f"[WARNING] Marker '{marker_name}' not in local positions. Skipping.")
                continue
            flag = -1 if any(x in marker_name for x in ["KNE",  "HEE", "TOE", "5MHD"]) else 0
          
            trans = np.array(mks_local_positions[marker_name]).reshape(3)
            frame_placement = pin.SE3(np.eye(3), trans)

            frame = pin.Frame(
                marker_name,          # str
                joint_id-flag,             # int (parent joint id)
                parent_frame_id,      # int (parent frame id)
                frame_placement,      # SE3
                pin.FrameType.OP_FRAME,
                inertia               # Inertia
            )

            model.addFrame(frame, False)

    return model
