import pinocchio as pin
import os
import sys 

import cv2
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

rt_cosmik_path = os.path.dirname(script_directory)
from src.rtcosmik.human_model.urdf_model import * 
from pinocchio.visualize import GepettoVisualizer
from src.rtcosmik.utils.read_write_utils import read_mks_data,udp_csv_to_dataframe,marker_data_to_dataframe, read_subject_info
import pandas as pd
from src.rtcosmik.human_model.urdf_model import * 
from src.rtcosmik.viewer.gv_viewer import place, gv_init, Rquat, add_marker, add_frames
from src.rtcosmik.config_loader import settings
from src.rtcosmik.human_model.model_utils import get_segment_length
from typing import Dict
import numpy as np

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from collections import defaultdict

from src.rtcosmik.human_model.pin_model import build_model
from src.rtcosmik.human_model.model_utils import construct_segments_frames, get_segments_mks_dict
from src.rtcosmik.ik.ik import RT_IK,RT_SWIKA

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

    # -------- Pretty-print fallback for Python < 3.9 --------
    def _fallback_indent(elem, level=0, space="  "):
        """Recursively pretty-print an ElementTree element (Python 3.8-safe)."""
        i = "\n" + level * space
        if len(elem):
            if not (elem.text and elem.text.strip()):
                elem.text = i + space
            for child in elem:
                _fallback_indent(child, level + 1, space)
                if not (child.tail and child.tail.strip()):
                    child.tail = i + space
            if not (elem.tail and elem.tail.strip()):
                elem.tail = i
        else:
            if not (elem.tail and elem.tail.strip()):
                elem.tail = i

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

    # Map joint IDs to BODY frame names (first BODY frame per joint)
    joint_id_to_link_name = {}
    for frame in scaled_model.frames:
        if frame.type == pin.FrameType.BODY:
            joint_id = frame.parentJoint
            if joint_id not in joint_id_to_link_name:
                joint_id_to_link_name[joint_id] = frame.name

    # Collect BODY frames and unique inertial data per parent joint
    body_frames = [frame for frame in scaled_model.frames if frame.type == pin.FrameType.BODY]
    body_frames_by_joint = defaultdict(list)
    unique_masses = {}
    unique_coms = {}
    unique_inertia_matrices = {}

    for frame in scaled_model.frames:
        if frame.type == pin.FrameType.BODY:  # fixed typo here
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
        # Default inertial values
        mass = 0.0
        com = np.zeros(3)
        inertia_matrix = np.zeros((3, 3))

        # Assign inertial props only to the first non-virtual body tied to a joint
        for i, sublist in enumerate(list(body_frames_by_joint.values())):
            if link_name in sublist:
                non_virtual_indices = [idx for idx, name in enumerate(sublist) if 'virtual' not in name]
                if non_virtual_indices:
                    first_non_virtual_index = min(non_virtual_indices)
                    j = sublist.index(link_name)
                    if j == first_non_virtual_index:
                        mass = list(unique_masses.values())[i]
                        com = list(unique_coms.values())[i]
                        inertia_matrix = list(unique_inertia_matrices.values())[i]
                break

        link_elem = ET.SubElement(urdf, "link", name=link_name)
        inertial_elem = ET.SubElement(link_elem, "inertial")
        ET.SubElement(inertial_elem, "mass", value=str(mass))
        ET.SubElement(inertial_elem, "origin",
                      xyz=f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}", rpy="0 0 0")
        ET.SubElement(inertial_elem, "inertia",
                      ixx=str(inertia_matrix[0, 0]), ixy=str(inertia_matrix[0, 1]), ixz=str(inertia_matrix[0, 2]),
                      iyy=str(inertia_matrix[1, 1]), iyz=str(inertia_matrix[1, 2]), izz=str(inertia_matrix[2, 2]))

        # Visual geometries
        if visual_model:
            frame_id = scaled_model.getFrameId(link_name)
            for geom in visual_model.geometryObjects:
                if geom.parentFrame == frame_id:
                    visual_elem = ET.SubElement(link_elem, "visual")
                    placement = geom.placement
                    xyz = placement.translation
                    rpy = pin.rpy.matrixToRpy(placement.rotation)
                    ET.SubElement(visual_elem, "origin",
                                  xyz=f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
                                  rpy=f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}")
                    geometry_elem = ET.SubElement(visual_elem, "geometry")
                    mesh_path = geom.meshPath
                    scale = geom.meshScale
                    ET.SubElement(geometry_elem, "mesh", filename=mesh_path,
                                  scale=f"{scale[0]:.6f} {scale[1]:.6f} {scale[2]:.6f}")
                    material_name = ("body_color_L" if "left_" in link_name
                                     else "body_color_R" if "right_" in link_name
                                     else "body_color")
                    ET.SubElement(visual_elem, "material", name=material_name)

        # Collision geometries
        if collision_model:
            frame_id = scaled_model.getFrameId(link_name)
            for geom in collision_model.geometryObjects:
                if geom.parentFrame == frame_id:
                    collision_elem = ET.SubElement(link_elem, "collision")
                    placement = geom.placement
                    xyz = placement.translation
                    rpy = pin.rpy.matrixToRpy(placement.rotation)
                    ET.SubElement(collision_elem, "origin",
                                  xyz=f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
                                  rpy=f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}")
                    geometry_elem = ET.SubElement(collision_elem, "geometry")
                    mesh_path = geom.meshPath
                    scale = geom.meshScale
                    ET.SubElement(geometry_elem, "mesh", filename=mesh_path,
                                  scale=f"{scale[0]:.6f} {scale[1]:.6f} {scale[2]:.6f}")

    # Joint type mapping
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

    # Active joints
    for i in range(2, scaled_model.njoints):  # skip universe (0) and often root (1)
        joint = scaled_model.joints[i]
        joint_name = scaled_model.names[i]
        parent_idx = scaled_model.parents[i]
        parent_link = joint_id_to_link_name.get(parent_idx, "middle_pelvis")
        child_link = joint_id_to_link_name.get(i)
        if not child_link:
            print(f"Warning: No BODY frame for joint {joint_name}, skipping")
            continue

        joint_shortname = joint.shortname() if hasattr(joint, 'shortname') else "Unknown"
        actual_shortname = joint_shortname.split("JointModel")[1] if "JointModel" in joint_shortname else joint_shortname
        joint_type = joint_type_map.get(actual_shortname, "fixed")

        joint_elem = ET.SubElement(urdf, "joint", name=joint_name, type=joint_type)
        ET.SubElement(joint_elem, "parent", link=parent_link)
        ET.SubElement(joint_elem, "child", link=child_link)

        placement = scaled_model.jointPlacements[i]
        xyz = placement.translation
        rpy = pin.rpy.matrixToRpy(placement.rotation)
        ET.SubElement(joint_elem, "origin",
                      xyz=f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
                      rpy=f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}")

        if joint_type == "revolute":
            joint_data = joint.createData()
            joint.calc(joint_data, np.zeros(joint.nq))
            axis = joint_data.S[3:6]  # rotation axis
            ET.SubElement(joint_elem, "axis",
                          xyz=f"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f}")

            idx_q = joint.idx_q
            idx_v = joint.idx_v
            lower = scaled_model.lowerPositionLimit[idx_q]
            upper = scaled_model.upperPositionLimit[idx_q]
            effort = scaled_model.effortLimit[idx_v]
            velocity = scaled_model.velocityLimit[idx_v]
            ET.SubElement(joint_elem, "limit", effort=str(effort), velocity=str(velocity),
                          lower=str(lower), upper=str(upper))

    # Fixed joints between multiple BODY frames sharing same parentJoint
    frames_by_joint = defaultdict(list)
    for frame in body_frames:
        frames_by_joint[frame.parentJoint].append(frame)
    for joint_id, frames in frames_by_joint.items():
        if len(frames) > 1:
            main_link = joint_id_to_link_name.get(joint_id, frames[0].name)
            main_frame = next(f for f in frames if f.name == main_link)
            for other_frame in frames:
                if other_frame.name != main_link:
                    other_link = other_frame.name
                    joint_name = f"fixed_{main_link}_to_{other_link}"
                    joint_elem = ET.SubElement(urdf, "joint", name=joint_name, type="fixed")
                    ET.SubElement(joint_elem, "parent", link=main_link)
                    ET.SubElement(joint_elem, "child", link=other_link)
                    # NOTE: If you truly want the relative placement, use:
                    # rel_placement = main_frame.placement.inverse() * other_frame.placement
                    rel_placement = main_frame.placement
                    xyz = rel_placement.translation
                    rpy = pin.rpy.matrixToRpy(rel_placement.rotation)
                    ET.SubElement(joint_elem, "origin",
                                  xyz=f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
                                  rpy=f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}")

    # Registered markers
    for frame in scaled_model.frames:
        if frame.type == pin.FrameType.OP_FRAME:
            marker_name = frame.name
            parent_joint_id = frame.parentJoint
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
            ET.SubElement(joint_elem, "origin",
                          xyz=f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
                          rpy=f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}")

    # Save the URDF file (pretty-printing compatible with Python 3.8)
    tree = ET.ElementTree(urdf)
    # Try stdlib pretty printer (3.9+); otherwise fall back
    indent = getattr(ET, "indent", None)
    if callable(indent):
        ET.indent(tree, space="  ", level=0)
    else:
        _fallback_indent(urdf, space="  ")

    os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
    tree.write(new_model_path, encoding="utf-8", xml_declaration=True)
    print(f"Scaled URDF saved to: {new_model_path}")

SUBJECTS = ['Alessandro','Anais','Anastasia','Batiste','Bilal','Claire_','Clement','Emmanuelle','Flavie','Guilhem','Herbert','Kahina','Marie_M','Mathis','Maxime_','Mohamed','Nicolas','Zoe']

subject_ids = {
    "Nicolas": 2307,
    "Mohamed": 1602,
    "Clement": 1118,
    "Mathis": 3361,
    "Claire_": 4827,
    "Anais": 4687,
    "Emmanuelle": 4801,
    "Maxime_": 1847,
    "Alessandro": 4279,
    "Marie_M": 2112,
    "Anastasia": 4216,
    "Flavie": 1012,
    "Zoe": 4162,
    "Kahina": 4665,
    "Herbert": 1508,
    "Guilhem": 4509,
    "Bilal": 4612,
    "Batiste": 2198
}

mks_to_skip = ['LForearm','LUArm', 'RUArm', 'RHJC_study','LHJC_study','r_pelvis','l_pelvis','LHL2','LHM5','RHL2','RHM5',
               'LHand', 'RForearm','RHand', 'L_sh1_study', 'L_thigh1_study','r_sh1_study', 'r_thigh1_study']
#read mks data

for subject in SUBJECTS:

    no_trial = subject
    task = "static" #hitting sat probleme
    path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mocap/{task}/mocap_downsampled_to_40hz.csv"

    info_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/info.txt"
    subject_height,subject_mass, gender = read_subject_info(info_path)


    start_sample=0
    mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
                'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
                'BHD','RHD','LHD','FHD',
                'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
                'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
                'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
                'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
                'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
                'r_pelvis', 'l_pelvis']
    # df_raw = pd.read_data_to_dataframe(df_raw, mks_names) #marker data are string 
    # mks_data = udp_csv_to_dataframe(path_to_csv, mks_names) #float
    df_wide = pd.read_csv(path_to_csv)
    result_markers, start_sample_dict = read_mks_data(df_wide, start_sample=start_sample,converter = 1000.0) #check the function of read 

    #load urdf
    human = Robot('/root/workspace/ros_ws/src/rt-cosmik/urdf/human.urdf',rt_cosmik_path,isFext=True) 
    human_model = human.model
    human_data = human.data
    human_collision_model = human.collision_model
    human_visual_model = human.visual_model

    #scale the model to data
    human_model = scale_human_model(human_model, start_sample_dict,with_hand=True,gender=gender,subject_height=subject_height)
    human_model= mks_registration(human_model,start_sample_dict, with_hand=True)
    human_data = pin.Data(human_model)

    # # # # # Save the modified URDF using the function
    new_urdf_name = f"{subject_ids[no_trial]}_scaled.urdf"
    new_urdf_path = os.path.join("/root/workspace/ros_ws/src/rt-cosmik/urdf", new_urdf_name)
    save_scaled_urdf(new_urdf_name, new_urdf_path, human_model, human_visual_model , human_collision_model )

    #load new urdf
    new_human = Robot(new_urdf_path,rt_cosmik_path,isFext=True) 
    new_human_model = new_human.model
    new_human_data = new_human.data
    new_human_collision_model = new_human.collision_model
    new_human_visual_model = new_human.visual_model


    # viz = GepettoVisualizer(human_model,human_collision_model,human_visual_model)
    # try:
    #     viz.initViewer()
    # except ImportError as err:
    #     print("Install gepetto-viewer.")
    #     sys.exit(0)

    # try:
    #     viz.loadViewerModel("pinocchio")
    # except AttributeError as err:
    #     print("Start gepetto-viewer before running this script.")
    #     sys.exit(0)

    # viz2 = GepettoVisualizer(new_human_model,new_human_collision_model,new_human_visual_model)
    # viz2.initViewer()
    # viz2.loadViewerModel("model_rescaled")

    # q0 = pin.neutral(human_model)
    # q1 = pin.neutral(new_human_model)
    # q1[2]+=0.5

    # viz.display(q0)
    # viz2.display(q1)