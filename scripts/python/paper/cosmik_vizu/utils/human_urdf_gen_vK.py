import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import os

def get_segment_names() -> list[str]:
    """Returns list of segment names"""
    return [
        'middle_pelvis',
        'left_upperleg', 'left_lowerleg', 'left_foot',
        'middle_abdomen', 'middle_thorax', 'middle_head',
        'left_upperarm', 'left_lowerarm', 'left_hand',
        'right_upperarm', 'right_lowerarm', 'right_hand',
        'right_upperleg', 'right_lowerleg', 'right_foot'
    ]

def get_joint_names() -> list[str]:
    """Returns list of joint names"""
    return [
        'left_hip_Z', 'left_knee', 'left_ankle_Z',
        'middle_lumbar_Z', 'middle_thoracic_Z', 'middle_cervical_Z',
        'left_clavicle_joint_X', 'left_shoulder_Z', 'left_elbow_Z', 'left_wrist_Z',
        'right_clavicle_joint_X', 'right_shoulder_Z', 'right_elbow_Z', 'right_wrist_Z',
        'right_hip_Z', 'right_knee', 'right_ankle_Z'
    ]

def get_mass_ratios(gender: str = 'f') -> np.array:
    """
    Return anthropometric table mass ratio for male ('m') or female ('f') body
    """
    pelvis_r = 0.146 if gender == 'f' else 0.142
    abdomen_r = 0.042 if gender == 'f' else 0.027
    thorax_r = 0.263 if gender == 'f' else 0.304
    head_r = 0.067 if gender == 'f' else 0.067
    upperarm_r = 0.022 if gender == 'f' else 0.024
    lowerarm_r = 0.013 if gender == 'f' else 0.017
    hand_r = 0.005 if gender == 'f' else 0.006
    upperleg_r = 0.146 if gender == 'f' else 0.123
    lowerleg_r = 0.045 if gender == 'f' else 0.048
    foot_r = 0.01 if gender == 'f' else 0.012
    return np.array([
        pelvis_r,
        upperleg_r, lowerleg_r, foot_r,
        abdomen_r, thorax_r, head_r,
        upperarm_r, lowerarm_r, hand_r,
        upperarm_r, lowerarm_r, hand_r,
        upperleg_r, lowerleg_r, foot_r])


def get_length_ratios(gender: str = 'f') -> np.array:
    """
    Return anthropometric table segment length ratio for male ('m') or female ('f') body
    """
    upperarm_r = 0.1439 if gender == 'f' else 0.1455
    lowerarm_r = 0.1462 if gender == 'f' else 0.1519
    hand_r = 0.0989 if gender == 'f' else 0.1014
    upperleg_r = 0.2244 if gender == 'f' else 0.2319
    lowerleg_r = 0.2297 if gender == 'f' else 0.2324
    foot_r = 0.0977 if gender == 'f' else 0.0982
    pelvis_r = 0.0634 if gender == 'f' else 0.0505
    abdomen_r = 0.1270 if gender == 'f' else 0.0797
    thorax_r = 0.1270 if gender == 'f' else 0.1763
    head_r = 0.1308 if gender == 'f' else 0.1310
    h = upperleg_r + lowerleg_r + foot_r + pelvis_r + abdomen_r + thorax_r + head_r
    assert h == 1.
    return np.array([
        pelvis_r,
        upperleg_r, lowerleg_r, foot_r,
        abdomen_r, thorax_r, head_r,
        upperarm_r, lowerarm_r, hand_r,
        upperarm_r, lowerarm_r, hand_r,
        upperleg_r, lowerleg_r, foot_r])


def calculate_segments_masses(body_mass: float, gender: str) -> dict:
    """ Function returns scaled segment masses (we use anthropometric table) """
    seg_names = get_segment_names()
    mass_ratio = get_mass_ratios(gender)
    assert np.sum(mass_ratio) == 1
    return {
        k: np.round(body_mass * mass_ratio[i], 2) for i, k in enumerate(seg_names)
    }


def calculate_segments_lengths(body_length: float, gender: str) -> dict:
    """ Function returns scaled segment masses (we use anthropometric table) """
    seg_names = get_segment_names()
    length_ratio = get_length_ratios(gender)
    return {
        k: np.round(body_length * length_ratio[i], 3) for i, k in enumerate(seg_names)
    }


def calculate_joints_pos(segments_lengths: dict, body_size: float, gender: str
                         ) -> dict:
    """Returns joint positions for every joint"""
    joints_names = get_joint_names()
    joints_pos = {}

    SJ_thorax_X_r = 0.0043 if gender == 'f' else 0.0046
    SJ_thorax_Y_r = -0.0449 if gender == 'f' else -0.0416
    SJ_thorax_Z_r = 0.1108 if gender == 'f' else 0.1164
    HJ_pelvis_X_r = 0.0296 if gender == 'f' else 0.0305
    HJ_pelvis_Y_r = -0.0567 if gender == 'f' else -0.0428
    HJ_pelvis_Z_r = 0.0548 if gender == 'f' else 0.0457
    for j in joints_names:
        if j == 'left_hip_Z':
            j_pos = body_size * np.array([HJ_pelvis_X_r, HJ_pelvis_Y_r, -HJ_pelvis_Z_r])
        elif j == 'right_hip_Z':
            j_pos = body_size * np.array([HJ_pelvis_X_r, HJ_pelvis_Y_r, HJ_pelvis_Z_r])
        elif j in ['left_knee', 'right_knee']:
            j_pos = np.array([0, -segments_lengths['left_upperleg'], 0])
        elif j in ['left_ankle_Z', 'right_ankle_Z']:
            j_pos = np.array([0, -segments_lengths['left_lowerleg'], 0])
        elif j == 'middle_lumbar_Z':
            j_pos = np.array([0, 0, 0])
        elif j == 'middle_thoracic_Z':
            j_pos = np.array([0, segments_lengths['middle_abdomen'], 0])
        elif j in ['middle_cervical_Z', 'left_clavicle_joint_X',
                   'right_clavicle_joint_X']:
            j_pos = np.array([0, segments_lengths['middle_thorax'], 0])
        elif j == 'left_shoulder_Z':
            j_pos = body_size * np.array([SJ_thorax_X_r, SJ_thorax_Y_r, -SJ_thorax_Z_r])
        elif j == 'right_shoulder_Z':
            j_pos = body_size * np.array([SJ_thorax_X_r, SJ_thorax_Y_r, SJ_thorax_Z_r])
        elif j in ['left_elbow_Z', 'right_elbow_Z']:
            j_pos = np.array([0, -segments_lengths['left_upperarm'], 0])
        elif j in ['left_wrist_Z', 'right_wrist_Z']:
            j_pos = np.array([0, -segments_lengths['left_lowerarm'], 0])
        else:
            raise ValueError(f"Behavior for {j} joint is not defined")
        joints_pos[j] = np.round(j_pos, 3)

    return joints_pos


def calculate_inertial_parameters(segments_mass: dict,
                                  segments_length: dict,
                                  gender: str) -> dict:
    """
    Returns dictionary with keys as link names and values as another dictionary
    containing mass, Center Of Mass and inertia
    """
    segments_names = get_segment_names()
    inertial_segments_values = {}
    for seg in segments_names:
        if gender == 'f':
            if seg == 'middle_pelvis':
                com = segments_length[seg] * np.array([-0.009, -0.232, 0.002])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.91) ** 2,
                    -(segments_length[seg] * 0.34) ** 2,
                    -(segments_length[seg] * 0.01) ** 2,
                    (segments_length[seg] * 1) ** 2,
                    -(segments_length[seg] * 0.01) ** 2,
                    (segments_length[seg] * 0.79) ** 2
                ])
            elif seg in ['left_upperleg', 'right_upperleg']:
                com = segments_length[seg] * np.array([-0.077, -0.377, 0.009])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.31) ** 2,
                    (segments_length[seg] * 0.07) ** 2,
                    -(segments_length[seg] * 0.02) ** 2,
                    (segments_length[seg] * 0.19) ** 2,
                    -(segments_length[seg] * 0.07) ** 2,
                    (segments_length[seg] * 0.32) ** 2
                ])
            elif seg in ['left_lowerleg', 'right_lowerleg']:
                com = segments_length[seg] * np.array([-0.049, -0.404, 0.031])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.28) ** 2,
                    (segments_length[seg] * 0.02) ** 2,
                    (segments_length[seg] * 0.01) ** 2,
                    (segments_length[seg] * 0.1) ** 2,
                    (segments_length[seg] * 0.06) ** 2,
                    (segments_length[seg] * 0.28) ** 2])
            elif seg in ['left_foot', 'right_foot']:
                com = segments_length[seg] * np.array([0.27, -0.218, 0.039])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.17) ** 2,
                    -(segments_length[seg] * 0.10) ** 2,
                    (segments_length[seg] * 0.06) ** 2,
                    (segments_length[seg] * 0.36) ** 2,
                    -(segments_length[seg] * 0.04) ** 2,
                    (segments_length[seg] * 0.35) ** 2
                ])
            elif seg in ['middle_abdomen', 'middle_thorax']:
                com = segments_length[seg] * np.array([-0.016, 0.564, -0.006])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.29) ** 2,
                    (segments_length[seg] * 0.22) ** 2,
                    (segments_length[seg] * 0.05) ** 2,
                    (segments_length[seg] * 0.27) ** 2,
                    -(segments_length[seg] * 0.05) ** 2,
                    (segments_length[seg] * 0.29) ** 2
                ])
                # TODO: write comment why we do this
                inertia += np.array([
                    segments_mass[seg] * segments_length[seg] ** 2,
                    0, 0, 0, 0,
                    segments_mass[seg] * segments_length[seg] ** 2
                ])
            elif seg == 'middle_head':
                com = segments_length[seg] * np.array([-0.07, 0.597, 0])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.32) ** 2,
                    -(segments_length[seg] * 0.06) ** 2,
                    (segments_length[seg] * 0.01) ** 2,
                    (segments_length[seg] * 0.27) ** 2,
                    -(segments_length[seg] * 0.01) ** 2,
                    (segments_length[seg] * 0.34) ** 2
                ])
            elif seg in ['left_upperarm', 'right_upperarm']:
                com = segments_length[seg] * np.array([-0.073, -0.454, -0.028])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.33) ** 2,
                    (segments_length[seg] * 0.03) ** 2,
                    -(segments_length[seg] * 0.05) ** 2,
                    (segments_length[seg] * 0.17) ** 2,
                    (segments_length[seg] * 0.14) ** 2,
                    (segments_length[seg] * 0.33) ** 2
                ])
            elif seg in ['left_lowerarm', 'right_lowerarm']:
                com = segments_length[seg] * np.array([0.021, -0.411, 0.019])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.26) ** 2,
                    (segments_length[seg] * 0.1) ** 2,
                    (segments_length[seg] * 0.04) ** 2,
                    (segments_length[seg] * 0.14) ** 2,
                    -(segments_length[seg] * 0.13) ** 2,
                    (segments_length[seg] * 0.25) ** 2
                ])
            elif seg in ['left_hand', 'right_hand']:
                com = segments_length[seg] * np.array([0.077, -0.768, 0.048])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.63) ** 2,
                    (segments_length[seg] * 0.29) ** 2,
                    (segments_length[seg] * 0.23) ** 2,
                    (segments_length[seg] * 0.43) ** 2,
                    -(segments_length[seg] * 0.28) ** 2,
                    (segments_length[seg] * 0.58) ** 2
                ])
        elif gender == 'm':
            if seg == 'middle_pelvis':
                com = segments_length[seg] * np.array([0.028, -0.28, -0.006])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 1.01) ** 2,
                    -(segments_length[seg] * 0.25) ** 2,
                    -(segments_length[seg] * 0.12) ** 2,
                    (segments_length[seg] * 1.06) ** 2,
                    -(segments_length[seg] * 0.08) ** 2,
                    (segments_length[seg] * 0.95) ** 2])
            elif seg in ['left_upperleg', 'right_upperleg']:
                com = segments_length[seg] * np.array([-0.041, -0.429, 0.033])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.29) ** 2,
                    (segments_length[seg] * 0.07) ** 2,
                    -(segments_length[seg] * 0.02) ** 2,
                    (segments_length[seg] * 0.15) ** 2,
                    -(segments_length[seg] * 0.07) ** 2,
                    (segments_length[seg] * 0.3) ** 2
                ])
            elif seg in ['left_lowerleg', 'right_lowerleg']:
                com = segments_length[seg] * np.array([-0.048, -0.41, 0.007])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.28) ** 2,
                    -(segments_length[seg] * 0.04) ** 2,
                    -(segments_length[seg] * 0.02) ** 2,
                    (segments_length[seg] * 0.1) ** 2,
                    (segments_length[seg] * 0.05) ** 2,
                    (segments_length[seg] * 0.28) ** 2
                ])
            elif seg in ['left_foot', 'right_foot']:
                com = segments_length[seg] * np.array([0.382, -0.151, 0.026])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.17) ** 2,
                    (segments_length[seg] * 0.13) ** 2,
                    -(segments_length[seg] * 0.08) ** 2,
                    (segments_length[seg] * 0.37) ** 2,
                    (segments_length[seg] * 0) ** 2,
                    (segments_length[seg] * 0.36) ** 2
                ])
            elif seg in ['middle_abdomen', 'middle_thorax']:
                com = segments_length[seg] * np.array([-0.036, 0.58, -0.002])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.27) ** 2,
                    (segments_length[seg] * 0.18) ** 2,
                    (segments_length[seg] * 0.02) ** 2,
                    (segments_length[seg] * 0.25) ** 2,
                    -(segments_length[seg] * 0.04) ** 2,
                    (segments_length[seg] * 0.28) ** 2
                ])
                inertia += np.array([
                    segments_mass[seg] * segments_length[seg] ** 2,
                    0, 0, 0, 0,
                    segments_mass[seg] * segments_length[seg] ** 2
                ])
            elif seg == 'middle_head':
                com = segments_length[seg] * np.array([-0.062, 0.555, 0.001])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.31) ** 2,
                    -(segments_length[seg] * 0.09) ** 2,
                    -(segments_length[seg] * 0.02) ** 2,
                    (segments_length[seg] * 0.25) ** 2,
                    (segments_length[seg] * 0.03) ** 2,
                    (segments_length[seg] * 0.33) ** 2
                ])
            elif seg in ['left_upperarm', 'right_upperarm']:
                com = segments_length[seg] * np.array([0.017, -0.452, -0.026])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.31) ** 2,
                    (segments_length[seg] * 0.06) ** 2,
                    (segments_length[seg] * 0.05) ** 2,
                    (segments_length[seg] * 0.14) ** 2,
                    (segments_length[seg] * 0.02) ** 2,
                    (segments_length[seg] * 0.32) ** 2
                ])
            elif seg in ['left_lowerarm', 'right_lowerarm']:
                com = segments_length[seg] * np.array([0.01, -0.417, 0.014])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.28) ** 2,
                    (segments_length[seg] * 0.03) ** 2,
                    (segments_length[seg] * 0.02) ** 2,
                    (segments_length[seg] * 0.11) ** 2,
                    -(segments_length[seg] * 0.08) ** 2,
                    (segments_length[seg] * 0.27) ** 2
                ])
            elif seg in ['left_hand', 'right_hand']:
                com = segments_length[seg] * np.array([0.082, -0.839, 0.074])
                inertia = segments_mass[seg] * np.array([
                    (segments_length[seg] * 0.61) ** 2,
                    (segments_length[seg] * 0.22) ** 2,
                    (segments_length[seg] * 0.15) ** 2,
                    (segments_length[seg] * 0.38) ** 2,
                    -(segments_length[seg] * 0.2) ** 2,
                    (segments_length[seg] * 0.56) ** 2
                ])
            else:
                raise ValueError(f"Behavior of inertia parameters for {seg} segment is "
                                 f"not defined")
        inertial_segments_values[seg] = {
            'mass': segments_mass[seg],
            'com': np.round(com, 3),
            'inertias': np.round(inertia, 6)}
    return inertial_segments_values


def generate_custom_urdf(human_blank_path: str,
                         save_path: str,
                         joints_pos: dict,
                         inertia_p: dict):
    """Generates new urdf with the scaled parameters"""
    tree = ET.parse(human_blank_path)

    inertia_order = ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']
    for link in tree.findall('link'):
        inertia = link.find('inertial')
        if inertia is not None:
            l_name = link.get('name')
            if l_name in inertia_p.keys():
                origin = ' '.join([str(c) for c in inertia_p[l_name]['com']])
                inertia.find('mass').set('value', str(inertia_p[l_name]['mass']))
                inertia.find('origin').set('xyz', origin)
                inertia.find('origin').set('rpy', '0 0 0')
                for name, value in zip(inertia_order, inertia_p[l_name]['inertias']):
                    inertia.find('inertia').set(name, str(value))
            else:
                print(l_name, ' NOT set inertia')

    for joint in tree.findall('joint'):
        if joint.get('type') == 'revolute':
            j_name = joint.get('name')
            if j_name in joints_pos.keys():
                j_origin = joint.find('origin')
                j_origin.set('xyz', ' '.join([str(c) for c in joints_pos[j_name]]))
                j_origin.set('rpy', '0 0 0')

    tree.write(save_path)


if __name__ == "__main__":

    gender = "f"
    body_mass = 59.25
    body_size = 1.77
    save_filename = "human_generated.urdf"#f"generated_{gender}_mass{body_mass}_size{body_size}.urdf"

    urdf_name = "human_blank.urdf"
    script_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(script_directory)

    
   
    human_blank_path = os.path.join(parent_directory,"model/human_urdf/urdf/",urdf_name)#Path(__file__).resolve().joinpath('human_blank.urdf')
    save_path = os.path.join(parent_directory,"model/human_urdf/urdf/",save_filename)#Path(__file__).resolve().parent.joinpath(save_filename)
   
    
    assert gender in ['f', 'm']
    segments_masses = calculate_segments_masses(body_mass, gender)
    segments_lengths = calculate_segments_lengths(body_size, gender)
    joints_pos = calculate_joints_pos(segments_lengths, body_size, gender)
    inertial_param = calculate_inertial_parameters(segments_masses,
                                                   segments_lengths,
                                                   gender)
    # print(segments_masses)
    # print(segments_lengths)
    # print(joints_pos)
    # print(inertial_param)

    generate_custom_urdf(human_blank_path, save_path, joints_pos, inertial_param)
