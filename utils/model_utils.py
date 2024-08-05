import hppfcl as fcl
import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from typing import List, Tuple, Dict
from utils.linear_algebra_utils import col_vector_3D
from scipy.spatial.transform import Rotation as R
import pandas as pd

#Build inertia matrix from 6 inertia components
def make_inertia_matrix(ixx:float, ixy:float, ixz:float, iyy:float, iyz:float, izz:float)->np.ndarray:
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

#Function that takes as input a matrix and orthogonalizes it
#Its mainly used to orthogonalize rotation matrices constructed by hand
def orthogonalize_matrix(matrix:np.ndarray)->np.ndarray:
    print(matrix)
    print(np.linalg.norm(matrix[:3,0]))
    print(np.linalg.norm(matrix[:3,1]))
    print(np.linalg.norm(matrix[:3,2]))
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

#construct upperarm frame and get its pose
def get_upperarmR_pose(mocap_mks_positions):
    pose = np.eye(4,4)
    X, Y, Z, shoulder_center = [], [], [], []
    if 'RShoulder' in mocap_mks_positions:
    
        # Calcule du centre de l'épaule comme la position du marqueur 'RShoulder'
        shoulder_center = mocap_mks_positions['RShoulder'].reshape(3,1)
        # Calcule le centre du coude comme la moyenne des positions des marqueurs 'r_melbow_study' et 'r_lelbow_study'
        elbow_center = (mocap_mks_positions['r_melbow_study'] + mocap_mks_positions['r_lelbow_study']).reshape(3,1)/2.0
        
        # Vecteur Y : direction du bras du centre de l'épaule au centre du coude, normalisé
        Y = shoulder_center - elbow_center
        Y = Y/np.linalg.norm(Y)
        # Vecteur Z : direction de l'avant-bras, normalisé
        Z = (mocap_mks_positions['r_lelbow_study'] - mocap_mks_positions['r_melbow_study']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        # Vecteur X : produit vectoriel de Y et Z, normalisé
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
    else:
        # Si 'RShoulder' n'est pas présent, calcule le centre du coude autrement
        elbow_center = (mocap_mks_positions['RHLE'] + mocap_mks_positions['RHME']).reshape(3,1)/2.0

        # Calcule la pose du torse et la distance bi-acromiale
        torso_pose = get_torso_pose(mocap_mks_positions)
        bi_acromial_dist = np.linalg.norm(mocap_mks_positions['RSAT'] - mocap_mks_positions['LSAT'])

        # Calcule le centre de l'épaule comme une position ajustée
        shoulder_center = mocap_mks_positions['RSAT'] + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0)

        # Vecteur Y : direction du bras du centre de l'épaule au centre du coude, normalisé
        Y = shoulder_center - elbow_center
        Y = Y/np.linalg.norm(Y)

        # Vecteur Z : direction de l'avant-bras, normalisé
        Z = (mocap_mks_positions['RHLE'] - mocap_mks_positions['RHME']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)

        # Vecteur X : produit vectoriel de Y et Z, normalisé
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
        
    # Remplissage des colonnes de la matrice pose avec les vecteurs X, Y, Z et la position de l'épaule
    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = shoulder_center.reshape(3,)

    # Orthogonalisation de la partie rotationnelle de la matrice pose
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose # Retourne la matrice de transformation représentant la pose du bras supérieur droit




def get_upperarmL_pose(mocap_mks_positions):
    pose = np.eye(4,4)
    X, Y, Z, shoulder_center = [], [], [], []
    if 'LShoulder' in mocap_mks_positions:
    
        # Calcule du centre de l'épaule comme la position du marqueur 'RShoulder'
        shoulder_center = mocap_mks_positions['LShoulder'].reshape(3,1)
        # Calcule le centre du coude comme la moyenne des positions des marqueurs 'r_melbow_study' et 'r_lelbow_study'
        elbow_center = (mocap_mks_positions['L_melbow_study'] + mocap_mks_positions['L_lelbow_study']).reshape(3,1)/2.0
        
        # Vecteur Y : direction du bras du centre de l'épaule au centre du coude, normalisé
        Y = shoulder_center - elbow_center
        Y = Y/np.linalg.norm(Y)
        # Vecteur Z : direction de l'avant-bras, normalisé
        Z = (mocap_mks_positions['L_melbow_study'] - mocap_mks_positions['L_lelbow_study']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        # Vecteur X : produit vectoriel de Y et Z, normalisé
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
    else:
        elbow_center = (mocap_mks_positions['LHLE'] + mocap_mks_positions['LHME']).reshape(3,1)/2.0
        torso_pose = get_torso_pose(mocap_mks_positions)
        bi_acromial_dist = np.linalg.norm(mocap_mks_positions['LSAT'] - mocap_mks_positions['RSAT'])
        shoulder_center = mocap_mks_positions['LSAT'] + torso_pose[:3, :3] @ col_vector_3D(0., -0.17*bi_acromial_dist, 0)
        Y = shoulder_center - elbow_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['LHME'] - mocap_mks_positions['LHLE']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = shoulder_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose


#construct lowerarm frame and get its pose
def get_lowerarmR_pose(mocap_mks_positions):
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
    pose = np.eye(4,4)
    X, Y, Z, elbow_center = [], [], [], []
    if 'L_melbow_study' in mocap_mks_positions:
        elbow_center = (mocap_mks_positions['L_melbow_study'] + mocap_mks_positions['L_lelbow_study']).reshape(3,1)/2.0
        wrist_center = (mocap_mks_positions['L_mwrist_study'] + mocap_mks_positions['L_lwrist_study']).reshape(3,1)/2.0
        
        Y = elbow_center - wrist_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['L_mwrist_study'] - mocap_mks_positions['L_lwrist_study']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
    else:
        elbow_center = (mocap_mks_positions['LHLE'] + mocap_mks_positions['LHME']).reshape(3,1)/2.0
        wrist_center = (mocap_mks_positions['LRSP'] + mocap_mks_positions['LUSP']).reshape(3,1)/2.0
        
        Y = elbow_center - wrist_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['LUSP'] - mocap_mks_positions['LRSP']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = elbow_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose



#construct pelvis frame and get its pose
def get_pelvis_pose(mocap_mks_positions):
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

#construct thigh frame and get its pose
def get_thighR_pose(mocap_mks_positions):
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
        # print(mocap_mks_positions["RIAS"])
        # print(mocap_mks_positions["LIAS"])
        # print(mocap_mks_positions["RFLE"])
        # print(mocap_mks_positions["RFME"])
        dist_rPL_lPL = np.linalg.norm(mocap_mks_positions["RIAS"]-mocap_mks_positions["LIAS"])
        # print(dist_rPL_lPL)
        pelvis_pose = get_pelvis_pose(mocap_mks_positions)
        # print(pelvis_pose)
        hip_center = pelvis_pose[:3, 3].reshape(3,1)
        hip_center = hip_center + pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(-0.14*dist_rPL_lPL, 0.0, 0.0)
        hip_center = hip_center + pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, -0.3*dist_rPL_lPL, 0.0)
        hip_center = hip_center + pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, 0.0, 0.22*dist_rPL_lPL)
        knee_center = (mocap_mks_positions['RFLE'] + mocap_mks_positions['RFME']).reshape(3,1)/2.0
        # print(hip_center)
        # print(knee_center)
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
        hip_center = hip_center + pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, 0.0, -0.22*dist_rPL_lPL)
        knee_center = (mocap_mks_positions['LFLE'] + mocap_mks_positions['LFME']).reshape(3,1)/2.0
        Y = hip_center - knee_center
        Y = Y/np.linalg.norm(Y)
        Z = (mocap_mks_positions['LFME'] - mocap_mks_positions['LFLE']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = hip_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose



#construct shank frame and get its pose
def get_shankR_pose(mocap_mks_positions):
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
        Z = (mocap_mks_positions['LFME'] - mocap_mks_positions['LFLE']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = knee_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose




#construct foot frame and get its pose
def get_footR_pose(mocap_mks_positions):
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
        X = (toe_pos- mocap_mks_positions['LFCC']).reshape(3,1)
        X = X/np.linalg.norm(X)
        Z = (mocap_mks_positions['LTAM'] - mocap_mks_positions['LFAL']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        Y = np.cross(Z, X, axis=0)
        Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ankle_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose


#construct foot frame and get its pose
def get_insoleR_pose(mocap_mks_positions):
    pose = np.eye(4,4)
    X, Y, Z, ankle_center = [], [], [], []

    ankle_center = (mocap_mks_positions['RTAM'] + mocap_mks_positions['RFAL'] + mocap_mks_positions['RFM5']+mocap_mks_positions['RFM1']).reshape(3,1)/4.0
    toe_pos = (mocap_mks_positions['RFM5']+mocap_mks_positions['RFM1'])/2.0
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



#Construct challenge segments frames from mocap mks
# - mocap_mks_positions is a dictionnary of mocap mks names and 3x1 global positions
# - returns sgts_poses which correspond to a dictionnary to segments poses and names, constructed from mks global positions
def construct_segments_frames_challenge(mocap_mks_positions):    
    # torso_pose = get_torso_pose(mocap_mks_positions)
    # upperarmR_pose = get_upperarmR_pose(mocap_mks_positions)
    # lowerarmR_pose = get_lowerarmR_pose(mocap_mks_positions)
    # upperarmL_pose = get_upperarmL_pose(mocap_mks_positions)
    # lowerarmL_pose = get_lowerarmL_pose(mocap_mks_positions)
    pelvis_pose = get_pelvis_pose(mocap_mks_positions)
    thighR_pose = get_thighR_pose(mocap_mks_positions)
    shankR_pose = get_shankR_pose(mocap_mks_positions)
    footR_pose = get_footR_pose(mocap_mks_positions)
    thighL_pose = get_thighL_pose(mocap_mks_positions)
    shankL_pose = get_shankL_pose(mocap_mks_positions)
    footL_pose = get_footL_pose(mocap_mks_positions)
    
    
    # Constructing the dictionary to store segment poses
    sgts_poses = {
        # "torso": torso_pose,
        # "upperarmR": upperarmR_pose,
        # "lowerarmR": lowerarmR_pose,
        # "upperarmL": upperarmL_pose,
        # "lowerarmL": lowerarmL_pose,
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

def compare_offsets(mocap_mks_positions, lstm_mks_positions): #utiliser dans check_mocap_data
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

def get_segments_mocap_mks()->Dict:
    #This fuction returns a dictionnary containing the segments names, and the corresponding list of mocap
    #mks names attached to the segment
    # Constructing the dictionary to store segment poses
    sgts_mks_dict = {
        # "torso": ['RSAT', 'LSAT', 'SJN', 'CV7', 'HeadR', 'HeadL'],
        # "upperarmR": ['RHLE', 'RHME'],
        # "lowerarmR": ['RRSP', 'RUSP'],
        # "upperarmL" : ['LHLE', 'LHME'],
        # "lowerarmL": ['LRSP', 'LUSP'],
        "pelvis": ['RIAS', 'LIAS', 'LIPS', 'RIPS'],
        "thighR": ['RFLE', 'RFME'],
        "thighL": ['LFLE', 'LFME'],
        "shankR": ['RTAM', 'RFAL'],
        "shankL": ['LTAM', 'LFAL'],
        "footR": ['RFCC', 'RFM5', 'RFM1'],
        "footL": ['LFCC', 'LFM5', 'LFM1']
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

def get_local_mks_positions(sgts_poses: Dict, lstm_mks_positions: Dict, sgts_mks_dict: Dict)-> Dict:
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

    # Define a helper function to get the translation part from a 4x4 transformation matrix
    def get_translation(matrix):
        return matrix[:3, 3]

    # Compute local positions for each segment
    # Torso with respect to pelvis
    if "torso" in sgts_poses:
        torso_global = sgts_poses["torso"]
        local_positions["torso"] = (np.linalg.inv(pelvis_pose) @ torso_global @ np.array([0, 0, 0, 1]))[:3] #inv pour avoir globale dans local
        
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

def get_segment_length(mocap_mks_positions: Dict, path_for_segment_length: str):
    sgts_poses = construct_segments_frames_challenge(mocap_mks_positions)
    local_segments_positions = get_local_segments_positions(sgts_poses)
    print('local_segments_positions', local_segments_positions)
    # Calculate norms to get length of segments
    norms = {}
    norms['upperlegR'] = np.linalg.norm(local_segments_positions['shankR'])
    norms['lowerlegR'] = np.linalg.norm(local_segments_positions['footR'])
    norms['upperlegL'] = np.linalg.norm(local_segments_positions['shankL'])
    norms['lowerlegL'] = np.linalg.norm(local_segments_positions['footL'])

    # Save to CSV
    df = pd.DataFrame(list(norms.items()), columns=['Segment_Pair', 'Norm']).T
    df.to_csv(path_for_segment_length, index=False, header=False)



def build_model_challenge(mocap_mks_positions: Dict, lstm_mks_positions: Dict, meshes_folder_path: str)->Tuple[pin.Model,pin.Model, Dict]:
    """_Build the biomechanical model associated to one exercise for one subject_

    Args:
        mocap_mks_positions (Dict): _mocap_mks_positions is a dictionnary of mocap mks names and 3x1 global positions_
        lstm_mks_positions (Dict): _lstm_mks_positions is a dictionnary of lstm mks names and 3x1 global positions_
        meshes_folder_path (str): _meshes_folder_path is the path to the folder containing the meshes_

    Returns:
        Tuple[pin.Model,pin.GeomModel, Dict]: _returns the pinocchio model, geometry model, and a dictionnary with visuals._
    """
    sgts_poses = construct_segments_frames_challenge(mocap_mks_positions)
    sgts_mks_dict =  get_segments_mocap_mks()
    lstm_mks_local_positions = get_local_mks_positions(sgts_poses, lstm_mks_positions, sgts_mks_dict)
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
    model= pin.Model()
    geom_model = pin.GeometryModel()

    # pelvis with Freeflyer
    IDX_PELV_JF = model.addJoint(0,pin.JointModelFreeFlyer(),pin.SE3(np.array([[1,0,0],[0,0,-1],[0,1,0]]), np.matrix([0,0,0]).T),'pelvis_freeflyer')
    pelvis = pin.Frame('pelvis',IDX_PELV_JF,0,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_PELV_SF = model.addFrame(pelvis,False)
    # Add markers data
    idx_frame = IDX_PELV_SF
    for i in sgts_mks_dict["pelvis"]:
        frame = pin.Frame(i,IDX_PELV_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    pelvis_visual = pin.GeometryObject('pelvis', IDX_PELV_SF, IDX_PELV_JF, mesh_loader.load(meshes_folder_path+'/pelvis_mesh.STL'), pin.SE3(rtorso.as_matrix(), np.matrix([-0.15, -0.17, 0.13]).T), meshes_folder_path+'/pelvis_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), False, np.array([0, 1, 1, 1]))
    geom_model.addGeometryObject(pelvis_visual)
    visuals_dict["pelvis"] = pelvis_visual


    # Right Hip ZXY
    IDX_HIP_Z_JF = model.addJoint(IDX_PELV_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['thighR']).T),'Hip_Z_R') 
    thighR = pin.Frame('thigh_z',IDX_HIP_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_THIGH_SF = model.addFrame(thighR,False)
    idx_frame = IDX_THIGH_SF

    IDX_HIP_X_JF = model.addJoint(IDX_HIP_Z_JF,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'Hip_X_R') 
    thighR = pin.Frame('thigh_x',IDX_HIP_X_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_THIGH_SF = model.addFrame(thighR,False)
    idx_frame = IDX_THIGH_SF

    IDX_HIP_Y_JF = model.addJoint(IDX_HIP_X_JF,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'Hip_Y_R') 
    thighR = pin.Frame('thighR',IDX_HIP_Y_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_THIGH_SF = model.addFrame(thighR,False)
    idx_frame = IDX_THIGH_SF

    for i in sgts_mks_dict["thighR"]:
        frame = pin.Frame(i,IDX_HIP_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    upperleg_visual_R = pin.GeometryObject('upperleg_R',IDX_THIGH_SF, IDX_HIP_Y_JF, mesh_loader.load(meshes_folder_path+'/upperleg_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.05, -0.37, 0.1]).T), meshes_folder_path+'/upperleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(upperleg_visual_R)
    visuals_dict["upperleg_R"] = upperleg_visual_R


    # Right Knee Z
    IDX_KNEE_Z_JF = model.addJoint(IDX_HIP_Y_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['shankR']).T),'Knee_Z_R') 
    shankR = pin.Frame('shankR',IDX_KNEE_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SHANK_SF = model.addFrame(shankR,False)
    idx_frame = IDX_SHANK_SF

    for i in sgts_mks_dict["shankR"]:
        frame = pin.Frame(i,IDX_KNEE_Z_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    knee_visual = pin.GeometryObject('knee_R',IDX_SHANK_SF, IDX_KNEE_Z_JF, mesh_loader.load(meshes_folder_path+'/knee_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.05, 0, -0.015]).T), meshes_folder_path+'/knee_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(knee_visual)
    visuals_dict["knee_R"] = knee_visual

    lowerleg_visual_R = pin.GeometryObject('lowerleg_R',IDX_SHANK_SF, IDX_KNEE_Z_JF, mesh_loader.load(meshes_folder_path+'/lowerleg_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.03, -0.40, 0.1]).T), meshes_folder_path+'/lowerleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(lowerleg_visual_R)
    visuals_dict["lowerleg_R"] = lowerleg_visual_R


    # Right Ankle Z
    IDX_ANKLE_Z_JF_R = model.addJoint(IDX_KNEE_Z_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['footR']).T),'Ankle_Z_R') 
    foot_Z_R = pin.Frame('foot_Z_R',IDX_ANKLE_Z_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_foot_Z_R= model.addFrame(foot_Z_R,False)
    idx_frame = IDX_foot_Z_R

    IDX_ANKLE_X_JF_R = model.addJoint(IDX_ANKLE_Z_JF_R,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'Ankle_X_R') 
    foot_X_R = pin.Frame('footR',IDX_ANKLE_X_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_foot_X_R= model.addFrame(foot_X_R,False)
    idx_frame = IDX_foot_X_R

    for i in sgts_mks_dict["footR"]:
        frame = pin.Frame(i,IDX_ANKLE_X_JF_R,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    foot_visual_R = pin.GeometryObject('foot_R',IDX_foot_X_R, IDX_ANKLE_X_JF_R, mesh_loader.load(meshes_folder_path+'/foot_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.03, -0.07, 0.09]).T), meshes_folder_path+'/foot_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(foot_visual_R)
    visuals_dict["foot_R"] = foot_visual_R




    # Left Hip ZXY
    IDX_HIP_Z_JF_L = model.addJoint(IDX_PELV_JF, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix(local_segments_positions['thighL']).T), 'Hip_Z_L') 
    thighL = pin.Frame('thigh_z_L', IDX_HIP_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_TGH_SF_L = model.addFrame(thighL, False)
    idx_frame = IDX_TGH_SF_L

    IDX_HIP_X_JF_L = model.addJoint(IDX_HIP_Z_JF_L,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'Hip_X_L') 
    thighL = pin.Frame('thigh_x_L',IDX_HIP_X_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_THIGH_SF = model.addFrame(thighL,False)
    idx_frame = IDX_TGH_SF_L

    IDX_HIP_Y_JF_L = model.addJoint(IDX_HIP_X_JF_L, pin.JointModelRY(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'Hip_Y_L') 
    thighL = pin.Frame('thighL', IDX_HIP_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    IDX_TGH_SF_L = model.addFrame(thighL, False)
    idx_frame = IDX_TGH_SF_L

    for i in sgts_mks_dict["thighL"]:
        frame = pin.Frame(i, IDX_HIP_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix(lstm_mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
        idx_frame = model.addFrame(frame, False)

    thigh_visual_L = pin.GeometryObject('upperleg_L', IDX_TGH_SF_L, IDX_HIP_Y_JF_L, mesh_loader.load(meshes_folder_path+'/upperleg_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.05, -0.37, -0.105]).T), meshes_folder_path+'/upperleg_mesh.STL', np.array([0.0060, 0.0060, 0.0060]), False, np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(thigh_visual_L)
    visuals_dict["upperleg_L"] = thigh_visual_L



    # Left knee Z

    IDX_KNEE_Z_JF_L = model.addJoint(IDX_HIP_Y_JF_L,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['shankL']).T),'Knee_Z_L') 
    shankR = pin.Frame('shankL',IDX_KNEE_Z_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SHANK_SF_L = model.addFrame(shankR,False)
    idx_frame = IDX_SHANK_SF_L

    for i in sgts_mks_dict["shankL"]:
        frame = pin.Frame(i,IDX_KNEE_Z_JF_L,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    knee_visual = pin.GeometryObject('knee_L',IDX_SHANK_SF_L, IDX_KNEE_Z_JF_L, mesh_loader.load(meshes_folder_path+'/knee_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.05, 0, -0.07]).T), meshes_folder_path+'/knee_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(knee_visual)
    visuals_dict["knee_L"] = knee_visual

    lowerleg_visual_L = pin.GeometryObject('lowerleg_L',IDX_SHANK_SF_L, IDX_KNEE_Z_JF_L, mesh_loader.load(meshes_folder_path+'/lowerleg_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.03, -0.40, 0.05]).T), meshes_folder_path+'/lowerleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(lowerleg_visual_L)
    visuals_dict["lowerleg_L"] = lowerleg_visual_L


    # Left Ankle Z

    IDX_ANKLE_Z_JF_L = model.addJoint(IDX_KNEE_Z_JF_L,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['footL']).T),'Ankle_Z_L') 
    foot_Z_L = pin.Frame('foot_Z_L',IDX_ANKLE_Z_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_foot_Z_L= model.addFrame(foot_Z_L,False)
    idx_frame = IDX_foot_Z_L

    IDX_ANKLE_X_JF_L = model.addJoint(IDX_ANKLE_Z_JF_L,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'Ankle_X_L') 
    foot_X_L = pin.Frame('footL',IDX_ANKLE_X_JF_L,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_foot_X_L= model.addFrame(foot_X_L,False)
    idx_frame = IDX_foot_X_L

    for i in sgts_mks_dict["footL"]:
        frame = pin.Frame(i,IDX_ANKLE_X_JF_L,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    foot_visual_L = pin.GeometryObject('foot_L',IDX_ANKLE_X_JF_L, IDX_ANKLE_X_JF_L, mesh_loader.load(meshes_folder_path+'/foot_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.03, -0.07, 0.04]).T), meshes_folder_path+'/foot_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(foot_visual_L)
    visuals_dict["foot_L"] = foot_visual_L


    # # Lombaire L5-S1 flexion/extension
    # IDX_L5S1_JF = model.addJoint(IDX_PELV_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T),'L5S1_FE') 
    # torso = pin.Frame('torso_z',IDX_L5S1_JF,idx_frame,pin.SE3(np.eye(3),np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    # IDX_TORSO_SF = model.addFrame(torso,False)
    # idx_frame = IDX_TORSO_SF

    # IDX_L5S1_R_EXT_INT_JF = model.addJoint(IDX_L5S1_JF,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T),'L5S1_R_EXT_INT') 
    # torso = pin.Frame('torso',IDX_L5S1_R_EXT_INT_JF,idx_frame,pin.SE3(np.eye(3), np.matrix(local_segments_positions['torso']).T),pin.FrameType.OP_FRAME, inertia)
    # IDX_TORSO_SF = model.addFrame(torso,False)
    # idx_frame = IDX_TORSO_SF

    # for i in sgts_mks_dict["torso"]:
    #     frame = pin.Frame(i,IDX_L5S1_R_EXT_INT_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]+ local_segments_positions['torso']).T),pin.FrameType.OP_FRAME, inertia) 
    #     idx_frame = model.addFrame(frame,False)

    # torso_visual = pin.GeometryObject('torso', IDX_TORSO_SF, IDX_L5S1_JF, mesh_loader.load(meshes_folder_path+'/torso_mesh.STL'), pin.SE3(rtorso.as_matrix(), np.matrix([-0.15, 0.17, 0.13]).T), meshes_folder_path+'/torso_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), False, np.array([0, 1, 1, 1]))
    # geom_model.addGeometryObject(torso_visual)
    # visuals_dict["torso"] = torso_visual

    # abdomen_visual = pin.GeometryObject('abdomen', IDX_TORSO_SF, IDX_L5S1_JF, mesh_loader.load(meshes_folder_path+'/abdomen_mesh.STL'), pin.SE3(rtorso.as_matrix(), np.matrix([-0.12, 0.05, 0.09]).T), meshes_folder_path+'/abdomen_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), False, np.array([0, 1, 1, 1]))
    # geom_model.addGeometryObject(abdomen_visual)
    # visuals_dict["abdomen"] = abdomen_visual

    # # Right Shoulder ZXY
    # IDX_SH_Z_JF_R = model.addJoint(IDX_L5S1_R_EXT_INT_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['upperarmR'] + local_segments_positions['torso']).T),'Shoulder_Z_R') 
    # upperarmR = pin.Frame('upperarm_z_R',IDX_SH_Z_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    # IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    # idx_frame = IDX_UPA_SF_R

    # shoulder_visual_R = pin.GeometryObject('shoulder_R', IDX_UPA_SF_R, IDX_SH_Z_JF_R, mesh_loader.load(meshes_folder_path+'/shoulder_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.16, -0.045, -0.045]).T), meshes_folder_path+'/shoulder_mesh.STL',np.array([0.0055, 0.0055, 0.0055]), False , np.array([0,1,1,0.5]))
    # geom_model.addGeometryObject(shoulder_visual_R)
    # visuals_dict["shoulder_R"] = shoulder_visual_R

    # IDX_SH_X_JF_R = model.addJoint(IDX_SH_Z_JF_R,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'Shoulder_X_R') 
    # upperarmR = pin.Frame('upperarm_x_R',IDX_SH_X_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    # IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    # idx_frame = IDX_UPA_SF_R

    # IDX_SH_Y_JF_R = model.addJoint(IDX_SH_X_JF_R,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'Shoulder_Y_R') 
    # upperarmR = pin.Frame('upperarmR',IDX_SH_Y_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    # IDX_UPA_SF_R = model.addFrame(upperarmR,False)
    # idx_frame = IDX_UPA_SF_R
    # for i in sgts_mks_dict["upperarmR"]:
    #     frame = pin.Frame(i,IDX_SH_Y_JF_R,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
    #     idx_frame = model.addFrame(frame,False)

    # upperarm_visual_R = pin.GeometryObject('upperarm_R', IDX_UPA_SF_R, IDX_SH_Y_JF_R, mesh_loader.load(meshes_folder_path+'/upperarm_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.07, -0.29, 0.18]).T), meshes_folder_path+'/upperarm_mesh.STL',np.array([0.0063, 0.0060, 0.007]), False , np.array([0,1,1,0.5]))
    # geom_model.addGeometryObject(upperarm_visual_R)
    # visuals_dict["upperarm_R"] = upperarm_visual_R



    # # Right Elbow ZY 
    # IDX_EL_Z_JF_R = model.addJoint(IDX_SH_Y_JF_R,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['lowerarmR']).T),'Elbow_Z_R') 
    # lowerarmR = pin.Frame('lowerarm_z',IDX_EL_Z_JF_R,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    # IDX_LOA_SF = model.addFrame(lowerarmR,False)
    # idx_frame = IDX_LOA_SF

    # elbow_visual = pin.GeometryObject('elbow', IDX_LOA_SF, IDX_EL_Z_JF_R, mesh_loader.load(meshes_folder_path+'/elbow_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.15, -0.02, -0.035]).T), meshes_folder_path+'/elbow_mesh.STL',np.array([0.0055, 0.0055, 0.0055]), False , np.array([0,1,1,0.5]))
    # geom_model.addGeometryObject(elbow_visual)
    # visuals_dict["elbow"] = elbow_visual

    # IDX_EL_Y_JF = model.addJoint(IDX_EL_Z_JF_R,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'Elbow_Y_R') 
    # lowerarmR = pin.Frame('lowerarmR',IDX_EL_Y_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    # IDX_LOA_SF = model.addFrame(lowerarmR,False)
    # idx_frame = IDX_LOA_SF

    # for i in sgts_mks_dict["lowerarmR"]:
    #     frame = pin.Frame(i,IDX_EL_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
    #     idx_frame = model.addFrame(frame,False)

    # lowerarm_visual_R = pin.GeometryObject('lowerarm',IDX_LOA_SF, IDX_EL_Y_JF, mesh_loader.load(meshes_folder_path+'/lowerarm_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.05, -0.25, 0.17]).T), meshes_folder_path+'/lowerarm_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , np.array([0,1,1,0.5]))
    # geom_model.addGeometryObject(lowerarm_visual_R)
    # visuals_dict["lowerarm_R"] = lowerarm_visual_R



    # # Left shoulder ZXY
    # IDX_SH_Z_JF_L = model.addJoint(IDX_L5S1_R_EXT_INT_JF, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix(local_segments_positions['upperarmL'] + local_segments_positions['torso']).T), 'Shoulder_Z_L') 
    # upperarmL = pin.Frame('upperarm_z_L', IDX_SH_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    # IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    # idx_frame = IDX_UPA_SF_L

    # shoulder_visual_L = pin.GeometryObject('shoulder_L', IDX_UPA_SF_L, IDX_SH_Z_JF_L, mesh_loader.load(meshes_folder_path+'/shoulder_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.16, -0.045, -0.045]).T), meshes_folder_path+'/shoulder_mesh.STL', np.array([0.0055, 0.0055, 0.0055]), False, np.array([0,1,1,0.5]))
    # geom_model.addGeometryObject(shoulder_visual_L)
    # visuals_dict["shoulder_L"] = shoulder_visual_L

    # IDX_SH_X_JF_L = model.addJoint(IDX_SH_Z_JF_L, pin.JointModelRX(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'Shoulder_X_L') 
    # upperarmL = pin.Frame('upperarm_x_L', IDX_SH_X_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    # IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    # idx_frame = IDX_UPA_SF_L

    # IDX_SH_Y_JF_L = model.addJoint(IDX_SH_X_JF_L, pin.JointModelRY(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'Shoulder_Y_L') 
    # upperarmL = pin.Frame('upperarmL', IDX_SH_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    # IDX_UPA_SF_L = model.addFrame(upperarmL, False)
    # idx_frame = IDX_UPA_SF_L

    # for i in sgts_mks_dict["upperarmL"]:
    #     frame = pin.Frame(i, IDX_SH_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix(lstm_mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
    #     idx_frame = model.addFrame(frame, False)

    # upperarm_visual_L = pin.GeometryObject('upperarm_L', IDX_UPA_SF_L, IDX_SH_Y_JF_L, mesh_loader.load(meshes_folder_path+'/upperarm_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.07, -0.29, 0.18]).T), meshes_folder_path+'/upperarm_mesh.STL', np.array([0.0063, 0.0060, 0.007]), False, np.array([0,1,1,0.5]))
    # geom_model.addGeometryObject(upperarm_visual_L)
    # visuals_dict["upperarm_L"] = upperarm_visual_L



    # # Coude gauche ZY
    # IDX_EL_Z_JF_L = model.addJoint(IDX_SH_Y_JF_L, pin.JointModelRZ(), pin.SE3(np.eye(3), np.matrix(local_segments_positions['lowerarmL']).T), 'Elbow_Z_L')
    # lowerarmL = pin.Frame('lowerarm_z_L', IDX_EL_Z_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    # IDX_LOA_SF_L = model.addFrame(lowerarmL, False)
    # idx_frame = IDX_LOA_SF_L

    # elbow_visual_L = pin.GeometryObject('elbow_L', IDX_LOA_SF_L, IDX_EL_Z_JF_L, mesh_loader.load(meshes_folder_path+'/elbow_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.15, -0.02, -0.035]).T), meshes_folder_path+'/elbow_mesh.STL', np.array([0.0055, 0.0055, 0.0055]), False, np.array([0,1,1,0.5]))
    # geom_model.addGeometryObject(elbow_visual_L)
    # visuals_dict["elbow_L"] = elbow_visual_L

    # IDX_EL_Y_JF_L = model.addJoint(IDX_EL_Z_JF_L, pin.JointModelRY(), pin.SE3(np.eye(3), np.matrix([0,0,0]).T), 'Elbow_Y_L') 
    # lowerarmL = pin.Frame('lowerarmL', IDX_EL_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix([0,0,0]).T), pin.FrameType.OP_FRAME, inertia)
    # IDX_LOA_SF_L = model.addFrame(lowerarmL, False)
    # idx_frame = IDX_LOA_SF_L

    # for i in sgts_mks_dict["lowerarmL"]:
    #     frame = pin.Frame(i, IDX_EL_Y_JF_L, idx_frame, pin.SE3(np.eye(3), np.matrix(lstm_mks_local_positions[i]).T), pin.FrameType.OP_FRAME, inertia)
    #     idx_frame = model.addFrame(frame, False)

    # lowerarm_visual_L = pin.GeometryObject('lowerarm_L', IDX_LOA_SF_L, IDX_EL_Y_JF_L, mesh_loader.load(meshes_folder_path+'/lowerarm_mesh.STL'), pin.SE3(rlowerarm.as_matrix(), np.matrix([-0.05, -0.25, 0.17]).T), meshes_folder_path+'/lowerarm_mesh.STL', np.array([0.0060, 0.0060, 0.0060]), False, np.array([0,1,1,0.5]))
    # geom_model.addGeometryObject(lowerarm_visual_L)
    # visuals_dict["lowerarm_L"] = lowerarm_visual_L

    data     = model.createData()
    # print(model.names)


    # # Sample a random configuration
    # q        = pin.neutral(model)
    # print('q: %s' % q.T)
    # # Perform the forward kinematics over the kinematic tree
    # pin.forwardKinematics(model,data,q)
    # # Print out the placement of each joint of the kinematic tree
    
    # for name, oMi in zip(model.names, data.oMi):
    #     print(("{:<24} : {: .2f} {: .2f} {: .2f}"
    #         .format( name, *oMi.translation.T.flat )))


    model.upperPositionLimit[7:] = np.array([
                                        np.pi,          #Hip_Z_R +
                                        2.0*np.pi/9,    #Hip_X_R +
                                        np.pi/2.0 ,     #Hip_Y_R +
                                        0.0,            #Knee_Z_R +
                                        np.pi/4,        #Ankle_Z_R +
                                        np.pi/6,        #Ankle_X_R +
                                        np.pi,          #Hip_Z_L +
                                        np.pi,          #Hip_X_L +
                                        np.pi/2,        #Hip_Y_L +
                                        0.0,            #Knee_Z_L +
                                        np.pi/4,        #Ankle_Z_L +
                                        np.pi/6,        #Ankle_X_L +
                                        ]) 

    model.lowerPositionLimit[7:] = np.array([
                                        -np.pi/2.0,         #Hip_Z_R -
                                        -np.pi,             #Hip_X_R -
                                        -np.pi/2,           #Hip_Y_R -
                                        -3*np.pi/4.0,       #Knee_Z_R -
                                        -np.pi/2,           #Ankle_Z_R -
                                        -np.pi/6,           #Ankle_X_R -
                                        -np.pi/2,           #Hip_Z_L -
                                        -2.0*np.pi/9,       #Hip_X_L -
                                        -np.pi/2,           #Hip_Y_L -
                                        -3*np.pi/4.0,       #Knee_Z_L -
                                        -np.pi/2,           #Ankle_Z_L -
                                        -np.pi/6,           #Ankle_X_L -
                                        ])
    # model.upperPositionLimit[7:] = np.array([0.305,0.524,3.142,3.142,1.22,2.53, 1.57, 0.52,0.52,0.785,0,0.35])
    # model.lowerPositionLimit[7:] = np.array([-0.393,-0.524,-1.047,-0.698,-1.57,0, -1.57, -2.27,-0.52,-0.785,-2.36,-0.87])

    return model, geom_model, visuals_dict

