import numpy as np
import copy
import json
import os
import time
import numpy as np
import onnxruntime as ort
import pandas as pd


def marker(buffer, keypoint_index):
    """
    Retrieves the 3D trajectory of a reference marker: midhip
    
    Args:
        buffer (np.array): Array of shape (num_frames, 26, 3), containing the 3D coordinates
                           of the 26 markers over num_frames frames.
        keypoint_index (int): The index of the reference marker.

    Returns:
        np.array: Trajectory of the specific marker, array of shape (num_frames, 3).
    """
    # Récupérer les coordonnées x, y, z du marqueur spécifique sur toutes les frames
    reference_marker_trajectory = np.empty((buffer.shape[0], 3))
    
    # Extract the 3D coordinates for the specified keypoint across all frames
    reference_marker_trajectory[:, 0] = buffer[:, keypoint_index, 0]  # x coordinate
    reference_marker_trajectory[:, 1] = buffer[:, keypoint_index, 1]  # y coordinate
    reference_marker_trajectory[:, 2] = buffer[:, keypoint_index, 2]  # z coordinate
    
    return reference_marker_trajectory

def loadModel(augmenterDir, augmenterModelName="LSTM",augmenter_model='v0.3', offset = True):
    """
    Load and initialize LSTM models for different augmenter types.
    Parameters:
    -----------
    augmenterDir : str
        Directory where the augmenter models are stored.
    augmenterModelName : str, optional
        Name of the augmenter model (default is "LSTM").
    augmenter_model : str, optional
        Version of the augmenter model to load (default is 'v0.3').
    offset : bool, optional
        Flag to indicate if offset should be applied (default is True).
    Returns:
    --------
    dict
        A dictionary where keys are augmenter model types and values are the corresponding ONNX inference sessions.
    Notes:
    ------
    - The function supports different augmenter model versions ('v0.0', 'v0.1', 'v0.2', and others).
    - Depending on the augmenter model version, different sets of feature and response markers are loaded.
    - The function initializes ONNX inference sessions for each augmenter model type and stores them in a dictionary.
    """
    
    # Remove the redundant definition of loadModel

    models = {}
     # Augmenter types
    if augmenter_model == 'v0.0':
        from utils.lstm_utils import getOpenPoseMarkers_fullBody
        feature_markers_full, response_markers_full = getOpenPoseMarkers_fullBody()         
        augmenterModelType_all = [augmenter_model]
        feature_markers_all = [feature_markers_full]
        response_markers_all = [response_markers_full]            
    elif augmenter_model == 'v0.1' or augmenter_model == 'v0.2':
        # Lower body           
        augmenterModelType_lower = '{}_lower'.format(augmenter_model)
        from utils.lstm_utils import getOpenPoseMarkers_lowerExtremity
        feature_markers_lower, response_markers_lower = getOpenPoseMarkers_lowerExtremity()
        # Upper body
        augmenterModelType_upper = '{}_upper'.format(augmenter_model)
        from utils.lstm_utils import getMarkers_upperExtremity_noPelvis
        feature_markers_upper, response_markers_upper = getMarkers_upperExtremity_noPelvis()        
        augmenterModelType_all = [augmenterModelType_lower, augmenterModelType_upper]
        feature_markers_all = [feature_markers_lower, feature_markers_upper]
        response_markers_all = [response_markers_lower, response_markers_upper]
    else:
        # Lower body           
        augmenterModelType_lower = '{}_lower'.format(augmenter_model)
        from utils.lstm_utils import getOpenPoseMarkers_lowerExtremity2
        feature_markers_lower, response_markers_lower = getOpenPoseMarkers_lowerExtremity2()
        # Upper body
        augmenterModelType_upper = '{}_upper'.format(augmenter_model)
        from utils.lstm_utils import getMarkers_upperExtremity_noPelvis2
        feature_markers_upper, response_markers_upper = getMarkers_upperExtremity_noPelvis2()   
             
        augmenterModelType_all = [augmenterModelType_lower, augmenterModelType_upper]
    
    for idx_augm, augmenterModelType in enumerate(augmenterModelType_all):
        augmenterModelDir = os.path.join(augmenterDir, augmenterModelName, 
                                         augmenterModelType)
        session = ort.InferenceSession(f"{augmenterModelDir}/model.onnx")

        models[augmenterModelType] = session

    return models

def augmentTRC(keypoints_buffer, subject_mass, subject_height,
               models, augmenterDir, augmenterModelName='LSTM', augmenter_model='v0.3', offset=True):
    """
    Augments the given keypoints buffer using specified models and parameters.
    Parameters:
    -----------
    keypoints_buffer : numpy.ndarray
        The buffer containing keypoints data.
    subject_mass : float
        The mass of the subject.
    subject_height : float
        The height of the subject.
    models : dict
        Dictionary containing pre-warmed models for augmentation.
    augmenterDir : str
        Directory where augmenter models are stored.
    augmenterModelName : str, optional
        Name of the augmenter model (default is 'LSTM').
    augmenter_model : str, optional
        Version of the augmenter model (default is 'v0.3').
    offset : bool, optional
        Whether to apply offset (default is True).
    Returns:
    --------
    numpy.ndarray
        The concatenated responses from the lower and upper body augmenters.
    """

    n_response_markers_all = 0
    featureHeight = True
    featureWeight = True
    
    outputs_all = {}
    marker_indices_lower = [18, 6, 5, 12, 11, 14, 13, 16, 15, 25, 24, 23, 22, 21, 20] #['Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 'RSmallToe', 'LSmallToe', 'RBigToe', 'LBigToe']
    marker_indices_upper = [18, 6, 5, 8, 7, 10, 9] #['Neck', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist']

    # Loop over augmenter types to handle separate augmenters for lower and
    # upper bodies.
    augmenterModelType_all = [f'{augmenter_model}_lower', f'{augmenter_model}_upper']

    # Loop over augmenter types to handle separate augmenters for lower and upper bodies
    for augmenterModelType in augmenterModelType_all:
        if 'lower' in augmenterModelType:
            feature_markers = marker_indices_lower
            response_markers=['r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study', 'L.PSIS_study', 'r_knee_study', 'r_mknee_study', 'r_ankle_study', 'r_mankle_study', 'r_toe_study', 'r_5meta_study', 'r_calc_study', 'L_knee_study', 'L_mknee_study', 'L_ankle_study', 'L_mankle_study', 'L_toe_study', 'L_calc_study', 'L_5meta_study', 'r_shoulder_study', 'L_shoulder_study', 'C7_study', 'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study', 'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study', 'r_sh1_study', 'r_sh2_study', 'r_sh3_study', 'L_sh1_study', 'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']
        else:
            feature_markers = marker_indices_upper
            response_markers=['r_lelbow_study', 'r_melbow_study', 'r_lwrist_study', 'r_mwrist_study', 'L_lelbow_study', 'L_melbow_study', 'L_lwrist_study', 'L_mwrist_study']

        augmenterModelDir = os.path.join(augmenterDir, augmenterModelName, 
                                         augmenterModelType)
        # Process the keypoints buffer
        referenceMarker_data = marker(keypoints_buffer, 19)  # midihip
        norm_buffer = np.zeros_like(keypoints_buffer)

        # Normalize based on the reference marker
        for i in feature_markers:
            norm_buffer[:, i, :] = keypoints_buffer[:, i, :] - referenceMarker_data

        # Normalize with subject's height
        norm2_buffer = norm_buffer / subject_height

        # Flatten the keypoints data
        inputs = norm2_buffer[:, feature_markers, :].reshape(norm2_buffer.shape[0], -1)

        # Add height and weight as features
        if featureHeight:
            inputs = np.concatenate((inputs, subject_height * np.ones((inputs.shape[0], 1))), axis=1)
        if featureWeight:
            inputs = np.concatenate((inputs, subject_mass * np.ones((inputs.shape[0], 1))), axis=1)

        # Load mean and std for normalization
        #print(augmenterModelDir)
        pathMean = os.path.join(augmenterModelDir, "mean.npy")
        pathSTD = os.path.join(augmenterModelDir, "std.npy")
        #print(pathMean)

        if os.path.isfile(pathMean):
            trainFeatures_mean = np.load(pathMean, allow_pickle=True)
            inputs -= trainFeatures_mean

        if os.path.isfile(pathSTD):
            trainFeatures_std = np.load(pathSTD, allow_pickle=True)
            inputs /= trainFeatures_std

        # Reshape inputs if necessary for LSTM model
        inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))

        # pre-warmed model
        model = models.get(augmenterModelType)

        # inference
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: inputs.astype(np.float32)})

        outputs = outputs[0]
        #Post-process the outputs
        if augmenterModelName == "LSTM":
            outputs = np.reshape(outputs, (outputs.shape[1], outputs.shape[2]))

        # Un-normalize the outputs
        unnorm_outputs = outputs * subject_height
        unnorm2_outputs = np.zeros((unnorm_outputs.shape[0], unnorm_outputs.shape[1]))

        for i in range(0, unnorm_outputs.shape[1], 3):
            unnorm2_outputs[:, i:i+3] = unnorm_outputs[:, i:i+3] + referenceMarker_data

        for c, m in enumerate(response_markers):
            x = unnorm2_outputs[:,c*3]
            y = unnorm2_outputs[:,c*3+1]
            z = unnorm2_outputs[:,c*3+2]
        
        outputs_all[augmenterModelType] = unnorm2_outputs
        last_output = unnorm2_outputs[-1, :]
        outputs_all[augmenterModelType] = last_output


    # Check for existence of each key and concatenate if present
    if 'v0.3_lower' in outputs_all:
        v0_3_lower = outputs_all['v0.3_lower']

    if 'v0.3_upper' in outputs_all:
        v0_3_upper = outputs_all['v0.3_upper']


    responses_all_conc = np.concatenate((v0_3_lower, v0_3_upper))

    # # Convert responses_all_conc to a pandas DataFrame
    # df = pd.DataFrame([responses_all_conc])

    # df.to_csv("responses_all_conc_rt.csv", mode='a',header= False, index=False)
    return responses_all_conc

