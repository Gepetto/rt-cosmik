import numpy as np
import os
import numpy as np
import onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

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

def loadModel(augmenterDir, augmenterModelName="LSTM",augmenter_model='v0.3', use_mocap="T", add_noise="F", fine_tune="F", 
            add_layer="T", use_weights="F", rot_prob=0.0, rot_max_deg=30.0, rotation_scheme="off", up_axis="z", n_rotations=1):
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
    Returns:
    --------
    dict
        A dictionary where keys are augmenter model types and values are the corresponding ONNX inference sessions.
    Notes:
    ------
    - The function initializes ONNX inference sessions for each augmenter model type and stores them in a dictionary.
    """
    
    # Remove the redundant definition of loadModel

    models = {}

    # Lower body           
    augmenterModelType_lower = '{}_lower'.format(augmenter_model)
    # Upper body
    augmenterModelType_upper = '{}_upper'.format(augmenter_model)
            
    augmenterModelType_all = [augmenterModelType_lower, augmenterModelType_upper]
    
    for idx_augm, augmenterModelType in enumerate(augmenterModelType_all):
        augmenterModelDir = os.path.join(augmenterDir, augmenterModelName, 
                                            augmenterModelType)
        if augmenterModelType == "{}_lower".format(augmenter_model):
            session = ort.InferenceSession(f"{augmenterModelDir}/model_{augmenterModelType[5:]}_ft{fine_tune}_al{add_layer}_m{use_mocap}_n{add_noise}_w{use_weights}_prot{rot_prob}_maxrot{rot_max_deg}_rotscheme{rotation_scheme}_up{up_axis}_nrot{n_rotations}.onnx", sess_options=so, providers=["CPUExecutionProvider"])
        else:
            session = ort.InferenceSession(f"{augmenterModelDir}/model_{augmenterModelType[5:]}_ft{fine_tune}_al{add_layer}_m{use_mocap}_n{add_noise}_wF_prot{rot_prob}_maxrot{rot_max_deg}_rotscheme{rotation_scheme}_up{up_axis}_nrot{n_rotations}.onnx", sess_options=so, providers=["CPUExecutionProvider"])
        models[augmenterModelType] = session

    return models

def augmentTRC(keypoints_buffer, subject_mass, subject_height,
               models, augmenterDir, augmenterModelName='LSTM', augmenter_model='v0.3', offset=True,
               use_mocap="T", add_noise="F", fine_tune="F", 
               add_layer="T", use_weights="F", rot_prob=0.0, rot_max_deg=30.0, rotation_scheme="off", n_rotations=1, up_axis="z"):
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
    if use_mocap == "T":
        marker_indices_lower = [2, 0, 1, 7, 8, 10, 11, 12, 13, 14, 15, 18, 19, 16, 17] #['Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 'RSmallToe', 'LSmallToe', 'RBigToe', 'LBigToe']
        marker_indices_upper = [2, 0, 1, 3, 4, 5, 6] #['Neck', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist']
    elif use_mocap == "F":
        marker_indices_lower = [18, 6, 5, 12, 11, 14, 13, 16, 15, 25, 24, 23, 22, 21, 20] #['Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 'RSmallToe', 'LSmallToe', 'RBigToe', 'LBigToe']
        marker_indices_upper = [18, 6, 5, 8, 7, 10, 9] #['Neck', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist']
    else:
        raise Exception("Input type not supported. Please select T or F.")
    

    # Loop over augmenter types to handle separate augmenters for lower and
    # upper bodies.
    augmenterModelType_all = [f'{augmenter_model}_lower', f'{augmenter_model}_upper']

    # Loop over augmenter types to handle separate augmenters for lower and upper bodies
    for augmenterModelType in augmenterModelType_all:
        if 'lower' in augmenterModelType:
            feature_markers = marker_indices_lower
            # response_markers=['r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study', 'L.PSIS_study', 'r_knee_study', 'r_mknee_study', 'r_ankle_study', 'r_mankle_study', 'r_toe_study', 'r_5meta_study', 'r_calc_study', 'L_knee_study', 'L_mknee_study', 'L_ankle_study', 'L_mankle_study', 'L_toe_study', 'L_calc_study', 'L_5meta_study', 'r_shoulder_study', 'L_shoulder_study', 'C7_study', 'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study', 'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study', 'r_sh1_study', 'r_sh2_study', 'r_sh3_study', 'L_sh1_study', 'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']
        else:
            feature_markers = marker_indices_upper
            # response_markers=['r_lelbow_study', 'r_melbow_study', 'r_lwrist_study', 'r_mwrist_study', 'L_lelbow_study', 'L_melbow_study', 'L_lwrist_study', 'L_mwrist_study']

        augmenterModelDir = os.path.join(augmenterDir, augmenterModelName, 
                                         augmenterModelType)
        # Process the keypoints buffer
        if use_mocap == "T":
            referenceMarker_data = marker(keypoints_buffer, 9)  # midihip
        elif use_mocap == "F":
            referenceMarker_data = marker(keypoints_buffer, 19)  # midhip
        else:
            raise Exception("Input type not supported. Please select T or F.")
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
        if "lower" in augmenterModelType:
            pathMean = os.path.join(augmenterModelDir, "stats_streaming", f"mean_train_ft{fine_tune}_al{add_layer}_mT_n{add_noise}_w{use_weights}_prot{rot_prob}_maxrot{rot_max_deg}_rotscheme{rotation_scheme}_up{up_axis}_nrot{n_rotations}.npy")
            pathSTD = os.path.join(augmenterModelDir, "stats_streaming", f"std_train_ft{fine_tune}_al{add_layer}_mT_n{add_noise}_w{use_weights}_prot{rot_prob}_maxrot{rot_max_deg}_rotscheme{rotation_scheme}_up{up_axis}_nrot{n_rotations}.npy")
        else:
            pathMean = os.path.join(augmenterModelDir, "stats_streaming", f"mean_train_ft{fine_tune}_al{add_layer}_mT_n{add_noise}_wF_prot{rot_prob}_maxrot{rot_max_deg}_rotscheme{rotation_scheme}_up{up_axis}_nrot{n_rotations}.npy")
            pathSTD = os.path.join(augmenterModelDir, "stats_streaming", f"std_train_ft{fine_tune}_al{add_layer}_mT_n{add_noise}_wF_prot{rot_prob}_maxrot{rot_max_deg}_rotscheme{rotation_scheme}_up{up_axis}_nrot{n_rotations}.npy")
        #print(pathMean)

        if os.path.isfile(pathMean):
            trainFeatures_mean = np.load(pathMean, allow_pickle=True)
            inputs -= trainFeatures_mean
        else: raise Exception(f"No mean file found at {pathMean}")

        if os.path.isfile(pathSTD):
            trainFeatures_std = np.load(pathSTD, allow_pickle=True)
            inputs /= trainFeatures_std
        else: raise Exception(f"No mean file found at {pathMean}")

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

        outputs_all[augmenterModelType] = unnorm2_outputs
        last_output = unnorm2_outputs[-1, :]
        outputs_all[augmenterModelType] = last_output


    # Check for existence of each key and concatenate if present
    if 'v0.3_lower' in outputs_all:
        v0_3_lower = outputs_all['v0.3_lower']

    if 'v0.3_upper' in outputs_all:
        v0_3_upper = outputs_all['v0.3_upper']


    responses_all_conc = np.concatenate((v0_3_lower, v0_3_upper))
    # print(responses_all_conc)
    return responses_all_conc


def loadModelOpenCap(augmenterDir, augmenterModelName="LSTM",augmenter_model='v0.3'):
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
    Returns:
    --------
    dict
        A dictionary where keys are augmenter model types and values are the corresponding ONNX inference sessions.
    Notes:
    ------
    - The function initializes ONNX inference sessions for each augmenter model type and stores them in a dictionary.
    """
    
    # Remove the redundant definition of loadModel

    models = {}

    # Lower body           
    augmenterModelType_lower = '{}_lower'.format(augmenter_model)
    # Upper body
    augmenterModelType_upper = '{}_upper'.format(augmenter_model)
            
    augmenterModelType_all = [augmenterModelType_lower, augmenterModelType_upper]
    
    for idx_augm, augmenterModelType in enumerate(augmenterModelType_all):
        augmenterModelDir = os.path.join(augmenterDir, augmenterModelName, 
                                         augmenterModelType)
        session = ort.InferenceSession(f"{augmenterModelDir}/model.onnx", sess_options=so, providers=["CPUExecutionProvider"])

        models[augmenterModelType] = session

    return models


def augmentTRCOpenCap(keypoints_buffer, subject_mass, subject_height,
               models, augmenterDir, augmenterModelName='LSTM', augmenter_model='v0.3', offset=True
               ):
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
    marker_indices_lower = [2, 0, 1, 7, 8, 10, 11, 12, 13, 14, 15, 18, 19, 16, 17] #['Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 'RSmallToe', 'LSmallToe', 'RBigToe', 'LBigToe']
    marker_indices_upper = [2, 0, 1, 3, 4, 5, 6] #['Neck', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist']
    # elif use_mocap == "F":
    #     marker_indices_lower = [18, 6, 5, 12, 11, 14, 13, 16, 15, 25, 24, 23, 22, 21, 20] #['Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 'RSmallToe', 'LSmallToe', 'RBigToe', 'LBigToe']
    #     marker_indices_upper = [18, 6, 5, 8, 7, 10, 9] #['Neck', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist']
    # else:
    #     raise Exception("Input type not supported. Please select T or F.")

    # Loop over augmenter types to handle separate augmenters for lower and
    # upper bodies.
    augmenterModelType_all = [f'{augmenter_model}_lower', f'{augmenter_model}_upper']

    # Loop over augmenter types to handle separate augmenters for lower and upper bodies
    for augmenterModelType in augmenterModelType_all:
        if 'lower' in augmenterModelType:
            feature_markers = marker_indices_lower
            # response_markers=['r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study', 'L.PSIS_study', 'r_knee_study', 'r_mknee_study', 'r_ankle_study', 'r_mankle_study', 'r_toe_study', 'r_5meta_study', 'r_calc_study', 'L_knee_study', 'L_mknee_study', 'L_ankle_study', 'L_mankle_study', 'L_toe_study', 'L_calc_study', 'L_5meta_study', 'r_shoulder_study', 'L_shoulder_study', 'C7_study', 'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study', 'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study', 'r_sh1_study', 'r_sh2_study', 'r_sh3_study', 'L_sh1_study', 'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']
        else:
            feature_markers = marker_indices_upper
            # response_markers=['r_lelbow_study', 'r_melbow_study', 'r_lwrist_study', 'r_mwrist_study', 'L_lelbow_study', 'L_melbow_study', 'L_lwrist_study', 'L_mwrist_study']

        augmenterModelDir = os.path.join(augmenterDir, augmenterModelName, 
                                         augmenterModelType)
        # Process the keypoints buffer
        referenceMarker_data = marker(keypoints_buffer, 9)  # midihip
        # elif use_mocap == "F":
        #     referenceMarker_data = marker(keypoints_buffer, 19)  # midhip
        # else:
        #     raise Exception("Input type not supported. Please select T or F.")
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
        pathMean = os.path.join(augmenterModelDir, f"mean.npy")
        pathSTD = os.path.join(augmenterModelDir, f"std.npy")
        #print(pathMean)

        if os.path.isfile(pathMean):
            trainFeatures_mean = np.load(pathMean, allow_pickle=True)
            inputs -= trainFeatures_mean
        else: raise Exception(f"No mean file found at {pathMean}")

        if os.path.isfile(pathSTD):
            trainFeatures_std = np.load(pathSTD, allow_pickle=True)
            inputs /= trainFeatures_std
        else: raise Exception(f"No mean file found at {pathMean}")

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

        outputs_all[augmenterModelType] = unnorm2_outputs
        last_output = unnorm2_outputs[-1, :]
        outputs_all[augmenterModelType] = last_output


        if augmenterModelType == "{}_lower".format(augmenter_model):

            ordrer_output = ["r.ASIS_study_x", "r.ASIS_study_y", "r.ASIS_study_z", "L.ASIS_study_x", "L.ASIS_study_y", "L.ASIS_study_z",
            "r.PSIS_study_x", "r.PSIS_study_y", "r.PSIS_study_z", "L.PSIS_study_x", "L.PSIS_study_y", "L.PSIS_study_z",
            "r_knee_study_x", "r_knee_study_y", "r_knee_study_z", "r_mknee_study_x", "r_mknee_study_y", "r_mknee_study_z",
            "r_ankle_study_x", "r_ankle_study_y", "r_ankle_study_z", "r_mankle_study_x", "r_mankle_study_y", "r_mankle_study_z",
            "r_toe_study_x", "r_toe_study_y", "r_toe_study_z", "r_5meta_study_x", "r_5meta_study_y", "r_5meta_study_z", 
            "r_calc_study_x", "r_calc_study_y", "r_calc_study_z", "L_knee_study_x", "L_knee_study_y", "L_knee_study_z",
            "L_mknee_study_x", "L_mknee_study_y", "L_mknee_study_z", "L_ankle_study_x", "L_ankle_study_y", "L_ankle_study_z",
            "L_mankle_study_x", "L_mankle_study_y", "L_mankle_study_z", "L_toe_study_x", "L_toe_study_y", "L_toe_study_z",
            "L_calc_study_x", "L_calc_study_y",	"L_calc_study_z", "L_5meta_study_x", "L_5meta_study_y", "L_5meta_study_z",
            "r_shoulder_study_x", "r_shoulder_study_y", "r_shoulder_study_z", "L_shoulder_study_x", "L_shoulder_study_y", "L_shoulder_study_z",
            "C7_study_x", "C7_study_y",	"C7_study_z"]

            outputs_all[augmenterModelType] = outputs_all[augmenterModelType][:len(ordrer_output)]

    # Check for existence of each key and concatenate if present
    if 'v0.3_lower' in outputs_all:
        v0_3_lower = outputs_all['v0.3_lower']

    if 'v0.3_upper' in outputs_all:
        v0_3_upper = outputs_all['v0.3_upper']


    responses_all_conc = np.concatenate((v0_3_lower, v0_3_upper))
    # print(responses_all_conc)
    return responses_all_conc


def loadModel_incHPE(augmenterDir, augmenterModelName="LSTM",augmenter_model='v0.3', add_noise="F", 
            use_weights="F", seq_len=30, exclude_trials="none"):
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
    Returns:
    --------
    dict
        A dictionary where keys are augmenter model types and values are the corresponding ONNX inference sessions.
    Notes:
    ------
    - The function initializes ONNX inference sessions for each augmenter model type and stores them in a dictionary.
    """
    
    # Remove the redundant definition of loadModel

    models = {}

    # Lower body           
    augmenterModelType_lower = '{}_lower'.format(augmenter_model)
    # Upper body
    augmenterModelType_upper = '{}_upper'.format(augmenter_model)
            
    augmenterModelType_all = [augmenterModelType_lower, augmenterModelType_upper]
    
    for idx_augm, augmenterModelType in enumerate(augmenterModelType_all):
        augmenterModelDir = os.path.join(augmenterDir, augmenterModelName, 
                                            augmenterModelType)
        if augmenterModelType == "{}_lower".format(augmenter_model):
            session = ort.InferenceSession(f"{augmenterModelDir}/model_{augmenterModelType[5:]}_n{add_noise}_w{use_weights}_sl{seq_len}_exclude{exclude_trials}.onnx", sess_options=so, providers=["CPUExecutionProvider"])
        else:
            session = ort.InferenceSession(f"{augmenterModelDir}/model_{augmenterModelType[5:]}_n{add_noise}_wF_sl{seq_len}_exclude{exclude_trials}.onnx", sess_options=so, providers=["CPUExecutionProvider"])
        models[augmenterModelType] = session

    return models

def augmentTRC_incHPE(keypoints_buffer, subject_mass, subject_height,
               models, augmenterDir, augmenterModelName='LSTM', augmenter_model='v0.3', offset=True,
               add_noise="F",
               use_weights="F", seq_len=30, exclude_trials="none"):
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
    marker_indices_lower = [2, 0, 1, 7, 8, 10, 11, 12, 13, 14, 15, 18, 19, 16, 17] #['Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 'RSmallToe', 'LSmallToe', 'RBigToe', 'LBigToe']
    marker_indices_upper = [2, 0, 1, 3, 4, 5, 6] #['Neck', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist']
    # elif use_mocap == "F":
    #     marker_indices_lower = [18, 6, 5, 12, 11, 14, 13, 16, 15, 25, 24, 23, 22, 21, 20] #['Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 'RSmallToe', 'LSmallToe', 'RBigToe', 'LBigToe']
    #     marker_indices_upper = [18, 6, 5, 8, 7, 10, 9] #['Neck', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist']
    # else:
    #     raise Exception("Input type not supported. Please select T or F.")
    

    # Loop over augmenter types to handle separate augmenters for lower and
    # upper bodies.
    augmenterModelType_all = [f'{augmenter_model}_lower', f'{augmenter_model}_upper']

    # Loop over augmenter types to handle separate augmenters for lower and upper bodies
    for augmenterModelType in augmenterModelType_all:
        if 'lower' in augmenterModelType:
            feature_markers = marker_indices_lower
            # response_markers=['r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study', 'L.PSIS_study', 'r_knee_study', 'r_mknee_study', 'r_ankle_study', 'r_mankle_study', 'r_toe_study', 'r_5meta_study', 'r_calc_study', 'L_knee_study', 'L_mknee_study', 'L_ankle_study', 'L_mankle_study', 'L_toe_study', 'L_calc_study', 'L_5meta_study', 'r_shoulder_study', 'L_shoulder_study', 'C7_study', 'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study', 'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study', 'r_sh1_study', 'r_sh2_study', 'r_sh3_study', 'L_sh1_study', 'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']
        else:
            feature_markers = marker_indices_upper
            # response_markers=['r_lelbow_study', 'r_melbow_study', 'r_lwrist_study', 'r_mwrist_study', 'L_lelbow_study', 'L_melbow_study', 'L_lwrist_study', 'L_mwrist_study']

        augmenterModelDir = os.path.join(augmenterDir, augmenterModelName, 
                                         augmenterModelType)
        # Process the keypoints buffer

        referenceMarker_data = marker(keypoints_buffer, 9)  # midihip
        # elif use_mocap == "F":
        #     referenceMarker_data = marker(keypoints_buffer, 19)  # midhip
        # else:
        #     raise Exception("Input type not supported. Please select T or F.")
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
        if "lower" in augmenterModelType:
            pathMean = os.path.join(augmenterModelDir, "stats_streaming", f"mean_train_final_n{add_noise}_w{use_weights}_sl{seq_len}_exclude{exclude_trials}.npy")
            pathSTD = os.path.join(augmenterModelDir, "stats_streaming", f"std_train_final_n{add_noise}_w{use_weights}_sl{seq_len}_exclude{exclude_trials}.npy")
        else:
            pathMean = os.path.join(augmenterModelDir, "stats_streaming", f"mean_train_final_n{add_noise}_wF_sl{seq_len}_exclude{exclude_trials}.npy")
            pathSTD = os.path.join(augmenterModelDir, "stats_streaming", f"std_train_final_n{add_noise}_wF_sl{seq_len}_exclude{exclude_trials}.npy")
        #print(pathMean)

        if os.path.isfile(pathMean):
            trainFeatures_mean = np.load(pathMean, allow_pickle=True)
            inputs -= trainFeatures_mean
        else: raise Exception(f"No mean file found at {pathMean}")

        if os.path.isfile(pathSTD):
            trainFeatures_std = np.load(pathSTD, allow_pickle=True)
            inputs /= trainFeatures_std
        else: raise Exception(f"No mean file found at {pathMean}")

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

        outputs_all[augmenterModelType] = unnorm2_outputs
        last_output = unnorm2_outputs[-1, :]
        outputs_all[augmenterModelType] = last_output


    # Check for existence of each key and concatenate if present
    if 'v0.3_lower' in outputs_all:
        v0_3_lower = outputs_all['v0.3_lower']

    if 'v0.3_upper' in outputs_all:
        v0_3_upper = outputs_all['v0.3_upper']


    responses_all_conc = np.concatenate((v0_3_lower, v0_3_upper))
    # print(responses_all_conc)
    return responses_all_conc