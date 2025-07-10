import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import copy
import numpy as np
import os
import onnxruntime as ort
import pandas as pd
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, TimeDistributed


class KeypointDataset(Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path)
        label_path = os.path.join(os.path.dirname(data_path), "mks_data_gapfilled.csv")
        labels = pd.read_csv(label_path)
        self.X = torch.tensor(data.iloc[1:, :].values)
        self.y = torch.tensor(labels.iloc[1:, :].values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
    

subject_mass = 72.0
subject_height = 1.80
augmenter_path_lower = '/mnt/c/Users/nicol/Desktop/Travail/LAAS/SFE Gepetto/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM/v0.3_lower'
augmenter_path_upper = '/mnt/c/Users/nicol/Desktop/Travail/LAAS/SFE Gepetto/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM/v0.3_upper'
markers_lower = ['r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study', 'L.PSIS_study', 
                 'r_knee_study', 'r_mknee_study', 'r_ankle_study', 'r_mankle_study', 
                 'r_toe_study', 'r_5meta_study', 'r_calc_study', 'L_knee_study', 'L_mknee_study', 
                 'L_ankle_study', 'L_mankle_study', 'L_toe_study', 'L_calc_study', 'L_5meta_study', 
                 'r_shoulder_study', 'L_shoulder_study', 'C7_study', 'r_thigh1_study', 'r_thigh2_study', 
                 'r_thigh3_study', 'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study', 'r_sh1_study', 
                 'r_sh2_study', 'r_sh3_study', 'L_sh1_study', 'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']
markers_upper = ['r_lelbow_study', 'r_melbow_study', 'r_lwrist_study', 'r_mwrist_study', 
                 'L_lelbow_study', 'L_melbow_study', 'L_lwrist_study', 'L_mwrist_study']
marker_indices_lower = [18, 6, 5, 12, 11, 14, 13, 16, 15, 25, 24, 23, 22, 21, 20] #['Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 'RSmallToe', 'LSmallToe', 'RBigToe', 'LBigToe']
marker_indices_upper = [18, 6, 5, 8, 7, 10, 9] #['Neck', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist']


# 🔧 Hyperparamètres
batch_size = 64
num_epochs = 50
learning_rate = 6e-6
num_classes_lower = len(markers_lower)*3 # 46 markers with 3D coordinates (x, y, z)
num_classes_upper = len(markers_upper)*3 # 46 markers with 3D coordinates (x, y, z)
patience = 10

# # 🔄 Transforms
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# 📂 Datasets
train_dataset = KeypointDataset("/mnt/c/Users/nicol/Desktop/Travail/LAAS/SFE Gepetto/Data/Test_train/3d_keypoints_filtered_2.csv")
val_dataset = KeypointDataset("/mnt/c/Users/nicol/Desktop/Travail/LAAS/SFE Gepetto/Data/Test_val/3d_keypoints_filtered_2.csv")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 🧠 Modèle
# %% Load model and weights
json_file = open(os.path.join(augmenter_path_lower, "model.json"), 'r')
pretrainedModel_json = json_file.read()
json_file.close()
model = model_from_json(pretrainedModel_json)
model.load_weights(os.path.join(augmenter_path_lower, "weights.h5"))
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes_lower)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

# 🛑 Early Stopping + Historique
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# 🔁 Boucle d'entraînement
for epoch in range(num_epochs):
    # Entraînement
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model.fit(inputs, verbose=2)
        outputs = outputs[marker_indices_lower]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)       
        correct += (np.abs(preds - labels)<0.02).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (np.abs(preds - labels)<0.02).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    # Sauvegarde historique
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Époque [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("⏹️ Early stopping déclenché.")
            break

# 🔁 Charger les meilleurs poids
model.load_state_dict(best_model_wts)

# 📊 Tracer les courbes
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.xlabel("Époques")
plt.ylabel("Loss")
plt.title("Courbe de perte")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label="Train Acc")
plt.plot(epochs_range, val_accuracies, label="Val Acc")
plt.xlabel("Époques")
plt.ylabel("Accuracy (%)")
plt.title("Courbe de précision")
plt.legend()

plt.tight_layout()
plt.show()










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



def loadModel(augmenterDir, augmenterModelName="LSTM",augmenter_model='v0.3'):
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
        session = ort.InferenceSession(f"{augmenterModelDir}/model.onnx")

        models[augmenterModelType] = session

    return models


def augmentTRC(keypoints_buffer, subject_mass, subject_height,
               models, augmenterDir, augmenterModelName='LSTM', augmenter_model='v0.3', offset=True, selected_augmenter="all"):
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
    if selected_augmenter == "all":
        augmenterModelType_all = [f'{augmenter_model}_lower', f'{augmenter_model}_upper']
    elif selected_augmenter == "lower":
        augmenterModelType_all = [f'{augmenter_model}_lower']
    elif selected_augmenter == "upper":
        augmenterModelType_all = [f'{augmenter_model}_upper']
    else:
        raise ValueError("selected_augmenter must be 'all', 'lower', or 'upper'.")

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

        outputs_all[augmenterModelType] = unnorm2_outputs
        last_output = unnorm2_outputs[-1, :]
        outputs_all[augmenterModelType] = last_output


    # Check for existence of each key and concatenate if present
    if 'v0.3_lower' in outputs_all:
        v0_3_lower = outputs_all['v0.3_lower']

    if 'v0.3_upper' in outputs_all:
        v0_3_upper = outputs_all['v0.3_upper']

    if selected_augmenter == "all":
        responses_all_conc = np.concatenate((v0_3_lower, v0_3_upper))
    elif selected_augmenter == "lower":
        responses_all_conc = v0_3_lower
    elif selected_augmenter == "upper":
        responses_all_conc = v0_3_upper

    # print(responses_all_conc)
    return responses_all_conc