import pandas as pd 
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt 
import os
import csv

def set_zero_data(df):
    # Isolate the right ankle coordinates for frame 1
    right_ankle_frame1 = df[(df['Frame'] == 1) & (df['Keypoint'] == 'Right Ankle')]

    # Extract the X, Y, Z coordinates
    x_ankle = right_ankle_frame1['X'].values[0]
    y_ankle = right_ankle_frame1['Y'].values[0]
    z_ankle = right_ankle_frame1['Z'].values[0]

    # Subtract these coordinates from the entire dataframe
    df['X'] = df['X'] - x_ankle
    df['Y'] = df['Y'] - y_ankle
    df['Z'] = df['Z'] - z_ankle


def read_lstm_data(file_name: str)->Tuple[Dict, Dict]:
    """_Creates two dictionnaries, one containing the 3D positions of all the markers output by the LSTM, another to map the number of the marks to the JC associated_
    Args:
        file_name (str): _The name of the file to process_

    Returns:
        Tuple[Dict, Dict]: _3D positions of all markers, mapping to JCP_
    """
    data = pd.read_csv(file_name).to_numpy()
    data = data[:,1:-1]
    time_vector = data[2:,0].astype(float)

    x = data[1,:] # Get label names
    labels = x[~pd.isnull(x)].tolist() #removes nan values

    x = data[0,:] # Get JCP names
    JCP = x[~pd.isnull(x)] #removes nan values
    JCP = JCP[1:].tolist() # removes time

    # Assert that the length of labels is three times the length of JCP
    assert len(labels) == 3 * len(JCP), "The length of labels must be three times the length of JCP."

    # Create the dictionary
    mapping = {}
    for i in range(len(JCP)):
        mapping[JCP[i]] = labels[i*3:(i*3)+3]

    labels = labels # add Time label

    data = data[2:,1:].astype(float)

    d=dict(zip(labels,data.T))
    
    # Create an empty dictionary for the combined arrays
    d3 = {'Time': time_vector}

    # Iterate over each key-value pair in d2
    for key, value in mapping.items():
        # Extract the arrays corresponding to the markers in value from d1
        arrays = [d[marker] for marker in value]
        # Combine the arrays into a single 3D array
        combined_array = np.array(arrays)
        # Transpose the array to have the shape (3, n), where n is the number of data points
        combined_array = np.transpose(combined_array)

        # Store the combined array in d3 with the key from d2
        d3[key] = combined_array

    return d3,mapping

def convert_to_list_of_dicts(dict_mks_data: Dict)-> List:
    """_This function converts a dictionnary of data outputed from read_lstm_data(), to a list of dictionnaries for each sample._

    Args:
        dict_mks_data (Dict): _ dictionnary of data outputed from read_lstm_data()_

    Returns:
        List: _ list of dictionnaries for each sample._
    """
    list_of_dicts = []
    for i in range(1):  #len(dict_mks_data['Time'])
        curr_dict = {}
        for name in dict_mks_data:
            curr_dict[name] = dict_mks_data[name][i]
            # print(dict_mks_data[name][i])
        list_of_dicts.append(curr_dict)
    return list_of_dicts

def get_lstm_mks_names(file_name: str):
    """_Gets the lstm mks names_
    Args:
        file_name (str): _The name of the file to process_

    Returns:
        mk_names (list): _lstm mks names_
    """
    mk_data = pd.read_csv(file_name)
    row = mk_data.iloc[0]#on chope la deuxième ligne
    mk_names = row[2:].tolist() #on enlève les deux premieres valeurs
    mk_names = [mot for mot in mk_names if pd.notna(mot)] #On enlève les nan correspondant aux cases vides du fichier csv
    return mk_names

#read the first and second line of mocap data, from a trc file
# def read_mocap_data(file_path: str)->Dict:
#     """_Gets the lstm mks names_
#     Args:
#         file_path (str): _The name of the file to process_

#     Returns:
#         mocap_mks_positions (list): _mocap mks positions and names dict_
#     """
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
    
#     # Extracting the anatomical landmarks names from the first line
#     landmarks = lines[0].strip().split(',')
#     print(landmarks)
    
#     # Extracting the 3D positions from the second line
#     positions = list(map(float, lines[1].strip().split(',')))
#     print(positions)
    
#     # Creating a dictionary to store the 3D positions of the landmarks
#     mocap_mks_positions = {}
#     for i, landmark in enumerate(landmarks):
#         # Each landmark has 3 positions (x, y, z)
#         mocap_mks_positions[landmark] = np.array(positions[3*i:3*i+3]).reshape(3,1)
    
#     return mocap_mks_positions

#read all mocap data from a csv file 
def read_mocap_data(file_path: str) -> list:
    """Gets the mocap markers names and positions from a file.
    
    Args:
        file_path (str): The name of the file to process.

    Returns:
        mocap_mks_positions (dict): Dictionary containing mocap markers names as keys 
                                    and their 3D positions as values.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # print(df.values[:,2:])
    positions = df.values[:,2:]

    # Extract landmarks names from the columns and remove 'Lowerbody:' prefix
    landmarks = [col.replace('Lowerbody:', '').strip() for col in df.columns[2::3]]
    
    list_of_dicts = [] #Each 
    for current_position in positions:
        
    # Initialize dictionary to store the 3D positions of the landmarks
        mocap_mks_positions = {landmark: (np.array(current_position[3*i:3*i+3]).reshape(3,1))*1e-3 for i, landmark in enumerate(landmarks)}
        list_of_dicts.append(mocap_mks_positions)
    
    return list_of_dicts

def formatting_keypoints_data(df) -> list:
    """ Format the keypoint data to be given to the IK 
    
    Args:
        data (pandas DF): Dataframe to format (initial format is Frame,Keypoint_name,X,Y,Z).

    Returns:
        jcp_positions (dict): Dictionary containing jcp names as keys 
                                    and their 3D positions as values.
    """
    # Read the CSV file into a DataFrame
    positions =  df[['X', 'Y', 'Z']].to_numpy(dtype=np.float64)

    # Extract landmarks names from the columns and remove 'Lowerbody:' prefix
    landmarks = df['Keypoint'].drop_duplicates().tolist()

    list_of_dicts = [] #Each 
    for jj in range(int(positions.shape[0]/len(landmarks))):
        current_position = positions[len(landmarks)*jj:len(landmarks)*(jj+1)]
        list_of_arrays = [np.array(current_position[i]).reshape(3, 1) for i in range(current_position.shape[0])]
        
        # Initialize dictionary to store the 3D positions of the landmarks
        mocap_mks_positions = dict(zip(landmarks,list_of_arrays))
        list_of_dicts.append(mocap_mks_positions)
    
    return list_of_dicts

# def write_joint_angle_results(directory_name: str, q:np.ndarray):
#     """_Write the joint angles obtained from the ik as asked by the challenge moderators_

#     Args:
#         directory_name (str): _Name of the directory to store the results_
#         q (np.ndarray): _Joint angle results_
#     """
#     dofs_names = ['Hip_Z_R', 'Hip_X_R', 'Hip_Y_R', 'Knee_Z_R', 'Ankle_Z_R', 'Ankle_X_R','Hip_X_L', 'Hip_Y_L', 'Knee_Z_L', 'Ankle_Z_L', 'Ankle_X_L']
#     for ii in range(q.shape[1]):
#         open(directory_name+'/'+dofs_names[ii]+'.csv', 'w').close() # clear the file 
#         np.savetxt(directory_name+'/'+dofs_names[ii]+'.csv', q[:,ii])

def write_joint_angle_results(directory_name: str, q: np.ndarray, type: str, task: str):
    """
    Write the joint angles obtained from the ik as asked by the challenge moderators

    Args:
        directory_name (str): Name of the directory to store the results
        q (np.ndarray): Joint angle results
        type (str): Type of task
        task (str): Specific task identifier
    """
    # List of degrees of freedom (DOFs) names
    dofs_names = [
        'FF_TX', 'FF_TY', 'FF_TZ', 'FF_Rquat0', 'FF_Rquat1', 'FF_Rquat2', 'FF_Rquat3',
        'Hip_Z_R', 'Hip_X_R', 'Hip_Y_R', 'Knee_Z_R', 'Ankle_Z_R', 'Ankle_X_R',
        'Hip_Z_L', 'Hip_X_L', 'Hip_Y_L', 'Knee_Z_L', 'Ankle_Z_L', 'Ankle_X_L'
    ]

    # Ensure the directory exists
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    # Define the output CSV file name
    csv_filename = os.path.join(directory_name, f"joint_angles_{type}_{task}.csv")

    # Write the joint angles to the CSV file
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(dofs_names)
        
        # Write the data
        for row in q:
            writer.writerow(row)

    print(f"Joint angles successfully written to {csv_filename}")


def write_markers_to_csv(markers_list, filename):
    # Get the marker names from the first dictionary
    marker_names = list(markers_list[0].keys())
    
    # Prepare the data to write
    data = []
    
    for marker_dict in markers_list:
        row = []
        for marker in marker_names:
            position = marker_dict[marker].flatten()
            row.extend(position)
        data.append(row)
    
    # Write the data to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        header = []
        for marker in marker_names:
            header.extend([f"{marker}_x", f"{marker}_y", f"{marker}_z"])
        writer.writerow(header)
        
        # Write the data rows
        for row in data:
            writer.writerow(row)


def remove_nans_from_list_of_dicts(list_of_dicts):
    for i in range(len(list_of_dicts)):
        for key, value in list_of_dicts[i].items():
            if np.isnan(value[0][0]):
                list_of_dicts[i][key] = list_of_dicts[i-1][key]
    return list_of_dicts


def plot_joint_angle_results(directory_name:str):
    """_Plots the corresponding joint angles_

    Args:
        directory_name (str): _Directory name where the data to plot are stored_
    """
    dofs_names = ['Hip_Z_R', 'Hip_X_R', 'Hip_Y_R', 'Knee_Z_R', 'Ankle_Z_R', 'Ankle_X_R','Hip_X_L', 'Hip_Y_L', 'Knee_Z_L', 'Ankle_Z_L', 'Ankle_X_L']
    for name in dofs_names: 
        q_i = np.loadtxt(directory_name+'/'+name+'.csv')
        plt.plot(q_i)
        plt.title(name)
        plt.show()

def read_joint_angles(directory_name:str)->np.ndarray:
    dofs_names = ['Hip_Z_R', 'Hip_X_R', 'Hip_Y_R', 'Knee_Z_R', 'Ankle_Z_R', 'Ankle_X_R','Hip_X_L', 'Hip_Y_L', 'Knee_Z_L', 'Ankle_Z_L', 'Ankle_X_L']
    q=[]
    for name in dofs_names: 
        q_i = np.loadtxt(directory_name+'/'+name+'.csv')
        q.append(q_i)
    
    q=np.array(q)
    return q

def read_joint_positions(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Read the joint names from the first line
    joint_names = lines[0].strip().split(',')

    # Initialize a dictionary with joint names as keys and empty lists as values
    joint_positions = {joint: [] for joint in joint_names}

    # Process each subsequent line
    for line in lines[1:]:
        positions = list(map(float, line.strip().split(',')))
        for i, joint in enumerate(joint_names):
            # Convert positions to NumPy arrays of 3D coordinates
            joint_positions[joint].append(np.array(positions[i*3:(i+1)*3]))

    num_samples = len(lines) - 1
    return joint_positions, num_samples

# Lecture des données du fichier resultat MMPose
def read_mmpose_file(nom_fichier):
    donnees = []
    with open(nom_fichier, 'r') as f:
        for ligne in f:
            ligne = ligne.strip().split(',')  # Séparer les valeurs par virgule
            donnees.append([float(valeur) for valeur in ligne[2:]])  # Convertir les valeurs en float, en excluant le num_sample
    # print('donnees=',donnees)
    return donnees

def read_mmpose_scores(liste_fichiers):
    all_scores= []
    for f in liste_fichiers :
        data= np.loadtxt(f, delimiter=',')
        all_scores.append(data[:, 1])
    return np.array(all_scores).transpose().tolist()

def get_cams_params_challenge()->dict:
    donnees = {   
        "26578": {
            "mtx" : np.array([[ 1677.425415046875, 0.0, 537.260986328125,], [ 0.0, 1677.491943359375, 959.772338875,], [ 0.0, 0.0, 1.0,],]),
            "dist" : [ -0.000715875, 0.002123484375, 4.828125e-06, 7.15625e-06,],
            "rotation" : [ -0.21337719060501195, 2.5532219190152965, -1.6205092826416467,],
            "translation" : [ 1.6091977643464546, 1.15829414576161, 3.032840223956974,],
        },

        "26585": {
            "rotation": [1.0597472109795532, -1.9011820504536852, 1.1957319917079565],
            "translation": [0.9595297628274925, 1.0464733874997356, 2.270212894656588],
            "dist" : [ -0.000745328125, 0.002053671875, 1.921875e-06, -5.140625e-06,],
            "mtx": np.array([[1674.0107421875, 0.00000000e+00, 534.026550296875],[0.00000000e+00, 1673.7362060625, 982.399719234375],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        },

        "26587": {
            "rotation": [1.677775501516754, -1.029276831328994, 0.6842023176393756],
            "translation": [-0.5569369630815385, 1.2934348024206597, 1.991617525249041],
            "dist" : [ -0.000744265625, 0.002104171875, 4.328125e-06, 3.109375e-06,],
            "mtx": np.array([ [ 1675.204223640625, 0.0, 540.106201171875,], [ 0.0, 1675.234985359375, 955.9697265625,], [ 0.0, 0.0, 1.0,],])
        },

        "26579": {
            "rotation": [ 1.473263729647568, -1.3161084173646604, 0.8079167854373644,],
            "translation": [ 0.057125196677030775, 1.3404023742147497, 2.355331127576366,],
            "dist" : [ -0.000690171875, 0.00212715625, 1.7359375e-05, -6.96875e-06,],
            "mtx": np.array([ [ 1680.458740234375, 0.0, 542.9306640625,], [ 0.0, 1680.66064453125, 923.3432006875,], [ 0.0, 0.0, 1.0,],])
        },

        "26580": {
            "rotation": [ 0.8298688840152564, 1.8694631842579277, -1.5617258826886953,],
            "translation": [ 0.9904842020364727, 1.0246693922638055, 4.957845631470926,],
            "dist" : [ -0.00071015625, 0.0021813125, 9.609375e-06, -6.109375e-06,],
            "mtx": np.array([ [ 1685.629028328125, 0.0, 510.6878356875,], [ 0.0, 1685.509521484375, 969.57385253125,], [ 0.0, 0.0, 1.0,],])
        },

        "26582": {
            "rotation": [ 1.9667687809278538, 0.43733173855510793, -0.7311269859165496,],
            "translation": [ -1.5197560224152755, 0.8110430837593252, 4.454761186711195,],
            "dist" : [ -0.000721609375, 0.002187234375, 9.5e-06, 1.078125e-05,],
            "mtx": np.array([ [ 1681.244873046875, 0.0, 555.02630615625,], [ 0.0, 1681.075439453125, 948.137390140625,], [ 0.0, 0.0, 1.0,],])
        },

        "26583": {
            "rotation": [ 1.2380223927668794, 1.2806411592382023, -1.098415193550419,],
            "translation": [ -0.2799787691771713, 0.4419235311792159, 5.345299193754642,],
            "dist" : [ -0.000747609375, 0.00213728125, 1.51875e-05, 4.546875e-06,],
            "mtx": np.array([ [ 1673.79724121875, 0.0, 534.494567875,], [ 0.0, 1673.729614265625, 956.774108890625,], [ 0.0, 0.0, 1.0,],])
        },

        "26584": {
            "rotation": [ 2.0458341465177643, 0.01911893903238088, -0.011457679397024361,],
            "translation": [ -1.6433009675366304, 0.9777773776650169, 2.54863840307948,],
            "dist" : [ -0.00071109375, 0.002051796875, 2.03125e-07, -2.94375e-05,],
            "mtx": np.array([ [ 1674.07165528125, 0.0, 569.56646728125,], [ 0.0, 1673.930786140625, 936.65380859375,], [ 0.0, 0.0, 1.0,],])
        },

        "26586": {
            "rotation": [ 0.7993494245198899, -2.2782754140077803, 1.252697486024887,],
            "translation": [ 1.4363111933696429, 0.627047250057601, 2.828701383630391,],
            "dist" : [ -0.000729765625, 0.00215034375, -8.46875e-06, -8.078125e-06,],
            "mtx": np.array([ [ 1681.598388671875, 0.0, 513.20837403125,], [ 0.0, 1681.509887703125, 964.994873046875,], [ 0.0, 0.0, 1.0,],]),
            # "projection" : np.array
        },
    }
    return donnees

