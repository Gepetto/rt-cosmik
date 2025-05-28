import pandas as pd 
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt 
import os
import csv

def set_zero_data_df(df, x=None, y=None, z=None):
    # Isolate the right ankle coordinates for frame 1
    right_ankle_frame1 = df[(df['Frame'] == 1) & (df['Keypoint'] == 'Right Ankle')]

    # Extract the X, Y, Z 
    if x is None: 
        x_ankle = right_ankle_frame1['X'].values[0]
    else :
        x_ankle = x

    if y is None: 
        y_ankle = right_ankle_frame1['Y'].values[0]
    else :
        y_ankle = y

    if z is None: 
        z_ankle = right_ankle_frame1['Z'].values[0]
    else :
        z_ankle = z

    # Subtract these coordinates from the entire dataframe
    df['X'] = df['X'] - x_ankle
    df['Y'] = df['Y'] - y_ankle
    df['Z'] = df['Z'] - z_ankle
    return df

def set_zero_data(keypoints, x, y, z):
    keypoints[:][0]-= x
    keypoints[:][1]-= y
    keypoints[:][2]-= z
    return keypoints


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

def formatting_keypoints(keypoints,keypoints_names):
    list_of_arrays = [np.array(keypoints[i][:]).reshape(3, 1) for i in range(keypoints.shape[0])]
    return dict(zip(keypoints_names, list_of_arrays))

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

def read_joint_angles_wholebody(file_path, start_sample):

    dofs_names  = ['FF_X', 'FF_Y', 'FF_Z', 'FF_quatx','FF_quaty',
                          'FF_quatz', 'FF_quatw', 'Lhip_flex_ext', 'Lhip_abd_add','Lhip_int_ext_rot','Lknee_flex_ext','Lankle_flex_ext','Lankle_abd_add',
                          'Lumbar_flex_ext', 'Lumbar_lateral_flex',
                          'thoracic_flex_ext','thoracic_lateral_flex','thoracic_rot_int_ext',
                          'Lcalvicule_x',
                          'Lshoulder_flex_ext','Lshoulder_abd_add', 'Lshoulder_int_ext_rot','Lelbow_flex_ext','Lelbow_pron_supi','Lwrist_flex_ext','Lwrist_x',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'rcalvicule_x',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot','Relbow_flex_ext', 'Relbow_pron_supi', 'Rwrist_flex_ext','Rwrist_x',
                          'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext', 'Rankle_abd_add']
    q = []
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        #Skip the header 
        header = next(csv_reader)
        
        # Read all rows into memory to count total rows
        all_rows = list(csv_reader)
        total_rows = len(all_rows)
        # print(total_rows)
        
        #  determine indices of columns
        indices = [header.index(name) for name in dofs_names if name in header]
        
        # Start reading from the start_sampleth row and stop before the last end_sample rows
        for row in all_rows[start_sample:total_rows]:
            # Extract values for the specified columns
            selected_values = [float(row[i]) for i in indices]
            q.append(selected_values)
    
    return np.array(q)

def read_specific_joint(file_path, dofs_list,start_sample):

    q = []
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        #Skip the header 
        header = next(csv_reader)
        
        # Read all rows into memory to count total rows
        all_rows = list(csv_reader)
        total_rows = len(all_rows)
        # print(total_rows)
        
        #  determine indices of columns
        missing_cols = [name for name in dofs_list if name not in header]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes : {missing_cols}")
        
        indices = [header.index(name) for name in dofs_list]
        # print("Indices des colonnes :", indices)
        
        
        # Start reading from the start_sampleth row and stop before the last end_sample rows
        for row in all_rows[start_sample:total_rows]:
            # Extract values for the specified columns
            selected_values = [float(row[i]) for i in indices]
            q.append(selected_values)
    
    return np.array(q)

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
            donnees.append([float(valeur) for valeur in ligne[1:]])  # Convertir les valeurs en float, en excluant le num_sample
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

def csv_to_dict_of_dicts(df):
    headers = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']
    result = {}
    for i, header in enumerate(headers):
        # Each header corresponds to three consecutive columns (x, y, z)
        x_col = df.iloc[:, i*3]
        y_col = df.iloc[:, i*3 + 1]
        z_col = df.iloc[:, i*3 + 2]
        
        # Store x, y, z values as lists in the dictionary
        result[header] = {
            'x': x_col.tolist(),
            'y': y_col.tolist(),
            'z': z_col.tolist()
        }
    return result

def init_csv(csv_path, first_row):
    with open(csv_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        # Write the header row
        csv_writer.writerow(first_row)

def save_3dpos_to_csv(csv_path, pos_in_world, names, frame_idx, formatted_timestamp):
    with open(csv_path, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        for jj in range(len(names)):
            # Write to CSV
            csv_writer.writerow([frame_idx, formatted_timestamp,names[jj], pos_in_world[jj][0], pos_in_world[jj][1], pos_in_world[jj][2]])

def save_q_to_csv(csv_path, q, frame_idx, formatted_timestamp):
    # Saving kinematics
    with open(csv_path, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        # Write to CSV
        csv_writer.writerow([frame_idx, formatted_timestamp]+q.tolist())  

def save_to_csv(data, output_path, header=None):
    """Save 3D keypoints to a CSV file with optional header."""
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, header=header if header is not None else False)
    print(f"Saved {len(data)} frames to {output_path}")

def read_mks_data(data_markers, start_sample=0):
    #the mks are ordered in a csv like this : "time,r.ASIS_study_x,r.ASIS_study_y,r.ASIS_study_z...."
    """    
    Parameters:
        data_markers (pd.DataFrame): The input DataFrame containing marker data.
        start_sample (int): The index of the sample to start processing from.
        time_column (str): The name of the time column in the DataFrame.
        
    Returns:
        list: A list of dictionaries where each dictionary contains markers with 3D coordinates.
        dict: A dictionary representing the markers and their 3D coordinates for the specified start_sample.
    """
    # Extract marker column names
    marker_columns = [col[:-2] for col in data_markers.columns if col.endswith("_x")]
    # print(marker_columns)
    
    # Initialize the result list
    result_markers = []
    
    # Iterate over each row in the DataFrame
    for _, row in data_markers.iterrows():
        frame_dict = {}
        for marker in marker_columns:
            x = row[f"{marker}_x"]
            y = row[f"{marker}_y"]
            z = row[f"{marker}_z"]
            frame_dict[marker] = np.array([x, y, z])  # Store as a NumPy array
        result_markers.append(frame_dict)
    
    # Get the data for the specified start_sample
    start_sample_mks = result_markers[start_sample]
    
    return result_markers, start_sample_mks

def parse_marker_csv(path_to_csv, mks_names, column_name='marker_data', delimiter=';'):
    """
    Parses a CSV containing marker data stored as a single delimited string per row.

    Args:
        path_to_csv (str): Path to the CSV file.
        mks_names (list): List of marker names.
        column_name (str): Name of the column containing the marker data. Default is 'marker_data'.
        delimiter (str): Delimiter used in the marker data string. Default is ';'.

    Returns:
        list of dict: Each element is a dictionary {marker_name: np.array([x, y, z])} for one frame.
    """
    df = pd.read_csv(path_to_csv)
    expected_values = len(mks_names) * 3

    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' column not found in CSV")

    mks_dict = []
    for row in df[column_name]:
        values = list(map(float, row.split(delimiter)))
        if len(values) != expected_values:
            raise ValueError(f"Expected {expected_values} values, got {len(values)}")

        coords = [np.array(values[i:i+3]) for i in range(0, len(values), 3)]
        frame_data = dict(zip(mks_names, coords))
        mks_dict.append(frame_data)

    return mks_dict

def marker_data_to_dataframe(df, mks_names, marker_column='marker_data', delimiter=';'):
    """
    Converts a single-column marker string data DataFrame to a wide format DataFrame.

    Parameters:
        df (pd.DataFrame): Original DataFrame with a 'marker_data' column.
        mks_names (list): List of marker names.
        marker_column (str): Name of the column with delimited marker data.
        delimiter (str): Delimiter used in the string (default ';').

    Returns:
        pd.DataFrame: A DataFrame with columns like marker_x, marker_y, marker_z.
    """
    all_data = []
    for row in df[marker_column]:
        values = list(map(float, row.split(delimiter)))
        all_data.append(values)
    
    wide_df = pd.DataFrame(all_data, columns=[
        f"{name}_{axis}" for name in mks_names for axis in ['x', 'y', 'z']
    ])
    
    return wide_df


def udp_csv_to_dataframe(csv_path, marker_names):
    """
    Preprocess a UDP CSV file into a DataFrame suitable for read_mks_data.

    Parameters:
        csv_path (str): Path to the CSV file.
        marker_names (list): List of marker base names (without _x/_y/_z).

    Returns:
        pd.DataFrame: A DataFrame with columns formatted as marker_x, marker_y, marker_z.
    """
    # 1. Open manually
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # 2. Skip the header
    lines = lines[1:]

    # 3. Prepare all rows
    all_rows = []
    for line in lines:
        # Remove newline, then split
        line = line.strip()
        if not line:
            continue  # skip empty lines
        parts = line.split(",")
        timestamp = parts[0]
        udp_values = [float(val) for val in parts[1:]]
        all_rows.append(udp_values)

    # 4. Now create a dataframe
    udp_df = pd.DataFrame(all_rows)

    # 5. Build column names
    new_columns = []
    for marker in marker_names:
        new_columns.extend([f"{marker}_x", f"{marker}_y", f"{marker}_z"])

    if udp_df.shape[1] != len(new_columns):
        raise ValueError(f"Mismatch between expected markers ({len(new_columns)}) and data columns ({udp_df.shape[1]}). Check marker list!")

    udp_df.columns = new_columns

    return udp_df
#plot markers trajectories
def plot_marker_trajectories(udp_df, marker_names):
    """
    Plot x, y, z trajectories of each marker in its own figure with 3 subplots.

    Parameters:
        udp_df (pd.DataFrame): Original marker data.
        marker_names (list): List of marker names (without _x/_y/_z).
    """
    for marker in marker_names:
        fig, axes = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
        axes_labels = ['x', 'y', 'z']

        for i, axis in enumerate(axes_labels):
            col = f"{marker}_{axis}"
            axes[i].plot(udp_df.index, udp_df[col],color='r', label='Original')
            axes[i].set_ylabel(f"{axis}-axis")
            axes[i].legend()
            axes[i].grid(True)

        axes[-1].set_xlabel("Frame")
        fig.suptitle(f"Marker: {marker}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

#plot mks trajectories to compare and get rmse.
def plot_marker_comparison(gt_data, pred_data, markers_to_plot=None, labels=('mocap', 'cosmik'), save_fig=False):
    """
    Plot and compare X, Y, Z trajectories over time for selected markers from two datasets.

    Parameters:
        gt_data (list of dict): Ground truth data [{marker_name: np.array([x, y, z])}, ...]
        pred_data (list of dict): Predicted data (same format as gt_data)
        markers_to_plot (list of str): Markers to plot. If None, plots all markers found in gt_data.
        labels (tuple): Labels for legend, e.g. ('Ground Truth', 'Prediction')
        save_fig (bool): If True, saves the figure instead of displaying.
    """
    if not gt_data or not pred_data:
        print("Error: One or both datasets are empty.")
        return

    # Determine which markers to plot
    all_markers = set()
    for frame in gt_data:
        all_markers.update(frame.keys())
    if markers_to_plot is None:
        markers_to_plot = sorted(all_markers)

    rmse_results = {}
    for marker in markers_to_plot:
        gt_x, gt_y, gt_z = [], [], []
        pred_x, pred_y, pred_z = [], [], []

        for gt_frame, pred_frame in zip(gt_data, pred_data):
            # Ground truth values
            if marker in gt_frame:
                gx, gy, gz = gt_frame[marker]
            else:
                gx, gy, gz = np.nan, np.nan, np.nan
            gt_x.append(gx)
            gt_y.append(gy)
            gt_z.append(gz)

            # Predicted values
            if marker in pred_frame:
                px, py, pz = pred_frame[marker]
            else:
                px, py, pz = np.nan, np.nan, np.nan
            pred_x.append(px)
            pred_y.append(py)
            pred_z.append(pz)

        # Convert to arrays
        gt_x, gt_y, gt_z = map(np.array, (gt_x, gt_y, gt_z))
        pred_x, pred_y, pred_z = map(np.array, (pred_x, pred_y, pred_z))

        # Compute RMSE (ignoring NaNs)
        rmse_x = np.sqrt(np.nanmean((gt_x - pred_x) ** 2))
        rmse_y = np.sqrt(np.nanmean((gt_y - pred_y) ** 2))
        rmse_z = np.sqrt(np.nanmean((gt_z - pred_z) ** 2))
        rmse_results[marker] = {'x': rmse_x, 'y': rmse_y, 'z': rmse_z}

        frames = np.arange(len(gt_data))
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        axs[0].plot(frames, gt_x, 'r-', label=labels[0])
        axs[0].plot(frames, pred_x, 'g--', label=labels[1])
        axs[0].set_title(f"X (RMSE: {rmse_x:.4f})")

        axs[1].plot(frames, gt_y, 'r-', label=labels[0])
        axs[1].plot(frames, pred_y, 'g--', label=labels[1])
        axs[1].set_title(f"X (RMSE: {rmse_y:.4f})")

        axs[2].plot(frames, gt_z, 'r-', label=labels[0])
        axs[2].plot(frames, pred_z, 'g--', label=labels[1])
        axs[2].set_title(f"X (RMSE: {rmse_z:.4f})")
        axs[2].set_xlabel("Frame")

        fig.suptitle(
            f"{marker}",
            fontsize=14
        )
        for ax in axs:
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_fig:
            plt.savefig(f"{marker}_comparison.png")
            plt.close()
        else:
            plt.show()


def load_transformation(file_path):
    """
    Loads the transformation parameters (R, d, s, rms) from a text file.

    Parameters:
    file_path: str
        Path to the file from which the transformation parameters will be read.

    Returns:
    R: ndarray
        Rotation matrix (3x3)
    d: ndarray
        Translation vector (3,)
    s: float
        Scale factor
    rms: float
        Root mean square fit error
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        R_start = lines.index("Rotation Matrix (R):\n") + 1
        R = np.loadtxt(lines[R_start:R_start + 3])
        d_start = lines.index("Translation Vector (d):\n") + 1
        d = np.loadtxt(lines[d_start:d_start + 1]).flatten()
        s_line = next(line for line in lines if line.startswith("Scale Factor (s):"))
        s = float(s_line.split(":")[1].strip())
        rms_line = next(line for line in lines if line.startswith("RMS Error:"))
        rms = float(rms_line.split(":")[1].strip())
    return R, d, s, rms

def transform_keypoints_list_cam0_to_mocap(keypoints_list, R_trans, d_trans):
    """Apply transformation to each frame of flattened 3D keypoints."""
    transformed_list = []

    for flat_coords in keypoints_list:
        # Convert to shape (N, 3)
        p3d_cam0 = np.array(flat_coords).reshape(-1, 3)  # (N, 3)
        # Apply transformation
        p3d_mocap = (R_trans @ p3d_cam0.T).T + d_trans  # (N, 3)
        # Flatten again
        transformed_list.append(p3d_mocap.flatten().tolist())

    return transformed_list
