import os
import time
import numpy as np
import pandas as pd
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
import sys

# Add additional paths so that human_model, ik, and viewer modules are found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Import required functions and classes.
from human_model.model_utils import construct_segments_frames, get_segments_mks_dict
from human_model.pin_model import build_model
from ik.ik import RT_IK
from viewer.gv_viewer import Rquat
from viewer.gv_viewer import place, gv_init, Rquat, add_frames

# -----------------------------
# Helper function to update sphere/axis pose.
def place(viz, name, M):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    viz.viewer.gui.refresh()

# -----------------------------
# Function to load marker data from CSV
def load_marker_data(csv_file, marker_names):
    """
    Load marker data from a CSV file and return a list of dictionaries.
    
    The CSV is expected to have a header row where each marker's coordinates 
    are stored in columns with names like '<marker>_x', '<marker>_y', and '<marker>_z'.
    Any markers not found in the file are skipped.
    """
    data = pd.read_csv(csv_file)
    # If the first column is a time column, drop it.
    if "time" in data.columns[0].lower():
        data = data.iloc[:, 1:]
    
    # Build a dictionary mapping marker name -> list of coordinate columns.
    marker_columns = {}
    for marker in marker_names:
        cols = [col for col in data.columns if col.lower().startswith(marker.lower() + "_")]
        if len(cols) < 3:
            print(f"Warning: Could not find 3 coordinate columns for marker '{marker}'.")
        else:
            marker_columns[marker] = sorted(cols)[:3]
    
    num_frames = data.shape[0]
    markers_list = []
    for i in range(num_frames):
        frame_dict = {}
        for marker, cols in marker_columns.items():
            try:
                x = data.iloc[i][cols[0]]
                y = data.iloc[i][cols[1]]
                z = data.iloc[i][cols[2]]
                frame_dict[marker] = np.array([x, y, z])
            except Exception as e:
                print(f"Error extracting marker '{marker}' in frame {i}: {e}")
        markers_list.append(frame_dict)
    
    return markers_list

# -----------------------------
# Function to load vertices data from CSV.
def load_vertices_data(csv_file):
    """
    Load vertices data from a CSV file.
    
    The CSV is expected to have a "Frame" column, and then columns:
      v_0_x, v_0_y, v_0_z, v_1_x, v_1_y, v_1_z, ..., etc.
    
    Returns:
      vertices: A numpy array of shape (num_frames, num_vertices, 3)
    """
    df = pd.read_csv(csv_file)
    if 'Frame' in df.columns:
        df = df.drop(columns=['Frame'])
    num_cols = df.shape[1]
    num_vertices = num_cols // 3
    vertices = df.to_numpy().reshape(-1, num_vertices, 3)
    return vertices

# -----------------------------
# Function to animate markers, vertices, and segments in Gepetto Viewer.
def animate_all(viz, markers_list, vertices, marker_names,q, M_model_list, with_vertices=False, vertex_factor=1, delay=0.03):
    """
    Animate markers, vertices, and body segments in Gepetto Viewer.
    
    For each frame:
      - Marker spheres are updated using marker positions.
      - Vertices (displayed as small spheres) are updated.
      - Segment frames are computed from marker positions and coordinate axes are updated.
    """
    num_frames, num_vertices, _ = vertices.shape
    selected_indices = np.arange(0, num_vertices, vertex_factor)
    
    if with_vertices:
        # Create the initial vertex spheres.
        for idx in selected_indices:
            viz.viewer.gui.addSphere(f'world/v_{idx}', 0.005, [0, 0, 1, 0.6])
    
    # Add marker spheres.
    for marker in marker_names:
        viz.viewer.gui.addSphere('world/' + marker, 0.008, [1, 0, 0, 1])
        viz.viewer.gui.addSphere('world/'+marker+"_m",0.008,[0,1,0,1])
        
    
    # Compute segments from the first frame and add coordinate axes for each segment.
    initial_segments = construct_segments_frames(markers_list[0], with_head=False)
    for seg_name in initial_segments.keys():
        viz.viewer.gui.addXYZaxis('world/' + seg_name, [255, 0, 0, 1], 0.01, 0.08)
    
    
    rmse_list = []
    # Animation loop.
    for i in range(num_frames):
        viz.display(q[i])
        # Update markers.
        for marker in marker_names:
            if marker in markers_list[i]:
                pos = markers_list[i][marker]
                T = pin.SE3(np.eye(3), pos.reshape(3, 1))
                place(viz, 'world/' + marker, T)
            else:
                print(f"Warning: Marker '{marker}' missing in frame {i}.")
        
        if M_model_list is not None and i < len(M_model_list):
            model_dict = M_model_list[i]
            for marker, M_model in model_dict.items():
                place(viz, 'world/' + marker + "_m", M_model)
    
        
        if with_vertices:
            # Update vertices.
            for idx in selected_indices:
                pos = vertices[i, idx, :]
                T = pin.SE3(np.eye(3), pos.reshape(3, 1))
                place(viz, f'world/v_{idx}', T)
        
        # Compute segment frames from marker data and update their axes.
        segments = construct_segments_frames(markers_list[i], with_head=False)
        for seg_name, pose in segments.items():
            M = pin.SE3(pose[:3, :3], pose[:3, 3].reshape(3, 1))
            place(viz, 'world/' + seg_name, M)
        

         # Compute RMSE for the current frame.
        squared_errors = []
        if M_model_list is not None and i < len(M_model_list):
            model_dict = M_model_list[i]
            for marker in marker_names:
                if marker in markers_list[i] and marker in model_dict:
                    # Experimental marker position.
                    pos = np.asarray(markers_list[i][marker]).flatten()
                    # Predicted marker position from model.
                    pred = np.asarray(model_dict[marker].translation).flatten()
                    squared_error = np.sum((pos - pred)**2)
                    squared_errors.append(squared_error)
            if squared_errors:
                frame_rmse = np.sqrt(np.mean(squared_errors))
                print(frame_rmse)
                rmse_list.append(frame_rmse)
                # Optionally, you can print the frame RMSE
                # print(f"Frame {i} RMSE: {frame_rmse}")

        time.sleep(delay)
    # Compute average RMSE over all frames.
    if rmse_list:
        average_rmse = np.mean(rmse_list)
        print("Average RMSE:", average_rmse)
    else:
        average_rmse = None
        print("No RMSE computed.")
    
    return average_rmse



def calculate_ik(human_model, markers_list):
    dt = 1/40  # Time step for IK
    # Use the same marker names as keys to track.
    keys_to_track_list = marker_names

    q = pin.neutral(human_model)
    ik_class = RT_IK(human_model, start_sample_dict, q, keys_to_track_list, dt)

    # Warm-start the IK with a CasADi-based solution.
    q, _ = ik_class.solve_ik_sample_casadi()
    viz.display(q)
    ik_class._q0 = q
    print("Initial configuration q:", q)
    input("Press Enter to start the IK for whole trajetory...")
    
    q_list = []
    cost_list = []
    M_model_list = []
    for ii in range(start_sample, len(markers_list)):
        mks_dict = markers_list[ii]
        ik_class._dict_m = mks_dict
        # Solve IK for the current frame.
        q, cost = ik_class.solve_ik_sample_quadprog()
        pin.forwardKinematics(human_model, human_data, q)
        pin.updateFramePlacements(human_model, human_data)

        M_model_frame = {}
        for marker in markers_list[ii].keys():

            M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_list[ii][marker][0],markers_list[ii][marker][1],markers_list[ii][marker][2]]).T))
            M_model = human_data.oMf[human_model.getFrameId(marker)]
            marker_pos=  M_model.translation 
            M_model = pin.SE3(Rquat(1, 0, 0, 0), np.matrix([M_model.translation[0],M_model.translation[1],M_model.translation[2]]).T)
            M_model_frame[marker] = M_model
            # print(M_model)
            # place(viz,'world/'+marker,M)
            # viz.viewer.gui.addSphere('world/'+marker+"_m",0.008,[0,1,0,1])
            # place(viz,'world/'+marker+"_m",M_model)

        M_model_list.append(M_model_frame)
        ik_class._q0 = q 
        q_list.append(q)
        cost_list.append(cost)
    
    return q_list, cost_list,M_model_list


# -----------------------------
# Main routine.
if __name__ == '__main__':

    motion = 'lift_box_poses'
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(script_dir)
    
    #LSTM marker names
    marker_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
                    'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
                    'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
                    'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
                    'C7_study','r_lelbow_study','r_melbow_study','r_lwrist_study','r_mwrist_study',
                    'L_lelbow_study','L_melbow_study','L_lwrist_study','L_mwrist_study']
    
    # Path to CSV file with marker trajectories.
    markers_csv = os.path.join(script_dir, 'amass', 'smplh', f'{motion}_mks_lstm.csv')
    markers_list = load_marker_data(markers_csv, marker_names)

    # ----------------- INVERSE KINEMATICS PHASE ---------------------
    # For IK, we use the markers from the first frame as our starting dictionary.
    start_sample = 0
    start_sample_dict = markers_list[start_sample]
    
    # Define the folder path for the human model meshes.
    meshes_folder_path = '/home/madjel/Projects/gitpackages/rt-cosmik/meshes'
    
    # Build and calibrate the human model.
    human_model, human_geom_model, visuals_dict = build_model(start_sample_dict, meshes_folder_path, with_head=False)
    
    print("nq:", human_model.nq)

    q = np.zeros((human_model.nq, 1))
    # Initialize configuration and data.
    q = pin.neutral(human_model)
    human_data = pin.Data(human_model)
    pin.framesForwardKinematics(human_model,human_data,q)
    pin.updateFramePlacements(human_model, human_data)


    # Path to CSV file with vertices trajectories.
    vertices_csv = os.path.join(script_dir, 'amass', 'smplh', f'{motion}_vertices.csv')
    vertices = load_vertices_data(vertices_csv)
    print("Loaded vertices with shape:", vertices.shape)
    
    # Initialize Gepetto Viewer.
    viz = GepettoVisualizer(human_model, human_geom_model,human_geom_model)
    try:
        viz.initViewer()
    except ImportError as err:
        print("Error initializing viewer. Install gepetto-viewer.")
        sys.exit(0)
    try:
        viz.loadViewerModel("pinocchio")
    except AttributeError as err:
        print("Error loading viewer model. Start gepetto-viewer.")
        sys.exit(0)

    # Set viewer background colors.
    viz.viewer.gui.setBackgroundColor1('python-pinocchio', [1, 1, 1, 1])
    viz.viewer.gui.setBackgroundColor2('python-pinocchio', [1, 1, 1, 1])
    viz.display(q)

    # Set viewer background colors.
    viz.viewer.gui.setBackgroundColor1('python-pinocchio', [1, 1, 1, 1])
    viz.viewer.gui.setBackgroundColor2('python-pinocchio', [1, 1, 1, 1])
    
    q_list, cost_list,M_model_list = calculate_ik(human_model, markers_list)

    cost = 0.0
    for cost_i in cost_list:
        cost+=cost_i
        print("cost=", cost_i)
    print("average cost:", cost/len(cost_list))
    
    # Animate markers, vertices, and segments.
    animate_all(viz, markers_list, vertices, marker_names, q_list, M_model_list,vertex_factor=1, delay=0.01)
    
    # Optionally, save the computed joint angles (q_list) and marker model positions (M_model_list)
    # For example:
    # headers = [f"q{i}" for i in range(len(q_list[0]))]
    # df = pd.DataFrame(q_list, columns=headers)
    # csv_file = os.path.join(script_dir, 'process_data', 'q_cosmik_qp_modele_mocap.csv')
    # df.to_csv(csv_file, index=False)
