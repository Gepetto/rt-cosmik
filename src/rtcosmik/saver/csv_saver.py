import csv
from collections import OrderedDict
import os

class CSVSaver:
    def __init__(self, save_dir, keypoints_header, markers_header, joint_angles_header):
        """Initialize CSV files and headers."""
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)

        # Define file paths
        self.keypoints_path = os.path.join(self._save_dir, "keypoints.csv")
        self.markers_path = os.path.join(self._save_dir, "markers.csv")
        self.joint_angles_path = os.path.join(self._save_dir, "joint_angles.csv")

        # Ensure headers are immutable and ordered
        self.keypoints_header = []
        for el in keypoints_header:
            el_split = el.split('_')
            if el_split[0] == 'Frame':
                self.keypoints_header.append(el)
            else : 
                self.keypoints_header.append(el+'_x')
                self.keypoints_header.append(el+'_y')
                self.keypoints_header.append(el+'_z')

        self.markers_header = []
        for el in markers_header:
            el_split = el.split('_')
            if el_split[0] == 'Frame':
                self.markers_header.append(el)
            else :
                self.markers_header.append(el+'_x')
                self.markers_header.append(el+'_y')
                self.markers_header.append(el+'_z')
                
        self.joint_angles_header = list(joint_angles_header)

        # Initialize files with error handling
        try:
            self.keypoints_file = open(self.keypoints_path, mode='w', newline='')
            self.markers_file = open(self.markers_path, mode='w', newline='')
            self.joint_angles_file = open(self.joint_angles_path, mode='w', newline='')
        except IOError as e:
            self.close()
            raise RuntimeError(f"Failed to open CSV files: {e}")

        # Initialize writers and write headers
        self.keypoints_writer = csv.writer(self.keypoints_file)
        self.keypoints_writer.writerow(self.keypoints_header)
        self.keypoints_file.flush()  # Add this line
        
        self.markers_writer = csv.writer(self.markers_file)
        self.markers_writer.writerow(self.markers_header)
        self.markers_file.flush()    # Add this line
        
        self.joint_angles_writer = csv.writer(self.joint_angles_file)
        self.joint_angles_writer.writerow(self.joint_angles_header)
        self.joint_angles_file.flush()  # Add this line

    def save_keypoints(self, keypoints_dict):
        if not isinstance(keypoints_dict, OrderedDict):
            raise ValueError("keypoints_dict must be an OrderedDict")
        self.keypoints_writer.writerow(list(keypoints_dict.values()))
        self.keypoints_file.flush()  # Ensure data is written immediately

    def save_markers(self, markers_dict):
        if not isinstance(markers_dict, OrderedDict):
            raise ValueError("markers_dict must be an OrderedDict")
        self.markers_writer.writerow(list(markers_dict.values()))
        self.markers_file.flush()

    def save_joint_angles(self, joint_angles_dict):
        if not isinstance(joint_angles_dict, OrderedDict):
            raise ValueError("joint_angles_dict must be an OrderedDict")
        self.joint_angles_writer.writerow(list(joint_angles_dict.values()))
        self.joint_angles_file.flush()

    def close(self):
        """Close all open CSV files."""
        for file in [self.keypoints_file, self.markers_file, self.joint_angles_file]:
            if file and not file.closed:
                file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()