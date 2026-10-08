import csv
from collections import OrderedDict
import os

class CSVSaver:

    def __init__(self, save_dir, markers_header=None, joint_angles_header=None):
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)

        # Define file paths
        self.markers_path = os.path.join(self._save_dir, "markers.csv")
        self.joint_angles_path = os.path.join(self._save_dir, "joint_angles.csv")

        # Ensure headers are immutable and ordered
        self.markers_header = self._expand_xyz(markers_header)
        self.joint_angles_header = (
            list(joint_angles_header) if joint_angles_header is not None else None
        )

        self.markers_file = self.joint_angles_file = None
        self.markers_writer = self.joint_angles_writer = None

        # Initialize files and writers with error handling
        try:
            self.markers_file, self.markers_writer = self._open(
                self.markers_path, self.markers_header)
            self.joint_angles_file, self.joint_angles_writer = self._open(
                self.joint_angles_path, self.joint_angles_header)
        except IOError as e:
            self.close()
            raise RuntimeError(f"Failed to open CSV files: {e}")

    @staticmethod
    def _expand_xyz(header):
        """Expand each marker/keypoint name into its three per-axis columns."""
        if header is None:
            return None
        expanded = []
        for el in header:
            el_split = el.split('_')
            if el_split[0] == 'Frame':
                expanded.append(el)
            else:
                expanded.append(el+'_x')
                expanded.append(el+'_y')
                expanded.append(el+'_z')
        return expanded

    @staticmethod
    def _open(path, header):
        """Open one stream and write its header, or skip it when unconfigured."""
        if header is None:
            return None, None
        handle = open(path, mode='w', newline='')
        writer = csv.writer(handle)
        writer.writerow(header)
        handle.flush()
        return handle, writer

    def save_markers(self, markers_dict):
        if not isinstance(markers_dict, OrderedDict):
            raise ValueError("markers_dict must be an OrderedDict")
        if self.markers_writer is None:
            raise RuntimeError("CSVSaver was created without a markers header")
        self.markers_writer.writerow(list(markers_dict.values()))
        self.markers_file.flush()

    def save_joint_angles(self, joint_angles_dict):
        if not isinstance(joint_angles_dict, OrderedDict):
            raise ValueError("joint_angles_dict must be an OrderedDict")
        if self.joint_angles_writer is None:
            raise RuntimeError("CSVSaver was created without a joint angles header")
        self.joint_angles_writer.writerow(list(joint_angles_dict.values()))
        self.joint_angles_file.flush()

    def close(self):
        """Close all open CSV files."""
        for file in [self.markers_file, self.joint_angles_file]:
            if file and not file.closed:
                file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
