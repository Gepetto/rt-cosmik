import numpy as np
import yaml


def read_subject_yaml(file_path):
    """
    Lit un fichier YAML et retourne directement id, height, weight et gender.
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    subject_id = data.get('id')
    height = data.get('height')
    weight = data.get('weight')
    gender = data.get('gender')

    return subject_id, height, weight, gender


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
