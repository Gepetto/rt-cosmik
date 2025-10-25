#!/usr/bin/env python3
# train_lstm_end2end.py
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import random
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
from tensorflow.keras.layers import LSTM
import optuna, gc
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rtcosmik.utils.read_write_utils import read_mks_data, default_mocap_mks_names, read_subject_info

# ─────────────── Args ───────────────
p = argparse.ArgumentParser(description="End-to-end LSTM training with streaming tf.data")
p.add_argument('--data-path', required=True, type=str)
p.add_argument('--pretrained-path', required=True, type=str)
p.add_argument('--body-part', choices=['upper','lower'], required=True)
p.add_argument('--add-noise', choices=['T','F'], default='T')
p.add_argument('--use-weights', choices=['T','F'], default='F')
p.add_argument('--seq-len', type=int, default=100)
p.add_argument('--batch-size', type=int, default=64)
p.add_argument('--epochs', type=int, default=300)
p.add_argument('--patience', type=int, default=10)
p.add_argument('--lr', type=float, default=5e-6)
p.add_argument('--test-size', type=int, default=2, help="# of subjects reserved for val (last N alphabetical)")
p.add_argument("--excluded-trials", type=str, default="none", help="Exclude trials from the dataset")
p.add_argument("--id", type=str, default="0", help="Experiment ID")
p.add_argument('--optuna', type=int, default=0, help='Run Optuna HPO (1=yes)')
p.add_argument('--n-trials', type=int, default=20, help='Optuna trials')

args = p.parse_args()
ADD_NOISE_TRAIN = (args.add_noise == 'T')
rotation_scheme = "max"
# ─────────────── Config derived from body part ───────────────
### Cette partie permet simplement de définir les inputs et outputs en fonction du body part
if args.body_part == "upper":
    kpts_input_lstm = ['Neck','RShoulder','LShoulder','RElbow','LElbow','RWrist','LWrist']
    mks_of_interest = [
        'r_lelbow_study','r_melbow_study','r_lwrist_study','r_mwrist_study',
        'L_lelbow_study','L_melbow_study','L_lwrist_study','L_mwrist_study'
    ]
    out_dim = len(mks_of_interest)*3
elif args.body_part == "lower":
    kpts_input_lstm = ['Neck','RShoulder','LShoulder','RHip','LHip',
                       'RKnee','LKnee','RAnkle','LAnkle','RHeel',
                       'LHeel','RSmallToe','LSmallToe','RBigToe','LBigToe']
    mks_of_interest = [
        'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
        'r_knee_study','r_mknee_study','r_ankle_study','r_mankle_study',
        'r_toe_study','r_5meta_study','r_calc_study',
        'L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
        'L_toe_study','L_calc_study','L_5meta_study',
        'r_shoulder_study','L_shoulder_study','C7_study'
    ]
    response_markers_lower = [
    'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
    'r_knee_study','r_mknee_study','r_ankle_study','r_mankle_study',
    'r_toe_study','r_5meta_study','r_calc_study',
    'L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
    'L_toe_study','L_calc_study','L_5meta_study',
    'r_shoulder_study','L_shoulder_study','C7_study',
    'r_thigh1_study','r_thigh2_study','r_thigh3_study',
    'L_thigh1_study','L_thigh2_study','L_thigh3_study',
    'r_sh1_study','r_sh2_study','r_sh3_study',
    'L_sh1_study','L_sh2_study','L_sh3_study',
    'RHJC_study','LHJC_study'
    ]

    out_dim = len(mks_of_interest)*3
else:
    raise ValueError("Unsupported body_part")

### Cette partie permet de définir les trials à exclure au cas où --excluded-trials est spécifié
if args.excluded_trials == "all":
    excluded_trials = ["static", "crouch", "crouch_object", "hitting", "hitting_sat", "jump", "lifting_fast", "lower",
             "overhead_front", "sanding",
             "sanding_sat", "sit_to_stand", "squat", "upper", "walk", "walk_front", "welding", "welding_sat"]
elif args.excluded_trials == "bugs":
    excluded_trials = ["lifting", "crouch", "crouch_object"]
elif args.excluded_trials == "none":
    excluded_trials = []
elif args.excluded_trials == "robweld": ### Par exemple ici, on exclut tous les trials sauf robot_welding pour se focus sur cette tâche
    excluded_trials = ["static", "crouch", "crouch_object", "hitting_sat", "jump", "lifting_fast", "lower",
             "overhead_front", "sanding",
             "sanding_sat", "sit_to_stand", "squat", "upper", "walk", "walk_front", "welding", "welding_sat",
             "robot_sanding", "hitting", "bolting", "bolting_sat", "lifting", "overhead"]
else:
    raise ValueError(f"Unknown value for --exclude-trials: {args.excluded_trials}")

# ─────────────── Files discovery ───────────────
### Ici on définit les sujets à mettre dans le train et val set de manière random avec le random.shuffle qui shuffle les noms des sujets
root = Path(args.data_path)
subjects = [d.name for d in root.iterdir() if d.is_dir()]
#random.shuffle(subjects)
if len(subjects) < args.test_size + 1:
    raise RuntimeError("Not enough subjects to split.")
print("subjetcs",subjects)
train_subjects = subjects[:-args.test_size]
val_subjects   =subjects[-args.test_size:]

print("val_set :", val_subjects)
print("train_subjects :", train_subjects)

### Cette fonction retourne les path des fichiers .npz qui vont être utilisés lors du learning
### yield permet de retourner les éléments un par un à chaque demande de la part de la fonction finale (ici .fit), 
### c'est ce qui correspond au data_generator 
def enumerate_trials(subject_list):
    """Yield dicts describing usable trials with paths & metadata."""
    for s in subject_list:
        print(s)
        if s == "koko":
            continue  # skip this subject
        sp = root/s
        h, w, _ = read_subject_info(sp/'info.txt')
        for trial in sorted([d.name for d in sp.iterdir() if d.is_dir()]):
            if trial in excluded_trials:
                continue
            trial_dir = sp/trial
            jcp_name  = f"{trial}_joint_center_positions_with_offsets.npz"
            mocap_name= f"{trial}_trajectories.npz"
            if not (trial_dir/mocap_name).exists() or not (trial_dir/jcp_name).exists():
                raise FileNotFoundError(f"Some files are missing in {s} : {trial}")
            yield {
                'subject': s,
                'height': h,
                'weight': w,
                'trial': trial,
                'jcp_npz': str(trial_dir/jcp_name),
                'gt_npz':  str(trial_dir/mocap_name),
            }

### on constitue donc le traning et le val set avec les fonctions enumerate_trials avec en entrée la liste des sujets des 2 sets
train_trials = list(enumerate_trials(train_subjects))
val_trials   = list(enumerate_trials(val_subjects))
if not train_trials or not val_trials:
    raise RuntimeError("No usable trials found in train/val.")

# ─────────────── Windowing helpers and data augmentation ───────────────
### Convertisseur de type
def listdicts_to_array(ld, names):
    T = len(ld); P = len(names)
    arr = np.zeros((T,P,3), dtype=np.float32)
    for i, fr in enumerate(ld):
        for j, k in enumerate(names):
            arr[i,j,:] = fr[k]
    return arr

### Cette fonction génère une matrice de rotation d'angle theta_rad autour de l'axe z (vertical)
def _yaw_rotation_matrix(theta_rad: float):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[ c, -s, 0.],
                        [ s,  c, 0.],
                        [0.,  0., 1.]], dtype=np.float32)

### Cette fonction génère une matrice de rotation 3D aléatoire uniforme d'après la méthode d'Euler
def random_rotation_matrix(seed=None):
    """
    Génère une matrice de rotation 3D aléatoire uniforme.
    
    Args:
        method: 'quaternion' (recommandé) ou 'euler' ou 'axis_angle'
        seed: graine pour la reproductibilité (optionnel)
    
    Returns:
        np.array: matrice 3x3 de rotation orthogonale
    """
    if seed is not None:
        np.random.seed(seed)

    # Méthode angles d'Euler (moins uniforme mais simple)
    roll = np.random.uniform(0, 2*np.pi)   # rotation X
    pitch = np.random.uniform(0, 2*np.pi)  # rotation Y  
    yaw = np.random.uniform(0, 2*np.pi)    # rotation Z
    
    # Matrices de rotation élémentaires
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    return (Rz @ Ry @ Rx).astype(np.float32)

def small_xy_rotation_matrix(max_deg=2.0, seed=None):
    """
    Generate a small random rotation around X and Y axes only.
    The Z axis is unchanged (no yaw rotation).

    Args:
        max_deg (float): maximum absolute rotation in degrees for X and Y.
        seed (int, optional): random seed for reproducibility.

    Returns:
        np.ndarray: 3x3 rotation matrix (float32).
    """
    if seed is not None:
        np.random.seed(seed)

    # sample small random angles in radians
    roll  = np.deg2rad(np.random.uniform(-max_deg, max_deg))  # rotation around X
    pitch = np.deg2rad(np.random.uniform(-max_deg, max_deg))  # rotation around Y

    # rotation around X
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    # rotation around Y
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # combine: R = Ry * Rx  (Y then X)
    R = Ry @ Rx
    return R.astype(np.float32)


### Cette fonction est le coeur de la gestion de la data, c'est ici que les modifs du learning et les ajouts doivent être fait
### Il s'agit du data_generator qui vient sélectionner les fenêtres de samples et applique les transformations souhaitées (normalisation, noise,
### rotations). Elle termine par yield les fenêtres d'input et ground_truth. Elle est commentée plus en détails
def trial_to_windows(
    trial, ### Un élément issu de enumerate_trials
    seq_len, ### La longueur des fenêtres d'après args.seq_len
    rotation_scheme="max", ### le type de rotation, ici seul max a été codé
    add_noise=True
):
    """Load one trial from disk, produce windowed (inp, out) samples.
       If rotation_scheme == 'det', yields n_rotations evenly-spaced yaw copies per window (like Stanford circleRotation)."""

    ### Partie loading et mise au bon format du trial en cours de processing
    # read inputs (jcp) and gt (markers)
    jcp = np.load(trial['jcp_npz'], allow_pickle=True)
    gt  = np.load(trial['gt_npz'], allow_pickle=True)

    arr_in  = jcp["data"]         # numpy array (T, n_features)
    cols_in = jcp["columns"]      # array de strings (n_features,)

    arr_gt  = gt["data"]
    cols_gt = gt["columns"]

    df_in = pd.DataFrame(arr_in, columns=cols_in)
    df_gt = pd.DataFrame(arr_gt, columns=cols_gt)

    k_list, _ = read_mks_data(df_in, converter=1.0)        # includes 'midHip'
    m_list, _ = read_mks_data(df_gt, converter=1000)

    # Build arrays
    kpts_arr = listdicts_to_array(k_list, kpts_input_lstm)         # [T, Pin, 3]
    gt_arr   = listdicts_to_array(m_list, default_mocap_mks_names) # [T, Pall, 3]

    # mid-hip reference
    ### On récupère les midhip mocap et hpe pour le recentrage des jcp hpe et des mks mocap
    mid = np.zeros((len(k_list), 3), dtype=np.float32)
    for i, fr in enumerate(k_list):
        mid[i] = fr['midHip']

    # select GT markers of interest
    sel = [default_mocap_mks_names.index(m) for m in mks_of_interest]
    gt_sel = gt_arr[:, sel, :]  # [T, Pout, 3]

    ### On récupère la hauteur h et le poids w du sujet qui sont contenus dans trial renvoyé par enumerate_trials
    h = float(trial['height']); w = float(trial['weight'] or 0.0)
    inv_h = 1.0 / h

    ### On calcule le nombre de fenêtres d'après args.seq_len et la longueur totale du trial en samples (comme fenêtre glissante 
    ### on a len - taille_fenêtre + 1 fenêtres dans un trial)
    Ttot = kpts_arr.shape[0]
    n_w  = max(Ttot - seq_len + 1, 0)

    ### Boucle qui génère les fenêtres d'input et ground_truth du trial et les yield
    for start in range(n_w):
        ### end = indice de fin de la fenêtre en cours
        end  = start + seq_len
        ### kbuf = fenêtre de kpts de jcp_mocap
        kbuf = kpts_arr[start:end]
        ### ybuf = fenêtre de ground_truth correspondante
        ybuf = gt_sel[start:end]           # [L, Pout, 3]
        ### ref = midhip du mocap correspondant
        ref  = mid[start:end]              # [L, 3]

        ### recentrage input et ground_truth autour de leurs midhip respectifs
        din = kbuf - ref[:, None, :]
        dout = ybuf - ref[:, None, :]

        ### Normalisation de input et gt par la taille du sujet
        din = din * inv_h
        dout = dout * inv_h

        ### Data augmentation si rotation_scheme == "max" (en gros pour le train set actuellement)
        if rotation_scheme == "max":
            ### Je fais 9 rotations, identité sans rotation, 6 autour de z et 2 random
            for i in range(9):
                if i == 0:
                    R = np.eye(3) #no rot
                ### les 2 random
                elif i == 6 or i == 7:
                    R = small_xy_rotation_matrix() 
                ### les 6 autour de z
                else:
                    ### theta = angle random de rotation autour de z entre -60 et 60 degres (toutes les rot possibles)
                    theta = np.deg2rad(np.random.uniform(-60.0, 60.0))
                    ### Génération de la matrice de rotation
                    R = _yaw_rotation_matrix(theta).T
                ### On applique la rotation sur les inputs et ground_truth
                din_r = (din.reshape(-1,3)  @ R).reshape(seq_len, -1, 3)
                dout_r = (dout.reshape(-1,3) @ R).reshape(seq_len, -1, 3)

                # 1) build CLEAN (no-noise) tensors with consistent shapes
                inp_clean = din_r.reshape(seq_len, -1)  # [L, Pin*3]
                hw = np.concatenate([
                        np.full((seq_len,1), h, dtype=np.float32),
                        np.full((seq_len,1), w, dtype=np.float32)
                    ], axis=1)                          # [L, 2]
                inp_clean = np.concatenate([inp_clean, hw], axis=1).astype(np.float32)  # [L, Pin*3 + 2]
                out_flat  = dout_r.reshape(seq_len, -1).astype(np.float32)              # [L, Pout*3]

                # yield CLEAN sample (always)
                yield inp_clean, out_flat

                # 2) optionally yield a NOISY version (same shapes)
                if add_noise:
                    #print("noiiiiiiiiiiiiiiiiiiiiiiiiise train set")
                    din_noisy = (din_r + np.random.normal(0.0, 0.018, din_r.shape).astype(np.float32))
                    inp_noisy = din_noisy.reshape(seq_len, -1)
                    inp_noisy = np.concatenate([inp_noisy, hw], axis=1).astype(np.float32)
                    yield inp_noisy, out_flat  # out stays noise-free

        ### Partie sans data augmentation pour le set de validation
        else:
            R = np.eye(3)
            din_r = (din.reshape(-1,3)  @ R).reshape(seq_len, -1, 3)
            dout_r = (dout.reshape(-1,3) @ R).reshape(seq_len, -1, 3)

            # optional Gaussian noise on features only (XYZ)
            din_noisy = din_r
            if add_noise:
                #print("noiiiiiiiiiiiiiiiiiiiiiise val set")
                din_noisy = din_noisy + np.random.normal(0.0, 0.018, din_noisy.shape).astype(np.float32)

            # flatten & append height/weight (not rotated)
            inp = din_noisy.reshape(seq_len, -1)
            hw  = np.concatenate([
                    np.full((seq_len,1), h, dtype=np.float32),
                    np.full((seq_len,1), w, dtype=np.float32)
                ], axis=1)
            inp = np.concatenate([inp, hw], axis=1)  # [L, Pin*3 + 2]
            out = dout_r.reshape(seq_len, -1)        # [L, Pout*3]
            yield inp.astype(np.float32), out.astype(np.float32)


# Spec (needed for tf.data.from_generator)
feature_dim = len(kpts_input_lstm)*3 + 2

### spec des tailles de output et input pour le data_generator
output_signature = (
    tf.TensorSpec(shape=(args.seq_len, feature_dim), dtype=tf.float32),
    tf.TensorSpec(shape=(args.seq_len, out_dim),     dtype=tf.float32)
)

### Le data generator est déroulé dans cette fonction, lorsqu'elle est appelée, elle génère les fenêtres d'input et ground_truth
### quand elles sont demandées par .fit
def make_dataset(trials, seq_len, batch, shuffle_windows=True, rotation_scheme="max", add_noise=True):
    def gen():
        # interleave trials deterministically; you can randomize order here
        for t in trials:
            for inp, out in trial_to_windows(
                t, seq_len, rotation_scheme=rotation_scheme, add_noise=add_noise
            ):
                yield inp, out

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle_windows:
        ds = ds.shuffle(buffer_size=max(8192, 2048 * 2), reshuffle_each_iteration=True)

    ds = ds.batch(batch, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds

pretrained_dir = Path(args.pretrained_path) / f"v0.3_{args.body_part}"
pathMean = os.path.join(pretrained_dir, "mean.npy")
pathSTD  = os.path.join(pretrained_dir, "std.npy")

mu_acc = tf.zeros([feature_dim], tf.float32)
m2_acc = tf.zeros([feature_dim], tf.float32)
count  = 0.0
n_batches = 0

# --- Remplacement par loading ---
if os.path.isfile(pathMean):
    mean_train = np.load(pathMean, allow_pickle=True)
    mu_acc = tf.convert_to_tensor(mean_train, dtype=tf.float32)

if os.path.isfile(pathSTD):
    std_train = np.load(pathSTD, allow_pickle=True)
    m2_acc = tf.convert_to_tensor(std_train, dtype=tf.float32)

# --- Pour compatibilité avec le reste du code ---
mean_train = mu_acc.numpy()
std_train  = m2_acc.numpy()

# map normalization using captured constants
### Format Tensorflow
mt = tf.constant(mean_train, dtype=tf.float32)
st = tf.constant(std_train,  dtype=tf.float32)

### Fonction qui normalise les inputs par mean et std mais ne touche pas à la ground truth
def normalize_xy(x, y):
    x = (x - mt) / st
    return x, y

### génération du train set avec normalisation on the fly
train_ds = make_dataset(
    train_trials, args.seq_len, batch=args.batch_size,
    shuffle_windows=True,
    rotation_scheme=rotation_scheme,
    add_noise=ADD_NOISE_TRAIN
).map(normalize_xy, num_parallel_calls=tf.data.AUTOTUNE)

### génération du val set avec normalisation on the fly, pas de data augmentation pour le set de validation
val_ds = make_dataset(
    val_trials, args.seq_len, batch=args.batch_size,
    shuffle_windows=False,
    rotation_scheme='off',                # keep val clean (their practice)
    add_noise=False
).map(normalize_xy, num_parallel_calls=tf.data.AUTOTUNE)


##############################" loss ################################"""
# Optional weighted loss for lower body
response_mks_lower = mks_of_interest
### définition des poids des markers pour la loss
marker_weights = {"r_toe_study":2,"r_5meta_study":2,"r_calc_study":2,"L_toe_study":2,"L_5meta_study":2,"L_calc_study":2}

### Fonction qui construit le vecteur des poids des markers pour la loss
def build_weights_vector():
    ### Pas de poids sur upper donc on skip même si True sur upper
    if args.body_part != "lower" or args.use_weights != "T":
        return None
    w = np.ones(len(response_mks_lower)*3, dtype=np.float32)
    ### Boucle qui met les poids sur les bons indices en fonction des mks names
    for i, m in enumerate(response_mks_lower):
        if m in marker_weights:
            w[i*3:(i+1)*3] = marker_weights[m]
    return tf.constant(w, dtype=tf.float32)

W_loss = build_weights_vector()
### On définit la loss qui prend en compte les poids s'il y en a, il s'agit de la mse weighted
def weighted_l2(weights):
    def loss(y_true, y_pred):
        if weights is None:
            return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        sq = tf.square(y_true - y_pred)
        sq = sq * weights  # broadcast on last dim
        # sq = sq + tf.reduce_mean(tf.square(y_pred))*1e-3
        return tf.reduce_mean(sq, axis=-1)
    return loss

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def _last_lstm_name(keras_model):
    # find last LSTM layer name 
    lstm_layers = [l.name for l in keras_model.layers if isinstance(l, LSTM)]
    return lstm_layers[-1] if lstm_layers else None

def apply_freeze_strategy(model, strategy, head_prefix='time_distributed'):
    """Sets .trainable flags according to strategy: none | head | head+last"""
    if strategy == "none":
        for l in model.layers: l.trainable = True
        return

    # default: freeze everything
    for l in model.layers: l.trainable = False

    if strategy in ("head", "head+last"):
        # unfreeze final time_distributed head(s)
        for l in model.layers:
            if l.name.startswith(head_prefix):
                l.trainable = True

    if strategy == "head+last":
        # unfreeze the last LSTM block
        lname = _last_lstm_name(model)
        if lname is not None:
            model.get_layer(lname).trainable = True

# Build a fresh base (pretrained) from JSON/H5 (no SelectFeatures inside)
def build_pretrained_base(pretrained_dir):
    with open(pretrained_dir/"model.json", "r") as f:
        base = model_from_json(f.read())
    base.load_weights(str(pretrained_dir/"weights.h5"))
    return base

@keras.utils.register_keras_serializable(package="pose")
class SelectFeatures(keras.layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self._indices_list = list(indices)
        self.indices = tf.constant(self._indices_list, dtype=tf.int32)
    def call(self, x):
        return tf.gather(x, self.indices, axis=-1)
    def get_config(self):
        return {"indices": self._indices_list, **super().get_config()}

def wrap_lower_with_selector(base, feat_indices):
    out = SelectFeatures(feat_indices, name="lower_body")(base.output)
    return keras.Model(inputs=base.input, outputs=out, name="lower_model_select")

def build_feat_indices_lower():
    marker_idx = {m: i for i, m in enumerate(response_markers_lower)}
    missing = [m for m in mks_of_interest if m not in marker_idx]
    assert not missing, f"Missing marker names: {missing}"
    feat_indices = []
    for m in mks_of_interest:
        i = marker_idx[m]
        feat_indices += [i*3 + d for d in (0,1,2)]
    return feat_indices



###load pretrained model for evaluation to compare before and after finetuning
base_for_eval = build_pretrained_base(pretrained_dir)
optimizer_eval=Adam(args.lr)

#base_for_eval
if args.body_part == "lower":
    marker_idx = {m: i for i, m in enumerate(response_markers_lower)}  # 33 names -> idx
    feat_indices = []
    for m in mks_of_interest:            # your 21 marker names
        i = marker_idx[m]
        feat_indices += [i*3 + d for d in (0,1,2)]   # expand to x,y,z

    idx_tf = tf.constant(feat_indices, dtype=tf.int32)
    y21 = tf.keras.layers.Lambda(lambda x: tf.gather(x, idx_tf, axis=-1),
                                name="lower_body_eval")(base_for_eval.output)
    model_lower_21 = tf.keras.Model(inputs=base_for_eval.input, outputs=y21)
    ####
    model_lower_21.compile(optimizer=optimizer_eval, loss=weighted_l2(W_loss),metrics=[rmse])
    print("=== Baseline (pretrained base-only) BEFORE fine-tuning ===")
    print("train set")
    model_lower_21.evaluate(train_ds)
    print("val set")
    model_lower_21.evaluate(val_ds)
else : 
    base_for_eval.compile(optimizer=optimizer_eval, loss=weighted_l2(W_loss),metrics=[rmse])
    base_for_eval.summary()
    print("=== Baseline (pretrained base-only) BEFORE fine-tuning ===")
    print("train set")
    base_for_eval.evaluate(train_ds)
    print("val set")
    base_for_eval.evaluate(val_ds)


### Cette classe permet de print le lr à chaque epoch, c'est utile pour voir si le learning évolue bien quand on met un scheduler
class LRLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            # Méthode qui marche avec CosineDecay et LR constants
            current_lr = self.model.optimizer.learning_rate
            
            if callable(current_lr):
                # Pour les schedulers comme CosineDecay
                step = self.model.optimizer.iterations
                lr = float(current_lr(step).numpy())
            else:
                # Pour les LR constants
                lr = float(tf.keras.backend.get_value(current_lr))
                
            print(f"[INFO] Epoch {epoch+1}: lr = {lr:.2e}")
            
            # Ajoute aussi aux logs pour les autres callbacks
            logs = logs or {}
            logs['lr'] = lr
            
        except Exception as e:
            print(f"[WARNING] Could not retrieve learning rate at epoch {epoch+1}: {e}")

def objective(trial: optuna.Trial):
    gc.collect(); tf.keras.backend.clear_session()

    # --- search space ---
   # lr = trial.suggest_loguniform("lr", 3e-6, 3e-4)
    lr = trial.suggest_float("lr", 3e-6, 3e-4, log=True)
    freeze = trial.suggest_categorical("freeze", ["none", "head", "head+last"])

    # --- build fresh model from pretrained every trial ---
    base = build_pretrained_base(pretrained_dir)

    if args.body_part == "lower":
        feat_indices = build_feat_indices_lower()
        model_t = wrap_lower_with_selector(base, feat_indices)
    else:
        model_t = base  # upper: no selector

    # apply freeze & compile
    apply_freeze_strategy(model_t, freeze)
    opt = Adam(learning_rate=lr)
    model_t.compile(optimizer=opt, loss=weighted_l2(W_loss), metrics=[rmse])

    trial_prefix   = pretrained_dir / f"optuna_trial_{trial.number}"
    ckpt_weights   = trial_prefix.with_suffix(".best.weights.h5")     # best-on-val weights only
    arch_json_path = trial_prefix.with_suffix(".arch.json")         

    # save architecture ONCE (before training)
    with open(arch_json_path, "w") as f:
        f.write(model_t.to_json())

    # callbacks
    ckpt_path = pretrained_dir / f"optuna_trial_best.keras"
    cbs = [
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=0),
        ModelCheckpoint(str(ckpt_path), monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=0),
        optuna.integration.TFKerasPruningCallback(trial, monitor="val_loss"),
    ]
    cbs.append(LRLogger())

    ##evaluate modele before finetuning 
    print("===OBJECTIVE FUNCTION ------BEFORE fine-tuning:")
    print("train set")
    model_t.evaluate(train_ds)
    print("val set")
    model_t.evaluate(val_ds)
    # train
    history = model_t.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cbs,
        verbose=2
    )
    model_t.load_weights(str(ckpt_weights))
    final_w = pretrained_dir / f"weights_finetuned_offset_{args.id}_{trial.number}.h5"
    model_t.save_weights(str(final_w))

    # record paths for convenience
    trial.set_user_attr("ckpt_weights", str(ckpt_weights))
    trial.set_user_attr("arch_json",    str(arch_json_path))

    # evaluate
    print("=== ===OBJECTIVE FUNCTION ------ model AFTER fine-tuning:")
    print("train set")
    model_t.evaluate(train_ds)
    print("val set")
    model_t.evaluate(val_ds)

    # evaluate
    val_metrics = model_t.evaluate(val_ds, verbose=0)
    val_loss = float(val_metrics[0])
    trial.set_user_attr("val_rmse", float(val_metrics[1]) if len(val_metrics) > 1 else None)
    trial.set_user_attr("ckpt_path", str(ckpt_path))
    trial.set_user_attr("params_repr", f"lr={lr:.2e}, freeze={freeze}")

    return val_loss

if args.optuna:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== Optuna finished ===")
    print("Best val_loss:", study.best_value)
    print("Best params  :", study.best_trial.params)
    print("Attrs        :", study.best_trial.user_attrs)

    # ─────────────── SAVE BEST HYPERPARAMETERS ───────────────
    best_hparams = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "experiment_id": args.id,
        "body_part": args.body_part,
        "val_loss": study.best_value,
        "best_params": study.best_trial.params,
        "user_attrs": study.best_trial.user_attrs,
        "n_trials": args.n_trials,
    }

    json_path = pretrained_dir / f"optuna_best_params_{args.id}.json"
    with open(json_path, "w") as f:
        json.dump(best_hparams, f, indent=4)
    print(f"[Optuna] Saved best hyperparameters to {json_path}")

    # ─────────────── RETRAIN BEST CONFIG (your existing code) ───────────────
    best_lr = study.best_trial.params["lr"]
    best_freeze = study.best_trial.params["freeze"]

    base_best = build_pretrained_base(pretrained_dir)
    if args.body_part == "lower":
        feat_indices = build_feat_indices_lower()
        model_best = wrap_lower_with_selector(base_best, feat_indices)
    else:
        model_best = base_best

    apply_freeze_strategy(model_best, best_freeze)
    model_best.compile(optimizer=Adam(best_lr), loss=weighted_l2(W_loss), metrics=[rmse])

    ckpt_best = pretrained_dir / f"best_finetuned_weights_offset_{args.id}.h5"
    callbacks_best = [
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(ckpt_best), monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
    ]
    print(f"[Optuna] Retraining best config: lr={best_lr:.2e}, freeze={best_freeze}")
    model_best.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks_best, verbose=2)
    print("=== ===AFTER TRAINING WITH BEST CONFIG ------ model AFTER fine-tuning:")
    print("train set")
    model_best.evaluate(train_ds)
    print("val set")
    model_best.evaluate(val_ds)

    final_w = pretrained_dir / f"weights_finetuned_final_offset_{args.id}.h5"
    model_best.save_weights(str(final_w))
    print(f"[Done] Saved best weights: {final_w}")
    
    model_json_path = pretrained_dir / f"model_finetuned_offset_{args.id}.json"
    with open(model_json_path, "w") as f:
        f.write(model_best.to_json())
    model_best.save(pretrained_dir / f"model_finetuned_offset_{args.id}.keras")
    sys.exit(0)

if not args.optuna:
    # build once from pretrained with user LR and no freezing
    base = build_pretrained_base(pretrained_dir)
    if args.body_part == "lower":
        feat_indices = build_feat_indices_lower()
        model = wrap_lower_with_selector(base, feat_indices)
    else:
        model = base

    # no freezing by default
    apply_freeze_strategy(model, "none")
    model.compile(optimizer=Adam(args.lr), loss=weighted_l2(W_loss), metrics=[rmse])

    ckpt_path = pretrained_dir / f"best_finetuned_weights_offset_{args.id}.h5"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(ckpt_path), monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
        LRLogger(),
    ]

    print("=== Training (no Optuna) ===")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)

    final_w = pretrained_dir / f"weights_finetuned_final_offset_{args.id}.h5"
    model.save_weights(str(final_w))
    print(f"[Done] Saved: {final_w}")
    sys.exit(0)
