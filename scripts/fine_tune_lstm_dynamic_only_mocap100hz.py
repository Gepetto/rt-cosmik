#!/usr/bin/env python3
# train_lstm_end2end.py
import os, sys, json, argparse, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import random
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.layers import TimeDistributed, Dense
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Layer, Rescaling, Multiply
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import CosineDecay
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

args = p.parse_args()
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
# train_subjects = subjects[:-args.test_size]
# val_subjects   =subjects[-args.test_size:]

train_subjects = ["Maxime","Zoe","Kahina"]
val_subjects   = ["Maxime","Zoe","Kahina"]

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
            #print("rooooooooooooooooooooooot")
            ### Je fais 8 rotations, 6 autour de z et 2 random
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

                # din_noisy = din_r
                # if add_noise:
                #     din_noisy = din_noisy + np.random.normal(0.0, 0.018, din_noisy.shape).astype(np.float32)

                # ### reshaping et ajout de height et weight au bout des samples
                # inp = din_noisy.reshape(seq_len, -1)
                # hw  = np.concatenate([
                #         np.full((seq_len,1), h, dtype=np.float32),
                #         np.full((seq_len,1), w, dtype=np.float32)
                #     ], axis=1)
                # inp = np.concatenate([inp, hw], axis=1)  # [L, Pin*3 + 2]
                # out = dout_r.reshape(seq_len, -1)        # [L, Pout*3]
                # ### yield les fenêtres d'input et ground_truth pour génération on the fly, la normalisation par mean et std est faite
                # ### au moment de la génération de la data
                # yield inp.astype(np.float32), out.astype(np.float32)

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


# ─────────────── Two-pass normalization (streaming) ───────────────
# Pass 1: compute mean/std over TRAIN only
### On fait une premère passe sur le train set pour calculer les mean et std, ce qui prend du temps au début du learning
### ca peut aller jusqu'à une heure si on prend toute la data 
train_raw = make_dataset(
    train_trials, args.seq_len, batch=256,
    shuffle_windows=False,
    rotation_scheme=rotation_scheme,
    add_noise=args.add_noise
)

# ### Fonction qui calcule les mean et std de chaque feature de l'input recentré / normalisé par la taille du sujet (pas utilisée ici)
# @tf.function
# def batch_stats(x):
#     # x: [B, L, F]; we compute per-feature mean/std over both axes
#     mu  = tf.reduce_mean(x, axis=[0,1])
#     var = tf.reduce_mean((x - mu)**2, axis=[0,1])
#     return mu, tf.sqrt(var + 1e-8)

### Initialisation des mean et std et des conteurs de batches 
n_batches = 0
mu_acc = tf.zeros([feature_dim], tf.float32)
m2_acc = tf.zeros([feature_dim], tf.float32)
count  = 0.0

### Boucle qui calcule les mean et std de chaque batch de l'input et aggrège les stats pour obtenir mean et std final à la fin de la boucle
### (10+20)/2=15 et si je rajoute 17 je peux maj la moyenne en faisant 15+(17-15)*(1/3)=17.6667 nouvelle moyenne
### ici count = 2, tot = 3, b = 1, delta = 17-15, mu_acc = 15. Ensuite même type de formule pour la variance (je n'ai pas vérifié si ça marche c'est ChatGpt)
for x_batch, _ in train_raw:
    b = tf.cast(tf.shape(x_batch)[0]*tf.shape(x_batch)[1], tf.float32)  # batch * seq_len
    xb = tf.reshape(x_batch, [-1, feature_dim])                          # collapse time
    mu_b = tf.reduce_mean(xb, axis=0)
    var_b= tf.math.reduce_variance(xb, axis=0)
    # online update (Chan)
    delta = mu_b - mu_acc
    tot   = count + b
    mu_acc = mu_acc + delta * (b/tot)
    m2_acc = m2_acc + var_b*b + (delta**2)*count*b/tot
    count  = tot
    n_batches += 1

### On récupère les mean et std finales
mean_train = mu_acc.numpy()
std_train  = np.sqrt((m2_acc.numpy() / max(count,1.0)) + 1e-8)
### recupération de la taille moyenne et de la std de la taille (printable pour vérif, si diff 
### ça peut venir du float32 au lieu de float64 pour la précision numérique, ça m'a déjà fait des très gros écarts)
### A noter qu'il s'agit de mean et std calculés à partir du nombre de fois où la height apparaît dans les samples et comme les nombres de samples
### diffèrent entre les sujets il ne s'agit pas de simplement la moyenne des heights donc vérif difficile
mean_train_height = float(mean_train[-2])
std_train_height  = float(std_train[-2])

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
    add_noise=args.add_noise
).map(normalize_xy, num_parallel_calls=tf.data.AUTOTUNE)

### génération du val set avec normalisation on the fly, pas de data augmentation pour le set de validation
val_ds = make_dataset(
    val_trials, args.seq_len, batch=args.batch_size,
    shuffle_windows=False,
    rotation_scheme='off',                # keep val clean (their practice)
    add_noise=False
).map(normalize_xy, num_parallel_calls=tf.data.AUTOTUNE)

### génération du val set de monitoring qui permet ensuite de print pred vs gt (optionnel, à enlever si on utilise pas PreddictionLogger)
val_ds_monitor = make_dataset(
    val_trials, args.seq_len, batch=args.batch_size,
    shuffle_windows=True,
    rotation_scheme='off',
    add_noise=False
).map(normalize_xy, num_parallel_calls=tf.data.AUTOTUNE)

# ─────────────── Model (reuse your pretrained JSON/weights; adjust last layer when needed) ───────────────
### Loading des weights de OpenCap
pretrained_dir = Path(args.pretrained_path) / f"v0.3_{args.body_part}"
with open(pretrained_dir/"model.json", 'r') as f:
    base = model_from_json(f.read())
base.load_weights(str(pretrained_dir/"weights.h5"))

with open(pretrained_dir/"model.json", "r") as f:
    base_for_eval = model_from_json(f.read())
base_for_eval.load_weights(str(pretrained_dir/"weights.h5"))

if args.body_part == "lower":
    y21 = tf.keras.layers.Lambda(lambda x: x[..., :21*3])(base_for_eval.output)
    model_lower_21 = Model(inputs=base_for_eval.input, outputs=y21)

### Ajout de la layer supplémentaire initialisée comme Pontonnier
weight_decay = 0.01
initializer = RandomNormal(mean=0.0, stddev=0.022)
proj = TimeDistributed(Dense(out_dim, kernel_initializer=initializer, bias_initializer='zeros', kernel_regularizer=l2(weight_decay)), name="layer_added")(base.output)
model = Model(inputs=base.input, outputs=proj)

# Optional weighted loss for lower body
response_mks_lower = [
    'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
    'r_knee_study','r_mknee_study','r_ankle_study','r_mankle_study',
    'r_toe_study','r_5meta_study','r_calc_study',
    'L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
    'L_toe_study','L_calc_study','L_5meta_study',
    'r_shoulder_study','L_shoulder_study','C7_study'
]
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

optimizer=Adam(args.lr)

model.compile(optimizer=optimizer, loss=weighted_l2(W_loss),metrics=[rmse])

model.summary()

base_for_eval.compile(optimizer=optimizer, loss=weighted_l2(W_loss),metrics=[rmse])
print("=== Baseline (pretrained base-only) BEFORE fine-tuning ===")
print("train set")
base_for_eval.evaluate(train_ds, verbose=1)
print("val set")
base_for_eval.evaluate(val_ds, verbose=1)

### On sauvegarde le modele qui correspond au config de finetune
# Save model definition that matches finetune config
model_json_path = pretrained_dir / f"model_finetuned_offset_{args.id}.json"
with open(model_json_path, "w") as f:
    f.write(model.to_json())


# ─────────────── Callbacks ───────────────
### Cette longue classe permet de save predictions vs gt, attention cela peut générer beaucoup de fichiers il vaut mieux l'enlever si pas utile
class PredictionLogger(tf.keras.callbacks.Callback):
    def __init__(self, ds, save_dir, mks_names,
                 every_n_epochs=3, seq_len=30, seed=42,
                 n_samples=50, include_height=True,
                 mean_height=None, std_height=None,
                 preds_are_div_by_height=True):
        """
        - ds: tf.data.Dataset (x,y) normalisé comme en training
        - mks_names: liste des markers (P)
        - mean_height/std_height: floats pour dénormaliser la height si elle est z-scalée.
          Si None, on considère que la height est déjà en mètres dans x.
        - preds_are_div_by_height: True si vos y/pred sont encore en '/height'
          (cas sans wrapper to_meters) → on re-multiplie par la height pour logger en mètres.
        """
        super().__init__()
        self.ds = ds
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)
        self.every = every_n_epochs
        self.seq_len = seq_len
        self.P = len(mks_names)
        self.mks_names = list(mks_names)
        self.rng = np.random.default_rng(seed)
        self.n_samples = n_samples
        self.include_height = include_height
        self.mean_h = mean_height
        self.std_h  = std_height
        self.preds_are_div_by_height = preds_are_div_by_height

    def _collect_random_samples(self):
        """
        Balaye quelques batches de ds et prélève au total n_samples séquences aléatoires.
        Retourne x_sel, y_sel : [N, L, F], [N, L, out_dim]
        """
        X_sel, Y_sel = [], []
        remaining = self.n_samples

        # on balaye un nombre raisonnable de batches jusqu'à remplir notre quota
        # (évite de matérialiser tout le dataset)
        for x_b, y_b in self.ds.take(100):
            x_b = x_b.numpy()
            y_b = y_b.numpy()
            B = x_b.shape[0]
            if B == 0:
                continue
            k = min(remaining, B)
            idxs = self.rng.choice(B, size=k, replace=False)
            X_sel.append(x_b[idxs])
            Y_sel.append(y_b[idxs])
            remaining -= k
            if remaining <= 0:
                break

        if not X_sel:
            return None, None
        x_sel = np.concatenate(X_sel, axis=0)
        y_sel = np.concatenate(Y_sel, axis=0)
        # au cas où on a dépassé (peu probable)
        if x_sel.shape[0] > self.n_samples:
            x_sel = x_sel[:self.n_samples]
            y_sel = y_sel[:self.n_samples]
        return x_sel, y_sel

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every != 0:
            return

        x_sel, y_sel = self._collect_random_samples()
        if x_sel is None:
            print("[PredictionLogger] no samples collected; skipping")
            return

        # prédictions par batch (une seule passe)
        preds = self.model.predict(x_sel, verbose=0)  # [N, L, out_dim]

        # dernière frame seulement
        t_last = self.seq_len - 1
        x_last = x_sel[:, t_last, :]     # [N, F]
        y_last = y_sel[:, t_last, :]     # [N, out_dim]
        p_last = preds[:, t_last, :]     # [N, out_dim]

        # reshape en [N, P, 3]
        try:
            y_last = y_last.reshape(-1, self.P, 3)
            p_last = p_last.reshape(-1, self.P, 3)
        except ValueError:
            print("[PredictionLogger] reshape failed; check out_dim vs P*3")
            return

        # height réelle (m)
        # - si mean/std fournis → on dénormalise: h = z*std + mean
        # - sinon on suppose déjà en mètres dans x
        h_norm = x_last[:, -2]  # [N], feature 'height' (z-scalée ou non)
        if (self.mean_h is not None) and (self.std_h is not None):
            h_real = h_norm * self.std_h + self.mean_h
        else:
            h_real = h_norm
        h_real = h_real.astype(np.float32)  # [N]

        # si vos sorties sont en '/height' (pas de wrapper to_meters), loggez en mètres :
        if self.preds_are_div_by_height:
            y_last = y_last * h_real[:, None, None]
            p_last = p_last * h_real[:, None, None]

        # construire le DataFrame (une ligne par séquence)
        rows = {}
        rows["Seq"] = np.arange(y_last.shape[0])
        if self.include_height:
            rows["height"] = h_real

        for j, name in enumerate(self.mks_names):
            for a, ax in enumerate(["x", "y", "z"]):
                rows[f"GT.{name}.{ax}"]   = y_last[:, j, a]
                rows[f"Pred.{name}.{ax}"] = p_last[:, j, a]

        # df = pd.DataFrame(rows)
        # out = self.save_dir / f"pred_epoch{epoch+1}_laststep_{self.n_samples}.csv"
        # df.to_csv(out, index=False)
        # print(f"[INFO] Saved {out}")

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

### Définition des paths où save et des callbacks comme le Earlystopping, le checkpoint, le predictionLogger et le LRLogger
ckpt_path = pretrained_dir / f"best_finetuned_weights_offset_{args.id}.h5"
callbacks = [
    EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1),
    ModelCheckpoint(str(ckpt_path), monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
]
callbacks.append(
    PredictionLogger(
        ds=val_ds_monitor,
        save_dir=pretrained_dir,             # ou un sous-dossier 'pred_logs'
        mks_names=mks_of_interest,
        every_n_epochs=3,
        seq_len=args.seq_len,
        seed=42,
        n_samples=50,
        include_height=True,
        mean_height=mean_train_height,       # passe None si height déjà en mètres
        std_height=std_train_height,
        preds_are_div_by_height=True         # False si ton modèle sort déjà des mètres
    )
)
callbacks.append(LRLogger())

# ─────────────── Save stats + Train ───────────────
### On sauvegarde les stats de mean et std pour pouvoir les utiliser à l'inférence (test)
print(f"[Info] Feature mean/std from TRAIN: mean shape {mean_train.shape}, std shape {std_train.shape}")
stats_dir = Path(args.pretrained_path) / f"v0.3_{args.body_part}" / "stats_streaming"
stats_dir.mkdir(parents=True, exist_ok=True)
np.save(stats_dir / f"mean_train_{args.id}.npy", mean_train)
np.save(stats_dir / f"std_train_{args.id}.npy",  std_train)
### La ligne qui lance le learning avec training sur train_ds et validation sur val_ds
print("=== Extended model (base + layer_added) BEFORE fine-tuning:")
print("train set")
model.evaluate(train_ds,verbose=1)
print("val set")
model.evaluate(val_ds,verbose=1)
history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
print("=== Extended model AFTER fine-tuning:")
print("train set")
model.evaluate(train_ds,verbose=1)
print("val set")
model.evaluate(val_ds,verbose=1)
# ─────────────── Save weights ───────────────
### A la fin on sauvegarde les weights et un norm_meta.json qui contient les infos de la config de finetune
final_w = pretrained_dir / f"weights_finetuned_final_offset_{args.id}.h5"
model.save_weights(str(final_w))
with open(stats_dir / f"norm_meta_{args.id}.json", "w") as f:
    json.dump({
        "feature_dim": int(feature_dim),
        "seq_len": int(args.seq_len),
        "kpts_input_lstm": kpts_input_lstm,
        "body_part": args.body_part
    }, f, indent=2)

print(f"[Done] Saved: {final_w}")
