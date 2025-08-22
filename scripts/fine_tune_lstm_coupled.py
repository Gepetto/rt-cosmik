#!/usr/bin/env python3
# train_joint_lstm_with_coupling.py
import os, sys, json, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import TimeDistributed, Dense, Input, Lambda
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# project utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rtcosmik.utils.read_write_utils import read_mks_data, default_mocap_mks_names

# ──────────────────────────────── Args ────────────────────────────────
p = argparse.ArgumentParser(description="Joint upper+lower LSTM training with shoulder–arm proximity coupling")
p.add_argument('--data-path', required=True, type=str)

# pretrained heads
p.add_argument('--pretrained-path-upper', required=True, type=str, help="Folder containing v0.3_upper/model.json & weights.h5")
p.add_argument('--pretrained-path-lower', required=True, type=str, help="Folder containing v0.3_lower/model.json & weights.h5")

# training
p.add_argument('--seq-len', type=int, default=30)
p.add_argument('--batch-size', type=int, default=64)
p.add_argument('--epochs', type=int, default=500)
p.add_argument('--patience', type=int, default=8)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--weight-decay', type=float, default=1e-2)

# sets & reproducibility
p.add_argument('--test-size', type=int, default=2, help="# subjects reserved for val (last N alphabetical)")
p.add_argument('--seed', type=int, default=42)

# augmentation
p.add_argument('--add-noise', choices=['T','F'], default='F')
p.add_argument('--rotation-scheme', choices=['off','prob','det'], default='off')
p.add_argument('--n-rotations', type=int, default=1, help="When det: emit this many evenly-spaced yaw copies per window")
p.add_argument('--rot-prob', type=float, default=0.0, help='Prob to apply a random yaw per window (prob scheme)')
p.add_argument('--rot-max-deg', type=float, default=30.0)
p.add_argument('--up-axis', choices=['y','z'], default='y', help='Your data vertical axis')

# fine-tune & weighting
p.add_argument('--fine-tune', choices=['T','F'], default='F', help="Freeze all but last projection layer(s)")
p.add_argument('--add-layer-upper', choices=['T','F'], default='F')
p.add_argument('--add-layer-lower', choices=['T','F'], default='F')
p.add_argument('--use-weights-lower', choices=['T','F'], default='F')

# coupling
p.add_argument('--lambda-couple', type=float, default=0.10, help="Weight of coupling loss")
p.add_argument('--couple-radius', type=float, default=0.60, help="Max allowed shoulder→(elbow/wrist) distance (height units) before penalty")

args = p.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# ───────────────────── Marker sets & dims ─────────────────────
kpts_input_lstm_upper = ['Neck','RShoulder','LShoulder','RElbow','LElbow','RWrist','LWrist']
mks_of_interest_upper = [
    'r_lelbow_study','r_melbow_study','r_lwrist_study','r_mwrist_study',
    'L_lelbow_study','L_melbow_study','L_lwrist_study','L_mwrist_study'
]
out_dim_upper = len(mks_of_interest_upper)*3

kpts_input_lstm_lower = ['Neck','RShoulder','LShoulder','RHip','LHip',
                         'RKnee','LKnee','RAnkle','LAnkle','RHeel',
                         'LHeel','RSmallToe','LSmallToe','RBigToe','LBigToe']
mks_of_interest_lower = [
    'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
    'r_knee_study','r_mknee_study','r_ankle_study','r_mankle_study',
    'r_toe_study','r_5meta_study','r_calc_study',
    'L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
    'L_toe_study','L_calc_study','L_5meta_study',
    'r_shoulder_study','L_shoulder_study','C7_study'
]
out_dim_lower = len(mks_of_interest_lower)*3

feature_dim_upper = len(kpts_input_lstm_upper)*3 + 2
feature_dim_lower = len(kpts_input_lstm_lower)*3 + 2

# ───────────────────── Datasets discovery ─────────────────────
root = Path(args.data_path)
subjects = sorted([d.name for d in root.iterdir() if d.is_dir()])
if len(subjects) < args.test_size + 1:
    raise RuntimeError("Not enough subjects to split. Found: {}".format(len(subjects)))
train_subjects = subjects[:-args.test_size]
val_subjects   = subjects[-args.test_size:]

def read_subject_info(info_path: Path):
    height = weight = None
    with info_path.open('r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'): continue
            for sep in ['=', ':']: line = line.replace(sep, ' ')
            parts = line.split()
            if len(parts) < 2: continue
            key, val = parts[0].lower(), parts[1]
            if key.startswith('height'):
                v = float(val)
                height = v/100.0 if v > 3.5 else v
            elif key.startswith('weight'):
                weight = float(val)
    if height is None:
        raise ValueError(f"Missing height in {info_path}")
    return float(height), float(weight if weight is not None else 0.0)

def enumerate_trials(subject_list):
    for s in subject_list:
        sp = root/s
        h, w = read_subject_info(sp/'info.txt')
        for trial in sorted([d.name for d in sp.iterdir() if d.is_dir()]):
            trial_dir = sp/trial
            jcp_name  = f"{trial}_jcp_mocap.csv"
            mocap_name= f"{trial}_trajectories.csv"
            if not (trial_dir/jcp_name).exists() or not (trial_dir/mocap_name).exists():
                continue
            yield {
                'subject': s, 'height': h, 'weight': w, 'trial': trial,
                'jcp_csv': str(trial_dir/jcp_name), 'gt_csv':  str(trial_dir/mocap_name)
            }

train_trials = list(enumerate_trials(train_subjects))
val_trials   = list(enumerate_trials(val_subjects))
if not train_trials or not val_trials:
    raise RuntimeError("No usable trials found in train/val.")

# ─────────────────── Windowing & augmentation ───────────────────
def listdicts_to_array(ld, names):
    T = len(ld); P = len(names)
    arr = np.zeros((T,P,3), dtype=np.float32)
    for i, fr in enumerate(ld):
        for j, k in enumerate(names):
            arr[i,j,:] = fr[k]
    return arr

def _yaw_rotation_matrix(theta_rad: float, up_axis: str = 'z'):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    if up_axis == 'y':     # rotate in XZ (Y-up)
        return np.array([[ c, 0.,  s],
                         [0., 1., 0.],
                         [-s, 0.,  c]], dtype=np.float32)
    elif up_axis == 'z':   # rotate in XY (Z-up)
        return np.array([[ c, -s, 0.],
                         [ s,  c, 0.],
                         [0.,  0., 1.]], dtype=np.float32)
    else:
        raise ValueError("up_axis must be 'y' or 'z'")

def _even_yaw_angles(n: int):
    if n <= 1: return [0.0]
    return [2.0*np.pi*k/n for k in range(n)]

def trial_to_windows_dual(
    trial, seq_len,
    add_noise=False,
    rotation_scheme='off', n_rotations=1,
    rot_prob=0.0, rot_max_deg=30.0, up_axis='y'
):
    """Yield ((x_up, x_lo), (y_up, y_lo)) from a single trial with shared midHip/height/rotation."""
    # read inputs (jcp) and gt (markers)
    df_in  = pd.read_csv(trial['jcp_csv'])
    df_gt  = pd.read_csv(trial['gt_csv'])
    k_list, _ = read_mks_data(df_in, converter=1000)  # includes 'midHip'
    m_list, _ = read_mks_data(df_gt, converter=1000)

    # Build arrays
    kpts_up = listdicts_to_array(k_list, kpts_input_lstm_upper)         # [T, PinU, 3]
    kpts_lo = listdicts_to_array(k_list, kpts_input_lstm_lower)         # [T, PinL, 3]
    gt_all  = listdicts_to_array(m_list, default_mocap_mks_names)       # [T, Pall, 3]

    sel_up = [default_mocap_mks_names.index(m) for m in mks_of_interest_upper]
    sel_lo = [default_mocap_mks_names.index(m) for m in mks_of_interest_lower]
    gt_up  = gt_all[:, sel_up, :]                                       # [T, PoutU, 3]
    gt_lo  = gt_all[:, sel_lo, :]                                       # [T, PoutL, 3]

    # mid-hip ref
    Ttot = len(k_list)
    mid = np.zeros((Ttot,3), dtype=np.float32)
    for i, fr in enumerate(k_list):
        mid[i] = fr['midHip']

    h = float(trial['height']); w = float(trial['weight'])
    inv_h = 1.0 / max(h, 1e-6)
    n_w  = max(Ttot - seq_len + 1, 0)

    # prepare yaw set
    yaw_set = [None]
    if rotation_scheme == 'det':
        yaw_set = _even_yaw_angles(max(1, int(n_rotations)))

    for start in range(n_w):
        end = start + seq_len

        # translate to mid-hip
        din_up = kpts_up[start:end] - mid[start:end][:, None, :]
        din_lo = kpts_lo[start:end] - mid[start:end][:, None, :]
        dout_up= gt_up[start:end]   - mid[start:end][:, None, :]
        dout_lo= gt_lo[start:end]   - mid[start:end][:, None, :]

        # height normalize (features & labels)
        din_up  = din_up  * inv_h
        din_lo  = din_lo  * inv_h
        dout_up = dout_up * inv_h
        dout_lo = dout_lo * inv_h

        if rotation_scheme == 'det':
            thetas = yaw_set
        else:
            thetas = [None]

        for theta in thetas:
            din_u, din_l, dout_u, dout_l = din_up, din_lo, dout_up, dout_lo

            if rotation_scheme == 'prob' and (rot_prob > 0.0) and (np.random.rand() < rot_prob) and (rot_max_deg > 0.0):
                theta = np.deg2rad(np.random.uniform(-rot_max_deg, rot_max_deg))

            if theta is not None:
                R = _yaw_rotation_matrix(theta, up_axis=up_axis).T
                din_u  = (din_u.reshape(-1,3)   @ R).reshape(seq_len, -1, 3)
                din_l  = (din_l.reshape(-1,3)   @ R).reshape(seq_len, -1, 3)
                dout_u = (dout_u.reshape(-1,3)  @ R).reshape(seq_len, -1, 3)
                dout_l = (dout_l.reshape(-1,3)  @ R).reshape(seq_len, -1, 3)

            if add_noise:
                din_u = din_u + np.random.normal(0.0, 0.018, din_u.shape).astype(np.float32)
                din_l = din_l + np.random.normal(0.0, 0.018, din_l.shape).astype(np.float32)

            # flatten & append height/weight
            x_up = din_u.reshape(seq_len, -1)
            x_lo = din_l.reshape(seq_len, -1)
            hw   = np.concatenate([np.full((seq_len,1), h, dtype=np.float32),
                                   np.full((seq_len,1), w, dtype=np.float32)], axis=1)
            x_up = np.concatenate([x_up, hw], axis=1)  # [L, PinU*3+2]
            x_lo = np.concatenate([x_lo, hw], axis=1)  # [L, PinL*3+2]
            y_up = dout_u.reshape(seq_len, -1)         # [L, 3*PoutU]
            y_lo = dout_l.reshape(seq_len, -1)         # [L, 3*PoutL]
            yield (x_up.astype(np.float32), x_lo.astype(np.float32)), (y_up.astype(np.float32), y_lo.astype(np.float32))

# ─────────────────── tf.data builders ───────────────────
input_signature = (
    (tf.TensorSpec(shape=(args.seq_len, feature_dim_upper), dtype=tf.float32),
     tf.TensorSpec(shape=(args.seq_len, feature_dim_lower), dtype=tf.float32)),
    (tf.TensorSpec(shape=(args.seq_len, out_dim_upper), dtype=tf.float32),
     tf.TensorSpec(shape=(args.seq_len, out_dim_lower), dtype=tf.float32))
)

def make_dataset_dual(trials, seq_len, batch, shuffle_windows=True,
                      add_noise=False, rotation_scheme='off', n_rotations=1,
                      rot_prob=0.0, rot_max_deg=30.0, up_axis='y'):
    def gen():
        for t in trials:
            for (xu, xl), (yu, yl) in trial_to_windows_dual(
                t, seq_len, add_noise=add_noise,
                rotation_scheme=rotation_scheme, n_rotations=n_rotations,
                rot_prob=rot_prob, rot_max_deg=rot_max_deg, up_axis=up_axis
            ):
                yield (xu, xl), (yu, yl)
    ds = tf.data.Dataset.from_generator(gen, output_signature=input_signature)
    if shuffle_windows:
        ds = ds.shuffle(buffer_size=max(8192, 2048 * n_rotations), reshuffle_each_iteration=True)
    ds = ds.batch(batch, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds

# ─────────────────── Two-pass normalization (separate for each stream) ───────────────────
train_raw = make_dataset_dual(
    train_trials, args.seq_len, batch=256, shuffle_windows=False,
    add_noise=(args.add_noise=='T'),
    rotation_scheme=args.rotation_scheme, n_rotations=args.n_rotations,
    rot_prob=args.rot_prob, rot_max_deg=args.rot_max_deg, up_axis=args.up_axis
)

mu_u = tf.zeros([feature_dim_upper], tf.float32);  m2_u = tf.zeros_like(mu_u);  c_u = 0.0
mu_l = tf.zeros([feature_dim_lower], tf.float32);  m2_l = tf.zeros_like(mu_l);  c_l = 0.0

for (x_u, x_l), _ in train_raw:
    bu = tf.cast(tf.shape(x_u)[0]*tf.shape(x_u)[1], tf.float32)
    bl = tf.cast(tf.shape(x_l)[0]*tf.shape(x_l)[1], tf.float32)
    xu = tf.reshape(x_u, [-1, feature_dim_upper])
    xl = tf.reshape(x_l, [-1, feature_dim_lower])
    mu_bu = tf.reduce_mean(xu, axis=0); var_bu = tf.math.reduce_variance(xu, axis=0)
    mu_bl = tf.reduce_mean(xl, axis=0); var_bl = tf.math.reduce_variance(xl, axis=0)

    # Chan update (upper)
    du = mu_bu - mu_u
    tu = c_u + bu
    mu_u = mu_u + du * (bu/tu)
    m2_u = m2_u + var_bu*bu + (du**2)*c_u*bu/tu
    c_u  = tu

    # Chan update (lower)
    dl = mu_bl - mu_l
    tl = c_l + bl
    mu_l = mu_l + dl * (bl/tl)
    m2_l = m2_l + var_bl*bl + (dl**2)*c_l*bl/tl
    c_l  = tl

mean_u = mu_u.numpy()
std_u  = np.sqrt((m2_u.numpy() / max(c_u,1.0)) + 1e-8)
mean_l = mu_l.numpy()
std_l  = np.sqrt((m2_l.numpy() / max(c_l,1.0)) + 1e-8)

MT_U = tf.constant(mean_u, dtype=tf.float32); ST_U = tf.constant(std_u, dtype=tf.float32)
MT_L = tf.constant(mean_l, dtype=tf.float32); ST_L = tf.constant(std_l, dtype=tf.float32)

def norm_map(inputs, targets):
    x_u, x_l = inputs
    y_u, y_l = targets
    x_u = (x_u - MT_U) / ST_U
    x_l = (x_l - MT_L) / ST_L
    return (x_u, x_l), (y_u, y_l)

train_ds = make_dataset_dual(
    train_trials, args.seq_len, batch=args.batch_size, shuffle_windows=True,
    add_noise=(args.add_noise=='T'),
    rotation_scheme=args.rotation_scheme, n_rotations=args.n_rotations,
    rot_prob=args.rot_prob, rot_max_deg=args.rot_max_deg, up_axis=args.up_axis
).map(norm_map, num_parallel_calls=tf.data.AUTOTUNE)

val_ds = make_dataset_dual(
    val_trials, args.seq_len, batch=args.batch_size, shuffle_windows=False,
    add_noise=False, rotation_scheme='off', up_axis=args.up_axis
).map(norm_map, num_parallel_calls=tf.data.AUTOTUNE)

# ─────────────────── Load pretrained submodels ───────────────────
def load_head(pretrained_dir: Path, out_dim: int, add_layer: bool, *, replace_last_if_mismatch: bool):
    with open(pretrained_dir/"model.json", 'r') as f:
        base = model_from_json(f.read())
    base.load_weights(str(pretrained_dir/"weights.h5"))

    if add_layer:
        # Always add a projection on top
        x = base.output
        proj = TimeDistributed(
            Dense(out_dim,
                  kernel_initializer=RandomNormal(0., 0.022),
                  bias_initializer='zeros',
                  kernel_regularizer=l2(args.weight_decay)),
            name="added_td_dense"
        )(x)
        return tf.keras.Model(inputs=base.input, outputs=proj)

    # No extra layer requested
    if replace_last_if_mismatch and (base.output_shape[-1] != out_dim):
        # Mirror old script: replace the last layer
        # (assumes the last layer is the TD Dense head)
        x = base.layers[-2].output
        new_out = TimeDistributed(
            Dense(out_dim,
                  kernel_initializer=RandomNormal(0., 0.022),
                  bias_initializer='zeros',
                  kernel_regularizer=l2(args.weight_decay)),
            name="replaced_last_layer"
        )(x)
        return tf.keras.Model(inputs=base.input, outputs=new_out)

    # Output shape already matches (or we’re not replacing)
    return base

pre_u_dir = Path(args.pretrained_path_upper) / "v0.3_upper"
pre_l_dir = Path(args.pretrained_path_lower) / "v0.3_lower"

upper_head = load_head(
    pre_u_dir, out_dim_upper, add_layer=(args.add_layer_upper=='T'),
    replace_last_if_mismatch=False  # same as old script: assume match
)
lower_head = load_head(
    pre_l_dir, out_dim_lower, add_layer=(args.add_layer_lower=='T'),
    replace_last_if_mismatch=True   # same as old script: replace last if mismatch (105→63)
)

if args.fine_tune == 'T':
    # Freeze all but the last TD Dense (or last layer) of each head.
    for m in [upper_head, lower_head]:
        for ly in m.layers[:-1]:
            ly.trainable = False

# ─────────────────── Build joint model with coupling ───────────────────
inp_up = Input(shape=(args.seq_len, feature_dim_upper), name="inp_upper")
inp_lo = Input(shape=(args.seq_len, feature_dim_lower), name="inp_lower")

# pred_up = tf.identity(upper_head(inp_up), name="upper_pred")  # [B, L, 3*Pup]
# pred_lo = tf.identity(lower_head(inp_lo), name="lower_pred")  # [B, L, 3*Plo]

from tensorflow.keras.layers import Lambda, Activation

upper_out = Lambda(lambda x: x, name="upper_pred")(upper_head(inp_up))
lower_out = Lambda(lambda x: x, name="lower_pred")(lower_head(inp_lo))


# indices for coupling
def idx_map(names):
    return {n:i for i,n in enumerate(names)}

imap_up = idx_map(mks_of_interest_upper)
imap_lo = idx_map(mks_of_interest_lower)
# shoulders from LOWER
IDX_R_SHO = imap_lo['r_shoulder_study']*3
IDX_L_SHO = imap_lo['L_shoulder_study']*3
# elbows/wrists from UPPER (medial markers for stability)
IDX_R_ELB = imap_up['r_melbow_study']*3
IDX_L_ELB = imap_up['L_melbow_study']*3
IDX_R_WRI = imap_up['r_mwrist_study']*3
IDX_L_WRI = imap_up['L_mwrist_study']*3

def slice_xyz(x, start3):
    return x[..., start3:start3+3]  # [B, L, 3]

def hinge_proximity(u, a, R):
    d = tf.norm(u - a, axis=-1)          # [B, L]
    excess = tf.nn.relu(d - R)
    return tf.reduce_mean(tf.square(excess))

def coupling_loss(y_up, y_lo, R=0.6):
    rs = slice_xyz(y_lo, IDX_R_SHO)
    ls = slice_xyz(y_lo, IDX_L_SHO)
    loss = 0.0
    loss += hinge_proximity(slice_xyz(y_up, IDX_R_ELB), rs, R)
    loss += hinge_proximity(slice_xyz(y_up, IDX_R_WRI), rs, R)
    loss += hinge_proximity(slice_xyz(y_up, IDX_L_ELB), ls, R)
    loss += hinge_proximity(slice_xyz(y_up, IDX_L_WRI), ls, R)
    return loss

lambda_couple = tf.constant(float(args.lambda_couple), dtype=tf.float32)
couple_R      = tf.constant(float(args.couple_radius), dtype=tf.float32)


# Add coupling as a model-level loss
reg_tensor = Lambda(lambda t: lambda_couple * coupling_loss(t[0], t[1], couple_R),
                    name="coupling_reg")([upper_out, lower_out])
# Keras will pick it up from model.add_loss below

joint_model = tf.keras.Model(
    inputs=[inp_up, inp_lo],
    outputs=[upper_out, lower_out],
    name="joint_upper_lower"
)
joint_model.add_metric(reg_tensor, name="couple_loss", aggregation="mean")
joint_model.add_loss(reg_tensor)

# ─────────────────── Losses (upper: MSE; lower: optional weighted MSE) ───────────────────
# lower weighting (optional)
marker_weights_lower = {"r_toe_study":2.0,"r_5meta_study":2.0,"r_calc_study":2.0,
                        "L_toe_study":2.0,"L_5meta_study":2.0,"L_calc_study":2.0}
if args.use_weights_lower == 'T':
    w = np.ones(out_dim_lower, dtype=np.float32)
    for i, m in enumerate(mks_of_interest_lower):
        if m in marker_weights_lower:
            w[i*3:(i+1)*3] = marker_weights_lower[m]
    W_LO = tf.constant(w, dtype=tf.float32)
else:
    W_LO = None

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

def weighted_mse_lower(y_true, y_pred):
    if W_LO is None:
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    sq = tf.square(y_true - y_pred) * W_LO
    return tf.reduce_mean(sq, axis=-1)

joint_model.compile(
    optimizer=Adam(args.lr),
    loss={"upper_pred": mse_loss, "lower_pred": weighted_mse_lower},
    loss_weights={"upper_pred": 1.0, "lower_pred": 1.0},
)

joint_model.summary()

# ─────────────────── Train ───────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1),
]

history = joint_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────── Save outputs ───────────────────
# save finetuned sub-head weights
out_root_u = Path(args.pretrained_path_upper) / "v0.3_upper"
out_root_l = Path(args.pretrained_path_lower) / "v0.3_lower"

upper_head.save_weights(str(out_root_u / "weights_finetuned_joint.h5"))
lower_head.save_weights(str(out_root_l / "weights_finetuned_joint.h5"))

# save stats
stats_dir = Path(args.pretrained_path_upper) / "stats_streaming_joint"
stats_dir.mkdir(parents=True, exist_ok=True)
np.save(stats_dir / "mean_upper.npy", mean_u)
np.save(stats_dir / "std_upper.npy",  std_u)
np.save(stats_dir / "mean_lower.npy", mean_l)
np.save(stats_dir / "std_lower.npy",  std_l)
with open(stats_dir / "norm_meta.json", "w") as f:
    json.dump({
        "feature_dim_upper": int(feature_dim_upper),
        "feature_dim_lower": int(feature_dim_lower),
        "seq_len": int(args.seq_len),
        "kpts_input_lstm_upper": kpts_input_lstm_upper,
        "kpts_input_lstm_lower": kpts_input_lstm_lower,
        "mks_of_interest_upper": mks_of_interest_upper,
        "mks_of_interest_lower": mks_of_interest_lower,
        "lambda_couple": float(args.lambda_couple),
        "couple_radius": float(args.couple_radius)
    }, f, indent=2)

print("[Done] Saved:")
print("  Upper weights:", out_root_u / "weights_finetuned_joint.h5")
print("  Lower weights:", out_root_l / "weights_finetuned_joint.h5")
print("  Stats:", stats_dir)
