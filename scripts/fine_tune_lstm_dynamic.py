#!/usr/bin/env python3
# train_lstm_end2end.py
import os, sys, json, argparse, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.layers import TimeDistributed, Dense
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─────────────── Args ───────────────
p = argparse.ArgumentParser(description="End-to-end LSTM training with streaming tf.data")
p.add_argument('--data-path', required=True, type=str)
p.add_argument('--pretrained-path', required=True, type=str)
p.add_argument('--body-part', choices=['upper','lower'], required=True)
p.add_argument('--use-mocap', choices=['T','F'], default='T', required=True)        # must be 'T' for this script (mocap JCP + mocap GT)
p.add_argument('--add-noise', choices=['T','F'], default='F')
p.add_argument('--fine-tune', choices=['T','F'], default='F')
p.add_argument('--add-layer', choices=['T','F'], default='F')
p.add_argument('--use-weights', choices=['T','F'], default='F')
p.add_argument('--seq-len', type=int, default=30)
p.add_argument('--batch-size', type=int, default=64)
p.add_argument('--epochs', type=int, default=500)
p.add_argument('--patience', type=int, default=5)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--weight-decay', type=float, default=0.01)
p.add_argument('--test-size', type=int, default=2, help="# of subjects reserved for val (last N alphabetical)")
p.add_argument('--seed', type=int, default=42)
args = p.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# ─────────────── Config derived from body part ───────────────
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

# If you rely on your project's utils for parsing CSV into list-of-dicts, import them.
# Fallbacks below emulate the column->dict conversion used in your code.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.rtcosmik.utils.read_write_utils import read_mks_data, default_mocap_mks_names
except Exception:
    # minimal fallback if the project import isn't available
    default_mocap_mks_names = mks_of_interest  # will work if your GT CSVs have exactly these names tripled (_x,_y,_z)
    def read_mks_data(df: pd.DataFrame):
        # Convert wide dataframe with columns like "Neck_x" into list[dict[name]->np(3,)]
        names = sorted(set([c.rsplit('_',1)[0] for c in df.columns if c.endswith(('_x','_y','_z'))]))
        frames = []
        for _, row in df.iterrows():
            d = {}
            for n in names:
                d[n] = np.array([row[f"{n}_x"], row[f"{n}_y"], row[f"{n}_z"]], dtype=np.float32)
            frames.append(d)
        return frames, names

# ─────────────── Files discovery ───────────────
root = Path(args.data_path)
subjects = sorted([d.name for d in root.iterdir() if d.is_dir()])
if len(subjects) < args.test_size + 1:
    raise RuntimeError("Not enough subjects to split.")

train_subjects = subjects[:-args.test_size]
val_subjects   = subjects[-args.test_size:]

def read_subject_info(info_path: Path):
    height = weight = gender = None
    with info_path.open('r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            for sep in ['=', ':']:
                line = line.replace(sep, ' ')
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].lower()
            val = parts[1]
            if key.startswith('height'):
                v = float(val)
                height = v/100.0 if v > 3.5 else v
            elif key.startswith('weight'):
                try: weight = float(val)
                except: pass
            elif key.startswith('gender'):
                gender = val.strip().lower()
    if height is None:
        raise ValueError(f"Missing height in {info_path}")
    return height, weight, gender

def enumerate_trials(subject_list):
    """Yield dicts describing usable trials with paths & metadata."""
    for s in subject_list:
        sp = root/s
        h, w, _ = read_subject_info(sp/'info.txt')
        for trial in sorted([d.name for d in sp.iterdir() if d.is_dir()]):
            trial_dir = sp/trial
            jcp_name  = f"{trial}_jcp_mocap.csv"
            mocap_name= f"{trial}_trajectories.csv"
            if args.use_mocap != 'T':
                raise RuntimeError("This streaming script is set for --use-mocap T.")
            if not (trial_dir/jcp_name).exists() or not (trial_dir/mocap_name).exists():
                continue
            yield {
                'subject': s,
                'height': h,
                'weight': w,
                'trial': trial,
                'jcp_csv': str(trial_dir/jcp_name),
                'gt_csv':  str(trial_dir/mocap_name),
            }

train_trials = list(enumerate_trials(train_subjects))
val_trials   = list(enumerate_trials(val_subjects))
if not train_trials or not val_trials:
    raise RuntimeError("No usable trials found in train/val.")

# ─────────────── Windowing helpers ───────────────
def listdicts_to_array(ld, names):
    T = len(ld); P = len(names)
    arr = np.zeros((T,P,3), dtype=np.float32)
    for i, fr in enumerate(ld):
        for j, k in enumerate(names):
            arr[i,j,:] = fr[k]
    return arr

def trial_to_windows(trial, seq_len, add_noise=False):
    """Load one trial from disk, produce windowed (inp, out) samples."""
    # read inputs (jcp) and gt (markers)
    df_in  = pd.read_csv(trial['jcp_csv'])
    df_gt  = pd.read_csv(trial['gt_csv'])

    k_list, k_names = read_mks_data(df_in)       # includes 'midHip' key
    m_list, _       = read_mks_data(df_gt)

    # Build arrays
    kpts_arr = listdicts_to_array(k_list, kpts_input_lstm)         # [T, P_in, 3]
    gt_arr   = listdicts_to_array(m_list, default_mocap_mks_names) # [T, P_all, 3]

    # mid-hip ref
    mid = np.zeros((len(k_list), 3), dtype=np.float32)
    for i, fr in enumerate(k_list):
        mid[i] = fr['midHip']

    # select GT markers of interest
    sel = [default_mocap_mks_names.index(m) for m in mks_of_interest]
    gt_sel = gt_arr[:, sel, :]  # [T, P_out, 3]

    h = float(trial['height']); w = float(trial['weight'] or 0.0)

    Ttot = kpts_arr.shape[0]
    n_w  = max(Ttot - seq_len + 1, 0)
    for start in range(n_w):
        end = start + seq_len
        kbuf = kpts_arr[start:end]           # [L, P_in, 3]
        ref  = mid[start:end]                # [L, 3]

        inp = kbuf - ref[:,None,:]
        inp = inp / (h if h>0 else 1.0)
        if add_noise:
            inp = inp + np.random.normal(0, 0.018, inp.shape).astype(np.float32)
        inp = inp.reshape(seq_len, -1)       # flatten joints
        # append height & weight per frame (as in your script)
        hw  = np.concatenate([np.full((seq_len,1), h, dtype=np.float32),
                              np.full((seq_len,1), w, dtype=np.float32)], axis=1)
        inp = np.concatenate([inp, hw], axis=1)  # [L, P_in*3 + 2]

        ybuf = gt_sel[start:end]             # [L, P_out, 3]
        out  = (ybuf - ref[:,None,:]) / (h if h>0 else 1.0)
        out  = out.reshape(seq_len, -1)      # [L, out_dim]
        yield inp.astype(np.float32), out.astype(np.float32)

# Spec (needed for tf.data.from_generator)
feature_dim = len(kpts_input_lstm)*3 + 2
out_dim     = out_dim

output_signature = (
    tf.TensorSpec(shape=(args.seq_len, feature_dim), dtype=tf.float32),
    tf.TensorSpec(shape=(args.seq_len, out_dim),     dtype=tf.float32)
)

def make_dataset(trials, seq_len, batch, shuffle_windows=True, add_noise=False):
    def gen():
        # interleave trials deterministically; you can randomize order here
        for t in trials:
            for inp, out in trial_to_windows(t, seq_len, add_noise=add_noise):
                yield inp, out
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle_windows:
        ds = ds.shuffle(buffer_size=8192, reshuffle_each_iteration=True)
    ds = ds.batch(batch, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds

# ─────────────── Two-pass normalization (streaming) ───────────────
# Pass 1: compute mean/std over TRAIN only
train_raw = make_dataset(train_trials, args.seq_len, batch=256, shuffle_windows=False, add_noise=(args.add_noise=='T'))

@tf.function
def batch_stats(x):
    # x: [B, L, F]; we compute per-feature mean/std over both axes
    mu  = tf.reduce_mean(x, axis=[0,1])
    var = tf.reduce_mean((x - mu)**2, axis=[0,1])
    return mu, tf.sqrt(var + 1e-8)

n_batches = 0
mu_acc = tf.zeros([feature_dim], tf.float32)
m2_acc = tf.zeros([feature_dim], tf.float32)
count  = 0.0

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

mean_train = mu_acc.numpy()
std_train  = np.sqrt((m2_acc.numpy() / max(count,1.0)) + 1e-8)

# map normalization using captured constants
mt = tf.constant(mean_train, dtype=tf.float32)
st = tf.constant(std_train,  dtype=tf.float32)

def normalize_xy(x, y):
    x = (x - mt) / st
    return x, y

train_ds = make_dataset(train_trials, args.seq_len, batch=args.batch_size, shuffle_windows=True, add_noise=(args.add_noise=='T')).map(normalize_xy, num_parallel_calls=tf.data.AUTOTUNE)
val_ds   = make_dataset(val_trials,   args.seq_len, batch=args.batch_size, shuffle_windows=False, add_noise=False).map(normalize_xy, num_parallel_calls=tf.data.AUTOTUNE)

# ─────────────── Model (reuse your pretrained JSON/weights; adjust last layer when needed) ───────────────
pretrained_dir = Path(args.pretrained_path) / f"v0.3_{args.body_part}"
with open(pretrained_dir/"model.json", 'r') as f:
    base = model_from_json(f.read())
base.load_weights(str(pretrained_dir/"weights.h5"))

initializer = RandomNormal(mean=0.0, stddev=0.022)
if args.body_part == "upper":
    if args.add_layer == "T":
        proj = TimeDistributed(Dense(out_dim, kernel_initializer=initializer, bias_initializer='zeros', kernel_regularizer=l2(args.weight_decay)), name="layer_added")(base.output)
        model = Model(inputs=base.input, outputs=proj)
    else:
        model = Model(inputs=base.input, outputs=base.output)  # assumes base already matches out_dim
else:
    if args.add_layer == "T":
        proj = TimeDistributed(Dense(out_dim, kernel_initializer=initializer, bias_initializer='zeros', kernel_regularizer=l2(args.weight_decay)), name="layer_added")(base.output)
        model = Model(inputs=base.input, outputs=proj)
    else:
        x = base.layers[-2].output
        new_out = TimeDistributed(Dense(out_dim, kernel_initializer=initializer, bias_initializer='zeros', kernel_regularizer=l2(args.weight_decay)), name="replaced_last_layer")(x)
        model = Model(inputs=base.input, outputs=new_out)

# Fine-tune freezing policy
if args.fine_tune == "T":
    for layer in base.layers[:-1]:
        layer.trainable = False
    if args.add_layer == "T":
        base.layers[-1].trainable = False

# Optional weighted loss for lower body
response_mks_lower = [
    'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
    'r_knee_study','r_mknee_study','r_ankle_study','r_mankle_study',
    'r_toe_study','r_5meta_study','r_calc_study',
    'L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
    'L_toe_study','L_calc_study','L_5meta_study',
    'r_shoulder_study','L_shoulder_study','C7_study'
]
marker_weights = {"r_toe_study":2,"r_5meta_study":2,"r_calc_study":2,"L_toe_study":2,"L_5meta_study":2,"L_calc_study":2}

def build_weights_vector():
    if args.body_part != "lower" or args.use_weights != "T":
        return None
    w = np.ones(len(response_mks_lower)*3, dtype=np.float32)
    for i, m in enumerate(response_mks_lower):
        if m in marker_weights:
            w[i*3:(i+1)*3] = marker_weights[m]
    return tf.constant(w, dtype=tf.float32)

W_loss = build_weights_vector()
def weighted_l2(weights):
    def loss(y_true, y_pred):
        if weights is None:
            return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        sq = tf.square(y_true - y_pred)
        sq = sq * weights  # broadcast on last dim
        return tf.reduce_mean(sq, axis=-1)
    return loss

model.compile(optimizer=Adam(args.lr), loss=weighted_l2(W_loss))

# Save model definition that matches finetune config
model_json_path = pretrained_dir / f"model_finetuned_{args.body_part}_ft{args.fine_tune}_al{args.add_layer}_m{args.use_mocap}_n{args.add_noise}_w{args.use_weights}.json"
with open(model_json_path, "w") as f:
    f.write(model.to_json())

# ─────────────── Callbacks ───────────────
ckpt_path = pretrained_dir / f"best_finetuned_weights_{args.body_part}_ft{args.fine_tune}_al{args.add_layer}_m{args.use_mocap}_n{args.add_noise}_w{args.use_weights}.h5"
callbacks = [
    EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1),
    ModelCheckpoint(str(ckpt_path), monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
]

# ─────────────── Train ───────────────
print(f"[Info] Feature mean/std from TRAIN: mean shape {mean_train.shape}, std shape {std_train.shape}")
history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

# ─────────────── Save weights + stats ───────────────
final_w = pretrained_dir / f"weights_finetuned_{args.body_part}_ft{args.fine_tune}_al{args.add_layer}_m{args.use_mocap}_n{args.add_noise}_w{args.use_weights}.h5"
model.save_weights(str(final_w))
stats_dir = Path(args.data_path) / args.body_part / "stats_streaming"
stats_dir.mkdir(parents=True, exist_ok=True)
np.save(stats_dir / f"mean_train_m{args.use_mocap}_n{args.add_noise}.npy", mean_train)
np.save(stats_dir / f"std_train_m{args.use_mocap}_n{args.add_noise}.npy",  std_train)
with open(stats_dir / f"norm_meta.json", "w") as f:
    json.dump({
        "feature_dim": int(feature_dim),
        "seq_len": int(args.seq_len),
        "kpts_input_lstm": kpts_input_lstm,
        "body_part": args.body_part
    }, f, indent=2)

print(f"[Done] Saved: {final_w}")
