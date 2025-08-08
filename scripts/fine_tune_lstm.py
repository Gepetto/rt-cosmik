#!/usr/bin/env python3
# fine_tune_lstm.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import TimeDistributed, Dense
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
import argparse



# === Args ===
parser = argparse.ArgumentParser(description='LSTM retraining/finetuning arguments')
parser.add_argument('--data-path',
                        help='data path',
                        dest='data_path',
                        default='',
                        type=str)
parser.add_argument('--pretrained-path',
                        help='pretrained path',
                        dest='pretrained_path',
                        default='',
                        type=str)
parser.add_argument("--body-part",
                    help="body part",
                    dest="body_part",
                    default='',
                    type=str)
parser.add_argument("--fine-tune",
                    help="Fine tune or retrain",
                    dest="fine_tune",
                    type=str,
                    default="F")
parser.add_argument("--add-layer",
                    help="Add a final layer or not",
                    dest="add_layer",
                    type=str,
                    default="F")
parser.add_argument('--use-mocap',
                        help='use mocap or hpe',
                        dest='use_mocap',
                        default='',
                        type=str)
opt = parser.parse_args()

# === Hyperparams ===
data_dir = os.path.join(opt.data_path, opt.body_part)
pretrained_dir = os.path.join(opt.pretrained_path, f"v0.3_{opt.body_part}")
body_part = opt.body_part
fine_tune = opt.fine_tune
add_layer = opt.add_layer
use_mocap = opt.use_mocap
json_path      = os.path.join(pretrained_dir, "model.json")
weights_path   = os.path.join(pretrained_dir, "weights.h5")

batch_size   = 64
epochs       = 100
patience     = 5
learning_rate= 6e-6
initializer = RandomNormal(mean=0.0, stddev=0.022)
weight_decay = 0.01

# === Load pretrained model ===
with open(json_path, 'r') as f:
    base = model_from_json(f.read())
base.load_weights(weights_path)

if fine_tune == "T":
    # ❄️ Freeze tous les layers du modèle de base
    for layer in base.layers[:-1]:
        layer.trainable = False
    if add_layer == "T":
        base.layers[-1].trainable = False


# ====== Set LSTM to output only last-step (last vector of the predicted window), then only with the markers of interest
if body_part == "upper":
    if add_layer == "T":
        projection = TimeDistributed(Dense(24, 
                                    kernel_initializer=initializer, 
                                    bias_initializer='zeros', 
                                    kernel_regularizer=l2(weight_decay)),
                                    name="layer_added"
                                    )(base.output)
        # Final model
        model = Model(inputs=base.input, outputs=projection)
    else:
        # Final model
        model = Model(inputs=base.input, outputs=base.output)
elif body_part == "lower":
    if add_layer == "T":
        projection = TimeDistributed(Dense(63, 
                                    kernel_initializer=initializer, 
                                    bias_initializer='zeros', 
                                    kernel_regularizer=l2(weight_decay)),
                                    name="layer_added"
                                    )(base.output)
        # Final model
        model = Model(inputs=base.input, outputs=projection)
    else:
        x = base.layers[-2].output
        new_output = TimeDistributed(Dense(63,
                                    kernel_initializer=initializer, 
                                    bias_initializer='zeros', 
                                    kernel_regularizer=l2(weight_decay)),
                                    name="replaced_last_layer"
                                    )(x)
        model = Model(inputs=base.input, outputs=new_output)
else:
    raise Exception("Body part not supported. Please select upper or lower.")
model.summary()
model.compile(optimizer=Adam(learning_rate), loss='mse')

# Sauvegarde de l'architecture dans un fichier JSON
with open(os.path.join(pretrained_dir, f"model_finetuned_{body_part}_ft{fine_tune}_al{add_layer}_m{use_mocap}.json"), "w") as f:
    f.write(model.to_json())


# === Load data ===
X_train = np.load(os.path.join(data_dir, "train", f"X_train_m{use_mocap}.npy"))
Y_train = np.load(os.path.join(data_dir, "train", "Y_train.npy"))
X_val = np.load(os.path.join(data_dir, "val", f"X_val_m{use_mocap}.npy"))
Y_val = np.load(os.path.join(data_dir, "val", "Y_val.npy"))
mean_train = np.load(os.path.join(data_dir, "stats", f"mean_train_m{use_mocap}.npy"))
std_train = np.load(os.path.join(data_dir, "stats", f"std_train_m{use_mocap}.npy"))

# Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset   = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# === Train ===
checkpoint = ModelCheckpoint(
    filepath=os.path.join(pretrained_dir, f"best_finetuned_weights_{body_part}_ft{fine_tune}_al{add_layer}_m{use_mocap}.h5"),    
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,            # True si tu veux sauvegarder seulement les poids
    verbose=1
)
es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[es, checkpoint]
)


# Save fine-tuned weights
model.save_weights(os.path.join(pretrained_dir, f"weights_finetuned_{body_part}_ft{fine_tune}_al{add_layer}_m{use_mocap}.h5"))
print(f"Fine-tuning complete. Saved to weights_finetuned_{body_part}_ft{fine_tune}_al{add_layer}_m{use_mocap}.h5")


