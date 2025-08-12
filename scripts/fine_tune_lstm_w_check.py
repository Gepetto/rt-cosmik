#!/usr/bin/env python3
# fine_tune_lstm.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import TimeDistributed, Dense
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
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
parser.add_argument('--add-noise',
                        help='if mocap, add noise to the data',
                        dest='add_noise',
                        default='F',
                        type=str)
parser.add_argument('--weights',
                        help='use foot weights for loss function',
                        dest='weights',
                        default='F',
                        type=str)
opt = parser.parse_args()

# === Hyperparams ===
data_dir = os.path.join(opt.data_path, opt.body_part)
pretrained_dir = os.path.join(opt.pretrained_path, f"v0.3_{opt.body_part}")
body_part = opt.body_part
fine_tune = opt.fine_tune
add_layer = opt.add_layer
use_mocap = opt.use_mocap
add_noise = opt.add_noise
weights = opt.weights
monitoring = True
monitoring_step = 25
json_path      = os.path.join(pretrained_dir, "model.json")
weights_path   = os.path.join(pretrained_dir, "weights.h5")

batch_size   = 64
epochs       = 300
patience     = 3
learning_rate= 6e-5
initializer = RandomNormal(mean=0.0, stddev=0.022)
weight_decay = 0.01

response_mks_lower = [
    'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
    'r_knee_study','r_mknee_study','r_ankle_study','r_mankle_study',
    'r_toe_study','r_5meta_study','r_calc_study',
    'L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
    'L_toe_study','L_calc_study','L_5meta_study',
    'r_shoulder_study','L_shoulder_study','C7_study'
    ]
marker_weights = {"r_toe_study": 2,
                "r_5meta_study": 2,
                "r_calc_study": 2,
                "L_toe_study": 2,
                "L_5meta_study": 2,
                "L_calc_study": 2}

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

if body_part == "lower" and weights == "T":
    # Create vector of weights for loss function.
    weights_loss = np.ones(len(response_mks_lower)*3)
    for i, marker in enumerate(response_mks_lower):
        if marker in marker_weights:
            weights_loss[i*3:(i+1)*3] = marker_weights[marker]
else:
    weights_loss = None 
# Weightedd L2 loss function.
def weighted_l2_loss(weights):
    def loss(y_true, y_pred):
        squared_diff = K.square(y_true - y_pred)
        weighted_squared_diff = squared_diff * weights
        return K.mean(weighted_squared_diff, axis=-1)
    return loss
if weights == "T":
    model.compile(optimizer=Adam(learning_rate), loss=weighted_l2_loss(weights))
elif weights == "F":
    model.compile(optimizer=Adam(learning_rate), loss='mse')
else:
    raise Exception("Weights argument must be T or F.")

# Sauvegarde de l'architecture dans un fichier JSON
with open(os.path.join(pretrained_dir, f"model_finetuned_{body_part}_ft{fine_tune}_al{add_layer}_m{use_mocap}_n{add_noise}.json"), "w") as f:
    f.write(model.to_json())


# === Load data ===
X_train = np.load(os.path.join(data_dir, "train", f"X_train_m{use_mocap}_n{add_noise}.npy"))
Y_train = np.load(os.path.join(data_dir, "train", f"Y_train_m{use_mocap}_n{add_noise}.npy"))
X_val = np.load(os.path.join(data_dir, "val", f"X_val_m{use_mocap}_n{add_noise}.npy"))
Y_val = np.load(os.path.join(data_dir, "val", f"Y_val_m{use_mocap}_n{add_noise}.npy"))
mean_train = np.load(os.path.join(data_dir, "stats", f"mean_train_m{use_mocap}_n{add_noise}.npy"))
std_train = np.load(os.path.join(data_dir, "stats", f"std_train_m{use_mocap}_n{add_noise}.npy"))

# Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset   = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

class PredictionLogger(tf.keras.callbacks.Callback):
    def __init__(self, val_data, save_dir, every_n_epochs=1):
        super().__init__()
        self.val_data = val_data
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            # Récupère un batch complet de validation
            X_val, Y_val = next(iter(self.val_data))
            preds = self.model.predict(X_val)

            # Sauvegarde dans un DataFrame
            df = pd.DataFrame({
                "truth": Y_val.numpy().flatten(),
                "prediction": preds.flatten()
            })

            # Sauvegarde en CSV
            csv_path = os.path.join(self.save_dir, f"predictions_epoch_{epoch+1}.csv")
            df.to_csv(csv_path, index=False)
            print(f"[INFO] Sauvegarde des prédictions dans {csv_path}")

pred_logger = PredictionLogger(
    val_data=val_dataset,
    save_dir=pretrained_dir,
    every_n_epochs=monitoring_step
)

# === Train ===
checkpoint = ModelCheckpoint(
    filepath=os.path.join(pretrained_dir, f"best_finetuned_weights_{body_part}_ft{fine_tune}_al{add_layer}_m{use_mocap}_n{add_noise}.h5"),    
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,            # True si tu veux sauvegarder seulement les poids
    verbose=1
)
es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, mode="auto", verbose=1)
if monitoring:
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[es, checkpoint, pred_logger]
    )
else:
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[es, checkpoint]
    )

# Save fine-tuned weights
model.save_weights(os.path.join(pretrained_dir, f"weights_finetuned_{body_part}_ft{fine_tune}_al{add_layer}_m{use_mocap}_n{add_noise}.h5"))
print(f"Fine-tuning complete. Saved to weights_finetuned_{body_part}_ft{fine_tune}_al{add_layer}_m{use_mocap}_n{add_noise}.h5")


