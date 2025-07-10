from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# === Charger le CSV ===
df_inputs = pd.read_csv("../data/lstm_training/3d_keypoints_filtered_4.csv")
df_outputs = pd.read_csv("../data/lstm_training/mks_data_gapfilled.csv")
df_outputs = df_outputs.drop(df_outputs.columns[0], axis=1)  # Supprimer la colonne inutile

# === Conversion en tableau NumPy ===
data_inputs = df_inputs.to_numpy()
data_outputs = df_outputs.to_numpy()

# === Séparation entrée / sortie ===
x = data_inputs[:, [18*3, 18*3+1, 18*3+2, 6*3, 6*3+1, 6*3+2, 5*3, 5*3+1, 5*3+2, 
                   8*3, 8*3+1, 8*3+2, 7*3, 7*3+1, 7*3+2, 10*3, 10*3+1, 10*3+2, 9*3, 9*3+1, 9*3+2]]
y = data_outputs[:, [8*3, 8*3+1, 8*3+2, 9*3, 9*3+1, 9*3+2, 10*3, 10*3+1, 10*3+2, 15*3, 
                 15*3+1, 15*3+2, 16*3, 16*3+1, 16*3+2, 18*3, 18*3+1, 18*3+2, 19*3, 19*3+1, 19*3+2,
                 24*3, 24*3+1, 24*3+2, 25*3, 25*3+1, 25*3+2, 27*3, 27*3+1, 27*3+2, 28*3, 28*3+1, 28*3+2]]


# === Reshape en (batch, time, features) ===
x = x.reshape((-1, 21))
y = y.reshape((-1, 33))

print("Shape of x:", x.shape)  # (nb_samples, timesteps, input_dim)
print("Shape of y:", y.shape)  # (nb_samples, timesteps, output_dim)


# === Split train / val ===
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.reshape((-1, 1, 21))  # (nb_samples, timesteps, input_dim)
x_val = x_val.reshape((-1, 1, 21))      # (nb_samples, timesteps, input_dim)
y_train = y_train.reshape((-1, 1, 33))  # (nb_samples, timesteps, output_dim)
y_val = y_val.reshape((-1, 1, 33))      # (nb_samples, timesteps, output_dim)

# === Hyperparamètres ===
input_dim = 21       # nb de features en entrée
output_dim = 33      # nb de coordonnées/keypoints à prédire (par exemple)
timesteps = None     # ou fixe si besoin

# === Reconstruit le modèle jusqu’à l’avant-dernière couche ===
inputs = Input(shape=(timesteps, input_dim), name="input")
x = LSTM(128, return_sequences=True)(inputs)
x = LSTM(128, return_sequences=True)(x)
x = LSTM(128, return_sequences=True)(x)
x = LSTM(128, return_sequences=True)(x)

# === Remplace la sortie par une nouvelle couche ===
outputs = TimeDistributed(Dense(output_dim, activation="linear"))(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # ou autre métrique si classification
model.summary()






early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)





plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# MAE (ou autre)
if 'mae' in history.history:
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title("MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend()

plt.tight_layout()
plt.show()
