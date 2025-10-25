import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tf2onnx

@keras.utils.register_keras_serializable(package="pose")
class SelectFeatures(keras.layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self._indices_list = list(indices)
        self.indices = tf.constant(self._indices_list, dtype=tf.int32)
    def call(self, x):
        # Works for [B, T, F] or [B, F]; gathers along the last axis
        return tf.gather(x, self.indices, axis=-1)
    def get_config(self):
        return {"indices": self._indices_list, **super().get_config()}
    
KERAS_PATH= "src/rtcosmik/augmenter/augmentation_model/LSTM/v0.3_lower/optuna_trial_0_best.keras"
ONNX_OUT = "src/rtcosmik/augmenter/augmentation_model/LSTM/v0.3_lower/optuna_trial_0_best.onnx"

# ---------- 3) Load WITHOUT compiling (avoids custom loss/metric issues) ----------
model = load_model(
    KERAS_PATH,
    custom_objects={"SelectFeatures": SelectFeatures},
    compile=False                     # <— crucial fix
)

print(model.summary())
# print("Input shape:", model.input_shape)

# # ---------- 4) Build an input signature that matches the model ----------
# # Use the saved model's input shape (e.g., (None, None, 47))
# spec = (tf.TensorSpec(shape=model.input_shape, dtype=tf.float32, name="input"),)

# # ---------- 5) Convert to ONNX ----------
# onnx_model, _ = tf2onnx.convert.from_keras(
#     model,
#     input_signature=spec,
#     opset=17  # 13+ is fine; 17 is a good modern default
# )

# with open(ONNX_OUT, "wb") as f:
#     f.write(onnx_model.SerializeToString())

# print(f"[OK] Exported ONNX to: {ONNX_OUT}")