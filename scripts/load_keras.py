import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import tf2onnx

# (Optional) for stricter numerical reproducibility / to silence the oneDNN message:
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ---------- Define the custom layer EXACTLY as in training ----------
@keras.saving.register_keras_serializable(package="pose")  # keras>=3; for TF2.x older, utils.register_keras_serializable also works
class SelectFeatures(keras.layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self._indices_list = list(indices)
        self.indices = tf.constant(self._indices_list, dtype=tf.int32)
    def call(self, x):
        # gather along feature axis; works for [B,F] or [B,T,F]
        return tf.gather(x, self.indices, axis=-1)
    def get_config(self):
        return {"indices": self._indices_list, **super().get_config()}

# ---------- Paths ----------
json_path    = "/root/workspace/ros_ws/src/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM/v0.3_lower/optuna_trial_0.arch.json"

# ---------- Load architecture + weights ----------
with open(json_path, "r") as f:
    model_json = f.read()

model = model_from_json(model_json, custom_objects={"SelectFeatures": SelectFeatures})

model.summary()
# print("Output shape:", model.output_shape)

# # ---------- (Optional) sanity: check there is a SelectFeatures op ----------
# assert any(l.__class__.__name__ == "SelectFeatures" for l in model.layers), "SelectFeatures not found in graph."

# # ---------- Convert to ONNX ----------
# # If your input is sequences of 47-dim features: [B, T, 47]
# input_sig = [tf.TensorSpec(shape=[None, None, 47], dtype=tf.float32, name="input")]
# onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_sig, opset=17)  # opset 13+ is fine; 17 is current-safe

# with open(onnx_path, "wb") as f:
#     f.write(onnx_model.SerializeToString())

# print("Saved ONNX to:", onnx_path)
