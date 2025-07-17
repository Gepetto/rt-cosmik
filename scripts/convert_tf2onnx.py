import tensorflow as tf
import os
from tensorflow.keras.models import model_from_json
import tf2onnx

# Paths to your files
json_path = "/root/workspace/ros_ws/src/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM/v0.4_upper/model.json"
weights_path = "/root/workspace/ros_ws/src/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM/v0.4_upper/weights.h5"

# Load the model architecture
with open(json_path, 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Load the weights
model.load_weights(weights_path)

input_dim= model.input_shape

spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)

# Replace `input_dim` with the correct shape
# If your model input shape is (None, 128), then input_dim = 128

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save to file
with open("/root/workspace/ros_ws/src/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM/v0.4_upper/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
