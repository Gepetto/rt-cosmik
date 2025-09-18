import tensorflow as tf
import os
from tensorflow.keras.models import model_from_json
import tf2onnx

# Paths to your files
json_path = "/root/workspace/ros_ws/src/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM_finetuned/v0.3_upper/model_finetuned_032.json"
weights_path = "/root/workspace/ros_ws/src/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM_finetuned/v0.3_upper/best_finetuned_weights_032.h5"

# Load the model architecture
with open(json_path, 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Print model summary, which includes output shapes at each layer
model.summary()
print("Model output shape:", model.output_shape)

# Load the weights
model.load_weights(weights_path)

# Print shapes of weights per layer
for layer in model.layers:
    weights = layer.get_weights()
    if weights:  # skip layers without weights
        for i, w in enumerate(weights):
            print(f"Layer '{layer.name}' weight {i} shape: {w.shape}")

input_dim= model.input_shape

spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save to file
with open("/root/workspace/ros_ws/src/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM_finetuned/v0.3_upper/model_finetuned.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
    
