import tensorflow as tf
import os
from tensorflow.keras.models import model_from_json
import tf2onnx
import argparse


# ─────────────── Args ───────────────
p = argparse.ArgumentParser(description="End-to-end LSTM training with streaming tf.data")
p.add_argument('--model-json-path', type=str, required=True)
p.add_argument('--model-weights-path', type=str, required=True)

args = p.parse_args()

model_json_path = args.model_json_path
model_weights_path = args.model_weights_path
model_name = os.path.basename(args.model_json_path)[:-5]
body_part = os.path.dirname(args.model_json_path)

# Paths to your files
json_path = model_json_path
weights_path = model_weights_path

# Load the model architecture
with open(json_path, 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Print model summary, which includes output shapes at each layer
# model.summary()
print("Model output shape:", model.output_shape)

# Load the weights
model.load_weights(weights_path)

# Print shapes of weights per layer
for layer in model.layers:
    weights = layer.get_weights()
    if weights:  # skip layers without weights
        for i, w in enumerate(weights):
            print(f"Layer '{layer.name}' weight {i} shape: {w.shape}")

input_dim = model.input_shape

spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save to file
with open(os.path.join(body_part, f"{model_name}.onnx"), 'wb') as f:
    f.write(onnx_model.SerializeToString())
    
