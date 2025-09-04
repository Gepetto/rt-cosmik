import tensorflow as tf
import os
from tensorflow.keras.models import model_from_json
import tf2onnx
import argparse


# ─────────────── Args ───────────────
p = argparse.ArgumentParser(description="End-to-end LSTM training with streaming tf.data")
p.add_argument('--model-id', type=str, required=True)
p.add_argument('--body-part', type=str, required=True)

args = p.parse_args()

LSTM_repo_path = "/home/ngouget/Codes/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM"

model_id = args.model_id
json_file = "model_finetuned_" + model_id + ".json"
weights_file = "best_finetuned_weights_" + model_id + ".h5"
body_part = args.body_part

model_name = "model_finetuned_" + model_id

# Paths to your files
json_path = os.path.join(LSTM_repo_path, f"v0.3_{body_part}", json_file)
weights_path = os.path.join(LSTM_repo_path, f"v0.3_{body_part}", weights_file)

# Load the model architecture
with open(json_path, 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Print model summary, which includes output shapes at each layer
# model.summary()
print("Model output shape:", model.output_shape)

# Load the weights
model.load_weights(weights_path)

# # Print shapes of weights per layer
# for layer in model.layers:
#     weights = layer.get_weights()
#     if weights:  # skip layers without weights
#         for i, w in enumerate(weights):
#             print(f"Layer '{layer.name}' weight {i} shape: {w.shape}")

input_dim = model.input_shape

spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save to file
with open(os.path.join(LSTM_repo_path, f"v0.3_{body_part}", f"{model_name}.onnx"), 'wb') as f:
    f.write(onnx_model.SerializeToString())
    
