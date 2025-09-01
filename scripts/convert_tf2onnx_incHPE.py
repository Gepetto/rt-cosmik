import tensorflow as tf
import os
from tensorflow.keras.models import model_from_json
import tf2onnx
import argparse


# ─────────────── Args ───────────────
p = argparse.ArgumentParser(description="End-to-end LSTM training with streaming tf.data")
p.add_argument('--body-part', choices=['upper','lower'], required=True)
p.add_argument('--add-noise', choices=['T','F'], default='F')
p.add_argument('--use-weights', choices=['T','F'], default='F')
p.add_argument('--seq-len', type=int, default=30)

args = p.parse_args()

# Paths to your files
base_path = "/home/ngouget/Codes/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM"
json_path = os.path.join(base_path, f"v0.3_{args.body_part}", f"model_finetuned_optimised_n{args.add_noise}_w{args.use_weights}_sl{args.seq_len}.json")
weights_path = os.path.join(base_path, f"v0.3_{args.body_part}", f"best_finetuned_weights_final_n{args.add_noise}_w{args.use_weights}_sl{args.seq_len}.h5")

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
with open(os.path.join(base_path, f"v0.3_{args.body_part}", f"model_{args.body_part}_n{args.add_noise}_w{args.use_weights}_sl{args.seq_len}.onnx"), 'wb') as f:
    f.write(onnx_model.SerializeToString())
    
