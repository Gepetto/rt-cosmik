import tensorflow as tf
import os
from tensorflow.keras.models import model_from_json
import tf2onnx
import argparse


# ─────────────── Args ───────────────
p = argparse.ArgumentParser(description="End-to-end LSTM training with streaming tf.data")
p.add_argument('--body-part', choices=['upper','lower'], required=True)
p.add_argument('--use-mocap', choices=['T','F'], default='T', required=True)        # must be 'T' for this script (mocap JCP + mocap GT)
p.add_argument('--add-noise', choices=['T','F'], default='F')
p.add_argument('--fine-tune', choices=['T','F'], default='F')
p.add_argument('--add-layer', choices=['T','F'], default='F')
p.add_argument('--use-weights', choices=['T','F'], default='F')
p.add_argument('--rot-prob', type=float, default=0.0, help='Probability to apply a random yaw rotation per window (0..1)')
p.add_argument('--rot-max-deg', type=float, default=30.0, help='Max absolute rotation in degrees (uniform in [-max, max])')
p.add_argument('--up-axis', choices=['y','z'], default='z', help='Which axis is vertical in your data (usually y or z)')
p.add_argument('--rotation-scheme', choices=['off','prob','det'], default='off',
               help="off: no rotation; prob: single random yaw per window using --rot-prob/--rot-max-deg; det: emit n evenly-spaced yaws per window")
p.add_argument('--n-rotations', type=int, default=1,
               help="When --rotation-scheme det, emit this many evenly-spaced yaw angles per window (full circle).")
# tip: set --up-axis z for your dataset
args = p.parse_args()

# Paths to your files
base_path = "/home/ngouget/Codes/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM"
json_path = os.path.join(base_path, f"v0.3_{args.body_part}", f"model_finetuned_momo_{args.body_part}_ft{args.fine_tune}_al{args.add_layer}_m{args.use_mocap}_n{args.add_noise}_w{args.use_weights}_prot{args.rot_prob}_maxrot{args.rot_max_deg}_rotscheme{args.rotation_scheme}_up{args.up_axis}_nrot{args.n_rotations}.json")
weights_path = os.path.join(base_path, f"v0.3_{args.body_part}", f"best_finetuned_weights_momo_{args.body_part}_ft{args.fine_tune}_al{args.add_layer}_m{args.use_mocap}_n{args.add_noise}_w{args.use_weights}_prot{args.rot_prob}_maxrot{args.rot_max_deg}_rotscheme{args.rotation_scheme}_up{args.up_axis}_nrot{args.n_rotations}.h5")

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
with open(os.path.join(base_path, f"v0.3_{args.body_part}", f"model_{args.body_part}_ft{args.fine_tune}_al{args.add_layer}_m{args.use_mocap}_n{args.add_noise}_w{args.use_weights}_prot{args.rot_prob}_maxrot{args.rot_max_deg}_rotscheme{args.rotation_scheme}_up{args.up_axis}_nrot{args.n_rotations}.onnx"), 'wb') as f:
    f.write(onnx_model.SerializeToString())
    