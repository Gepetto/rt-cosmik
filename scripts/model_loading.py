import tensorflow as tf
import os
import keras
from keras.models import model_from_json
import tf2onnx
import sys

# Paths to your files
json_path = sys.argv[1]  # Path to the model JSON file
weights_path = sys.argv[2]  # Path to the model weights file

# Load the model architecture
with open(json_path, 'r') as json_file:
    model_json = json_file.read()


model = model_from_json(model_json)
# Print model summary, which includes output shapes at each layer
model.summary()
print("Model output shape:", model.output_shape)
