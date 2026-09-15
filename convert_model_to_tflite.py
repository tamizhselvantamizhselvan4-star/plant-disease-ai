"""
Convert the existing Plant Disease AI Keras model to TFLite.

Run this on the Windows PC where plant_disease_resnet50.keras already exists.
It does NOT upload or modify the original .keras file.
"""

import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_resnet50.keras")
OUTPUT_PATH = os.path.join(BASE_DIR, "plant_disease_resnet50.tflite")

print("=" * 60)
print("PLANT DISEASE AI - KERAS -> TFLITE")
print("=" * 60)
print("Input :", MODEL_PATH)
print("Output:", OUTPUT_PATH)
print()

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        "plant_disease_resnet50.keras was not found in the project folder."
    )

print("Loading existing Keras model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)
print("Converting to float32 TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]
tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

print()
print("SUCCESS")
print("Created:", OUTPUT_PATH)
print(f"Size: {size_mb:.2f} MB")
print()
print("Keep the original .keras file.")
print("The new .tflite file is what Render will use.")
