import tensorflow as tf
import os

MODEL_PATH = 'model/lip_model.h5'

if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found.")
    exit(1)

print(f"Attempting to load {MODEL_PATH}...")

try:
    # Try loading as a full model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("SUCCESS: Loaded as Full Model")
    model.summary()
except Exception as e:
    print(f"FAILED to load as full model: {e}")
    
    try:
        # Try loading as weights (requires architecture to be defined)
        # For now, just check if it's a valid H5 file
        import h5py
        with h5py.File(MODEL_PATH, 'r') as f:
            print("SUCCESS: Valid H5 file structure detected (likely weights only)")
            print("Keys in file:", list(f.keys()))
    except Exception as e2:
        print(f"FAILED to read as H5 file: {e2}")
