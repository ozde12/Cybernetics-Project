import os
import numpy as np
from audio.features import extract_features

def process_all_recordings(input_dir, output_dir):
    """
    Extract features for all .wav files in input_dir and save to output_dir as .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        print("Processing:", fname)
        if fname.lower().endswith(".wav"):
            print("Found .wav file.")
            fpath = os.path.join(input_dir, fname)
            features = extract_features(fpath)

            base_name = os.path.splitext(fname)[0]
            save_path = os.path.join(output_dir, f"{base_name}_logmel.npy")
            np.save(save_path, features["log_mel"])

            print(f"✅ Saved features for {fname} → {save_path}")
        else:
            print("Skipping non-audio file.")
