import os
import numpy as np
from visualization.plots import plot_logmel_feature

def visualize_all_features(feature_dir, output_dir=None):
    """
    Loads all .npy log-mel files and plots them.
    Saves plots to 'output_dir' if provided,
    otherwise defaults to '<base>/data/plots'.
    """
    # Default output directory if none provided
    if output_dir is None:
        base_data_dir = os.path.abspath(os.path.join(feature_dir, ".."))  # one level up (data/)
        output_dir = os.path.join(base_data_dir, "plots")

    # Make sure the folder exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Visualizing features from: {feature_dir}")
    print(f"Plots will be saved to: {output_dir}")

    for fname in os.listdir(feature_dir):
        print("Processing:", fname)
        if fname.endswith("_logmel.npy"):
            print(" Found log-mel feature file.")
            fpath = os.path.join(feature_dir, fname)
            save_path = os.path.join(output_dir, fname.replace(".npy", ".png"))
            plot_logmel_feature(fpath, save_path=save_path)
        else:
            print("Skipping non-log-mel file.")
