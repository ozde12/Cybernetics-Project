import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os

def plot_logmel_feature(file_path, save_path=None):
    """
    Load and plot a saved log-mel spectrogram (.npy file).
    Optionally saves the plot as an image.
    """
    log_mel = np.load(file_path)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Log-Mel Spectrogram: {os.path.basename(file_path)}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

def plot_basic_features(rms, centroid, zcr, sr=16000, hop_length=512, save_path=None):
    """
    Plot basic 1D features: RMS, Spectral Centroid, and Zero-Crossing Rate
    """
    frames = range(len(rms))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, rms, label='RMS Energy')
    plt.title("RMS Energy")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, centroid, color='orange', label='Spectral Centroid')
    plt.title("Spectral Centroid (Brightness)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, zcr, color='green', label='Zero-Crossing Rate')
    plt.title("Zero-Crossing Rate (Noisiness)")
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved feature plots to: {save_path}")
    else:
        plt.show()
