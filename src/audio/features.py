import librosa
import numpy as np
import os

def extract_features(file_path, n_mels=64, hop_length=512, n_fft=1024, sr=16000):
    """
    Extracts log-mel spectrogram and basic spectral features from an audio file.

    Returns:
        features (dict): contains log-mel spectrogram, RMS, centroid, and zero-crossing rate
    """
    print(f"Extracting features from: {file_path}")
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    
    # 1. Log-Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                              n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # 2. RMS Energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # 3. Spectral Centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

    # 4. Zero Crossing Rate (noisiness)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]

    features = {
        "log_mel": log_mel,
        "rms": rms,
        "centroid": centroid,
        "zcr": zcr,
        "sr": sr
    }

    return features
