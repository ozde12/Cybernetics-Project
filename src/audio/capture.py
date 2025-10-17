import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
from datetime import datetime

def record_audio(duration=2.0, fs=16000, save_path=None):
    """
    Record audio from the microphone and optionally save to a file.
    """
    print(f"Recording {duration} seconds of audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)

    if save_path:
        # Make sure the folder exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Convert from float32 to int16 for saving as .wav
        wav.write(save_path, fs, (audio * 32767).astype(np.int16))
        print(f"Saved recording to:\n{save_path}")

    return audio
