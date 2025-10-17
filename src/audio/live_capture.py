import sounddevice as sd
import numpy as np

def record_chunk(duration=1.0, fs=16000):
    """Record a short chunk of audio (non-blocking)."""
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)
