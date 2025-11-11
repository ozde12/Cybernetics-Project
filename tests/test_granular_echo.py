import sys, os
# Add the project root (one level above src/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.algorithms.audio.granular_echo import GranularEcho
import sounddevice as sd
import numpy as np
import librosa

def record_chunk(duration=3.0, fs=16000):
    print("🎙️ Recording ambient sound or short phrase...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def test_granular_echo():
    fs = 16000
    algo = GranularEcho(sr=fs)

    # record input
    y = record_chunk(duration=3.0, fs=fs)
    print(f"Recorded {len(y)} samples")

    # fill the algorithm’s memory manually to simulate longer context
    algo.memory[:len(y)] = y

    # extract features (dummy for now)
    feats = {
        "recent_audio": y,
        "chunk_duration": 1.0,
    }

    # run algorithm several times to hear evolving texture
    for i in range(5):
        print(f"✨ Generating echo iteration {i+1}")
        result = algo.process(feats, memory={})
        out = result["audio"]
        sd.play(out, fs)
        sd.wait()

if __name__ == "__main__":
    test_granular_echo()
