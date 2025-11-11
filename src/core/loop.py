import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa

from algorithms.router import choose_audio_algorithm, choose_visual_algorithm
from algorithms.audio.complementary_harmony import ComplementaryHarmony
from algorithms.audio.granular_echo import GranularEcho  # you’ll add
#from algorithms.audio.rhythm_responder import RhythmResponder  # you’ll add
from algorithms.visual.fractal_mandelbrot import MandelbrotVisual
from algorithms.visual.sacred_flower import SacredFlowerVisual
from algorithms.visual.lsystem import LSystemVisual

def compute_live_features(y, sr, hop=512):
    # basic features (you can expand)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0].mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)[0].mean()
    on_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_rate = (on_env > (on_env.mean()+on_env.std())).mean() * (sr/hop)
    # note: you can add a better key/tonal_conf here
    return {
        "rms_mean": float(rms),
        "centroid_mean": float(centroid),
        "zcr_mean": float(zcr),
        "onset_rate": float(onset_rate),
    }

def run_cybernetic_loop(fs=16000, chunk_duration=0.5, history=4.0, feedback_mix=0.15):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,5))

    audio_algos = {
        "complementary_harmony": ComplementaryHarmony(sr=fs),
        "granular_echo": GranularEcho(sr=fs),            # implement similarly
        #"rhythm_responder": RhythmResponder(sr=fs),      # implement similarly
    }
    visual_algos = {
        "fractal_mandelbrot": MandelbrotVisual(ax=ax),
        "sacred_flower": SacredFlowerVisual(ax=ax),
        "lsystem": LSystemVisual(ax=ax),
    }
    memory = {}

    # feedback buffer to mix system output back into next input
    fb = np.zeros(int(fs*chunk_duration), dtype=np.float32)

    try:
        while True:
            # ---- capture ----
            y = sd.rec(int(chunk_duration*fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            y = y[:,0]

            # mix previous system audio (feedback)
            y_fb = np.clip(y + feedback_mix*fb[:len(y)], -1.0, 1.0)

            # ---- features ----
            feats = compute_live_features(y_fb, fs)
            feats["recent_audio"] = y_fb
            feats["chunk_duration"] = chunk_duration

            # ---- choose algorithms ----
            a_name = choose_audio_algorithm(feats)
            v_name = choose_visual_algorithm(feats)
            a_algo = audio_algos[a_name]
            v_algo = visual_algos[v_name]

            # ---- run algorithms ----
            a_out = a_algo.process(feats, memory)  # {"audio": buffer, "sr": fs, "meta": ...}
            fb = a_out["audio"]

            # play system output (heard by mic next iteration via speakers)
            sd.play(fb, fs, blocking=False)

            # draw visuals
            v_algo.draw(feats, memory)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        plt.ioff()
        plt.show()
