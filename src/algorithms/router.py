"""
LIVE ALGORITHM ROUTER
---------------------
Continuously listens to the microphone and, for each short audio chunk, decides:

- If there's no clear pitched sound  → run Granular Echo
- If there's a clear, bright pitch  → run Major Harmony (major 3rd above f0)
- If there's a clear, darker pitch  → run Minor Harmony (minor 3rd above f0)

This is a "conductor" that triggers one of three sound algorithms.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import sounddevice as sd
import librosa
from scipy.signal import butter, sosfilt

# ===== Config =====
FS = 16000
BLOCK = 1024
HOP = 256
FMIN = librosa.note_to_hz('A2')
FMAX = librosa.note_to_hz('C7')
HP_CUTOFF = 100
RMS_GATE = 0.0015        # below this: treat as very quiet
YIN_CONF_GATE = 0.6      # how confident pitch must be to count as “pitched”
BRIGHT_CENTROID_HZ = 2000.0  # above → "bright" → major, below → minor

OUT_GAIN = 0.25
GRANULAR_MIX = 0.8       # how strong the granular echo is vs dry


# ===== Filters & helpers =====
def hp_filter_design(sr=FS, cutoff=HP_CUTOFF):
    return butter(4, cutoff, btype='highpass', fs=sr, output='sos')

HP_SOS = hp_filter_design(FS)

def safe_rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


# ===== Pitch detection (YIN) =====
def detect_f0_yin(chunk, sr=FS):
    x = sosfilt(HP_SOS, chunk)

    if safe_rms(x) < RMS_GATE:
        return None, 0.0

    try:
        f0_series = librosa.yin(
            x, fmin=FMIN, fmax=FMAX,
            sr=sr,
            frame_length=min(1024, len(x)),
            hop_length=HOP,
            trough_threshold=0.1
        )
        f0_series = f0_series[np.isfinite(f0_series)]
        if len(f0_series) == 0:
            return None, 0.0
        f0_med = float(np.median(f0_series))
        spread = float(np.median(np.abs(f0_series - f0_med))) / (f0_med + 1e-6)
        conf = max(0.0, 1.0 - 4.0 * spread)
        return (f0_med if conf > 0 else None), conf
    except Exception:
        return None, 0.0


# ===== Feature computation =====
def compute_features(chunk, sr=FS):
    x = sosfilt(HP_SOS, chunk)
    rms = safe_rms(x)

    # spectral centroid for brightness
    if np.all(x == 0):
        centroid = 0.0
    else:
        cent = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
        centroid = float(np.mean(cent))

    f0, conf = detect_f0_yin(chunk, sr=sr)
    return {
        "rms": rms,
        "centroid": centroid,
        "f0": f0,
        "f0_conf": conf
    }


# ===== Algorithms =====

class HarmonyEngine:
    """Simple sine-based harmony at a fixed ratio above f0 (major or minor)."""
    def __init__(self, fs, ratio, gain=OUT_GAIN):
        self.fs = fs
        self.ratio = ratio
        self.gain = gain
        self.phase = 0.0

    def synth_chunk(self, f0, nframes):
        if f0 is None or f0 <= 0:
            return np.zeros(nframes, dtype=np.float32)
        f_harm = f0 * self.ratio
        omega = 2 * np.pi * f_harm / self.fs
        phase_series = self.phase + np.cumsum(np.full(nframes, omega, dtype=np.float64))
        y = np.sin(phase_series)
        self.phase = float(phase_series[-1] % (2*np.pi))

        fade_len = min(nframes // 8, int(0.005 * self.fs))
        if fade_len > 0:
            env = np.ones(nframes, dtype=np.float32)
            fade = np.linspace(0, 1, fade_len, dtype=np.float32)
            env[:fade_len] *= fade
            env[-fade_len:] *= fade[::-1]
            y = y.astype(np.float32) * env
        else:
            y = y.astype(np.float32)

        return self.gain * y


class GranularEchoEngine:
    """
    Very simple echo-ish engine:
    - keeps a delay buffer
    - outputs input + soft delayed copy
    It's not fully granular, but it's a good starting point for the “echo mode”.
    """
    def __init__(self, fs, delay_s=0.4, feedback=0.4, mix=GRANULAR_MIX):
        self.fs = fs
        self.delay_samples = int(delay_s * fs)
        self.feedback = feedback
        self.mix = mix
        self.buffer = np.zeros(self.delay_samples, dtype=np.float32)
        self.idx = 0

    def process_chunk(self, x):
        n = len(x)
        out = np.zeros(n, dtype=np.float32)
        buf = self.buffer

        for i in range(n):
            delayed = buf[self.idx]
            # write current input + feedback
            buf[self.idx] = x[i] + self.feedback * delayed
            # read out
            out[i] = (1 - self.mix) * x[i] + self.mix * delayed
            self.idx = (self.idx + 1) % self.delay_samples

        return out


# ===== Router logic =====

def decide_mode(features, prev_mode):
    """
    Decide which algorithm to use based on features.
    Returns 'granular', 'major', or 'minor'.
    """
    rms = features["rms"]
    centroid = features["centroid"]
    f0 = features["f0"]
    conf = features["f0_conf"]

    # No strong pitch or very quiet → granular
    if f0 is None or conf < YIN_CONF_GATE or rms < RMS_GATE * 1.5:
        return "granular"

    # Pitched sound: choose between major / minor by brightness
    if centroid >= BRIGHT_CENTROID_HZ:
        return "major"
    else:
        return "minor"


def main():
    print("🎙️ Live Router running. Speak/sing/whistle/play near the mic. Ctrl+C to stop.")
    print("   Modes:")
    print("    - Granular Echo: when quiet or no clear pitch")
    print("    - Major Harmony: bright, pitched sounds")
    print("    - Minor Harmony: darker, pitched sounds\n")

    major_engine = HarmonyEngine(fs=FS, ratio=2 ** (4/12.0), gain=OUT_GAIN)  # major third
    minor_engine = HarmonyEngine(fs=FS, ratio=2 ** (3/12.0), gain=OUT_GAIN)  # minor third
    echo_engine = GranularEchoEngine(fs=FS)

    router_state = {
        "mode": "granular",
        "pending_mode": None,
        "streak": 0
    }
    HYSTERESIS_FRAMES = 4  # need 4 consecutive chunks requesting new mode

    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            # print(status)
            pass

        x = indata[:, 0].copy()
        feats = compute_features(x, sr=FS)

        desired_mode = decide_mode(feats, router_state["mode"])

        # hysteresis: avoid rapid flipping
        if desired_mode != router_state["mode"]:
            if router_state["pending_mode"] == desired_mode:
                router_state["streak"] += 1
            else:
                router_state["pending_mode"] = desired_mode
                router_state["streak"] = 1

            if router_state["streak"] >= HYSTERESIS_FRAMES:
                router_state["mode"] = desired_mode
                router_state["pending_mode"] = None
                router_state["streak"] = 0
                print(f"🔀 Switched mode → {router_state['mode']}")
        else:
            router_state["pending_mode"] = None
            router_state["streak"] = 0

        mode = router_state["mode"]

        if mode == "granular":
            y = echo_engine.process_chunk(x)
        elif mode == "major":
            y = major_engine.synth_chunk(feats["f0"], frames)
        else:  # "minor"
            y = minor_engine.synth_chunk(feats["f0"], frames)

        outdata[:, 0] = y
        if outdata.shape[1] > 1:
            outdata[:, 1] = y

    try:
        with sd.Stream(
            samplerate=FS,
            blocksize=BLOCK,
            dtype='float32',
            channels=1,
            callback=audio_callback,
            latency='low'
        ):
            print("✅ Stream started. Listening & routing...")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
