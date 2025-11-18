"""
BASE IMPLEMNTATION OF LIVE HARMONY, MAJOR IMPLEMENTATION
Live Harmony Responder
- Listens to microphone in real time
- Detects dominant pitch (note) continuously
- Plays back a MAJOR-THIRD harmony (4 semitones up) with low latency
- Prints live note updates
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import sounddevice as sd
import librosa
from scipy.signal import butter, sosfilt

# ===== Config =====
FS = 16000                 # sample rate
BLOCK = 1024               # frames per block (~64 ms at 16 kHz)
HOP = 256                  # for YIN inside the block
FMIN = librosa.note_to_hz('A2')
FMAX = librosa.note_to_hz('C7')
HP_CUTOFF = 100            # high-pass to reduce rumble
RMS_GATE = 0.002           # ignore very quiet frames
YIN_CONF_GATE = 0.65       # require decent yin confidence 0..1
HARM_RATIO = 2 ** (3 /12.0) # minor third as pure frequency ratio
OUT_GAIN = 0.25

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]



# ===== Simple high-pass =====
def hp_filter_design(sr=FS, cutoff=HP_CUTOFF):
    return butter(4, cutoff, btype='highpass', fs=sr, output='sos')

HP_SOS = hp_filter_design(FS)

def hz_to_note_name(hz):
    try:
        return librosa.hz_to_note(float(hz))
    except Exception:
        return None

def safe_rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0

class HarmonyEngine:
    """
    Maintains oscillator phase so the synthesized harmony is continuous.
    """
    def __init__(self, fs=FS, gain=OUT_GAIN):
        self.fs = fs
        self.gain = gain
        self.phase = 0.0
        self.last_freq = None
        self.display_note = None
        self.stable_counter = 0

    def synth_chunk(self, f0, nframes):
        """
        Generate a sine harmony at major third above f0 for nframes,
        with phase continuity and soft attack/decay.
        """
        if f0 is None or f0 <= 0:
            # output silence
            return np.zeros(nframes, dtype=np.float32)

        f_harm = f0 * HARM_RATIO
        # phase-continuous oscillator
        t = (np.arange(nframes) / self.fs)
        omega = 2 * np.pi * f_harm / self.fs  # per-sample increment
        # accumulate phase efficiently
        phase_series = self.phase + np.cumsum(np.full(nframes, omega, dtype=np.float64))
        y = np.sin(phase_series)
        self.phase = float(phase_series[-1] % (2*np.pi))

        # simple 5ms fade-in/out to prevent clicks
        fade_len = min(nframes//8, int(0.005*self.fs))
        if fade_len > 0:
            env = np.ones(nframes, dtype=np.float32)
            fade = np.linspace(0, 1, fade_len, dtype=np.float32)
            env[:fade_len] *= fade
            env[-fade_len:] *= fade[::-1]
            y = y.astype(np.float32) * env
        else:
            y = y.astype(np.float32)

        return self.gain * y

    def pretty_print_note(self, f0):
        if f0 is None: 
            return
        note_name = hz_to_note_name(f0)
        if note_name is None: 
            return
        
        # just show frequencies, no snapping to named notes
        if self.last_freq is None or abs(f0 - self.last_freq) / (self.last_freq + 1e-6) > 0.02:
            self.last_freq = f0
            print(f"🎶 f0 ≈ {f0:.1f} Hz | harmony ≈ {f0 * HARM_RATIO:.1f} Hz")
        # Only print when stable for a couple frames (reduce spam)
        """if self.display_note != note_name:
            self.stable_counter += 1
            if self.stable_counter >= 2:
                self.display_note = note_name
                self.stable_counter = 0
                # Also show harmony note
                harm_note = hz_to_note_name(f0 * (2 ** (HARM_INTERVAL_SEMITONES/12.0)))
                print(f"🎶 Detected: {note_name}  |  Harmony: {harm_note}")
        else:
            self.stable_counter = 0"""


def detect_f0_yin(chunk, sr=FS):
    """
    Lightweight pitch detection on the current audio block.
    Returns (f0_hz or None, confidence 0..1).
    """
    # high-pass to reduce low rumble
    x = sosfilt(HP_SOS, chunk)

    # gate silence
    if safe_rms(x) < RMS_GATE:
        return None, 0.0

    # YIN over the block
    # Use small frame_length to keep latency low
    try:
        f0_series = librosa.yin(x, fmin=FMIN, fmax=FMAX,
                                sr=sr, frame_length=min(1024, len(x)),
                                hop_length=HOP, trough_threshold=0.1)
        # confidence from librosa.yin: lower values = better trough (pitch), approx convert
        # We'll derive a rough confidence by inverting normalized trough strength if available
        # Here: just measure stability via IQR/median as a proxy
        f0_series = f0_series[np.isfinite(f0_series)]
        if len(f0_series) == 0:
            return None, 0.0
        f0_med = float(np.median(f0_series))
        spread = float(np.median(np.abs(f0_series - f0_med))) / (f0_med + 1e-6)
        conf = max(0.0, 1.0 - 4.0*spread)  # heuristic confidence 0..1
        return (f0_med if conf > 0 else None), conf
    except Exception:
        return None, 0.0


def main():
    print("🎙️ Live Harmony: speak/sing/whistle near the mic. Press Ctrl+C to stop.")
    print("   It will play a MINOR THIRD above your detected note in real-time.\n")

    # Pick default devices or set explicitly:
    # import sounddevice as sd; print(sd.query_devices()); sd.default.device = (input_idx, output_idx)

    engine = HarmonyEngine(fs=FS, gain=OUT_GAIN)

    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            # status overflow/underflow warnings etc.
            # print(status)
            pass

        x = indata[:, 0].copy()  # mono
        f0, conf = detect_f0_yin(x, sr=FS)

        if f0 is not None and conf >= YIN_CONF_GATE:
            engine.pretty_print_note(f0)
            y = engine.synth_chunk(f0, frames)
        else:
            # no reliable pitch -> output silence
            y = np.zeros(frames, dtype=np.float32)

        outdata[:, 0] = y  # mono out
        if outdata.shape[1] > 1:
            outdata[:, 1] = y  # duplicate to right if stereo

    try:
        with sd.Stream(samplerate=FS,
                       blocksize=BLOCK,
                       dtype='float32',
                       channels=1,            # mono in
                       callback=audio_callback,
                       latency='low',
                       finished_callback=None):
            print("✅ Stream started. Listening...")
            print("   Tips: get close to the mic, sustain notes; watch the console for detected notes.")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
