"""
melody_to_harmony_file.py
-------------------------
Processes an input audio file or generated sound (no microphone),
extracts its melody and rhythm, detects key, and generates a 
complementary harmony response using music theory.
"""

import sys, os, collections
import numpy as np
import sounddevice as sd
import librosa

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.algorithms.audio.complementary_harmony import ComplementaryHarmony


# === Constants ===
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


# === Utility Functions ===
def detect_key(notes):
    """Estimate key (root + mode) from detected note sequence."""
    if not notes:
        return "C", "major"

    counts = collections.Counter([n[0][:-1] for n in notes])  # remove octave number
    root = counts.most_common(1)[0][0]
    minor_likelihood = any("b" in n for n in counts.keys())
    return root, "minor" if minor_likelihood else "major"


def extract_notes(y, sr):
    """Extract monophonic notes (pitch + duration) from audio."""
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('A2'), fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0, sr=sr)

    notes = []
    prev_note, start_time = None, None
    for t, pitch, voiced in zip(times, f0, voiced_flag):
        if voiced and pitch is not None:
            note_name = librosa.hz_to_note(pitch)
            if prev_note is None:
                prev_note, start_time = note_name, t
            elif note_name != prev_note:
                duration = t - start_time
                notes.append((prev_note, duration))
                prev_note, start_time = note_name, t
        elif prev_note is not None:
            duration = t - start_time
            notes.append((prev_note, duration))
            prev_note, start_time = None, None
    return notes


def generate_harmony(notes, key_root='C', mode='major'):
    """Generate harmony using music theory intervals (major/minor third)."""
    harmonized = []
    interval = 4 if mode == "major" else 3  # semitone interval

    for note_name, dur in notes:
        hz = librosa.note_to_hz(note_name)
        harmonic_note = librosa.hz_to_note(hz * (2 ** (interval / 12)))
        harmonized.append((harmonic_note, dur))
    return harmonized


def synthesize_notes(notes, sr=16000):
    """Render each note as a sine tone."""
    audio = np.zeros(1)
    for note_name, dur in notes:
        hz = librosa.note_to_hz(note_name)
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        tone = 0.3 * np.sin(2 * np.pi * hz * t)
        audio = np.concatenate([audio, tone])
    return audio


# === Main Processing ===
def process_audio_input(y, sr):
    """Full pipeline: analyze melody, generate harmony, and play."""
    print("🎧 Extracting melody and harmony...")

    # Extract notes
    notes = extract_notes(y, sr)
    if not notes:
        print("⚠️ No clear notes detected.")
        return

    key_root, mode = detect_key(notes)
    print(f"🎼 Detected key: {key_root} {mode}")
    print(f"🎵 Detected notes: {notes}")

    # Generate harmony
    harmonized = generate_harmony(notes, key_root=key_root, mode=mode)
    print(f"🎶 Generated harmony: {harmonized}")

    # Synthesize and play
    response = synthesize_notes(harmonized, sr=sr)
    print("🔊 Playing harmony response...")
    sd.play(response, sr)
    sd.wait()
    print("✅ Done.\n")


# === Test Mode: choose input ===
def main():
    fs = 16000
    mode = "generate"  # change to "generate" or "mic" later if needed

    if mode == "file":
        # Example: load your own sound file here
        path = r"C:\Users\ozdep\Documents\example_input.wav"  # 👈 change this
        y, sr = librosa.load(path, sr=fs, mono=True)
        print(f"🎵 Loaded file: {path} (duration {len(y)/sr:.2f}s)")
        process_audio_input(y, sr)

    elif mode == "generate":
        # Generate a synthetic melody (A4–B4–C5)
        melody_notes = [("A4", 0.5), ("B4", 0.5), ("C5", 1.0)]
        y = synthesize_notes(melody_notes, sr=fs)
        print("🎹 Generated synthetic melody A–B–C.")
        process_audio_input(y, fs)

    else:
        print("🎙️ Microphone mode not used here.")


if __name__ == "__main__":
    main()
