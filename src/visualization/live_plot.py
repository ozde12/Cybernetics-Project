import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import threading

def extract_logmel_chunk(y, sr=16000, n_mels=64, hop_length=512, n_fft=1024):
    """Convert an audio chunk into a log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

def record_chunk(duration=1.0, fs=16000):
    """Record a short chunk of audio."""
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def live_logmel_visualizer(fs=16000, chunk_duration=0.5, history=8.0, n_mels=64, hop_length=512):
    """
    Continuously record and update a scrolling log-mel spectrogram.
    Press Enter or Ctrl+C to stop.
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    stop_flag = False

    def wait_for_enter():
        nonlocal stop_flag
        input("  Live recording started. Press Enter to stop...\n")
        stop_flag = True

    threading.Thread(target=wait_for_enter, daemon=True).start()

    max_chunks = int(history / chunk_duration)
    logmel_buffer = []

    try:
        while not stop_flag:
            # --- Record a new short chunk ---
            audio = record_chunk(duration=chunk_duration, fs=fs)
            rms = np.sqrt(np.mean(audio**2))
            print(f"Chunk RMS: {rms:.6f}")
            logmel = extract_logmel_chunk(audio, sr=fs, n_mels=n_mels, hop_length=hop_length)

            # --- Keep a sliding history buffer ---
            logmel_buffer.append(logmel)
            if len(logmel_buffer) > max_chunks:
                logmel_buffer.pop(0)

            combined = np.concatenate(logmel_buffer, axis=1)

            # --- Clear and redraw the spectrogram ---
            ax.clear()
            librosa.display.specshow(
                combined, sr=fs, hop_length=hop_length,
                x_axis='time', y_axis='mel', cmap='magma',
                ax=ax, vmin=-80, vmax=0)
            ax.set(title="Live Log-Mel Spectrogram")

            # --- Refresh plot ---
            plt.pause(0.001)

    except KeyboardInterrupt:
        print("\n Stopped by user.")
    finally:
        plt.ioff()
        plt.show()
