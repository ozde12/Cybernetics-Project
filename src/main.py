from audio.capture import record_audio
import os
from datetime import datetime
from core.feature_runner import process_all_recordings

def main():
    # Define your absolute path (Windows)
    base_path = r"C:\Users\ozdep\Documents\art & tech\Cybernetics-Project\cybernetic-sound-system\data\raw"

    # Create a timestamped filename (optional, to avoid overwriting)
    filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    save_path = os.path.join(base_path, filename)

    # Record and save
    audio = record_audio(duration=3.0, save_path=save_path)
    print("Audio shape:", audio.shape)

    input_dir = r"C:\Users\ozdep\Documents\art & tech\Cybernetics-Project\cybernetic-sound-system\data"
    output_dir = os.path.join(input_dir, "features")

    process_all_recordings(input_dir, output_dir)

if __name__ == "__main__":
    main()
