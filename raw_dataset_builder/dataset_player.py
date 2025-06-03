import os
import simpleaudio as sa
from pydub import AudioSegment
import numpy as np
from datetime import datetime

ROOT_PATH: str = "podcast_mp3_processed"

def play_processed_audio(label, root_folder=ROOT_PATH, sample_rate=44100, num_channels=2, bytes_per_sample=2):
    label_path = os.path.join(root_folder, label)
    print("debug >>> ", label_path)
    if not os.path.exists(label_path) or not os.path.isdir(label_path):
        print(f"Label '{label}' not found!")
        return
    
    audio_files = sorted([f for f in os.listdir(label_path) if f.endswith(".bin")])
    
    if not audio_files:
        print(f"No processed audio files found for label '{label}'!")
        return
    
    for file_name in audio_files:
        file_path = os.path.join(label_path, file_name)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Playing: {file_name}")
        
        with open(file_path, "rb") as bin_file:
            raw_data = np.frombuffer(bin_file.read(), dtype=np.int16)
        
        wave_obj = sa.play_buffer(raw_data, num_channels=num_channels, bytes_per_sample=bytes_per_sample, sample_rate=sample_rate)
        wave_obj.wait_done()
        
if __name__ == "__main__":
    label_to_play = input("Enter label name to play: ")
    play_processed_audio(label_to_play)
