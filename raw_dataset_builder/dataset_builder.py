import os
import numpy as np
from pydub import AudioSegment

def extract_audio_data(input_folder, output_folder, segment_duration=10_000, sample_rate=16000):
    os.makedirs(output_folder, exist_ok=True)
    
    for label in os.listdir(input_folder):
        label_path = os.path.join(input_folder, label)
        output_label_path = os.path.join(output_folder, label)
        os.makedirs(output_label_path, exist_ok=True)
        
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                if file_name.endswith(".mp3"):
                    input_file_path = os.path.join(label_path, file_name)
                    audio = AudioSegment.from_mp3(input_file_path)
                    
                    # Convert to mono and set sample rate
                    audio = audio.set_channels(1).set_frame_rate(sample_rate)
                    
                    num_segments = len(audio) // segment_duration + (1 if len(audio) % segment_duration > 0 else 0)
                    
                    base_name = os.path.splitext(file_name)[0]
                    
                    for i in range(num_segments):
                        start_time = i * segment_duration
                        end_time = min((i + 1) * segment_duration, len(audio))
                        segment = audio[start_time:end_time]
                        
                        pcm_data = np.array(segment.get_array_of_samples(), dtype=np.int16)
                        index_str = str(i).zfill(len(str(num_segments - 1)))
                        output_file_path = os.path.join(output_label_path, f"{base_name}_{index_str}.bin")
                        
                        with open(output_file_path, "wb") as bin_file:
                            bin_file.write(pcm_data.tobytes())
                        
                        print(f"Saved: {output_file_path}")

if __name__ == "__main__":
    input_root = "podcast_mp3"
    output_root = "podcast_mp3_processed"
    extract_audio_data(input_root, output_root)
