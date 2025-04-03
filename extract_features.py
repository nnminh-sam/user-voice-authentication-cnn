import os
import random
import numpy as np
from scipy.io import wavfile
import librosa
from pathlib import Path

from src.utils.data_processer import extract_features, read_pcm_binary


def generate_header_file(features_dict):
    header_content = """// Generated audio feature vectors
#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

#include <vector>
#include <string>

namespace audio_features {
"""

    # Add feature arrays for each speaker
    for speaker, features in features_dict.items():
        # Convert features to C++ array format
        features_str = ", ".join([f"{x:.6f}f" for x in features])
        header_content += f"""
    const float {speaker}_features[] = {{{features_str}}};
    const int {speaker}_features_size = {len(features)};
"""
    
    header_content += """
}

#endif // AUDIO_FEATURES_H
"""
    
    return header_content

def main():
    dataset_dir = Path("dataset")
    speakers = ["anh_ban_than", "giang_oi"]
    features_dict = {}
    
    for speaker in speakers:
        speaker_dir = dataset_dir / speaker
        if not speaker_dir.exists():
            print(f"Directory not found: {speaker_dir}")
            continue
            
        # Get list of audio files
        audio_files = list(speaker_dir.glob("*.bin"))
        if not audio_files:
            print(f"No audio files found in {speaker_dir}")
            continue
            
        # Select random audio file
        audio_file = random.choice(audio_files)
        print(f"Processing {audio_file}")
        
        data, sample_rate = read_pcm_binary(audio_file)
        
        # Extract features
        features = extract_features(data, sample_rate)
        features_dict[speaker] = features
    
    # Generate header file
    header_content = generate_header_file(features_dict)
    
    # Write to file
    output_file = Path("include/audio_features.h")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write(header_content)
    
    print(f"Generated header file: {output_file}")

if __name__ == "__main__":
    main() 