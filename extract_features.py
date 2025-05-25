import random
import numpy as np
from pathlib import Path
import tensorflow as tf

from src.utils.data_processer import extract_features, read_pcm_binary


def generate_header_file(features_dict):
    interpreter = tf.lite.Interpreter(
        model_path=f"output/model/voice_identification.tflite"
    )
    interpreter.allocate_tensors()

    header_content = """// Generated audio feature vectors
#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

#include <vector>
#include <string>

namespace audio_features {
"""

    for speaker, features in features_dict.items():
        features_str = ", ".join(str(x) for x in features.flatten())
        header_content += f"""
    const int {speaker}_features[] = {{{features_str}}};
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

        audio_files = list(speaker_dir.glob("*.bin"))
        if not audio_files:
            print(f"No audio files found in {speaker_dir}")
            continue

        audio_file = random.choice(audio_files)
        print(f"Processing {audio_file}")

        data, sample_rate = read_pcm_binary(audio_file)

        features = extract_features(data, sample_rate)
        features_dict[speaker] = features

    header_content = generate_header_file(features_dict)

    output_file = Path("output/test-embedded-model/audio_features.h")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write(header_content)

    print(f"Generated header file: {output_file}")


if __name__ == "__main__":
    main()
