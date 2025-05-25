import tensorflow as tf


def generate_header_file(features_dict, model_path):
    interpreter = tf.lite.Interpreter(model_path)
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
