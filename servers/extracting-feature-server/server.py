from flask import Flask, request, jsonify
import numpy as np
import librosa
from keras.api.models import load_model
import os

app = Flask(__name__)


def read_pcm_binary(file_path, dtype=np.int16, sample_rate=16_000):
    """Reads a PCM binary file and returns the audio signal as a NumPy array.

    Args:
        file_path (str): Path to audio file
        dtype (object, optional): Data type of reading data. Defaults to np.int16 which is raw PCM data.
        sample_rate (int, optional): Audio data sample rate. Defaults to 16_000.

    Returns:
        (data: NDArray, sample_rate: Audio sample rate): tuple
    """

    with open(file_path, "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=dtype)
    return raw_data.astype(np.float32) / np.iinfo(dtype).max, sample_rate


def extract_features(data, sample_rate):
    """Extracts MFCC and additional voice features.

    Args:
        data (NDArray): Audio data as NDArray
        sample_rate (int): Audio data sample rate

    Returns:
        Audio data features
    """

    # * Extract 12 MFCC features
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=12)
    mfcc_mean = np.mean(mfcc, axis=1)  # * Take the mean of each MFCC feature

    # * Additional features
    mean_freq = np.mean(librosa.fft_frequencies(sr=sample_rate))
    std_dev = np.std(data)
    amplitude = np.sum(np.abs(data))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(data))

    # * Combine all features into a single vector
    return np.hstack([mfcc_mean, mean_freq, std_dev, amplitude, zero_crossing_rate])


@app.route("/extract_features", methods=["POST"])
def process_audio():
    try:
        # Get JSON data from the request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()

        if "audio" not in data or "sample_rate" not in data:
            return jsonify({"error": "Missing audio data or sample rate"}), 400

        # Decode base64 audio data to bytes
        audio_path = data["audio"]
        sample_rate = int(data["sample_rate"])

        print("Received audio data and sample rate:", sample_rate)
        print("Audio data:", audio_path)

        data, sample_rate = read_pcm_binary(audio_path)

        feature_vector = extract_features(data, sample_rate)

        try:
            load_model_path = "/Users/nnminh/Workspaces/voice-authentication-service/output/model/voice_identification.keras"
            print(os.path.exists(load_model_path))
            model = load_model(load_model_path)
            prediction = model.predict(np.expand_dims(feature_vector, axis=0))
            print("Prediction shape:", prediction.shape)
            print("Prediction:", prediction)
            predicted_index = np.argmax(prediction)
            confidence = np.max(prediction)
            print("Predicted index:", predicted_index)
            print("Confidence:", confidence)
            if predicted_index == 0:
                print("Anh ban than")
            else:
                print("Giang oi")
        except Exception as e:
            print("Error loading model:", str(e))
            return jsonify({"error": "Model loading failed"}), 500

        feature_vector = feature_vector.tolist()

        return jsonify(
            {
                "message": "Audio data received successfully",
                "feature_vector": feature_vector,
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)  # Run Flask server
