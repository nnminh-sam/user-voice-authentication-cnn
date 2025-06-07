from flask import Flask, request, jsonify
import numpy as np
import librosa
from keras.api.models import load_model
import base64

app = Flask(__name__)


def process_audio_data(raw_data, sample_rate=16_000):
    """Process raw audio data and return the audio signal as a NumPy array."""
    # Convert the raw data to numpy array
    data = np.frombuffer(raw_data, dtype=np.int16)
    return data.astype(np.float32) / np.iinfo(np.int16).max, sample_rate


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
    feature_vector = np.hstack([mfcc_mean, mean_freq, std_dev, amplitude, zero_crossing_rate])
    
    # Try to load model and make prediction
    # try:
    #     # This model prediction is for testing purpose
    #     model_path = "/Users/nnminh/Workspaces/project-voice-authentication/voice-authentication-service/model-collection/model-1/model/voice_identification.keras"
    #     model = load_model(model_path)
    #     prediction = model.predict(np.expand_dims(feature_vector, axis=0))
    #     predicted_index = int(np.argmax(prediction))
    #     confidence = float(np.max(prediction))
    #     print("Prediction:", prediction)
    #     print("Predicted index:", predicted_index)
    #     print("Confidence:", confidence)
    #     if (predicted_index == 0):
    #         print("-------------------- Anh ban than ---------------")
    #     elif (predicted_index == 1):
    #         print("-------------------- background noise ---------------")
    #     else:
    #         print("-------------------- giang oi ---------------")
    # except Exception as e:
    #     predicted_index = -1
    #     confidence = 0.0
    #     print(f"Model prediction error: {str(e)}")

    return {
        "feature_vector": feature_vector.tolist(),
        "message": "Feature vector extracted"
    }


@app.route("/extract_features", methods=["POST"])
def process_audio():
    try:
        # Get JSON data from the request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()

        if "audio" not in data or "sample_rate" not in data:
            return jsonify({"error": "Missing audio data or sample rate"}), 400

        # Decode base64 audio data
        audio_data = base64.b64decode(data["audio"])
        sample_rate = int(data["sample_rate"])

        print("Received audio data, size:", len(audio_data), "bytes")
        print("Sample rate:", sample_rate)

        # Process audio data
        data, sample_rate = process_audio_data(audio_data, sample_rate)
        result = extract_features(data, sample_rate)

        return jsonify(result), 200

    except Exception as e:
        print("Error processing audio:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
