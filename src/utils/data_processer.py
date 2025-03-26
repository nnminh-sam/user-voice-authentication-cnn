import numpy as np
import librosa
import librosa.display


def read_pcm_binary(file_path, dtype=np.int16, sample_rate=16_000):
    """Reads a PCM binary file and returns the audio signal as a NumPy array.
    
    Args:
        file_path (str): Path to audio file
        dtype (object, optional): Data type of reading data. Defaults to np.int16 which is raw PCM data.
        sample_rate (int, optional): Audio data sample rate. Defaults to 16_000.

    Returns:
        (data: NDArray, sample_rate: Audio sample rate): tuple
    """

    with open(file_path, 'rb') as f:
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
