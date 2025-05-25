# Human Speech Features in Voice Authentication

---

The voice authentication model utilizes the human speech's features in the following ways:

1. **Fundamental Voice Characteristics**: The core principle of the voice authentication system is that each individual possesses a **unique voice signature based on inherent characteristics like tone, pitch, and volume**. These fundamental differences are what the system aims to capture and distinguish.

1. **Mel Frequency Cepstrum Coefficient (MFCC)**: The model employs MFCC as a primary method for **feature extraction from the voice signal**. The process involves several steps designed to mimic human auditory perception. By applying a pre-emphasis filter, segmenting the signal into frames, applying a Hamming window, performing Fast Fourier Transform, warping frequencies to the Mel scale, and then applying Discrete Cosine Transform, **12 MFCC coefficients are obtained**. These coefficients provide a compact and perceptually relevant representation of the spectral envelope of the voice, capturing important aspects of how a person pronounces sounds. These MFCC features are crucial for the CNN to learn the unique characteristics of each authorized user's voice.

1. **Additional Discriminatory Features**: To enhance the system's ability to differentiate between individuals, the model also extracts **four additional statistical features from the voice signal**:
    - **Mean Frequency**: This feature calculates the average frequency content of the voice signal, providing information about the overall spectral distribution.
    - **Standard Division (Standard Deviation)**: This measures the variability or spread of frequencies in the voice signal, reflecting the consistency or inconsistency in a person's pitch.
    - **Amplitude**: This feature quantifies the overall energy or loudness of the voice signal.
    - **Zero-Crossing Rate**: This measures the number of times the voice signal crosses the zero axis within a given frame. It provides insights into the noise characteristics and the rate of spectral change in the signal, contributing to the discriminatory power.

1. **Combined Feature Vector and CNN**: The extracted **12 MFCC features and the four additional features are combined to form a feature vector of 16 elements**. This feature vector serves as the input to the **1D Convolutional Neural Network (CNN)**. The CNN is a deep learning architecture designed to automatically learn complex patterns from the input data. During the training phase, the CNN analyzes the feature vectors of authorized users' voices, learning to identify the specific patterns and characteristics associated with each individual. In the testing or authentication phase, when a user attempts to access the cloud, their voice is processed to extract the same 16 features, and this feature vector is fed into the trained CNN. The CNN then classifies the input, determining whether it belongs to an authorized user and, if so, which user it is. The high accuracy achieved by the model (around 98%) indicates that these considered features, when processed by the deep learning CNN, are effective in uniquely identifying individuals based on their voice.
