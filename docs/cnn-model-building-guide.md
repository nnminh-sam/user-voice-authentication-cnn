# CNN Model Building Guide

---

## **🟢 Step 1: Data Preprocessing & Feature Extraction**
Before building the CNN model, we must extract meaningful **features** from raw speech signals.

### **1.1 Load and Process Audio Data**
- Read PCM audio data from a file using **`np.fromfile()`** or **`np.frombuffer()`**.
- Normalize the audio to a range of [-1, 1] to ensure consistency.

### **1.2 Extract Key Speech Features**
We extract a **16-element feature vector** from each speech sample:
1. **12 Mel Frequency Cepstrum Coefficients (MFCCs)**
   - Apply **pre-emphasis filter** to reduce noise.
   - Perform **Fast Fourier Transform (FFT)** and convert to the **Mel scale**.
   - Apply **Discrete Cosine Transform (DCT)** to extract **12 MFCC features**.

2. **4 Additional Features**
   - **Mean Frequency** → Average frequency distribution.
   - **Standard Deviation** → Variability of pitch.
   - **Amplitude** → Overall loudness of the speech.
   - **Zero-Crossing Rate** → Number of times the signal changes sign.

### **1.3 Convert Features into Training Data**
- Save the extracted **16-element feature vectors**.
- Split into **training (80%) and testing (20%)** sets.

---

## **🟠 Step 2: Build the 1D CNN Model**
Now, we construct the **1D Convolutional Neural Network (CNN)** for learning speaker-specific patterns.

### **2.1 Define the CNN Architecture**
Use **1D convolutional layers** since our input is a **1D feature vector**.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the CNN model
def build_voice_auth_cnn(input_shape=(16, 1), num_classes=10):
    model = models.Sequential([
        layers.Conv1D(filters=32, kernel_size=3, activation='leaky_relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(filters=64, kernel_size=3, activation='leaky_relu'),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters=128, kernel_size=3, activation='leaky_relu'),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Softmax for multi-user classification
    ])
    return model

# Build the model
cnn_model = build_voice_auth_cnn()
cnn_model.summary()
```

### **2.2 Model Components**
- **Input Layer:** Accepts the **16-element voice feature vector**.
- **Convolutional Layers:** Extract **local patterns** from the voice features.
- **Leaky ReLU Activation:** Used for better gradient flow.
- **MaxPooling Layers:** Reduce dimensionality while preserving important information.
- **Flatten Layer:** Converts feature maps into a **single vector**.
- **Dense Layers:** Fully connected layers for final classification.
- **SoftMax Output Layer:** Predicts the probability of the speaker being an **authorized user**.

---

## **🔵 Step 3: Train the Model**
### **3.1 Compile the Model**
- **Loss Function:** `categorical_crossentropy` (for multi-user classification).
- **Optimizer:** Adam (with **learning rate = 0.001**).
- **Metric:** Accuracy.

```python
cnn_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
```

### **3.2 Train the Model on Voice Data**
```python
history = cnn_model.fit(
    train_features, train_labels,  # Training data
    epochs=50,
    batch_size=32,
    validation_data=(test_features, test_labels)  # Validation set
)
```
- **Training Data:** Feature vectors from multiple users' voices.
- **Epochs:** 50 (Adjust based on performance).
- **Batch Size:** 32 (Adjust for hardware capacity).
- **Validation Data:** Used to prevent overfitting.

---

## **🟣 Step 4: Test & Evaluate the Model**
### **4.1 Test the Model on New Voice Samples**
```python
test_loss, test_acc = cnn_model.evaluate(test_features, test_labels)
print(f"Test Accuracy: {test_acc:.2f}")
```
- **Test accuracy > 95%** → The model is effective for voice authentication.
- If accuracy is **low**, revisit **feature selection or model tuning**.

### **4.2 Analyze Model Performance**
```python
import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```
- If **training accuracy is high** but **validation accuracy is low**, the model is overfitting.

---

## **🟢 Step 5: Authenticate Users Using the Model**
Now, the CNN model is deployed to authenticate users based on **voice input**.

### **5.1 Load a New Voice Sample**
```python
def authenticate_user(voice_sample, cnn_model, threshold=0.9):
    # Extract 16 speech features from the sample
    features = extract_features(voice_sample)  
    features = features.reshape(1, 16, 1)  # Reshape for CNN
    
    # Predict using trained CNN model
    predictions = cnn_model.predict(features)
    predicted_user = np.argmax(predictions)
    confidence = np.max(predictions)
    
    if confidence >= threshold:
        print(f"Authenticated as User {predicted_user} (Confidence: {confidence:.2f})")
        return predicted_user
    else:
        print("Authentication Failed: Unknown User")
        return None
```
### **5.2 Steps in Authentication**
1. **Preprocess the voice input** (extract MFCC & other features).
2. **Feed into the trained CNN model**.
3. **Predict the speaker ID**.
4. **Check confidence level** (Threshold = 0.9 for security).
5. **Allow or deny authentication**.

---

## **🎯 Summary**
| Step       | Task                                                                             |
|------------|----------------------------------------------------------------------------------|
| **Step 1** | Preprocess raw audio, extract **16 speech features**                             |
| **Step 2** | Build a **1D CNN model** with convolutional, pooling, and fully connected layers |
| **Step 3** | Train the CNN using **speech feature vectors** from multiple users               |
| **Step 4** | Test model accuracy and analyze performance                                      |
| **Step 5** | Authenticate new users based on their **voice signature**                        |

---

## **🚀 Final Thoughts**
- This **CNN-based voice authentication model** efficiently recognizes and verifies users based on their **voice features**.
- Key extracted features **(MFCCs, Mean Frequency, Amplitude, ZCR, etc.)** provide **robust biometric authentication**.
- The CNN automatically **learns patterns**, improving **security & reliability**.
- **Fine-tune the learning rate, batch size, and training epochs** to optimize accuracy.

💡 **Next Steps:** Deploy this model in a **real-time voice authentication system** using a Flask API or cloud service! 🚀
