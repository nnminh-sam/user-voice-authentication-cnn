import os
import json
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import data_processer as DataProcesser
import utils.logger as Logger
import model.model as Model


def read_data_and_extract_feature(dataset_path: str):
    """Read data from dataset folder and extract features from each audio data"""
    features = []
    labels = []
    
    for label_folder in os.listdir(dataset_path):
        label_folder_path = os.path.join(dataset_path, label_folder)
        
        if not os.path.isdir(label_folder_path):
            continue
        
        for file_name in os.listdir(label_folder_path):
            if not file_name.endswith(".bin"):
                Logger.log(f"Skipped file: {file_name}")
                continue
            
            Logger.log(f"Processing file: {file_name}")
            file_path = os.path.join(label_folder_path, file_name)
            
            data, sample_rate = DataProcesser.read_pcm_binary(file_path)
            Logger.log(f"Data from {file_name} is read")
            
            feature_vector = DataProcesser.extract_features(data, sample_rate)
            features.append(feature_vector)
            labels.append(label_folder)
    
    unique_labels = sorted(list(set(labels)))  # Sort to maintain consistency
    label_map = {name: i for i, name in enumerate(unique_labels)}
    reverse_label_map = {i: name for name, i in label_map.items()}  # Reverse mapping
    
    numeric_labels = np.array([label_map[name] for name in labels])
    
    return np.array(features), numeric_labels, label_map, reverse_label_map


def split_and_save_sets(features, file_names, train_path, val_path, test_path):
    """Split features and labels into training, validation and test sets

    Args:
        features (NPArray): Array of features
        file_names (NPArray): Array of label (audio file name)
        train_path (str): Path to train set
        val_path (str): Path to validation set
        test_path (str): Path to test set
    """

    # * 70% Train, 15% Validation, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(features, file_names, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # * Save the splits
    np.save(os.path.join(train_path, f"X_train.npy"), X_train)
    np.save(os.path.join(val_path, "X_val.npy"), X_val)
    np.save(os.path.join(test_path, "X_test.npy"), X_test)

    np.save(os.path.join(train_path, "y_train.npy"), y_train)
    np.save(os.path.join(val_path, "y_val.npy"), y_val)
    np.save(os.path.join(test_path, "y_test.npy"), y_test)

    Logger.log(f"Dataset split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    return (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
    )


def authenticate_user(voice_path, cnn_model, label_map_path, threshold=0.9):
    """User authentication using CNN model

    Args:
        voice_path (_type_): _description_
        cnn_model (_type_): _description_
        label_map_path (_type_): _description_
        threshold (float, optional): _description_. Defaults to 0.9.

    Returns:
        _type_: _description_
    """
    
    Logger.log(f"Identifing {voice_path}")
    with open(label_map_path, "r") as f:
        reverse_label_map = json.load(f)

    data, sample_rate = DataProcesser.read_pcm_binary(voice_path)
    Logger.log(f"Data from {voice_path} is read")
    
    feature_vector = DataProcesser.extract_features(data, sample_rate)
    feature_vector = feature_vector.reshape(1, 16, 1)
    
    # Predict using trained CNN model
    predictions = cnn_model.predict(feature_vector)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)
    
    predicted_label = reverse_label_map[str(predicted_index)]  # Convert key to string since JSON keys are strings
    
    if confidence >= threshold:
        print(f"Authenticated as {predicted_label} (Confidence: {confidence:.2f})")
        return predicted_label
    else:
        print("Authentication Failed: Unknown User")
        return None



def main():
    DATASET_PATH: str = "dataset"
    OUTPUT_PATH: str = "output"
    LABEL_MAP_PATH: str = "output/label_map.json"
    TRAIN_SET_PATH: str = "features/train"
    VALIDATION_SET_PATH: str = "features/validation"
    TEST_SET_PATH: str = "features/test"
    USAGE_TEST_PATH: str = "dataset/giang_oi/nguoi_lon_ra_o_rieng_72.bin"
    
    features, labels, label_map, reverse_label_map = read_data_and_extract_feature(DATASET_PATH)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_and_save_sets(
        features, labels, TRAIN_SET_PATH, VALIDATION_SET_PATH, TEST_SET_PATH
    )

    # Save the label map for later use
    os.makedirs("output", exist_ok=True)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(reverse_label_map, f)
    
    num_classes = len(set(labels))
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    X_train = X_train.reshape(X_train.shape[0], 16, 1)
    X_val = X_val.reshape(X_val.shape[0], 16, 1)
    X_test = X_test.reshape(X_test.shape[0], 16, 1)

    
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    model = Model.build_voice_auth_cnn(num_classes)
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    # history = model.fit(
    #     X_train, y_train,
    #     validation_data=(X_val, y_val),
    #     epochs=100,
    #     batch_size=32
    # )
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=callbacks)
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    Logger.log(f"Test Accuracy: {test_acc:.2f}")
    
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("output/model.png")

    authenticate_user(USAGE_TEST_PATH, model, LABEL_MAP_PATH)


main()
