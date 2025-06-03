import os
import json
import time
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import src.models.model as Models

# import src.model.model as Models
import src.utils.logger as Logger
from src.utils.logger import LogLevel
import src.utils.data_processer as DataProcesser
import src.utils.model_conversion as ModelConversion
import src.utils.dataset_config_loader as ConfigLoader


def read_data_and_extract_features(
    dataset_path: str, output_path: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """Reads audio data from the dataset folder and extracts features from each audio file.

    Args:
        dataset_path (str): The path to the dataset folder containing audio files organized in subfolders by label.

    Returns:
        tuple: A tuple containing:
            - features (np.ndarray): An array of extracted feature vectors.
            - numeric_labels (np.ndarray): An array of numeric labels corresponding to the features.
            - label_map (dict): A mapping from label names to numeric indices.
            - reverse_label_map (dict): A mapping from numeric indices back to label names.
    """

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

            file_path = os.path.join(label_folder_path, file_name)

            data, sample_rate = DataProcesser.read_pcm_binary(file_path)
            Logger.log(f"Loaded file: {file_name}")

            feature_vector = DataProcesser.extract_features(data, sample_rate)
            features.append(feature_vector)
            labels.append(label_folder)

    unique_labels = sorted(set(labels))
    label_map = {name: i for i, name in enumerate(unique_labels)}
    reverse_label_map = {i: name for name, i in label_map.items()}
    numeric_labels = np.array([label_map[name] for name in labels])

    label_output_path: str = f"{output_path}/dataset"
    os.makedirs(label_output_path, exist_ok=True)
    np.save(os.path.join(label_output_path, "numeric_labels.npy"), numeric_labels)
    Logger.log(f"Numeric labels saved at {label_output_path}/numeric_labels.npy")

    return np.array(features), numeric_labels, label_map, reverse_label_map


def split_and_save_datasets(
    features: np.ndarray, labels: np.ndarray, output_path: str
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Splits features and labels into training, validation, and test sets, and saves them to files.

    Args:
        features (np.ndarray): Array of extracted features.
        labels (np.ndarray): Array of numeric labels corresponding to the features.
        output_path (str): The output path where the datasets will be saved.

    Returns:
        tuple: A tuple containing:
            - (X_train, y_train): Training features and labels.
            - (X_val, y_val): Validation features and labels.
            - (X_test, y_test): Test features and labels.
    """

    train_path = f"{output_path}/dataset/features/train"
    val_path = f"{output_path}/dataset/features/validation"
    test_path = f"{output_path}/dataset/features/test"

    try:
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
    except Exception as e:
        Logger.log(e, LogLevel.ERROR)
        return

    # Split dataset into train, validation, and test sets with 70% Train, 15% Validation, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Save the splits
    try:
        np.save(os.path.join(train_path, "X_train.npy"), X_train)
        np.save(os.path.join(train_path, "y_train.npy"), y_train)
        np.save(os.path.join(val_path, "X_val.npy"), X_val)
        np.save(os.path.join(val_path, "y_val.npy"), y_val)
        np.save(os.path.join(test_path, "X_test.npy"), X_test)
        np.save(os.path.join(test_path, "y_test.npy"), y_test)

        Logger.log(
            f"Dataset split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}"
        )
        Logger.log(f"Train set saved at: {train_path}")
        Logger.log(f"Validation set saved at: {val_path}")
        Logger.log(f"Test set saved at: {test_path}")
    except Exception as e:
        Logger.log(e, LogLevel.ERROR)
        return

    return (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
    )


def categorical_and_reshape_datasets(
    labels: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Converts labels to categorical format and reshapes dataset for model input.

    Args:
        labels (list): List of original labels.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        tuple: A tuple containing:
            - (X_train, y_train): Reshaped training features and categorical labels.
            - (X_val, y_val): Reshaped validation features and categorical labels.
            - (X_test, y_test): Reshaped test features and categorical labels.
    """

    num_classes = len(set(labels))
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    X_train = X_train.reshape(X_train.shape[0], 16, 1)
    X_val = X_val.reshape(X_val.shape[0], 16, 1)
    X_test = X_test.reshape(X_test.shape[0], 16, 1)

    Logger.log(f"X_train shape: {X_train.shape}")
    Logger.log(f"X_val shape: {X_val.shape}")
    Logger.log(f"X_test shape: {X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def process_dataset(
    dataset_path: str, output_path: str
) -> Tuple[
    List[str],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Processes the dataset by reading, extracting features, splitting, and saving them.

    Args:
        dataset_path (str): The path to the dataset folder.
        output_path (str): The output path for saving processed datasets.

    Returns:
        tuple: A tuple containing:
            - labels (list): The original labels.
            - (X_train, y_train): Training features and labels.
            - (X_val, y_val): Validation features and labels.
            - (X_test, y_test): Test features and labels.
    """

    data_labeling_path = f"{output_path}/dataset"

    # Read dataset from files and extract features
    features, labels, label_map, reverse_label_map = read_data_and_extract_features(
        dataset_path=dataset_path, output_path=output_path
    )

    # Split dataset into different sets and save for future usage
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_and_save_datasets(
        features=features, labels=labels, output_path=output_path
    )

    # Save dataset label map to JSON file
    os.makedirs(data_labeling_path, exist_ok=True)
    with open(f"{data_labeling_path}/label_map.json", "w") as f:
        json.dump(reverse_label_map, f)

    return labels, (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_dataset(
    output_path: str,
) -> Tuple[
    List[str],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Loads the training, validation, and test datasets from saved files.

    Args:
        output_path (str): The path where the datasets are saved.

    Returns:
        tuple: A tuple containing:
            - (X_train, y_train): Loaded training features and labels.
            - (X_val, y_val): Loaded validation features and labels.
            - (X_test, y_test): Loaded test features and labels.
    """

    labels_path = f"{output_path}/dataset/numeric_labels.npy"

    labels = np.load(labels_path)

    train_path = f"{output_path}/dataset/features/train"
    val_path = f"{output_path}/dataset/features/validation"
    test_path = f"{output_path}/dataset/features/test"

    X_train = np.load(os.path.join(train_path, "X_train.npy"))
    y_train = np.load(os.path.join(train_path, "y_train.npy"))
    X_val = np.load(os.path.join(val_path, "X_val.npy"))
    y_val = np.load(os.path.join(val_path, "y_val.npy"))
    X_test = np.load(os.path.join(test_path, "X_test.npy"))
    y_test = np.load(os.path.join(test_path, "y_test.npy"))

    Logger.log(
        f"Loaded dataset: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}"
    )

    return labels, (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_model(
    model_name: str,
    model_output_path: str,
    load_sets_from_file: bool,
    sets_path: Optional[str] = None,
    labels: Optional[List[str]] = None,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Trains the CNN model using the provided datasets.

    Args:
        labels (list): List of original labels.
        model_name (str): The name of the model to be saved.
        model_output_path (str): The output path for saving the trained model.
        load_sets_from_file (bool): Whether to load datasets from files.
        sets_path (str, optional): Path for loading datasets from files.
        X_train (np.ndarray, optional): Training features.
        y_train (np.ndarray, optional): Training labels.
        X_val (np.ndarray, optional): Validation features.
        y_val (np.ndarray, optional): Validation labels.
        X_test (np.ndarray, optional): Test features.
        y_test (np.ndarray, optional): Test labels.

    Returns:
        tuple: A tuple containing:
            - model: The trained CNN model.
            - history: The training history.
    """

    # Load datasets from file if required
    if load_sets_from_file:
        labels, (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(
            output_path=model_output_path
        )
        Logger.log(
            f"Train, validation, and test sets loaded from files at: {sets_path}"
        )

    # Define callbacks for optimizing model training process
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    # Build model
    num_classes = len(set(labels))
    model = Models.CNNModel.build(num_classes)
    model.summary()
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    # Categoricalize and reshape dataset before fitting into model
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = (
        categorical_and_reshape_datasets(
            labels=labels,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
    )

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
    )

    # Test model with test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    Logger.log(f"Test Accuracy: {test_acc:.2f}")

    # Save trained model
    model_output_path = f"{model_output_path}/model"
    os.makedirs(model_output_path, exist_ok=True)
    model_save_path = os.path.join(model_output_path, f"{model_name}.keras")
    model.save(model_save_path)
    Logger.log(f"Model saved to {model_save_path}")

    # Plot model accuracy
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{model_output_path}/train-and-validation-accuracy.png")

    # Convert Keras model to a TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    tflite_model_path = f"{model_output_path}/{model_name}.tflite"
    open(tflite_model_path, "wb").write(tflite_model)
    Logger.log(f"TensorFlow Lite model saved at: {tflite_model_path}")

    # Write TFLite model to a C source (or header) file
    c_model_path = f"{model_output_path}/{model_name}.h"
    with open(c_model_path, "w") as file:
        file.write(ModelConversion.hex_to_c_array(tflite_model, model_name))
    Logger.log(f"C-Array model saved at: {c_model_path}")

    return model, history


def predict_data(
    voice_data_path: str,
    cnn_model: tf.keras.Model,
    dataset_path: str,
    threshold: float = 0.8,
) -> Optional[str]:
    """Authenticates user voice data using the trained CNN model.

    Args:
        voice_data_path (str): The path to the voice data file.
        cnn_model: The trained CNN model for prediction.
        dataset_path (str): The path to the dataset folder containing the label map.
        threshold (float, optional): Confidence threshold for prediction. Defaults to 0.9.

    Returns:
        str or None: The predicted label if confidence is above the threshold, otherwise None.
    """

    try:
        Logger.log(f"Identifying {voice_data_path}")
        with open(f"{dataset_path}/dataset/label_map.json", "r") as f:
            reverse_label_map = json.load(f)
        Logger.log(f"Label map loaded from {dataset_path}/dataset/label_map.json")
    except Exception as e:
        Logger.log(e, LogLevel.ERROR)

    data, sample_rate = DataProcesser.read_pcm_binary(voice_data_path)
    Logger.log(f"Data from {voice_data_path} has been read.")

    feature_vector = DataProcesser.extract_features(data, sample_rate)
    feature_vector = feature_vector.reshape(1, 16, 1)

    # Predict using trained CNN model with time monitoring
    start_time = time.time()
    predictions = cnn_model.predict(feature_vector)
    end_time = time.time()
    prediction_time = end_time - start_time
    Logger.log(f"Voice prediction time: {prediction_time:.4f} seconds")

    Logger.log(f"Prediction Threshold: {threshold}")
    Logger.log(f"Predictions: {predictions}")
    Logger.log(f"Predictions shape: {predictions.shape}")
    Logger.log(f"Predictions dtype: {predictions.dtype}")

    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    predicted_label = reverse_label_map[
        str(predicted_index)
    ]  # Convert key to string since JSON keys are strings

    if confidence >= threshold:
        Logger.log(
            f"Prediction success! Result: {predicted_label} (Confidence: {confidence:.2f})"
        )
        return predicted_label
    else:
        Logger.log("Prediction failed: Unknown class")
        return None


def main():
    configs: dict = ConfigLoader.load_config("config/application_config.yml")

    load_dataset_from_file: bool = False

    if load_dataset_from_file is False:
        labels, (X_train, y_train), (X_val, y_val), (X_test, y_test) = process_dataset(
            dataset_path=configs["dataset_path"],
            output_path=configs["output_path"],
        )

        model, _ = train_model(
            model_name=configs["model_name"],
            model_output_path=configs["output_path"],
            load_sets_from_file=load_dataset_from_file,
            labels=labels,
            X_val=X_val,
            y_val=y_val,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
    else:
        model, _ = train_model(
            model_name=configs["model_name"],
            model_output_path=configs["output_path"],
            load_sets_from_file=load_dataset_from_file,
        )

    predict_data(
        voice_data_path=configs["usage_test_path_1"],
        cnn_model=model,
        dataset_path=configs["output_path"],
    )
    predict_data(
        voice_data_path=configs["usage_test_path_2"],
        cnn_model=model,
        dataset_path=configs["output_path"],
    )
    predict_data(
        voice_data_path=configs["usage_test_path_3"],
        cnn_model=model,
        dataset_path=configs["output_path"],
    )

    # with open(f"{configs['output_path']}/dataset/label_map.json", "r") as f:
    #     reverse_label_map = json.load(f)

    # dataset_dir = Path(configs["dataset_path"])
    # for dataset_label in dataset_dir.iterdir():
    #     if dataset_label.is_dir():
    #         audio_files = list(dataset_label.glob("*.bin"))
    #         if not audio_files:
    #             Logger.log(f"No audio files found in {dataset_label}")
    #             continue

    #         audio_file = random.choice(audio_files)
    #         Logger.log(f"Processing {audio_file}")

    #         data, sample_rate = DataProcesser.read_pcm_binary(audio_file)
    #         Logger.log(f"Loaded file: {audio_file}")

    #         feature_vector = DataProcesser.extract_features(data, sample_rate)
    #         start_time = time.time()
    #         predictions = model.predict(feature_vector)
    #         end_time = time.time()
    #         prediction_time = end_time - start_time
    #         Logger.log(f"Voice prediction time: {prediction_time:.4f} seconds")

    #         Logger.log(f"Predictions: {predictions}")
    #         Logger.log(f"Predictions shape: {predictions.shape}")
    #         Logger.log(f"Predictions dtype: {predictions.dtype}")

    #         predicted_index = np.argmax(predictions)
    #         confidence = np.max(predictions)

    #         predicted_label = reverse_label_map[str(predicted_index)]

    #         Logger.log(f"Expected result: {dataset_label}")
    #         Logger.log(
    #             f"Predicted result: {predicted_label} (Confidence: {confidence:.2f})"
    #         )


if __name__ == "__main__":
    main()
