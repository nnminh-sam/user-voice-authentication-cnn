import argparse
import os
import tensorflow as tf

from src.core import (
    process_dataset,
    train_model,
    predict_data,
)

from src.utils.dataset_config_loader import load_config

import src.utils.logger as Logger


CONFIG_FILE = "application_config.yml"


def load_and_prepare_dataset():
    """Loads dataset, extracts features, and splits into train, validation, and test sets."""

    config: dict = load_config(CONFIG_FILE)

    labels, (X_train, y_train), (X_val, y_val), (X_test, y_test) = process_dataset(
        dataset_path=config["dataset_path"],
        output_path=config["output_path"],
    )

    Logger.log("Dataset processing completed.")
    return labels, (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_voice_model(load_from_file: bool = False):
    """Trains the model and saves it."""

    config = load_config(CONFIG_FILE)

    Logger.log("Starting model training...")

    if load_from_file is False:
        labels, (X_train, y_train), (X_val, y_val), (X_test, y_test) = process_dataset(
            dataset_path=config["dataset_path"],
            output_path=config["output_path"],
        )

        model, _ = train_model(
            model_name=config["model_name"],
            model_output_path=config["output_path"],
            load_sets_from_file=load_from_file,
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
            model_name=config["model_name"],
            model_output_path=config["output_path"],
            load_sets_from_file=load_from_file,
        )

    Logger.log(f"Model training complete. Model saved at {config['output_path']}")
    # return model


def predict_voice(voice_file: str):
    """Loads a trained model and makes a voice prediction."""
    config = load_config(CONFIG_FILE)

    # Load trained model
    model_path = os.path.join(
        config["output_path"], "model", f"{config['model_name']}.keras"
    )
    if not os.path.exists(model_path):
        Logger.log(
            f"Error: Model not found at {model_path}. Train the model first.",
            level="ERROR",
        )
        return

    model = tf.keras.models.load_model(model_path)
    prediction = predict_data(
        voice_data_path=voice_file,
        cnn_model=model,
        dataset_path=config["output_path"],
    )

    if prediction:
        Logger.log(f"Predicted User: {prediction}")
    else:
        Logger.log("Failed to authenticate user.")


def main():
    parser = argparse.ArgumentParser(description="Voice Authentication CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Load dataset command
    parser_dataset = subparsers.add_parser(
        "load-dataset", help="Load dataset, extract features, and split."
    )

    # Train model command
    parser_train = subparsers.add_parser(
        "train", help="Train the voice authentication model."
    )
    parser_train.add_argument(
        "--from-file",
        action="store_true",
        help="Load dataset from saved files instead of extracting.",
    )

    # Predict command
    parser_predict = subparsers.add_parser(
        "predict", help="Authenticate a user from a voice file."
    )
    parser_predict.add_argument(
        "voice_file", type=str, help="Path to the voice sample file (.bin)"
    )

    args = parser.parse_args()

    if args.command == "load-dataset":
        load_and_prepare_dataset()
    elif args.command == "train":
        train_voice_model(load_from_file=args.from_file)
    elif args.command == "predict":
        predict_voice(args.voice_file)


if __name__ == "__main__":
    main()
