import os
import numpy as np
import simpleaudio as sa

import utils.logger as Logger
import utils.dataset_config_loader as ConfigLoader


def play_processed_audio(
    label: str, dataset_path: str = "dataset", bytes_per_sample: int = 2
):
    label_path = os.path.join(dataset_path, label)
    Logger.log(f"Dataset path: {label_path}")

    dataset_config = ConfigLoader.load_config(f"{label_path}/config.yml")
    Logger.log(dataset_config)
    sample_rate: int = int(dataset_config["sample_rate"])
    num_channels: int = int(dataset_config["num_channels"])

    if not os.path.exists(label_path) or not os.path.isdir(label_path):
        Logger.log(f"Label '{label}' not found!", Logger.LogLevel.ERROR)
        return

    audio_files: list[str] = sorted(
        [file for file in os.listdir(label_path) if file.endswith(".bin")]
    )

    if not audio_files:
        Logger.log(
            f"No processed audio files found for label '{label}'!",
            Logger.LogLevel.WARNING,
        )
        return

    for file_name in audio_files:
        file_path = os.path.join(label_path, file_name)
        Logger.log(f"Playing: {file_name}")

        with open(file_path, "rb") as bin_file:
            raw_data = np.frombuffer(bin_file.read(), dtype=np.int16)

        wave_obj = sa.play_buffer(
            raw_data,
            num_channels=num_channels,
            bytes_per_sample=bytes_per_sample,
            sample_rate=sample_rate,
        )
        wave_obj.wait_done()


def play_dataset(label: str):
    project_config = ConfigLoader.load_config("project_config.yml")
    dataset_path: str = project_config["dataset_path"]
    Logger.log(f"Dataset path: {dataset_path}")
    play_processed_audio(label, dataset_path)


if __name__ == "__main__":
    project_config = ConfigLoader.load_config("project_config.yml")
    dataset_path: str = project_config["dataset_path"]
    Logger.log(f"Dataset path: {dataset_path}")

    label_to_play = input("Enter label name to play: ")
    play_processed_audio(label_to_play, dataset_path)
