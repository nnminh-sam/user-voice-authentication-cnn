# User voice authentication

---

## Microphone Server and ESP32 Microphone

- Server code is at: [`/servers/microphone-server`](/esp/microphone/microphone.ino)
- Microphone code is at: [`/esp/microphone`](/servers/microphone-server/server.js)
- Installation: [`/servers/microphone-server/README.md`](/servers/microphone-server/README.md)

---

## Model training application

### Dataset and output folder structure

**Dataset folder structure:**

- User voice dataset is placed in a folder named `dataset`. This is called "dataset folder".
- In the dataset folder, each user's voice data is stored inside a folder named after the username with a `config.yml` file for storing user dataset config. This config file is used to contain metadata about user voice and will be used to read dataset.

Exmaple `dataset` folder:

```
dataset
|-- user_a
|   |-- audio_1.bin
|   |-- audio_2.bin
|   |-- audio_3.bin
|   |-- audio_4.bin
|-- user_b
|   |-- audio_1.bin
|   |-- audio_2.bin
|   |-- audio_3.bin
|   |-- audio_4.bin
|-- user_c
    |-- audio_1.bin
    |-- audio_2.bin
    |-- audio_3.bin
    |-- audio_4.bin
```

This dataset has three classes for classification: `user_a`, `user_b` and `user_c`.

**Output foler structure:**

- Output folder contains dataset features.
- Train, validate and test sets for model training.
- Model files for integrations.
- Visualization files.
- Dataset labels map for prediction.

This is the structure of the output folder:

```
output
|-- dataset
|   |-- label_map.json
|   |-- features
|       |-- train
|       |   |-- X_train.npy
|       |   |-- y_train.npy
|       |-- validation
|       |   |-- X_val.npy
|       |   |-- y_val.npy
|       |-- test
|           |-- X_test.npy
|           |-- y_test.npy
|-- model
    |-- train-and-validation-accuracy.png
    |-- voice_identification.h
    |-- voice_identification.keras
    |-- voice_identification.tflite
```

This is an example of the output folder.

- The `output/dataset` folder structure must follow the given one for application integration.
- Model name can be configured in the `application_config.yml` file.
- Visualization figure file name is immuable.

### Application usage

> *Model training application is a small CLI built around the purpose of training voice authentication model, analysis model for model optimization. Out short-term goal is embedded the model into ESP32 module for user voice identification. In the future, we hope to develope a user voice authentication service for other application integration.*

**Version 0.1**

- This version is still in development but you can use it as a developer.

*Installation:*

- Create a virtual environment for python. This build is developing using Python 3.11.11.
- Install required packages for python from our `requirements.txt`.

*Usage:*

- Run the CLi by running the `app.py` file: 
    ```bash
    python app.py
    ```
- CLI help: 
    ```bash
    python app.py -h
    ```
- Load and process dataset:
    ```bash
    python app.py load-dataset
    ```
- Train model:
    - Train model using saved processed dataset:
        ```bash
        python app.py train --from-file
        ```
    - Load, process dataset and train model:
        ```bash
        python app.py train
        ```
- Predict audio file class (classification):
    ```bash
    python app.py predict <file path from project root>
    ```

*Note:*

- The current CLI is quite slow and lack of usage tracking.
- Log system is just straight to terminal without support for model analysis.
