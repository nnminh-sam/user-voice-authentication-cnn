# Voice Detection System

---

## Project Structure

- NodeJS server source code: [`./servers/microphone-server`](servers/microphone-server)
    - Server entry: [`./servers/microphone-server/server.js`](servers/microphone-server/server.js)

- Python server source code: [`./servers/extracting-feature-server`](servers/extracting-feature-server)
    - Server entry: [`./servers/extracting-feature-server/server.py`](servers/extracting-feature-server/server.py)

- ESP32 source code: [`./esp/microphone`](/esp/microphone)
    - ESP32 source code: [`./esp/microphone/microphone.ino`](esp/microphone/microphone.ino)
    - ESP32 embedded model: [`./esp/microphone/model.h`](esp/microphone/model.h)

- Data collecting script source code: [`./raw_dataset_builder`](raw_dataset_builder)
    - Dataset bulding script: [`./raw_dataset_builder/dataset_builder.py`](raw_dataset_builder/dataset_builder.py)
    - Dataset player script: [`./raw_dataset_builder/dataset_player.py`](raw_dataset_builder/dataset_player.py)

---

## Project Virtual Environment

To use this project, create a virtual environment to work with python.

```bash
python -m venv venv
```

This will create a python virtual environment, you can also add python version into the command to choose the correct python version. I suggest using python 3.11.

Then you will need to install required libraries.

```bash
pip install -r requirements.txt
```

---

## Model Training

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

---

## Run Servers & ESP32

Before running the servers and ESP32, you will need to get the IPv4 address of the computer that runs the servers. Then update this IP value at these places:

1. Audio client HTML file at line 172.

    - File location: [`servers/microphone-server/audio_client.html`](servers/microphone-server/audio_client.html)
    - Code:
        ```js
        const WS_URL = "ws:///192.168.1.56:8888";
        ```
    - Just replace the IP address, don't replace the port and prefix

1. ESP32 code at line 28.

    - File location: [`esp/microphone/microphone.ino`](esp/microphone/microphone.ino)
    - Code:
        ```cpp
        const char* websocket_server_host = "192.168.1.56";
        ```
    - Also update the wifi name and password for correct ESP32 wifi connection

### NodeJS Server

The NodeJS server will create a server that allow the ESP32 to transfer data through WebSocket protocol. This server also stream realtime audio data to a web client via WebSocket.

To run the server:

1. Change directory to `servers/microphone-server` directory
1. Install the dependencies

    ```bash
    npm install
    ```

1. Run the command

    ```bash
    node server.js
    ```

After the server started, you can navigate to `localhost:8000/audio` to access the Web App UI. Press connect to connect the Web App to the server to stream audio. A connection log will show on the console log.

### Python Server

The Python server will recevied PCM audio data from NodeJS server via HTTP API request and calculate it then return to the NodeJS server.

To run the server (required virtual environment and dependencies):

1. Change directory to `servers/extracting-feature-server`
1. Run the command

    ```bash
    python server.py
    ```

You will need to run both server up first then start to load the code and run ESP32. Sometime, ESP32 will take a long time to connect to the NodeJS server via WebSocket protocol, please wait 5 - 10 minutes.

---

## Project Members

1. Nguyễn Nhật Minh (*Leader*)
2. Trần Vũ Phương Nam
3. Nguyễn Tấn Lộc
4. Huỳnh Ngọc Tân

---
