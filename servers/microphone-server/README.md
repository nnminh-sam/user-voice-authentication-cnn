# Microphone Server and Microphone ESP32 Workflow

---

## Workflow

1. Node server create a websocket server for ESP32 to emit audio data and plotting audio data in realtime.
1. ESP32 will listen to Node server and send data to Node server via websocket. Server down requires ESP32 restart.
3. Node server will constantly store audio data into local as raw PCM in binary file.

---

## Start up steps

1. Install packages
    ```bash
    npm install
    ```

2. Start Node server to get websocket server
3. Config Websocket IP into `audio_client.html` here:
    ```js
    window.connect = function connect() {

        connectBtn.disabled = !connectBtn.disabled;
        pauseBtn.disabled = !pauseBtn.disabled;

        player = new PCMPlayer({
            inputCodec: 'Int16',
            channels: 1,
            sampleRate: 16000,
            // sampleRate: 44100,
        });
        const WS_URL = 'ws:///192.168.184.83:8888' // Change Websocket server IP here
        var ws = new WebSocket(WS_URL)
        ws.binaryType = 'arraybuffer'
        ws.addEventListener('message', function (event) {
            if(continueBtn.disabled){
                player.feed(event.data)
                worker.postMessage(event.data) // Remove if it makes the web browser slow.
            }
        });
    }
    ```
4. Config Websocket IP into ESP32.
5. Start ESP32.