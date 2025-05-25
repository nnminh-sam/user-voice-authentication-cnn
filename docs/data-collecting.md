# Data collecting

## Understand the audio data from INMP441

Let’s break down how the audio data is collected by the INMP441 microphone using your provided ESP32 code, and then calculate the size of the audio data for 1 second of recording time. I'll explain each part clearly and provide the necessary calculations.

### How the Audio Data is Collected by the Microphone

In your code, the ESP32 is configured to use the I2S (Inter-IC Sound) interface to communicate with the INMP441 digital microphone. Here’s how the audio data collection works:

#### 1. **Hardware Setup and I2S Configuration**
   - The INMP441 is a digital MEMS microphone that outputs audio data over the I2S interface. In your code, the pins are defined as follows:
     - `I2S_SD (32)`: Data output from the INMP441 (serial data line).
     - `I2S_WS (25)`: Word Select (Left/Right Clock) to indicate which channel (left or right) is being transmitted. In your case, it’s set to `I2S_CHANNEL_FMT_ONLY_LEFT`, meaning only the left channel is used.
     - `I2S_SCK (26)`: Bit Clock (serial clock) to synchronize the data transfer.

   - The I2S configuration (`i2s_config_t`) in the `i2s_install()` function specifies:
     - **Mode**: Master receive mode (`I2S_MODE_MASTER | I2S_MODE_RX`), meaning the ESP32 is receiving data from the microphone.
     - **Sample Rate**: 16,000 Hz (16 kHz), which means the microphone samples the audio 16,000 times per second.
     - **Bits Per Sample**: 16 bits, so each audio sample is represented by 16 bits (2 bytes).
     - **Channel Format**: Only the left channel is used (`I2S_CHANNEL_FMT_ONLY_LEFT`), which is typical for a single microphone.
     - **Buffer Settings**: The DMA (Direct Memory Access) buffer has `bufferCnt` (10) buffers, each of length `bufferLen` (1024 samples). This means the ESP32 can store up to 10 * 1024 = 10,240 samples in its DMA buffers before needing to process or transfer them.

#### 2. **Data Collection in `micTask`**
   - The `micTask` function runs on core 1 of the ESP32 and is responsible for continuously reading audio data from the INMP441 via I2S.
   - The `i2s_read` function is used to fetch audio data into the `sBuffer` array, which is defined as `int16_t sBuffer[bufferLen]` (an array of 1024 16-bit integers).
   - Each time `i2s_read` is called, it fills `sBuffer` with up to `bufferLen` (1024) samples, and the number of bytes actually read is stored in `bytesIn`.
   - If the read is successful (`ESP_OK`) and the WebSocket connection is active (`isWebSocketConnected`), the audio data in `sBuffer` is sent over WebSocket as binary data.

   - **Sample Format**: Since each sample is 16 bits (2 bytes) and the channel format is set to `ONLY_LEFT`, each element in `sBuffer` represents one audio sample from the microphone. The values in `sBuffer` are signed 16-bit integers (ranging from -32768 to 32767), corresponding to the amplitude of the audio signal at that moment.

#### 3. **Real-Time Data Flow**
   - The microphone continuously captures audio at 16 kHz, and the I2S peripheral on the ESP32 reads this data into the DMA buffers.
   - The `micTask` function processes these buffers as they fill up, sending the data to a WebSocket server. This implies that the audio data is streamed in real-time rather than being stored locally for a long duration (due to limited ESP32 memory).

### Calculating the Size of Audio Data for 1 Second of Recording Time

To determine the size of the audio data for 1 second of recording, we need to consider the following parameters from your code and configuration:

#### Parameters:
- **Sample Rate (`SAMPLE_RATE`)**: 16,000 Hz (16 kHz), meaning 16,000 samples are captured per second.
- **Bits Per Sample**: 16 bits (2 bytes) per sample, as specified in `bits_per_sample = i2s_bits_per_sample_t(16)`.
- **Channel**: Only one channel (left) is used, as set by `I2S_CHANNEL_FMT_ONLY_LEFT`. Since the INMP441 is a mono microphone, this is appropriate.

#### Calculation:

1. **Number of Samples Per Second**:
   The sample rate is 16,000 samples per second. So, in 1 second, the microphone will produce:

   ```
   Samples per second = Sample Rate = 16,000
   ```

2. **Size of Each Sample**:
   Each sample is 16 bits, which is equivalent to 2 bytes (since 1 byte = 8 bits, and 16 / 8 = 2).

   ```
   Bytes per sample = 2 bytes
   ```

3. **Total Size of Audio Data for 1 Second**:
   Multiply the number of samples per second by the size of each sample:

   ```
   Total bytes per second = Samples per second × Bytes per sample
   Total bytes per second = 16,000 × 2 = 32,000 bytes
   ```

   So, 1 second of audio data from the INMP441, as configured in your code, will occupy **32,000 bytes** (or 32 KB).

#### Additional Notes:
- **Buffer Size vs. 1 Second**: Your `bufferLen` is 1024 samples. The time corresponding to 1024 samples at 16 kHz is:

   ```
   Time for 1024 samples = Number of samples / Sample Rate
   Time for 1024 samples = 1024 / 16,000 = 0.064 seconds
   ```

   This means each buffer (`sBuffer`) holds approximately 64 milliseconds of audio. To capture 1 second of audio, you would need:

   ```
   Number of buffers for 1 second = 1 second / 0.064 seconds per buffer ≈ 15.625 buffers
   ```

   Since you have `bufferCnt` (10) buffers, each of `bufferLen` (1024 samples), the total capacity of your DMA buffers is:

   ```
   Total samples in DMA = bufferCnt × bufferLen = 10 × 1024 = 10,240 samples
   ```

   The time covered by these buffers is:

   ```
   Time covered = 10,240 / 16,000 ≈ 0.64 seconds
   ```

   This indicates that your current buffer setup can hold about 0.64 seconds of audio before it needs to be processed or overwritten. If you want to capture 1 full second, you might need to increase `bufferCnt` or handle the data more frequently.

- **WebSocket Transmission**: Your code sends each buffer (`sBuffer`) over WebSocket as soon as it’s filled, which is why the audio is streamed in real-time. The `bytesIn` variable ensures that only the actual number of bytes read is sent, which should be close to `bufferLen × 2` (since each sample is 2 bytes).

- **Memory Usage**: The ESP32 has limited RAM (520 KB total, with much of it used by the system and other tasks). Storing 32 KB per second of audio data is feasible for short durations, but for longer recordings, you’ll need to stream the data (as your code does) or offload it to external storage or a server.

### Summary
- **Audio Collection**: The INMP441 microphone sends digital audio data via I2S to the ESP32, which buffers it in `sBuffer` and streams it to a WebSocket server. Each sample is a 16-bit value, and the sample rate is 16 kHz.
- **Size for 1 Second**: 1 second of audio data at 16 kHz with 16 bits per sample (mono) results in **32,000 bytes** (32 KB).

If you need to adjust the sample rate, bits per sample, or buffer size, let me know, and I can help recalculate or modify the code. Additionally, if you want to process this audio data (e.g., for MFCC extraction) before sending it, we can integrate that into the `micTask` function.