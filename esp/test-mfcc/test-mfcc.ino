#include <driver/i2s.h>
#include <Arduino.h>
#include <fft.h>

#define I2S_SD   32
#define I2S_WS   25
#define I2S_SCK  26
#define I2S_PORT I2S_NUM_0

const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
};

const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
};

// const float PI = 3.14159;
const int num_filters = 26; // Number of Mel filters
const int num_ceps = 13; // Including c0, but we'll take first 12 for article
float hamming_window[512];
float mel_weights[26][256]; // N_fft/2 = 256 for 512-point FFT

const int NUMBER_SAMPLES_PER_READING = 512; // Adjust based on memory
int16_t audio_buffer[NUMBER_SAMPLES_PER_READING];

void setup() {
    Serial.begin(115200);
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_start(I2S_PORT);
}

void loop() {
    size_t bytes_read;
    esp_err_t result = i2s_read(I2S_PORT, audio_buffer, sizeof(audio_buffer), &bytes_read, portMAX_DELAY);
    if (result == ESP_OK) {
        int samples_read = bytes_read / sizeof(int16_t);
        // Process audio_buffer for feature extraction
        process_frame(audio_buffer, samples_read);
    }
    delay(10); // Adjust delay for real-time processing
}

void compute_hamming_window(float *window, int N) {
    for(int n = 0; n < N; n++) {
        window[n] = 0.54 - 0.46 * cos(2 * PI * n / (N - 1));
    }
}

float mel(float f) {
    return 2595 * log10(1 + f / 700);
}

float inverse_mel(float m) {
    return (pow(10, m / 2595) - 1) * 700;
}

void compute_mel_weights(float weights[26][256], int num_filters, int N_fft, float sample_rate) {
    int num_bins = N_fft / 2;
    float f_min = 0;
    float f_max = sample_rate / 2;
    float mel_min = mel(f_min);
    float mel_max = mel(f_max);
    int num_points = num_filters + 2;
    float mel_points[num_points];
    float f_points[num_points];

    float step_mel = (mel_max - mel_min) / (num_points - 1);
    for(int i = 0; i < num_points; i++) {
        mel_points[i] = mel_min + i * step_mel;
        f_points[i] = inverse_mel(mel_points[i]);
    }

    for(int i = 0; i < num_filters; i++) {
        float left_freq = f_points[i];
        float center_freq = f_points[i + 1];
        float right_freq = f_points[i + 2];
        for(int k = 0; k < num_bins; k++) {
            float freq_k = k * sample_rate / N_fft;
            if(freq_k < left_freq || freq_k > right_freq) {
                weights[i][k] = 0;
            } else if(freq_k >= left_freq && freq_k < center_freq) {
                weights[i][k] = (freq_k - left_freq) / (center_freq - left_freq);
            } else {
                weights[i][k] = (right_freq - freq_k) / (right_freq - center_freq);
            }
        }
    }
}

void process_frame(int16_t *frame, int length) {
    if(length < N_fft) return; // Ensure enough samples

    // Pre-emphasis (simplified for first frame, need state for real-time)
    float frame_pre[N_fft];
    pre_emphasis(frame, frame_pre, length);

    // Apply Hamming window
    float frame_windowed[N_fft];
    for(int i = 0; i < N_fft; i++) {
        frame_windowed[i] = frame_pre[i] * hamming_window[i];
    }

    // Compute FFT
    float vRe[N_fft], vIm[N_fft];
    memcpy(vRe, frame_windowed, sizeof(float) * N_fft);
    memset(vIm, 0, sizeof(float) * N_fft);
    FFT fft = FFT(N_fft);
    fft.Windowing(vRe, vIm, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
    fft.Compute(vRe, vIm, FFT_FORWARD);

    // Compute power spectrum (first N_fft/2 bins)
    float power_spectrum[256]; // N_fft/2 = 256
    for(int k = 0; k < 256; k++) {
        power_spectrum[k] = vRe[k] * vRe[k] + vIm[k] * vIm[k];
    }

    // Compute Mel energy
    float mel_energy[26];
    for(int i = 0; i < num_filters; i++) {
        mel_energy[i] = 0;
        for(int k = 0; k < 256; k++) {
            mel_energy[i] += power_spectrum[k] * mel_weights[i][k];
        }
        if(mel_energy[i] > 0) mel_energy[i] = log(mel_energy[i]);
        else mel_energy[i] = -100; // Avoid log(0)
    }

    // Compute DCT for MFCC (first 13, take 12 as per article)
    float mfcc_temp[13];
    for(int j = 0; j < num_ceps; j++) {
        mfcc_temp[j] = 0;
        for(int i = 0; i < num_filters; i++) {
            mfcc_temp[j] += mel_energy[i] * cos(PI * j * (i + 0.5) / num_filters);
        }
    }
    // Store first 12 for article (excluding c0 if needed, but include for now)
    for(int j = 0; j < 12; j++) mfcc[j] = mfcc_temp[j+1]; // Adjust based on convention

    // Compute additional features
    float mean_freq = compute_mean_frequency(power_spectrum, 256, sample_rate);
    float std_dev = compute_standard_deviation(power_spectrum, 256);
    float amplitude = compute_amplitude(frame_pre, N_fft);
    float zcr = compute_zero_crossing_rate(frame_pre, N_fft);

    // Output features
    Serial.print("MFCC: ");
    for(int j = 0; j < 12; j++) {
        Serial.print(mfcc[j]);
        Serial.print(" ");
    }
    Serial.println();
    Serial.print("Mean Frequency: "); Serial.println(mean_freq);
    Serial.print("Standard Deviation: "); Serial.println(std_dev);
    Serial.print("Amplitude: "); Serial.println(amplitude);
    Serial.print("Zero-Crossing Rate: "); Serial.println(zcr);
}

// Placeholder for additional feature computations
float compute_mean_frequency(float *power, int num_bins, float sample_rate) {
    float sum_freq_power = 0;
    float sum_power = 0;
    float freq_step = sample_rate / num_bins;
    for(int k = 0; k < num_bins; k++) {
        float freq = k * freq_step;
        sum_freq_power += freq * power[k];
        sum_power += power[k];
    }
    return sum_power > 0 ? sum_freq_power / sum_power : 0;
}

float compute_standard_deviation(float *power, int num_bins) {
    float mean_power = 0;
    for(int k = 0; k < num_bins; k++) mean_power += power[k];
    mean_power /= num_bins;
    float variance = 0;
    for(int k = 0; k < num_bins; k++) variance += (power[k] - mean_power) * (power[k] - mean_power);
    return sqrt(variance / num_bins);
}

float compute_amplitude(float *frame, int length) {
    float sum_squares = 0;
    for(int i = 0; i < length; i++) sum_squares += frame[i] * frame[i];
    return sqrt(sum_squares / length);
}

float compute_zero_crossing_rate(float *frame, int length) {
    int crossings = 0;
    for(int i = 1; i < length; i++) {
        if((frame[i] > 0 && frame[i-1] <= 0) || (frame[i] < 0 && frame[i-1] >= 0)) {
            crossings++;
        }
    }
    return (float)crossings / (length - 1);
}

