#include <driver/i2s.h>

#define I2S_NUM         I2S_NUM_0
#define I2S_BCLK_PIN    26  // Bit clock pin
#define I2S_LRCLK_PIN   25  // Left/right clock pin
#define I2S_DOUT_PIN    32  // Data output pin

void setup() {
  Serial.begin(115200);

  // I2S Configuration
  i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S_MSB,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_BCLK_PIN,
    .ws_io_num = I2S_LRCLK_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_DOUT_PIN
  };

  // Install and configure I2S driver
  i2s_driver_install(I2S_NUM, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM, &pin_config);

  Serial.println("INMP441 I2S Initialized");
}

void loop() {
  const int num_samples = 128;
  int32_t raw_samples[num_samples];
  size_t bytes_read;

  // Read data from INMP441
  i2s_read(I2S_NUM, &raw_samples, sizeof(raw_samples), &bytes_read, portMAX_DELAY);

  for (int i = 0; i < num_samples; i++) {
    int16_t sample = raw_samples[i] >> 14;  // Convert from 32-bit to meaningful 16-bit data
    Serial.println(sample);
  }

  delay(100);
}