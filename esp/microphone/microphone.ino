#include <WiFi.h>
#include <driver/i2s.h>
#include <ArduinoJson.h>
#include <ArduinoWebsockets.h>
#include <HTTPClient.h>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

#define I2S_SD 32   // Data Output from INMP441
#define I2S_WS 25   // Left/Right Clock
#define I2S_SCK 26  // Bit Clock
#define I2S_PORT I2S_NUM_0

#define bufferCnt 10
#define bufferLen 1024
int16_t sBuffer[bufferLen];

const char* ssid = "NHA TRO LE VAN VIET";
const char* password = "0902511322";

const char* websocket_server_host = "192.168.1.56";
const uint16_t websocket_server_port = 8888;  // <WEBSOCKET_SERVER_PORT>
const uint16_t http_server_port = 8000;  // <HTTP_SERVER_PORT>

// Tensor arena size (50 KB) - may need adjustment based on model requirements
constexpr int kTensorArenaSize = 20 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// TensorFlow Lite objects
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

using namespace websockets;
WebsocketsClient client;
bool isWebSocketConnected = false;

// ----

void setupModel() {
  // Load the model from flash memory
  model = tflite::GetModel(voice_identification);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }

  // Initialize interpreter with corrected error reporter
  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed - increase kTensorArenaSize if this persists");
    return;
  }

  // Optional: Check how much of the tensor arena is used
  Serial.print("Tensor arena used bytes: ");
  Serial.println(interpreter->arena_used_bytes());

  // Get pointers to input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void predict(const float* features, size_t feature_size, const char* message) {
  Serial.println(message);

  // Copy features into the input tensor (assuming float32 input)
  for (size_t i = 0; i < feature_size; ++i) {
    input->data.f[i] = features[i];
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  Serial.println("Output: ");
  Serial.println(output->data.f[0]);
  Serial.println(output->data.f[1]);
  Serial.println(output->data.f[2]);
  Serial.println("---------------");
  float prob_anh_ban_than = output->data.f[0];
  float prob_background_noise = output->data.f[1];
  float prob_giang_oi = output->data.f[2];
  
  float max_prob = max(max(prob_anh_ban_than, prob_giang_oi), prob_background_noise);
  String max_var = "";
  
  if (max_prob == prob_anh_ban_than) {
    max_var = "anh_ban_than";
  } else if (max_prob == prob_giang_oi) {
    max_var = "giang_oi"; 
  } else {
    max_var = "background_noise";
  }
  
  Serial.print("Largest probability: ");
  Serial.print(max_prob, 6);
  Serial.print(" from variable: ");
  Serial.println(max_var);

  // Send prediction result via HTTP POST
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String serverUrl = "http://" + String(websocket_server_host) + ":" + String(http_server_port) + "/update_speaker";
    
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");
    
    StaticJsonDocument<200> doc;
    doc["speaker"] = max_var;
    doc["confidence"] = max_prob;
    
    String jsonString;
    serializeJson(doc, jsonString);
    
    int httpResponseCode = http.POST(jsonString);
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("HTTP Response code: " + String(httpResponseCode));
      Serial.println("Response: " + response);
    } else {
      Serial.println("Error sending HTTP POST request");
    }
    
    http.end();
  }
}

void onEventsCallback(WebsocketsEvent event, String data) {
  if (event == WebsocketsEvent::ConnectionOpened) {
    Serial.println("Connection Opened");
    isWebSocketConnected = true;
  } else if (event == WebsocketsEvent::ConnectionClosed) {
    Serial.println("Connection Closed");
    isWebSocketConnected = false;
  } else if (event == WebsocketsEvent::GotPing) {
    Serial.println("Got a Ping!");
  } else if (event == WebsocketsEvent::GotPong) {
    Serial.println("Got a Pong!");
  }
}

void onMessageCallback(WebsocketsMessage message) {
  Serial.print("Received from Server: ");

  StaticJsonDocument<1024> doc;
  DeserializationError error = deserializeJson(doc, message.data());

  if (error) {
    Serial.print("deserializeJson() failed: ");
    Serial.println(error.c_str());
    return;
  }

  float featureVector[16];
  int index = 0;
  if (doc.is<JsonArray>()) {
    JsonArray array = doc.as<JsonArray>();
    Serial.println("vector of 16");
    for (JsonVariant value : array) {
      float num = value.as<float>();
      Serial.print(num, 6);
      Serial.print(", ");
      featureVector[index] = num;
      index++;
    }
    Serial.println();
    predict(featureVector, 16, "Predicting using [recevied] features");
  } else {
    Serial.println("Received data is not an array");
  }
}

void i2s_install() {
  // Set up I2S Processor configuration
  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = i2s_bits_per_sample_t(16),
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = bufferCnt,
    .dma_buf_len = bufferLen,
    .use_apll = false
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
}

void i2s_setpin() {
  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  i2s_set_pin(I2S_PORT, &pin_config);
}

void setup() {
  Serial.begin(115200);

  setupModel();
  connectWiFi();
  connectWSServer();
  xTaskCreatePinnedToCore(micTask, "micTask", 10000, NULL, 1, NULL, 1);
}

void loop() {
  client.poll();
}

void connectWiFi() {
  Serial.println("Connecting to wifi");
  Serial.println("SSID: " + String(ssid));
  Serial.println("Password: " + String(password));
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("wifi...");
  }

  Serial.print("ESP32 local IP: ");
  Serial.println(WiFi.localIP());
  Serial.println("");
  Serial.println("WiFi connected");
}

void connectWSServer() {
  Serial.println("Connecting to websocket");
  client.onEvent(onEventsCallback);
  client.onMessage(onMessageCallback);
  Serial.println("Host: " + String(websocket_server_host) + " Port: " + String(websocket_server_port));
  while (!client.connect(websocket_server_host, websocket_server_port, "/")) {
    delay(500);
    Serial.println("ws...");
  }
  Serial.println("Websocket Connected!");
}

void micTask(void* parameter) {
  i2s_install();
  i2s_setpin();
  i2s_start(I2S_PORT);

  size_t bytesIn = 0;
  while (1) {
    esp_err_t result = i2s_read(I2S_PORT, &sBuffer, bufferLen, &bytesIn, portMAX_DELAY);
    if (result == ESP_OK && isWebSocketConnected) {
      client.sendBinary((const char*)sBuffer, bytesIn);
      vTaskDelay(5 / portTICK_PERIOD_MS);
    }
    vTaskDelay(5 / portTICK_PERIOD_MS);
  }
}
