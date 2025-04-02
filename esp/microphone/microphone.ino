#include <driver/i2s.h>
#include <WiFi.h>
#include <ArduinoWebsockets.h>

#define I2S_SD 32   // Data Output from INMP441
#define I2S_WS 25   // Left/Right Clock
#define I2S_SCK 26  // Bit Clock
#define I2S_PORT I2S_NUM_0

#define bufferCnt 10
#define bufferLen 1024
int16_t sBuffer[bufferLen];

const char* ssid = "FPT Telecom";   // WiFi SSID
const char* password = "17092016";        // WiFi Password

const char* websocket_server_host = "192.168.100.171";
const uint16_t websocket_server_port = 8888;  // <WEBSOCKET_SERVER_PORT>

using namespace websockets;
WebsocketsClient client;
bool isWebSocketConnected = false;

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
  
  if (message.isBinary()) {
    // Xử lý dữ liệu nhị phân
    const uint8_t* data = (const uint8_t*)message.data().c_str();
    
    // Giả sử dữ liệu gửi 1 byte (0x01)
    int16_t receivedValue = data[0]; // Giả sử chỉ đọc 1 byte đầu tiên
    Serial.println(receivedValue);

    // Nếu bạn gửi nhiều dữ liệu (ví dụ: mảng 2 byte), xử lý như sau:
    // int16_t receivedValue = (data[1] << 8) | data[0]; // Nếu dữ liệu là 2 byte
  } else {
    // Nếu dữ liệu không phải nhị phân (trong trường hợp bạn gửi chuỗi)
    Serial.println("Received string data: ");
    Serial.println(message.data());
  }
}

void i2s_install() {
  // Set up I2S Processor configuration
  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,  // Adjust if necessary
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
  // Set I2S pin configuration
  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,  // Not used
    .data_in_num = I2S_SD
  };

  i2s_set_pin(I2S_PORT, &pin_config);
}

void setup() {
  Serial.begin(115200);

  connectWiFi();
  connectWSServer();
  xTaskCreatePinnedToCore(micTask, "micTask", 10000, NULL, 1, NULL, 1);
}

void loop() {
    client.poll(); 
}

void connectWiFi() {
  Serial.println("Connecting to wifi");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("wifi...");
  }
  Serial.println("");
  Serial.println("WiFi connected");
}

void connectWSServer() {
  Serial.println("Connecting to websocket");
  client.onEvent(onEventsCallback);
  client.onMessage(onMessageCallback);
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
      client.sendBinary((const char*)sBuffer, bytesIn);  // Gửi dữ liệu nhị phân
      vTaskDelay(5 / portTICK_PERIOD_MS);
    }
    vTaskDelay(5 / portTICK_PERIOD_MS);
  }
}
