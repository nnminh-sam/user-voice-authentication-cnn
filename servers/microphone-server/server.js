const path = require("path");
const express = require("express");
const WebSocket = require("ws");
const fs = require("fs");
const app = express();

const WS_PORT = process.env.WS_PORT || 8888;
const HTTP_PORT = process.env.HTTP_PORT || 8000;

const wsServer = new WebSocket.Server({ port: WS_PORT }, () =>
  console.log(`WS server is listening at ws://localhost:${WS_PORT}`)
);

let connectedClients = [];
let buffer = []; // Buffer to store data temporarily
let esp32Client = null // store esp32Client

const processData = (data) => "1";

function processFileAndSendBack(filePath) {
  // Đọc dữ liệu từ file đã lưu
  fs.readFile(filePath, (err, data) => {
    if (err) {
      console.error("Error reading file:", err);
      return;
    }

    console.log("Read file data:", data);

    const processedData = processData(data); // denoise

    // Gửi lại dữ liệu đã xử lý cho ESP32 nếu có kết nối
    if (esp32Client && esp32Client.readyState === WebSocket.OPEN) {
      console.log("Sending processed data back to ESP32");
      esp32Client.send(processedData);
    } else {
      console.log("ESP32 client is not connected.");
    }
  });
}

wsServer.on("connection", (ws, req) => {
  console.log("Connected");
  connectedClients.push(ws);
  const clientIP = req.socket.remoteAddress;
  console.log("New client connected from IP:", clientIP);

  // Lấy phần IP từ địa chỉ IPv6-mapped IPv4 address
  const formattedIP = clientIP.replace("::ffff:", "");

  // Kiểm tra IP của client (nếu là ESP32, ví dụ ESP32 có IP 192.168.100.113)
  // check IP ESP32 first
  if (formattedIP === '192.168.100.113') {
    console.log("This is ESP32 client");
    esp32Client = ws;
  }

  ws.on("message", (data) => {
    buffer.push(data); // Store data in buffer

    // Gửi dữ liệu cho tất cả client khác
    connectedClients.forEach((client) => {
      if (client !== esp32Client && client.readyState === WebSocket.OPEN) {
        client.send(data); // Gửi dữ liệu cho client khác
      }
    });
  });

});

// Function to write data to a file every 10 seconds
setInterval(() => {
  if (buffer.length > 0) {
    const timestamp = Date.now();
    const filePath = path.join(__dirname, `data/audio_${timestamp}.bin`);

    fs.writeFile(filePath, Buffer.concat(buffer), (err) => {
      if (err) console.error("Error writing file:", err);
      else {
        console.log(`Saved audio to ${filePath}`);
        processFileAndSendBack(filePath);
      }
    });

    buffer = []; // Clear buffer after writing
  }
}, 10000); // Every 10 seconds

// HTTP stuff
app.use("/image", express.static("image"));
app.use("/js", express.static("js"));
app.get("/audio", (req, res) =>
  res.sendFile(path.resolve(__dirname, "./audio_client.html"))
);
app.listen(HTTP_PORT, () =>
  console.log(`HTTP server listening at http://localhost:${HTTP_PORT}`)
);
