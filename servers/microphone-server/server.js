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

wsServer.on("connection", (ws, req) => {
  console.log("Connected");
  connectedClients.push(ws);

  ws.on("message", (data) => {
    buffer.push(data); // Store data in buffer

    connectedClients.forEach((client, i) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(data);
      } else {
        connectedClients.splice(i, 1);
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
      else console.log(`Saved audio to ${filePath}`);
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
