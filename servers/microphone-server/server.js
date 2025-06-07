const path = require("path");
const express = require("express");
const WebSocket = require("ws");
const fs = require("fs");
const app = express();

const WS_PORT = process.env.WS_PORT || 8888;
const HTTP_PORT = process.env.HTTP_PORT || 8000;
const EXTRACTING_FEATURE_SERVER_PORT =
  process.env.EXTRACTING_FEATURE_SERVER_PORT || 8001;

const wsServer = new WebSocket.Server({ port: WS_PORT }, () =>
  console.log(`WS server is listening at ws://localhost:${WS_PORT}`)
);

let connectedClients = [];
let buffer = [];
let esp32Client = null;

function processFileAndSendBack(audioData) {
  const base64Audio = audioData.toString("base64");

  fetch(`http://localhost:${EXTRACTING_FEATURE_SERVER_PORT}/extract_features`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      audio: base64Audio,
      sample_rate: 16000,
    }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      if (esp32Client && esp32Client.readyState === WebSocket.OPEN) {
        console.log("Sending processed data back to ESP32");
        console.dir(data, { depth: null });
        esp32Client.send(JSON.stringify(data.feature_vector));
      } else {
        console.log("ESP32 client is not connected.");
      }
    })
    .catch((error) => {
      console.error("Error fetching from Extracting feature server:", error);
    });
}

wsServer.on("connection", (ws, req) => {
  console.log("Connected");
  connectedClients.push(ws);
  const clientIP = req.socket.remoteAddress;
  console.log("New client connected from IP:", clientIP);

  esp32Client = ws;

  ws.on("message", (data) => {
    buffer.push(data);

    connectedClients.forEach((client) => {
      if (client !== esp32Client && client.readyState === WebSocket.OPEN) {
        client.send(data);
      }
    });
  });
});

// Function to process data every 1 second
setInterval(() => {
  if (buffer.length > 0) {
    const audioData = Buffer.concat(buffer);
    processFileAndSendBack(audioData);
    buffer = [];
  }
}, 1000);

// HTTP stuff
app.use("/image", express.static("image"));
app.use("/js", express.static("js"));
app.use(express.json());

app.get("/audio", (req, res) =>
  res.sendFile(path.resolve(__dirname, "./audio_client.html"))
);

// Add endpoint for speaker updates
app.post("/update_speaker", (req, res) => {
  const { speaker, confidence } = req.body;
  console.log(
    "Received speaker update:",
    speaker,
    "with confidence:",
    confidence
  );

  // Broadcast to all web clients
  connectedClients.forEach((client) => {
    if (client !== esp32Client && client.readyState === WebSocket.OPEN) {
      client.send(
        JSON.stringify({
          type: "speaker_update",
          speaker: speaker,
          confidence: confidence,
        })
      );
    }
  });

  res.json({ status: "success" });
});

app.listen(HTTP_PORT, () =>
  console.log(`HTTP server listening at http://localhost:${HTTP_PORT}`)
);
