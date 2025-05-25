import os
import uuid
import asyncio
import aiofiles
import websockets
import socketserver
from datetime import datetime
from http.server import SimpleHTTPRequestHandler

# Ports
WS_PORT = int(os.environ.get("WS_PORT", 8888))
HTTP_PORT = int(os.environ.get("HTTP_PORT", 8000))

# Global variables
connected_clients = []
buffer = []  # Buffer to store data temporarily
esp32_client = None  # Store ESP32 client


async def process_data(data):
    """Process the data (similar to Node.js processData). Returns a random UUID."""
    return str(uuid.uuid4())


async def process_file_and_send_back(file_path):
    """Read file data and send processed data back to ESP32."""
    async with aiofiles.open(file_path, "rb") as f:
        data = await f.read()

    print(f"Read file data: {data}")

    processed_data = await process_data(data)  # Denoise or process data
    print(f"Processed data: {processed_data}")

    # Send back to ESP32 if connected
    if esp32_client and esp32_client.open:
        print("Sending processed data back to ESP32")
        await esp32_client.send(processed_data)
    else:
        print("ESP32 client is not connected.")


async def handle_websocket(websocket, path):
    global connected_clients, esp32_client

    print(f"New WebSocket connection attempt from {websocket.remote_address}")
    connected_clients.append(websocket)

    client_ip = websocket.remote_address[0]
    print(f"New client connected from IP: {client_ip}")

    # Assume ESP32 has a specific IP (e.g., '192.168.100.113')
    if client_ip == "192.168.100.113":  # Replace with actual ESP32 IP
        print("This is ESP32 client")
        esp32_client = websocket
    else:
        print(f"Client IP {client_ip} is not ESP32")

    try:
        async for message in websocket:
            print(
                f"Received message: {message[:50]}..."
            )  # Print first 50 bytes of message
            buffer.append(message)  # Store data in buffer

            # Broadcast to all other clients except ESP32 and itself
            for client in connected_clients:
                if client != esp32_client and client.open:
                    await client.send(message)  # Send data to other clients

    except websockets.ConnectionClosed as e:
        print(f"Client disconnected: {e}")
        connected_clients.remove(websocket)
        if esp32_client == websocket:
            esp32_client = None
            print("ESP32 client disconnected")
    except Exception as e:
        print(f"Error handling WebSocket: {e}")


async def periodic_file_write():
    """Write buffered data to a file every 10 seconds."""
    global buffer

    while True:
        await asyncio.sleep(10)  # Every 10 seconds

        if buffer:
            timestamp = int(datetime.now().timestamp() * 1000)  # Similar to Date.now()
            file_path = os.path.join(os.getcwd(), f"data/audio_{timestamp}.bin")

            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

            async with aiofiles.open(file_path, "wb") as f:
                await f.write(b"".join(buffer))  # Concatenate all buffer data

            print(f"Saved audio to {file_path}")
            await process_file_and_send_back(file_path)

            buffer = []  # Clear buffer after writing


# Start WebSocket server
async def main():
    # Start WebSocket server
    ws_server = await websockets.serve(handle_websocket, "0.0.0.0", WS_PORT)
    print(f"WS server is listening at ws://localhost:{WS_PORT}")

    # Start periodic file writing
    asyncio.create_task(periodic_file_write())

    # Start HTTP server in a separate thread or process (simplified)
    class CustomHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def translate_path(self, path):
            if path.startswith("/image"):
                return os.path.join(os.getcwd(), "image", path[7:])
            elif path.startswith("/js"):
                return os.path.join(os.getcwd(), "js", path[4:])
            elif path == "/audio":
                return os.path.join(os.getcwd(), "audio_client.html")
            return super().translate_path(path)

    httpd = socketserver.TCPServer(("", HTTP_PORT), CustomHandler)
    print(f"HTTP server listening at http://localhost:{HTTP_PORT}")
    httpd.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
