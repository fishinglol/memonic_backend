import asyncio
import websockets

async def test():
    uri = "wss://8001-01kkh2et3bdjymj2fjq6jabg8k.cloudspaces.litng.ai/ws/audio"
    try:
        async with websockets.connect(uri) as websocket:
            print("Successfully connected!")
            await websocket.send("START")
            await websocket.send("STOP")
            response = await websocket.recv()
            print("Received:", response)
    except Exception as e:
        print("Failed:", e)

asyncio.run(test())
