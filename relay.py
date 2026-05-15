"""
relay.py — Mac audio relay
==========================
ESP32 → UDP (LAN, no TLS) → this script → WSS (TLS) → Lightning AI

Run: python relay.py
"""

import asyncio
import socket
import websockets
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("relay")

# ── Config ────────────────────────────────────────────────────
UDP_HOST   = "0.0.0.0"
UDP_PORT   = 5005                       # ESP32 sends audio here
WS_URL     = "wss://8001-01kkh2et3bdjymj2fjq6jabg8k.cloudspaces.litng.ai/ws/audio"
ESP32_PORT = 5006                       # relay sends commands back here
# ─────────────────────────────────────────────────────────────

esp32_addr = None   # (ip, port) — set when first UDP packet arrives


async def relay():
    global esp32_addr

    # UDP socket — receives audio from ESP32, sends commands back
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind((UDP_HOST, UDP_PORT))
    # blocking=True is fine — recvfrom runs in executor thread, not event loop
    log.info(f"UDP listening on :{UDP_PORT}")

    loop = asyncio.get_event_loop()

    while True:
        log.info(f"Connecting to {WS_URL} ...")
        try:
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=10) as ws:
                log.info("WSS connected to Lightning AI")

                # Send HELLO so server registers us as bracelet
                await ws.send("HELLO")

                # Two tasks run concurrently:
                # 1. UDP → WSS  (audio from ESP32 to server)
                # 2. WSS → UDP  (commands from server to ESP32)

                async def udp_to_ws():
                    global esp32_addr
                    while True:
                        try:
                            data, addr = await loop.run_in_executor(
                                None, lambda: udp.recvfrom(8192)
                            )
                            esp32_addr = (addr[0], ESP32_PORT)

                            # Text commands from ESP32 (HELLO, STOP, etc.)
                            try:
                                txt = data.decode("utf-8").strip()
                                log.info(f"ESP32 text → server: {txt}")
                                await ws.send(txt)
                            except UnicodeDecodeError:
                                # Binary audio → forward as bytes
                                await ws.send(data)
                        except Exception as e:
                            log.error(f"udp_to_ws error: {e}")
                            await asyncio.sleep(0.01)

                async def ws_to_udp():
                    async for msg in ws:
                        if esp32_addr is None:
                            continue
                        if isinstance(msg, str):
                            log.info(f"Server → ESP32: {msg}")
                            udp.sendto(msg.encode(), esp32_addr)
                        # Binary from server (shouldn't happen, but handle it)
                        else:
                            udp.sendto(msg, esp32_addr)

                await asyncio.gather(udp_to_ws(), ws_to_udp())

        except Exception as e:
            log.error(f"WSS error: {e} — reconnecting in 3s")
            await asyncio.sleep(3)


if __name__ == "__main__":
    import sys
    # Print local IP so user knows what to put in ESP32
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    local_ip = s.getsockname()[0]
    s.close()
    print(f"\n{'='*45}")
    print(f"  Mac relay starting")
    print(f"  Local IP: {local_ip}")
    print(f"  Set ESP32 serverHost = \"{local_ip}\"")
    print(f"  Set ESP32 serverPort = {UDP_PORT}")
    print(f"{'='*45}\n")

    asyncio.run(relay())
