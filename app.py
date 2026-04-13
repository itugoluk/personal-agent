#!/usr/bin/env python3
import threading
import time
import os
import signal
import uvicorn
import webview
from server import app as fastapi_app

PORT = 8765


def start_server():
    uvicorn.run(fastapi_app, host="127.0.0.1", port=PORT, log_level="error")


def on_closed():
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    time.sleep(0.8)  # allow server to bind

    window = webview.create_window(
        "Olympus",
        f"http://127.0.0.1:{PORT}",
        width=1100,
        height=760,
        min_size=(800, 580),
        resizable=True,
        text_select=True,
    )
    webview.start(on_closed=on_closed)
