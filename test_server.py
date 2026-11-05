#!/usr/bin/env python3
"""
test_server.py
--------------
Lightweight chat UI over the running vLLM server.

Usage:
    python test_server.py [--host 127.0.0.1] [--port 8000]
                         [--vllm-url http://127.0.0.1:8001]

If --vllm-url is omitted it connects to http://127.0.0.1:8001.

Open http://127.0.0.1:8000 in a browser to use the chat UI.

Raw JSON API (curl / scripts):
    POST /chat    { "message": "...", "max_tokens": 300, "temperature": 0.2 }
    GET  /health
    GET  /models

To debug just the proxy without the web server:
    from vllm_deploy.ui_server import VllmProxy
    proxy = VllmProxy("http://127.0.0.1:8001")
    print(proxy.health())
    print(proxy.chat("Hello"))
"""

import argparse

from vllm_deploy.ui_server import UiServer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--port",     default=8000, type=int)
    parser.add_argument("--vllm-url", default="",   dest="vllm_url")
    args = parser.parse_args()

    server = UiServer(
        vllm_url = args.vllm_url or "http://127.0.0.1:8001",
        host     = args.host,
        port     = args.port,
    )
    server.serve()


if __name__ == "__main__":
    main()
