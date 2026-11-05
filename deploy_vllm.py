#!/usr/bin/env python3
"""
deploy_vllm.py
--------------
Thin orchestrator — calls the modular classes in vllm_deploy/.

To debug a single step, import and call the class directly:

    from vllm_deploy.config       import Config
    from vllm_deploy.runner       import Runner
    from vllm_deploy.system_check import SystemCheck

    cfg = Config()
    runner = Runner(cfg)
    SystemCheck(cfg, runner).run()

Stop with Ctrl+C.
"""

import threading
import time
from pathlib import Path

from vllm_deploy.config        import Config
from vllm_deploy.runner        import Runner
from vllm_deploy.system_check  import SystemCheck
from vllm_deploy.installer     import Installer
from vllm_deploy.api_key       import ApiKeyManager
from vllm_deploy.server        import VllmServer
from vllm_deploy.benchmark     import Benchmark
from vllm_deploy.keepalive     import KeepAlive
from vllm_deploy.ui_server     import UiServer


def main() -> None:
    cfg    = Config()
    runner = Runner(cfg)

    cfg.workdir.mkdir(parents=True, exist_ok=True)

    # ── Cell 1 — environment check ───────────────────────────────────────────
    SystemCheck(cfg, runner).run()

    # ── Cell 2+3 — install uv + vLLM ─────────────────────────────────────────
    installer = Installer(cfg, runner)
    installer.install_uv()
    installer.install_vllm()

    # ── Cell 4 — API key ─────────────────────────────────────────────────────
    api_key = ApiKeyManager(cfg, runner).load_or_create()

    # ── Cell 5+6 — start server, wait ready ───────────────────────────────────
    vllm_server = VllmServer(cfg, runner)
    vllm_server.start()
    vllm_server.wait_ready()

    ui_server = UiServer.from_config(cfg)
    threading.Thread(target=ui_server.serve, daemon=True).start()

    # ── Cell 7+8 — smoke test + benchmark ────────────────────────────────────
    bench = Benchmark(cfg, runner, api_key)
    bench.smoke_test()
    bench.load_test(n=20)

    # ── Summary ──────────────────────────────────────────────────────────────
    runner.banner("Deployment complete")
    print(f"  Local  : {cfg.server_url}")
    print(f"  Chat UI: http://127.0.0.1:{cfg.ui_port}")
    print(f"  API key: {api_key}")
    print(f"  vLLM log: {cfg.vllm_log}")
    print("\nStarting keep-alive loop (Ctrl+C to stop) …")

    # ── Cell 11 — keep-alive ──────────────────────────────────────────────────
    KeepAlive(interval=600).start()

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nShutting down.")
        vllm_server.stop()


if __name__ == "__main__":
    main()
