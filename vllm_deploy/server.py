"""
server.py
---------
Cell 5 + Cell 6 — Launch the vLLM server in the background and wait until
its /health endpoint returns HTTP 200.

Debug standalone:
    python -c "
    from vllm_deploy.config  import Config
    from vllm_deploy.runner  import Runner
    from vllm_deploy.server  import VllmServer
    import os; api_key = open('.api_key').read().strip()
    s = VllmServer(Config(), Runner(Config()))
    proc = s.start(api_key)
    s.wait_ready()
    print('server pid:', proc.pid)
    "
"""

import subprocess
import time
import shlex

import requests

from .config import Config
from .runner import Runner


class VllmServer:
    """
    Manages the lifecycle of the background vLLM server process.

    Methods:
        start(api_key)  — Cell 5: spawn vLLM server in background
        wait_ready()    — Cell 6: poll /health until 200 or timeout
        stop()          — kill the server process if running
    """

    def __init__(self, cfg: Config, runner: Runner) -> None:
        self.cfg    = cfg
        self.runner = runner
        self._proc: subprocess.Popen | None = None

    # ── Cell 5 ──────────────────────────────────────────────────────────────

    def start(self) -> subprocess.Popen:
        """
        Build the vllm serve command and launch it in the background.
        No API key — authentication is handled at the UI proxy layer.

        Returns:
            subprocess.Popen handle (also stored in self._proc).
        """
        self.runner.banner("Cell 5 – Start vLLM server")

        cmd_parts = self._build_cmd()
        cmd       = " ".join(shlex.quote(str(p)) for p in cmd_parts)

        self._proc = self.runner.popen_bg(
            cmd,
            self.cfg.vllm_log,
            env={"CUDA_VISIBLE_DEVICES": "0"},
        )
        print(f"vLLM server started  (pid {self._proc.pid})")
        print(f"Log: {self.cfg.vllm_log}")
        return self._proc

    # ── Cell 6 ──────────────────────────────────────────────────────────────

    def wait_ready(self, timeout: int = 900) -> None:
        """
        Poll GET /health every 5 s until HTTP 200 is received.

        Args:
            timeout: Maximum seconds to wait before giving up (default 900 s).

        Raises:
            SystemExit if the server is not healthy within the timeout.
        """
        self.runner.banner("Cell 6 – Wait for server to be ready")
        t0 = time.time()

        while time.time() - t0 < timeout:
            if self._is_healthy():
                elapsed = time.time() - t0
                print(f"\nServer ready after {elapsed:.0f}s ✓")
                return
            elapsed = int(time.time() - t0)
            print(f"  waiting … {elapsed}s", end="\r", flush=True)
            time.sleep(5)

        # Timed out — dump recent log for diagnosis
        print("\n[ERROR] Server NOT ready within timeout. Last log lines:")
        self._dump_log_tail(3000)
        raise SystemExit("Server failed to start within timeout.")

    def stop(self) -> None:
        """Terminate the server process if it was started by this instance."""
        if self._proc and self._proc.poll() is None:
            print(f"Stopping vLLM server (pid {self._proc.pid}) …")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            print("Server stopped.")

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_cmd(self, api_key: str = "") -> list:
        """
        Return the vllm serve argv list.
        No --api-key flag: authentication is handled by the UI layer rate limiter.
        """
        return [
            str(self.cfg.venv / "bin" / "vllm"), "serve", self.cfg.model_id,
            "--served-model-name",      self.cfg.served_name,
            "--dtype",                  "float16",
            # Hermes requires at least a 64K context window. Qwen2.5-Coder
            # supports this length; vLLM advertises it through /v1/models.
            "--max-model-len",          "65536",
            "--max-num-seqs",           "20",
            "--gpu-memory-utilization", "0.85",
            "--enforce-eager",          # safer on Blackwell until CUDA graphs stabilize
            # Hermes emits OpenAI tool calls with tool_choice=auto. vLLM must
            # enable automatic tool selection and use the Hermes parser so
            # those requests are accepted and decoded correctly.
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--host", "127.0.0.1",      # bind only to localhost; UI proxy is the gateway
            "--port", str(self.cfg.server_port),
        ]

    def _is_healthy(self) -> bool:
        """Return True if GET /health → 200, False on any error."""
        try:
            return (
                requests.get(f"{self.cfg.server_url}/health", timeout=2).status_code
                == 200
            )
        except Exception:
            return False

    def _dump_log_tail(self, chars: int = 3000) -> None:
        """Print the last `chars` characters of the vLLM log file."""
        try:
            print(self.cfg.vllm_log.read_text()[-chars:])
        except FileNotFoundError:
            print(f"  (log file not found: {self.cfg.vllm_log})")
