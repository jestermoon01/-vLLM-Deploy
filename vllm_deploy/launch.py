"""Start the configured vLLM backend and expose its chat UI on WSL interfaces."""

import subprocess
import time
from pathlib import Path

import psutil
import requests

from .config import Config
from .runner import Runner
from .server import VllmServer


def _listener_addresses(port: int) -> set[str]:
    return {
        connection.laddr.ip
        for connection in psutil.net_connections(kind="inet")
        if connection.status == psutil.CONN_LISTEN
        and connection.laddr.port == port
    }


def _stop_loopback_ui(cfg: Config) -> None:
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            if (
                len(command) >= 2
                and Path(command[1]).resolve() == (cfg.workdir / "test_server.py").resolve()
                and "--port" in command
                and command[command.index("--port") + 1] == str(cfg.ui_port)
            ):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except psutil.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue


def _start_ui(cfg: Config) -> None:
    addresses = _listener_addresses(cfg.ui_port)
    if "0.0.0.0" not in addresses and "::" not in addresses:
        if addresses:
            _stop_loopback_ui(cfg)
            if _listener_addresses(cfg.ui_port):
                raise SystemExit(
                    f"Port {cfg.ui_port} is occupied by a process other than this project's chat UI."
                )

        log_path = cfg.workdir / "ui.log"
        with log_path.open("a") as log:
            subprocess.Popen(
                [
                    str(cfg.venv_py),
                    str(cfg.workdir / "test_server.py"),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(cfg.ui_port),
                    "--vllm-url",
                    cfg.server_url,
                ],
                env=cfg.enriched_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            health = requests.get(
                f"http://127.0.0.1:{cfg.ui_port}/health", timeout=2
            )
            health.raise_for_status()
            if health.json().get("status") == "ok":
                return
        except (requests.RequestException, ValueError):
            time.sleep(1)
    raise SystemExit(f"Chat UI did not become healthy on port {cfg.ui_port}.")


def _verify_chat(cfg: Config) -> None:
    response = requests.post(
        f"{cfg.server_url}/v1/chat/completions",
        json={
            "model": cfg.served_name,
            "messages": [{"role": "user", "content": "Reply exactly READY."}],
            "max_tokens": 32,
            "temperature": 0,
        },
        timeout=300,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip()
    if not content:
        raise SystemExit("vLLM chat test returned an empty response.")


def main() -> None:
    cfg = Config()
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    if not cfg.venv_py.is_file() or not (cfg.venv / "bin" / "vllm").is_file():
        raise SystemExit(
            f"vLLM is not installed at {cfg.venv}. Run the installation workflow first."
        )

    try:
        health = requests.get(f"{cfg.server_url}/health", timeout=2)
    except requests.RequestException:
        health = None

    if health is None or health.status_code != 200:
        server = VllmServer(cfg, Runner(cfg))
        command = server._build_cmd()
        with cfg.vllm_log.open("a") as log:
            log.write("\n=== Starting vLLM ===\n")
            log.flush()
            subprocess.Popen(
                command,
                env=cfg.enriched_env({"CUDA_VISIBLE_DEVICES": "0"}),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        server.wait_ready()

    models_response = requests.get(f"{cfg.server_url}/v1/models", timeout=10)
    models_response.raise_for_status()
    model_ids = {item.get("id") for item in models_response.json().get("data", [])}
    if cfg.served_name not in model_ids:
        raise SystemExit(
            f"vLLM is healthy but does not serve the expected model {cfg.served_name!r}."
        )

    _verify_chat(cfg)
    _start_ui(cfg)
    print(f"vLLM backend: {cfg.server_url} (localhost only)")
    print(f"Chat UI: http://127.0.0.1:{cfg.ui_port}")
    print(f"vLLM log: {cfg.vllm_log}")


if __name__ == "__main__":
    main()
