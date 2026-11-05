"""
config.py
---------
Central configuration: all paths, constants, and .env loading.

Debug:
    python -c "from vllm_deploy.config import Config; cfg = Config(); print(cfg)"
"""

import os
from pathlib import Path


class Config:
    """
    Single source of truth for every path and constant used across the package.
    Reads .env automatically on construction.
    """

    def __init__(self, workdir: Path | None = None) -> None:
        # ── Directory layout ────────────────────────────────────────────────
        self.workdir: Path = workdir or Path(__file__).parent.parent.resolve()
        self.venv:         Path = self.workdir / "venv"
        self.venv_py:      Path = self.venv / "bin" / "python"
        self.lock_file:    Path = self.workdir / "requirements.lock.txt"
        self.vllm_log:     Path = self.workdir / "vllm.log"
        self.cf_log:       Path = self.workdir / "cf.log"
        self.cf_bin:       Path = self.workdir / "cloudflared"
        self.key_file:     Path = self.workdir / ".api_key"
        self.cf_url_file:  Path = self.workdir / ".cf_url"

        # ── Server ──────────────────────────────────────────────────────────
        self.server_url:  str = "http://localhost:8001"   # vLLM backend
        self.server_host: str = "127.0.0.1"
        self.server_port: int = 8001                      # vLLM listens here
        self.ui_port:     int = 8000                      # chat UI listens here
        self.model_id:    str = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
        self.served_name: str = "qwen-coder"

        # ── Rate limiting (per IP) ──────────────────────────────────────────
        # Max requests per window; requests beyond this get HTTP 429
        self.rate_limit_requests: int = 10   # max requests per window
        self.rate_limit_window:   int = 60   # rolling window in seconds

        # ── vLLM wheel index (Blackwell / sm_120 safe nightly) ──────────────
        self.vllm_extra_index: str = "https://wheels.vllm.ai/nightly"

        # ── User-space gcc (extracted from .deb, no sudo) ────────────────────
        home             = Path.home()
        self.home:       Path = home
        self.gcc_bin:    Path = home / "gcc-local/root/usr/bin"
        self.gcc_libs:   str  = (
            f"{home}/gcc-local/root/usr/lib/x86_64-linux-gnu"
            f":{home}/gcc-local/root/usr/lib"
        )
        self.cc_path:  str = str(self.gcc_bin / "x86_64-linux-gnu-gcc-15")
        self.cxx_path: str = str(self.gcc_bin / "x86_64-linux-gnu-g++-15")

        # ── Load .env then populate os.environ ──────────────────────────────
        self._load_dotenv()

    # ── .env loader ─────────────────────────────────────────────────────────

    def _load_dotenv(self) -> None:
        """Parse KEY=VALUE pairs from workdir/.env into os.environ (no-override)."""
        env_file = self.workdir / ".env"
        if not env_file.exists():
            return
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

    # ── Environment enrichment ───────────────────────────────────────────────

    def enriched_env(self, extra: dict | None = None) -> dict:
        """
        Return os.environ merged with gcc paths and any caller-supplied extras.
        Pass the result to subprocess calls so gcc/uv are always on PATH.
        """
        env = {**os.environ, **(extra or {})}
        env["PATH"] = (
            f"{self.home}/.local/bin:{self.gcc_bin}:{env.get('PATH', '')}"
        )
        env["LD_LIBRARY_PATH"] = (
            f"{self.gcc_libs}:{env.get('LD_LIBRARY_PATH', '')}"
        )
        env["LIBRARY_PATH"] = (
            f"{self.home}/gcc-local/root/usr/lib/x86_64-linux-gnu:"
            f"{self.home}/gcc-local/root/usr/lib:{env.get('LIBRARY_PATH', '')}"
        )
        gcc_includes = (
            f"{self.home}/gcc-local/root/usr/include:"
            f"{self.home}/gcc-local/root/usr/include/x86_64-linux-gnu"
        )
        for variable in ("C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
            env[variable] = ":".join(
                path for path in (gcc_includes, env.get(variable, "")) if path
            )
        env.setdefault("CC",  self.cc_path)
        env.setdefault("CXX", self.cxx_path)
        return env

    # ── Pretty-print for debugging ───────────────────────────────────────────

    def __repr__(self) -> str:
        lines = [f"Config(workdir={self.workdir})"]
        attrs = [
            "venv", "venv_py", "lock_file", "vllm_log", "cf_log",
            "cf_bin", "key_file", "cf_url_file",
            "server_url", "model_id", "served_name",
            "cc_path", "cxx_path",
        ]
        for attr in attrs:
            val = getattr(self, attr)
            lines.append(f"  {attr:20s} = {val}")
        return "\n".join(lines)
