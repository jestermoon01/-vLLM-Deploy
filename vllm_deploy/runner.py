"""
runner.py
---------
Thin subprocess wrappers shared by every other module.
Keeps all Popen/run logic in one place so it's easy to mock during testing.

Debug:
    python -c "
    from vllm_deploy.config import Config
    from vllm_deploy.runner import Runner
    r = Runner(Config())
    r.run('echo hello')
    "
"""

import subprocess
import sys
from pathlib import Path

from .config import Config


class Runner:
    """Wraps subprocess.run and subprocess.Popen with shared env enrichment."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    # ── Foreground ───────────────────────────────────────────────────────────

    def run(
        self,
        cmd: str | list,
        *,
        check: bool = True,
        env: dict | None = None,
        capture: bool = False,
    ) -> subprocess.CompletedProcess:
        """
        Run a command synchronously.

        Args:
            cmd:     Shell string or argv list.
            check:   Raise SystemExit on non-zero exit code.
            env:     Extra env vars merged on top of enriched_env().
            capture: If True, capture stdout/stderr AND print them.

        Returns:
            subprocess.CompletedProcess with .stdout / .stderr when capture=True.
        """
        merged = self.cfg.enriched_env(env)
        if isinstance(cmd, str):
            print(f"$ {cmd}")
            result = subprocess.run(
                cmd, shell=True, env=merged, capture_output=capture, text=True
            )
        else:
            print("$ " + " ".join(str(c) for c in cmd))
            result = subprocess.run(
                cmd, env=merged, capture_output=capture, text=True
            )

        if capture and result.stdout:
            print(result.stdout, end="")
        if capture and result.stderr:
            print(result.stderr, end="", file=sys.stderr)

        if check and result.returncode != 0:
            raise SystemExit(
                f"Command failed (exit {result.returncode}): "
                + (cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd))
            )
        return result

    # ── Background ───────────────────────────────────────────────────────────

    def popen_bg(
        self,
        cmd: str,
        log_path: Path,
        env: dict | None = None,
    ) -> subprocess.Popen:
        """
        Start a command in the background, redirecting stdout+stderr to log_path.

        Returns:
            subprocess.Popen handle — caller is responsible for .wait()/.kill().
        """
        merged = self.cfg.enriched_env(env)
        print(f"$ {cmd}  >>  {log_path}  &")
        log_fh = open(log_path, "w")
        return subprocess.Popen(
            cmd, shell=True, env=merged, stdout=log_fh, stderr=log_fh
        )

    # ── uv resolver ─────────────────────────────────────────────────────────

    def uv_cmd(self) -> str:
        """Return the path to a working `uv` binary, or raise SystemExit."""
        candidates = [
            str(self.cfg.home / ".local/bin/uv"),
            "/usr/local/bin/uv",
            "uv",
        ]
        for candidate in candidates:
            r = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                env=self.cfg.enriched_env(),
            )
            if r.returncode == 0:
                return candidate
        raise SystemExit("uv not found — run the installer step first.")

    # ── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def banner(title: str) -> None:
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)
