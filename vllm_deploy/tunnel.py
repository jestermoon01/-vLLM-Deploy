"""
tunnel.py
---------
Cell 9 — Download cloudflared (if needed) and open a quick Cloudflare tunnel
so the server is reachable from outside the machine.

Debug standalone:
    python -c "
    from vllm_deploy.config import Config
    from vllm_deploy.runner import Runner
    from vllm_deploy.tunnel import CloudflareTunnel
    t = CloudflareTunnel(Config(), Runner(Config()))
    url = t.open()
    print('public URL:', url)
    "
"""

import subprocess
import time

from .config import Config
from .runner import Runner


class CloudflareTunnel:
    """
    Manages a temporary Cloudflare quick-tunnel for the vLLM server.

    Methods:
        open()  — download binary if needed, start tunnel, return public URL
    """

    # URL printed by cloudflared in the log
    _URL_PATTERN = r"https://[a-z0-9-]*\.trycloudflare\.com"
    _CF_RELEASE  = (
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64"
    )

    def __init__(self, cfg: Config, runner: Runner) -> None:
        self.cfg    = cfg
        self.runner = runner

    def open(self, wait_seconds: int = 15) -> str | None:
        """
        Start the Cloudflare tunnel.

        Args:
            wait_seconds: Seconds to wait for the tunnel URL to appear in the log.

        Returns:
            Public https:// URL, or None if it could not be parsed.
        """
        self.runner.banner("Cell 9 – Cloudflare quick tunnel")
        self._ensure_binary()
        self._start_tunnel()
        return self._wait_for_url(wait_seconds)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _ensure_binary(self) -> None:
        """Download cloudflared binary if it doesn't exist yet."""
        if not self.cfg.cf_bin.exists():
            self.runner.run(
                f"curl -sSL -o {self.cfg.cf_bin} {self._CF_RELEASE}"
            )
            self.cfg.cf_bin.chmod(0o755)
        self.runner.run(f"{self.cfg.cf_bin} --version")

    def _start_tunnel(self) -> None:
        """Launch cloudflared in the background, redirecting output to cf.log."""
        self.runner.popen_bg(
            f"{self.cfg.cf_bin} tunnel --url {self.cfg.server_url} --no-autoupdate",
            self.cfg.cf_log,
        )

    def _wait_for_url(self, wait_seconds: int) -> str | None:
        """
        Wait for the public URL to appear in cf.log.

        Returns:
            URL string or None if not found within the timeout.
        """
        print(f"Waiting {wait_seconds}s for tunnel URL …")
        time.sleep(wait_seconds)

        r = subprocess.run(
            rf"grep -o '{self._URL_PATTERN}' {self.cfg.cf_log} | head -1",
            shell=True,
            capture_output=True,
            text=True,
        )
        public_url = r.stdout.strip()

        if public_url:
            print(f"\n✅  Public URL: {public_url}")
            self.cfg.cf_url_file.write_text(public_url)
            return public_url

        print("[WARN] Could not find tunnel URL in cf.log.")
        self._dump_cf_log()
        return None

    def _dump_cf_log(self, chars: int = 2000) -> None:
        """Print the last `chars` of cf.log for manual inspection."""
        try:
            print(self.cfg.cf_log.read_text()[-chars:])
        except FileNotFoundError:
            print(f"  (cf.log not found: {self.cfg.cf_log})")
