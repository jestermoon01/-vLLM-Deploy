"""
api_key.py
----------
Cell 4 — Generate or load an API key and persist it to .api_key.

Debug standalone:
    python -c "
    from vllm_deploy.config  import Config
    from vllm_deploy.runner  import Runner
    from vllm_deploy.api_key import ApiKeyManager
    key = ApiKeyManager(Config(), Runner(Config())).load_or_create()
    print('key:', key)
    "
"""

import os
import secrets

from .config import Config
from .runner import Runner


class ApiKeyManager:
    """
    Responsible for one thing: produce a valid API key string.

    Priority order:
      1. Kaggle secrets (if running inside Kaggle kernel)
      2. Existing .api_key file on disk
      3. Generate a new random token and save it
    """

    def __init__(self, cfg: Config, runner: Runner) -> None:
        self.cfg    = cfg
        self.runner = runner

    def load_or_create(self) -> str:
        """
        Return the API key, creating and persisting it if necessary.
        Also sets os.environ["SERVER_API_KEY"] for child processes.
        """
        self.runner.banner("Cell 4 – Config & API key")
        api_key = self._try_kaggle() or self._try_file() or self._generate_new()
        self._persist(api_key)
        return api_key

    # ── Sources (tried in order) ─────────────────────────────────────────────

    def _try_kaggle(self) -> str | None:
        """Return key from Kaggle UserSecretsClient, or None if unavailable."""
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore
            key = UserSecretsClient().get_secret("VLLM_API_KEY")
            print("Loaded API key from Kaggle secrets.")
            return key
        except Exception:
            return None

    def _try_file(self) -> str | None:
        """Return key stored in .api_key, or None if the file doesn't exist."""
        if self.cfg.key_file.exists():
            key = self.cfg.key_file.read_text().strip()
            print(f"Reusing existing key from {self.cfg.key_file}")
            return key
        return None

    def _generate_new(self) -> str:
        """Generate a cryptographically random 32-byte URL-safe token."""
        key = secrets.token_urlsafe(32)
        print(f"Generated new API key: {key}")
        return key

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist(self, api_key: str) -> None:
        """Write key to disk (mode 0o600) and export to the process environment."""
        os.environ["SERVER_API_KEY"] = api_key
        self.cfg.key_file.write_text(api_key)
        self.cfg.key_file.chmod(0o600)
        print(f"Key saved to {self.cfg.key_file}")
