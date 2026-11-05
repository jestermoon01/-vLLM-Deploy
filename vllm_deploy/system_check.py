"""
system_check.py
---------------
Cell 1 — Environment verification: GPU, Python version, gcc, HF token.

Debug standalone:
    python -c "
    from vllm_deploy.config       import Config
    from vllm_deploy.runner       import Runner
    from vllm_deploy.system_check import SystemCheck
    SystemCheck(Config(), Runner(Config())).run()
    "
"""

import os
import sys
from pathlib import Path

from .config import Config
from .runner import Runner


class SystemCheck:
    """
    Verifies the host environment before any installation begins.

    Checks:
    - nvidia-smi (GPU present and driver loaded)
    - Python interpreter version
    - gcc / g++ binaries (user-space, no sudo)
    - HF_TOKEN presence in environment
    """

    def __init__(self, cfg: Config, runner: Runner) -> None:
        self.cfg    = cfg
        self.runner = runner

    def run(self) -> None:
        self.runner.banner("Cell 1 – Environment check")
        self._check_gpu()
        self._check_python()
        self._check_gcc()
        self._check_hf_token()

    # ── Individual checks (call independently for targeted debugging) ────────

    def _check_gpu(self) -> None:
        """Print GPU info via nvidia-smi. Raises SystemExit if not found."""
        self.runner.run(
            "nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version"
            " --format=csv"
        )

    def _check_python(self) -> None:
        """Print Python version of the current interpreter."""
        self.runner.run(f"{sys.executable} --version")

    def _check_gcc(self) -> None:
        """Confirm user-space gcc binaries exist at expected paths."""
        for label, path_str in [("CC", self.cfg.cc_path), ("CXX", self.cfg.cxx_path)]:
            exists = Path(path_str).exists()
            print(f"{label:4s} = {path_str}  (exists: {exists})")
            if not exists:
                print(f"  [WARN] {label} binary not found. "
                      "Extract gcc_4*.deb first or adjust Config.gcc_bin.")

    def _check_hf_token(self) -> None:
        """Warn if HF_TOKEN is not set (required to download gated models)."""
        token_set = "HF_TOKEN" in os.environ
        print(f"HF_TOKEN set: {token_set}")
        if not token_set:
            print("  [WARN] HF_TOKEN not set. "
                  "Add HF_TOKEN=<token> to your .env file.")
