"""
installer.py
------------
Cell 2 + Cell 3 — Install uv, create the Python 3.12 venv, install vLLM nightly,
write a lock file, and verify CUDA is reachable.

Debug standalone:
    python -c "
    from vllm_deploy.config    import Config
    from vllm_deploy.runner    import Runner
    from vllm_deploy.installer import Installer
    i = Installer(Config(), Runner(Config()))
    i.install_uv()     # Cell 2 only
    i.install_vllm()   # Cell 3 only
    "
"""

from .config import Config
from .runner import Runner


class Installer:
    """
    Handles all Python environment and vLLM package installation steps.

    Split into two public methods so you can run them independently:
        install_uv()   — ensure the `uv` package manager is present
        install_vllm() — create venv, pip-install vLLM nightly, write lock file
    """

    def __init__(self, cfg: Config, runner: Runner) -> None:
        self.cfg    = cfg
        self.runner = runner

    # ── Cell 2 ──────────────────────────────────────────────────────────────

    def install_uv(self) -> None:
        """Install the `uv` package manager if not already present."""
        self.runner.banner("Cell 2 – Install / verify uv")
        import subprocess

        r = subprocess.run(
            [str(self.cfg.home / ".local/bin/uv"), "--version"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            print(f"uv already installed: {r.stdout.strip()}")
            return

        self.runner.run("curl -LsSf https://astral.sh/uv/install.sh | sh")
        self.runner.run(f"{self.runner.uv_cmd()} --version")

    # ── Cell 3 ──────────────────────────────────────────────────────────────

    def install_vllm(self) -> None:
        """
        Create a Python 3.12 venv, install vLLM nightly (Blackwell sm_120 safe),
        write requirements.lock.txt, and verify CUDA visibility.
        """
        self.runner.banner("Cell 3 – Create venv + install vLLM (Blackwell-safe nightly)")
        uv = self.runner.uv_cmd()

        self._create_venv(uv)
        self._pip_install_vllm(uv)
        self._write_lock_file(uv)
        self._verify_cuda()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _create_venv(self, uv: str) -> None:
        """Create (or recreate) the venv at cfg.venv using Python 3.12."""
        self.runner.run(
            [uv, "venv", str(self.cfg.venv), "--python", "3.12", "--clear"]
        )

    def _pip_install_vllm(self, uv: str) -> None:
        """Install vLLM nightly pre-release into the venv."""
        self.runner.run(
            [
                uv, "pip", "install",
                "--python", str(self.cfg.venv_py),
                "--pre", "vllm",
                "--extra-index-url", self.cfg.vllm_extra_index,
            ],
            env={"TORCH_CUDA_ARCH_LIST": "12.0"},
        )

    def _write_lock_file(self, uv: str) -> None:
        """Freeze installed packages into requirements.lock.txt."""
        result = self.runner.run(
            [uv, "pip", "freeze", "--python", str(self.cfg.venv_py)],
            capture=True,
            check=True,
        )
        self.cfg.lock_file.write_text(result.stdout)
        print(f"Lock file written: {self.cfg.lock_file}")

    def _verify_cuda(self) -> None:
        """Run a quick Python probe inside the venv to confirm CUDA is usable."""
        probe = (
            "import vllm, torch; "
            "ok = torch.cuda.is_available(); "
            "dev = torch.cuda.get_device_name(0) if ok else 'N/A'; "
            "print('vllm', vllm.__version__, '| torch', torch.__version__, "
            "      '| cuda ok:', ok, '|', dev); "
            "exit(0 if ok else 1)"
        )
        result = self.runner.run(
            [str(self.cfg.venv_py), "-c", probe],
            check=False,
            capture=True,
        )
        if result.returncode != 0:
            print("\n[ERROR] CUDA not available or vllm import failed.")
            raise SystemExit(
                "Fix CUDA / vLLM before continuing. "
                f"Check {self.cfg.venv_py} and driver version."
            )
