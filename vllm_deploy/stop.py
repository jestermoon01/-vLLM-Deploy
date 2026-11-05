"""Stop processes started from this vLLM deployment."""

import argparse
import signal
from pathlib import Path

import psutil

from .config import Config


def _resolved(path: str | Path) -> Path:
    return Path(path).resolve()


def _project_process_role(
    command: list[str], workdirs: list[Path], venvs: list[Path]
) -> str | None:
    for workdir in workdirs:
        deploy_script = _resolved(workdir / "deploy_vllm.py")
        ui_script = _resolved(workdir / "test_server.py")
        for arg in command:
            try:
                path = _resolved(arg)
            except (OSError, RuntimeError, ValueError):
                continue
            if path == deploy_script:
                return "deployment orchestrator"
            if path == ui_script:
                return "chat UI"
            if path == _resolved(workdir / "cloudflared") and "tunnel" in command:
                return "Cloudflare tunnel"

        for venv in venvs:
            vllm_executable = _resolved(venv / "bin" / "vllm")
            if any(_resolved_arg_equals(arg, vllm_executable) for arg in command):
                if "serve" in command:
                    return "vLLM server"
            python_executable = _resolved(venv / "bin" / "python")
            if any(_resolved_arg_equals(arg, python_executable) for arg in command):
                if "vllm_deploy.launch" in command:
                    return "deployment launcher"
    return None


def _resolved_arg_equals(arg: str, expected: Path) -> bool:
    try:
        return _resolved(arg) == expected
    except (OSError, RuntimeError, ValueError):
        return False


def _collect_processes(
    workdirs: list[Path], venvs: list[Path]
) -> tuple[dict[int, psutil.Process], dict[int, str]]:
    targets: dict[int, psutil.Process] = {}
    roots: dict[int, str] = {}
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            role = _project_process_role(command, workdirs, venvs)
            if role is None:
                continue
            roots[process.pid] = role
            targets[process.pid] = process
            for child in process.children(recursive=True):
                targets[child.pid] = child
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied as exc:
            raise SystemExit(
                f"Permission denied while inspecting process {process.pid}."
            ) from exc
    return targets, roots


def stop_project(workdirs: list[Path], dry_run: bool = False) -> int:
    unique_workdirs = list(dict.fromkeys(_resolved(path) for path in workdirs))
    venvs = [workdir / "venv" for workdir in unique_workdirs]
    targets, roots = _collect_processes(unique_workdirs, venvs)

    if not roots:
        print("No running processes for this project were found.")
        return 0

    if dry_run:
        for pid, role in sorted(roots.items()):
            print(f"Would stop {role} (PID {pid})")
        print(f"Would also stop {len(targets) - len(roots)} child process(es).")
        return 0

    for pid, process in targets.items():
        try:
            process.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied as exc:
            raise SystemExit(f"Permission denied stopping process {pid}.") from exc

    _, alive = psutil.wait_procs(list(targets.values()), timeout=20)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied as exc:
            raise SystemExit(f"Permission denied killing process {process.pid}.") from exc

    _, survivors = psutil.wait_procs(alive, timeout=5)
    if survivors:
        pids = ", ".join(str(process.pid) for process in survivors)
        raise SystemExit(f"Project processes did not stop: {pids}")

    remaining_targets, remaining_roots = _collect_processes(unique_workdirs, venvs)
    if remaining_roots or remaining_targets:
        pids = ", ".join(str(pid) for pid in sorted(remaining_targets))
        raise SystemExit(f"Project processes remain after shutdown: {pids}")

    ports = {8000, 8001}
    remaining_listeners = [
        connection
        for connection in psutil.net_connections(kind="inet")
        if connection.status == psutil.CONN_LISTEN
        and connection.laddr.port in ports
    ]
    if remaining_listeners:
        listeners = ", ".join(
            f"{connection.laddr.ip}:{connection.laddr.port} (PID {connection.pid})"
            for connection in remaining_listeners
        )
        raise SystemExit(
            "Ports used by the project still have listeners; they were not "
            f"stopped because they may belong to another application: {listeners}"
        )

    for pid, role in sorted(roots.items()):
        print(f"Stopped {role} (PID {pid}).")
    print(f"Stopped {len(targets) - len(roots)} child process(es).")
    print("Verified: no project processes remain and ports 8000/8001 are closed.")
    return len(roots)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        action="append",
        default=[],
        help="Additional project directory to include in process matching.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching project processes without stopping them.",
    )
    args = parser.parse_args()

    cfg = Config()
    workdirs = [cfg.workdir, *(Path(path) for path in args.project_dir)]
    stop_project(workdirs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
