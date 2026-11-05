"""
benchmark.py
------------
Cell 7 + Cell 8 — Smoke test (single request) and 20-concurrent-user load test.

Debug standalone:
    python -c "
    from vllm_deploy.config    import Config
    from vllm_deploy.runner    import Runner
    from vllm_deploy.benchmark import Benchmark
    import os; api_key = open('.api_key').read().strip()
    b = Benchmark(Config(), Runner(Config()), api_key)
    b.smoke_test()      # Cell 7 only
    b.load_test(n=5)    # Cell 8 with fewer users
    "
"""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .config import Config
from .runner import Runner


class Benchmark:
    """
    Validates and stress-tests the running vLLM server.

    Methods:
        smoke_test()         — Cell 7: single palindrome check request
        load_test(n=20)      — Cell 8: N concurrent requests + OOM check
    """

    def __init__(self, cfg: Config, runner: Runner, api_key: str) -> None:
        self.cfg     = cfg
        self.runner  = runner
        self.api_key = api_key
        self._headers = {"Authorization": f"Bearer {api_key}"}

    # ── Cell 7 ──────────────────────────────────────────────────────────────

    def smoke_test(self) -> None:
        """Send a single 'is palindrome?' prompt and print the response."""
        self.runner.banner("Cell 7 – Smoke test")
        out = self._chat(
            "Write a Python function that checks if a string is a palindrome.",
            max_tokens=256,
        )
        print(out["choices"][0]["message"]["content"])
        print(out["usage"])

    # ── Cell 8 ──────────────────────────────────────────────────────────────

    def load_test(self, n: int = 20) -> None:
        """
        Fire `n` concurrent requests and report wall time, latency, and throughput.

        Args:
            n: Number of parallel users (default 20).
        """
        self.runner.banner(f"Cell 8 – {n} concurrent users benchmark")

        t0 = time.time()
        with ThreadPoolExecutor(n) as ex:
            results = list(ex.map(self._single_load_request, range(n)))
        wall = time.time() - t0

        latencies   = [r[0] for r in results]
        total_tokens = sum(r[1] for r in results)

        print(
            f"{n} users | wall {wall:.1f}s "
            f"| latency avg {sum(latencies)/n:.1f}s max {max(latencies):.1f}s "
            f"| {total_tokens/wall:.0f} tok/s total "
            f"~{total_tokens/wall/n:.1f} tok/s per user"
        )
        self._check_oom_warnings()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _chat(self, prompt: str, max_tokens: int = 300) -> dict:
        """POST to /v1/chat/completions and return parsed JSON."""
        r = requests.post(
            f"{self.cfg.server_url}/v1/chat/completions",
            headers=self._headers,
            timeout=300,
            json={
                "model":       self.cfg.served_name,
                "max_tokens":  max_tokens,
                "temperature": 0.2,
                "messages":    [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        return r.json()

    def _single_load_request(self, idx: int) -> tuple[float, int]:
        """
        Send one load-test request for worker `idx`.

        Returns:
            (elapsed_seconds, completion_token_count)
        """
        t = time.time()
        out = self._chat(
            f"Write Python function #{idx} that merges two sorted lists, "
            f"with docstring.",
            max_tokens=300,
        )
        return time.time() - t, out["usage"]["completion_tokens"]

    def _check_oom_warnings(self) -> None:
        """Grep the vLLM log for OOM / preemption warnings and print them."""
        r = subprocess.run(
            rf"grep -i -E 'preempt|out of memory|error' {self.cfg.vllm_log} | tail -5",
            shell=True,
            capture_output=True,
            text=True,
        )
        if r.stdout.strip():
            print("\nWarnings in vllm.log:\n" + r.stdout)
        else:
            print("No preemption / OOM warnings ✓")
