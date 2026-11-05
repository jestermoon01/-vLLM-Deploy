"""
keepalive.py
------------
Cell 11 — Background heartbeat thread that prints a timestamp every 10 min.
Useful on Kaggle / Colab kernels that auto-disconnect idle sessions.

Debug standalone:
    python -c "
    import time
    from vllm_deploy.keepalive import KeepAlive
    ka = KeepAlive(interval=5)  # 5 s for testing
    ka.start()
    time.sleep(12)
    ka.stop()
    print('done')
    "
"""

import threading
import time


class KeepAlive:
    """
    Daemon thread that emits a heartbeat log line at regular intervals.

    Args:
        interval: Seconds between heartbeat prints (default 600 = 10 min).
    """

    def __init__(self, interval: int = 600) -> None:
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the heartbeat daemon thread (non-blocking)."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="keepalive")
        self._thread.start()
        print(f"Keep-alive thread started (interval={self.interval}s).")

    def stop(self) -> None:
        """Signal the heartbeat thread to exit gracefully."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 1)
        print("Keep-alive thread stopped.")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval):
            print(f"[keep-alive] {time.strftime('%H:%M')}", flush=True)
