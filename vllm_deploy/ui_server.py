"""
ui_server.py
------------
Chat UI server with per-IP rate limiting. No API key required.

Classes:
    RateLimiter  — thread-safe sliding-window rate limiter keyed by IP
    VllmProxy    — urllib wrapper for the vLLM backend (no auth header)
    ChatHandler  — HTTP handler: routing + rate-limit enforcement
    UiServer     — assembles everything and runs on port 8000

Debug standalone:
    # Test rate limiter
    python -c "
    from vllm_deploy.ui_server import RateLimiter
    rl = RateLimiter(max_requests=3, window=10)
    for i in range(5):
        ok, remaining, reset = rl.check('127.0.0.1')
        print(f'request {i+1}: allowed={ok} remaining={remaining} reset_in={reset:.1f}s')
    "

    # Test proxy (vLLM must be running)
    python -c "
    from vllm_deploy.ui_server import VllmProxy
    p = VllmProxy('http://127.0.0.1:8001')
    print(p.health())
    "
"""

import collections
import http.server
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MAX_TOKENS  = 512
DEFAULT_TEMPERATURE = 0.2
SERVED_MODEL_NAME   = "qwen-coder"

# ── HTML UI ───────────────────────────────────────────────────────────────────

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen Coder \u2013 Test UI</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f1117; color: #e8eaed;
    display: flex; flex-direction: column; height: 100vh;
  }
  header {
    padding: 12px 20px; background: #1a1d27;
    border-bottom: 1px solid #2d3148;
    display: flex; align-items: center; gap: 12px;
  }
  header h1 { font-size: 1.1rem; font-weight: 600; }
  .badge { font-size: .7rem; background: #2563eb; padding: 2px 8px; border-radius: 99px; }
  #chat {
    flex: 1; overflow-y: auto; padding: 20px;
    display: flex; flex-direction: column; gap: 16px;
  }
  .msg {
    max-width: 80%; padding: 12px 16px; border-radius: 12px;
    line-height: 1.55; white-space: pre-wrap; word-break: break-word;
    font-size: .93rem;
  }
  .msg.user   { align-self: flex-end; background: #2563eb; }
  .msg.bot    { align-self: flex-start; background: #1e2030; border: 1px solid #2d3148; }
  .msg.error  { align-self: flex-start; background: #3b1a1a; border: 1px solid #7f1d1d; color: #fca5a5; }
  .msg.system { align-self: center; font-size: .78rem; color: #6b7280; font-style: italic; }
  pre  { background: #0f1117; padding: 10px 14px; border-radius: 8px;
         overflow-x: auto; margin-top: 8px; font-size: .85rem; }
  code { font-family: 'Cascadia Code','Fira Code',monospace; }
  footer {
    padding: 12px 20px; background: #1a1d27;
    border-top: 1px solid #2d3148;
    display: flex; gap: 10px;
  }
  #input {
    flex: 1; padding: 10px 14px; background: #0f1117;
    border: 1px solid #2d3148; border-radius: 8px; color: #e8eaed;
    font-size: .93rem; resize: none; height: 46px; max-height: 160px; overflow-y: auto;
  }
  #input:focus { outline: none; border-color: #2563eb; }
  button {
    padding: 10px 20px; background: #2563eb; color: #fff;
    border: none; border-radius: 8px; cursor: pointer;
    font-size: .93rem; font-weight: 500; transition: background .15s;
  }
  button:hover  { background: #1d4ed8; }
  button:active { background: #1e40af; }
  button:disabled { background: #374151; cursor: not-allowed; }
  .controls {
    padding: 8px 20px; background: #1a1d27;
    border-bottom: 1px solid #2d3148;
    display: flex; gap: 16px; align-items: center;
    font-size: .82rem; color: #9ca3af; flex-wrap: wrap;
  }
  .controls label { display: flex; align-items: center; gap: 6px; }
  .controls input[type=range]  { width: 100px; accent-color: #2563eb; }
  .controls input[type=number] {
    width: 70px; background: #0f1117; border: 1px solid #2d3148;
    border-radius: 6px; color: #e8eaed; padding: 2px 6px;
  }
  #ratelimit {
    margin-left: auto; font-size: .78rem; color: #6b7280;
    transition: color .3s;
  }
  #ratelimit.warn { color: #f59e0b; }
  #ratelimit.full { color: #ef4444; }
  #spinner {
    display: none; width: 18px; height: 18px;
    border: 2px solid #374151; border-top-color: #2563eb;
    border-radius: 50%; animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header>
  <h1>&#x1F916; Qwen2.5 Coder 7B</h1>
  <span class="badge">AWQ &middot; float16</span>
  <span id="status" style="margin-left:auto;font-size:.78rem;color:#6b7280">connecting&hellip;</span>
</header>
<div class="controls">
  <label>Max tokens
    <input type="number" id="maxTokens" value="512" min="64" max="4096" step="64">
  </label>
  <label>Temperature
    <input type="range" id="temp" min="0" max="1" step="0.05" value="0.2">
    <span id="tempVal">0.20</span>
  </label>
  <button onclick="clearChat()" style="padding:4px 12px;font-size:.78rem;background:#374151">Clear</button>
  <span id="ratelimit"></span>
</div>
<div id="chat">
  <div class="msg system">Send a message to start chatting with Qwen2.5 Coder 7B.</div>
</div>
<footer>
  <textarea id="input" placeholder="Ask the model anything&hellip;" rows="1"></textarea>
  <div id="spinner"></div>
  <button id="sendBtn" onclick="sendMessage()">Send</button>
</footer>
<script>
const chat       = document.getElementById('chat');
const input      = document.getElementById('input');
const sendBtn    = document.getElementById('sendBtn');
const spinner    = document.getElementById('spinner');
const status     = document.getElementById('status');
const rlBadge    = document.getElementById('ratelimit');
const tempSlider = document.getElementById('temp');
const tempVal    = document.getElementById('tempVal');

tempSlider.oninput = () => tempVal.textContent = parseFloat(tempSlider.value).toFixed(2);

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 160) + 'px';
});
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

function updateRateLimit(headers) {
  const remaining = headers.get('X-RateLimit-Remaining');
  const limit     = headers.get('X-RateLimit-Limit');
  const reset     = headers.get('X-RateLimit-Reset');
  if (remaining === null) return;
  const r = parseInt(remaining), l = parseInt(limit);
  rlBadge.textContent = `${r}/${l} requests left`;
  rlBadge.className   = r === 0 ? 'full' : r <= 2 ? 'warn' : '';
  if (reset) {
    const secs = Math.ceil(parseFloat(reset));
    if (secs > 0) rlBadge.textContent += ` (resets in ${secs}s)`;
  }
}

function renderMarkdown(text) {
  return text
    .replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = renderMarkdown(text);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function clearChat() {
  chat.innerHTML = '<div class="msg system">Chat cleared.</div>';
}

async function sendMessage() {
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  input.style.height = '46px';
  sendBtn.disabled = true;
  spinner.style.display = 'block';
  addMessage('user', msg);
  const thinking = addMessage('bot', '\u2026');
  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: msg,
        max_tokens: parseInt(document.getElementById('maxTokens').value),
        temperature: parseFloat(tempSlider.value)
      })
    });
    updateRateLimit(resp.headers);
    const data = await resp.json();
    if (resp.status === 429) {
      thinking.className = 'msg error';
      thinking.textContent = '\u26a0 Rate limit reached. ' + (data.error || '');
      return;
    }
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    thinking.className = 'msg bot';
    thinking.innerHTML = renderMarkdown(data.content);
  } catch (err) {
    thinking.className = 'msg error';
    thinking.textContent = '\u26a0 ' + err.message;
  } finally {
    sendBtn.disabled = false;
    spinner.style.display = 'none';
    chat.scrollTop = chat.scrollHeight;
  }
}

async function checkHealth() {
  try {
    const r = await fetch('/health');
    const data = await r.json();
    updateRateLimit(r.headers);
    status.textContent = r.ok ? '\u25cf online'  : '\u25cf degraded';
    status.style.color  = r.ok ? '#22c55e' : '#f59e0b';
  } catch {
    status.textContent = '\u25cf offline';
    status.style.color  = '#ef4444';
  }
}
checkHealth();
setInterval(checkHealth, 10000);
</script>
</body>
</html>
"""


# ════════════════════════════════════════════════════════════════════════════
# RateLimiter — sliding-window, per-IP, thread-safe
# ════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Sliding-window rate limiter keyed by client IP.

    Each IP gets `max_requests` calls within a rolling `window` seconds.
    Timestamps older than the window are discarded on every check.

    Args:
        max_requests: Allowed requests per window (default 10).
        window:       Rolling window in seconds (default 60).

    Returns from check():
        (allowed: bool, remaining: int, reset_in: float)
            allowed   — whether this request should proceed
            remaining — how many more requests are allowed right now
            reset_in  — seconds until the oldest timestamp expires (0 if empty)

    Debug:
        rl = RateLimiter(max_requests=3, window=10)
        for i in range(5):
            ok, rem, rst = rl.check('1.2.3.4')
            print(ok, rem, f'{rst:.1f}s')
    """

    def __init__(self, max_requests: int = 10, window: int = 60) -> None:
        self.max_requests = max_requests
        self.window       = window
        # {ip: deque of timestamps}
        self._buckets: dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    def check(self, ip: str) -> tuple[bool, int, float]:
        """
        Record a request from `ip` and decide whether to allow it.

        Returns:
            (allowed, remaining, reset_in_seconds)
        """
        now = time.monotonic()
        cutoff = now - self.window

        with self._lock:
            if ip not in self._buckets:
                self._buckets[ip] = collections.deque()

            bucket = self._buckets[ip]

            # Drop timestamps outside the window
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            count = len(bucket)
            if count >= self.max_requests:
                # Oldest timestamp tells us when a slot opens
                reset_in  = (bucket[0] + self.window) - now
                return False, 0, max(0.0, reset_in)

            bucket.append(now)
            remaining = self.max_requests - (count + 1)
            reset_in  = (bucket[0] + self.window) - now if bucket else 0.0
            return True, remaining, max(0.0, reset_in)

    def stats(self) -> dict:
        """Return per-IP usage snapshot for debugging."""
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            return {
                ip: len([t for t in dq if t > cutoff])
                for ip, dq in self._buckets.items()
            }


# ════════════════════════════════════════════════════════════════════════════
# VllmProxy — urllib calls to vLLM backend (no auth header)
# ════════════════════════════════════════════════════════════════════════════

class VllmProxy:
    """
    Thin wrapper around the vLLM OpenAI-compatible API.
    No authentication headers — vLLM runs without --api-key.

    Debug:
        proxy = VllmProxy("http://127.0.0.1:8001")
        print(proxy.health())
        print(proxy.chat("Hello", max_tokens=50))
    """

    def __init__(self, vllm_url: str) -> None:
        self.vllm_url = vllm_url

    def health(self) -> dict:
        """GET /health → {'status': 'ok', 'vllm_status': <http_code>}."""
        try:
            with urllib.request.urlopen(
                f"{self.vllm_url}/health", timeout=4
            ) as resp:
                return {"status": "ok", "vllm_status": resp.status}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    def models(self) -> dict:
        """GET /v1/models → parsed JSON."""
        with urllib.request.urlopen(
            f"{self.vllm_url}/v1/models", timeout=8
        ) as resp:
            return json.loads(resp.read())

    def chat(
        self,
        message: str,
        max_tokens: int   = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> dict:
        """
        POST /v1/chat/completions.

        Returns:
            {'content': <reply>, 'usage': {...}}

        Raises:
            urllib.error.HTTPError, Exception
        """
        payload = json.dumps({
            "model":       SERVED_MODEL_NAME,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "messages":    [{"role": "user", "content": message}],
        }).encode()

        req = urllib.request.Request(
            f"{self.vllm_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())

        return {
            "content": data["choices"][0]["message"]["content"],
            "usage":   data.get("usage", {}),
        }


# ════════════════════════════════════════════════════════════════════════════
# ChatHandler — routing + rate-limit enforcement
# ════════════════════════════════════════════════════════════════════════════

class ChatHandler(http.server.BaseHTTPRequestHandler):
    """
    HTTP handler. Delegates all business logic to VllmProxy.
    Rate limiter is checked on every /chat POST.

    proxy   and  rate_limiter  are injected as class attributes by UiServer.
    """

    proxy:        "VllmProxy"
    rate_limiter: "RateLimiter"

    # ── Routing ──────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        routes = {
            "/":           self._serve_ui,
            "/index.html": self._serve_ui,
            "/health":     self._handle_health,
            "/models":     self._handle_models,
        }
        handler = routes.get(self.path)
        if handler:
            handler()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path == "/chat":
            self._handle_chat()
        else:
            self._send_json({"error": "not found"}, 404)

    # ── GET handlers ─────────────────────────────────────────────────────────

    def _serve_ui(self) -> None:
        self._send_html(HTML_UI)

    def _handle_health(self) -> None:
        result = self.__class__.proxy.health()
        status = 200 if result.get("status") == "ok" else 502
        self._send_json(result, status)

    def _handle_models(self) -> None:
        try:
            self._send_json(self.__class__.proxy.models())
        except Exception as exc:
            self._send_json({"error": str(exc)}, 502)

    # ── POST handlers ─────────────────────────────────────────────────────────

    def _handle_chat(self) -> None:
        # ── Rate limit check ─────────────────────────────────────────────────
        ip = self.client_address[0]
        allowed, remaining, reset_in = self.__class__.rate_limiter.check(ip)

        rl = self.__class__.rate_limiter
        rl_headers = {
            "X-RateLimit-Limit":     str(rl.max_requests),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset":     f"{reset_in:.1f}",
            "X-RateLimit-Window":    str(rl.window),
        }

        if not allowed:
            self._send_json(
                {
                    "error": (
                        f"Rate limit exceeded: {rl.max_requests} requests "
                        f"per {rl.window}s. "
                        f"Retry in {reset_in:.0f}s."
                    )
                },
                status=429,
                extra_headers=rl_headers,
            )
            return

        # ── Parse body ───────────────────────────────────────────────────────
        body = self._read_json_body()
        if body is None:
            return  # error already sent

        message     = body.get("message", "").strip()
        max_tokens  = int(body.get("max_tokens",  DEFAULT_MAX_TOKENS))
        temperature = float(body.get("temperature", DEFAULT_TEMPERATURE))

        if not message:
            self._send_json({"error": "message is required"}, 400,
                            extra_headers=rl_headers)
            return

        # ── Proxy to vLLM ────────────────────────────────────────────────────
        try:
            result = self.__class__.proxy.chat(message, max_tokens, temperature)
            self._send_json(result, extra_headers=rl_headers)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode(errors="replace")
            self._send_json(
                {"error": f"vLLM HTTP {exc.code}: {err_body}"}, 502,
                extra_headers=rl_headers,
            )
        except Exception as exc:
            self._send_json({"error": str(exc)}, 502, extra_headers=rl_headers)

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _read_json_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length))
        except Exception as exc:
            self._send_json({"error": f"bad request: {exc}"}, 400)
            return None

    def _send_html(self, body: str, status: int = 200) -> None:
        encoded = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(
        self,
        obj: dict,
        status: int = 200,
        extra_headers: dict | None = None,
    ) -> None:
        encoded = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")


# ════════════════════════════════════════════════════════════════════════════
# UiServer — assembles and runs everything on port 8000
# ════════════════════════════════════════════════════════════════════════════

class UiServer:
    """
    Wires VllmProxy + RateLimiter into ChatHandler, then runs ThreadingHTTPServer.

    Args:
        vllm_url:        Backend URL (default http://127.0.0.1:8001)
        host:            Bind address (default 0.0.0.0)
        port:            UI port (default 8000)
        rate_limit_req:  Max requests per window per IP (default 10)
        rate_limit_win:  Window size in seconds (default 60)

    Debug (no full deploy needed):
        server = UiServer(vllm_url="http://127.0.0.1:8001", port=8000)
        server.serve()
    """

    def __init__(
        self,
        vllm_url:        str = "http://127.0.0.1:8001",
        host:            str = "0.0.0.0",
        port:            int = 8000,
        rate_limit_req:  int = 10,
        rate_limit_win:  int = 60,
    ) -> None:
        self.host           = host
        self.port           = port
        self.vllm_url       = vllm_url
        self.rate_limit_req = rate_limit_req
        self.rate_limit_win = rate_limit_win

    @classmethod
    def from_config(cls, cfg) -> "UiServer":
        """Construct a UiServer directly from a Config object."""
        return cls(
            vllm_url       = cfg.server_url,
            host           = cfg.server_host,
            port           = cfg.ui_port,
            rate_limit_req = cfg.rate_limit_requests,
            rate_limit_win = cfg.rate_limit_window,
        )

    def serve(self) -> None:
        """Inject proxy + rate limiter into handler, print banner, block."""
        ChatHandler.proxy        = VllmProxy(self.vllm_url)
        ChatHandler.rate_limiter = RateLimiter(self.rate_limit_req, self.rate_limit_win)

        server = http.server.ThreadingHTTPServer((self.host, self.port), ChatHandler)

        print(
            f"\n{'='*58}\n"
            f"  Chat UI  ->  http://{self.host}:{self.port}\n"
            f"  vLLM     ->  {self.vllm_url}\n"
            f"  Rate limit: {self.rate_limit_req} requests / {self.rate_limit_win}s per IP\n"
            f"{'='*58}\n"
            f"Open http://localhost:{self.port} in a browser.\n"
            f"Press Ctrl+C to stop.\n"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
            server.server_close()
