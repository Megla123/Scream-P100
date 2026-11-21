#!/usr/bin/env python3
"""hy4_openai.py [PORT] [SOCKET] [HOST] (2026-11-21): an OpenAI-style /v1/chat/completions in front of hy4_v1 --serve. Hy4's chat
template (the GGUF's, rendered with jinja2 as llama.cpp's --jinja would), llama.cpp's tokenizer, greedy writing (temperature is
ignored), one request at a time. reasoning_effort (top level or in chat_template_kwargs): high / max think, everything else is
no_think. The answer: content after the template's think end, the thought in reasoning_content; usage and llama-server-style
timings. Files (run.sh sets them): HY4_GGUF (the trunk, shard 1), HY4_TOKENIZE (llama.cpp's llama-tokenize), HY4_TEMPLATE (the
GGUF's chat template, which setup.sh writes out)."""
import base64
import codecs
import json
import os
import socket
import subprocess
import tempfile
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from jinja2 import Environment

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8097
SOCK = sys.argv[2] if len(sys.argv) > 2 else "hy4.sock"
HOST = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1"
GGUF = os.environ["HY4_GGUF"]
TOKENIZE = os.environ["HY4_TOKENIZE"]
TPL = Path(os.environ["HY4_TEMPLATE"]).read_text(encoding="utf-8")
THINK_END = "</think:opensource>"
TC_BEGIN, TC_END = "<tool_calls:opensource>", "</tool_calls:opensource>"
CALL_BEGIN, CALL_END = "<tool_call:opensource>", "</tool_call:opensource>"
KEY_B, KEY_E, VAL_B, VAL_E = "<arg_key:opensource>", "</arg_key:opensource>", "<arg_value:opensource>", "</arg_value:opensource>"
HY_END = "<｜hy_end:opensource｜>"

env = Environment(trim_blocks=True, lstrip_blocks=True)


def _raise(m):
    raise RuntimeError(m)


env.globals["raise_exception"] = _raise
# the template asks tojson(ensure_ascii=False) as llama.cpp's minja knows it (jinja2's filter has no such argument and escapes <>&)
env.filters["tojson"] = lambda v, ensure_ascii=False, indent=None, **kw: json.dumps(v, ensure_ascii=ensure_ascii, indent=indent)
template = env.from_string(TPL)
lock = threading.Lock()


def tokenize(text):
    # the text through a file: one argument is limited to 128 KiB (a long Koda prompt is more); the cards are full, so llama.cpp
    # must not touch them
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write(text)
        name = f.name
    try:
        r = subprocess.run([TOKENIZE, "-m", GGUF, "-f", name, "--ids", "--log-disable"], capture_output=True, text=True, timeout=600,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    finally:
        os.unlink(name)
    line = [l for l in r.stdout.splitlines() if l.startswith("[")][-1]
    return json.loads(line)


STOP = tokenize(HY_END)   # the end of an answer (a single special token)


def engine(tokens, max_tokens, lookup=-1, on_bytes=None):
    """one request to hy4_v1 --serve; on_bytes(bytes) for each written token as it comes (the engine's "stream" lines)"""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(7200)
    s.connect(SOCK)
    req = {"tokens": tokens, "max": max_tokens, "stop": STOP}
    if lookup >= 0:
        req["lookup"] = lookup
    if on_bytes:
        req["stream"] = 1
    s.sendall((json.dumps(req) + "\n").encode())
    f = s.makefile("rb")
    try:
        for line in f:
            r = json.loads(line)
            if "tok" in r:
                if on_bytes and r.get("b"):
                    on_bytes(base64.b64decode(r["b"]))
                continue
            return r
        return {"error": "the engine closed the connection"}
    finally:
        f.close()
        s.close()


class Splitter:
    """the stream into reasoning, content and the tool-call block: text is held back while it may be the start of a marker"""

    def __init__(self, thinking):
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.buf, self.tools = "", None
        self.mode = "reasoning" if thinking else "content"

    def feed(self, b, final=False):
        self.buf += self.dec.decode(b, final=final)
        out = []   # (kind, text)
        while True:
            if self.mode == "tools":
                self.tools += self.buf
                self.buf = ""
                break
            marker = THINK_END if self.mode == "reasoning" else TC_BEGIN
            self.buf = self.buf.replace(HY_END, "").replace("<think:opensource>", "")
            i = self.buf.find(marker)
            if i >= 0:
                if i:
                    out.append((self.mode, self.buf[:i]))
                self.buf = self.buf[i + len(marker):]
                if self.mode == "reasoning":
                    self.mode = "content"
                else:
                    self.mode, self.tools = "tools", TC_BEGIN
                continue
            keep = 0   # the longest tail that may be a marker still arriving
            if not final:
                for m in (marker, HY_END, "<think:opensource>"):
                    for k in range(min(len(m) - 1, len(self.buf)), 0, -1):
                        if self.buf.endswith(m[:k]):
                            keep = max(keep, k)
                            break
            cut = len(self.buf) - keep
            if cut > 0:
                out.append((self.mode, self.buf[:cut]))
                self.buf = self.buf[cut:]
            break
        return [(k, t) for k, t in out if t]


def history(messages):
    """OpenAI messages for the template: a tool call's arguments as a mapping (the template walks arguments.items())"""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a) if a.strip() else {}
                    except ValueError:
                        fn["arguments"] = {"arguments": a}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def tool_calls(text):
    """Hy4's <tool_calls:opensource> block: (text before it, OpenAI tool_calls or None)"""
    if TC_BEGIN not in text:
        return text, None
    head, rest = text.split(TC_BEGIN, 1)
    rest = rest.split(TC_END, 1)[0]
    calls = []
    for part in rest.split(CALL_BEGIN)[1:]:
        part = part.split(CALL_END, 1)[0]
        name = part.split(KEY_B, 1)[0].strip()
        args = {}
        for kv in part.split(KEY_B)[1:]:
            if KEY_E not in kv:
                continue
            k, v = kv.split(KEY_E, 1)
            v = v.split(VAL_B, 1)[-1].split(VAL_E, 1)[0]
            try:
                args[k.strip()] = json.loads(v) if v.strip()[:1] in "[{" or v.strip() in ("true", "false", "null") or v.strip().lstrip("-").replace(".", "", 1).isdigit() else v
            except ValueError:
                args[k.strip()] = v
        calls.append({"id": f"call_{len(calls)}_{int(time.time() * 1000) % 100000}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return head, calls or None


def effort_of(body):
    e = (body.get("chat_template_kwargs") or {}).get("reasoning_effort", body.get("reasoning_effort"))
    return e if e in ("high", "max") else "no_think"


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        sys.stderr.write(time.strftime("%H:%M:%S ") + fmt % a + "\n")

    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            return self._send(200, {"object": "list", "data": [{"id": "hy4-flash-iq3xxs", "object": "model"}]})
        if self.path.rstrip("/") == "/health":   # as llama-server: 503 until the engine takes connections
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as t:
                    t.settimeout(2)
                    t.connect(SOCK)
                return self._send(200, {"status": "ok"})
            except OSError:
                return self._send(503, {"status": "loading"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            eff = effort_of(body)
            text = template.render(messages=history(body["messages"]), add_generation_prompt=True, reasoning_effort=eff,
                                   tools=body.get("tools"))
            if body.get("stream"):
                return self._stream(body, eff, text)
            with lock:
                t0 = time.time()
                toks = tokenize(text)
                t1 = time.time()
                r = engine(toks, int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024),
                           int(body.get("lookup", -1)))
            if "error" in r:
                return self._send(500, {"error": r["error"]})
            out = r.get("text", "")
            finish = "length" if len(r.get("tokens", [])) >= int(body.get("max_tokens") or 1024) else "stop"
            reasoning = None
            if THINK_END in out:
                reasoning, out = out.split(THINK_END, 1)
                reasoning = reasoning.replace("<think:opensource>", "").strip()
            out, calls = tool_calls(out.replace(HY_END, ""))
            out = out.strip()
            if calls:
                finish = "tool_calls"
            n, w, rd = len(r.get("tokens", [])), r.get("write_s") or 1e-9, r.get("read_s") or 1e-9
            pn = r.get("prompt_n", len(toks)) - r.get("cached", 0)
            msg = {"role": "assistant", "content": out}
            if reasoning is not None:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = calls
            self._send(200, {
                "id": "hy4-" + str(int(t0 * 1000)), "object": "chat.completion", "created": int(t0), "model": "hy4-flash-iq3xxs",
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": len(toks), "completion_tokens": n, "total_tokens": len(toks) + n},
                "timings": {"prompt_n": pn, "prompt_ms": rd * 1000, "prompt_per_second": pn / rd, "predicted_n": n,
                            "predicted_ms": w * 1000, "predicted_per_second": n / w, "cache_n": r.get("cached", 0),
                            "tokenize_ms": (t1 - t0) * 1000, "hits": r.get("hits"), "misses": r.get("misses"),
                            "drafted": r.get("drafted"), "accepted": r.get("accepted")}})
        except Exception as ex:   # noqa: BLE001
            self._send(500, {"error": str(ex)})

    def _stream(self, body, eff, text):
        """stream: SSE chunks as llama-server sends them (delta.reasoning_content / delta.content, the tool calls at the end)"""
        cid, created = "hy4-" + str(int(time.time() * 1000)), int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        first = [True]

        def chunk(delta, finish=None, extra=None):
            if first[0]:   # the role goes with the first text: a chunk sent before the prompt is read looks like the first token
                first[0] = False   # to a client that times it (llama-benchy's prompt speed came out near zero)
                delta = {"role": "assistant", **delta}
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": "hy4-flash-iq3xxs",
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if extra:
                obj.update(extra)
            self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

        sp = Splitter(eff in ("high", "max"))

        def send(parts):
            for kind, t in parts:
                chunk({"reasoning_content": t} if kind == "reasoning" else {"content": t})

        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024)
        with lock:
            t0 = time.time()
            toks = tokenize(text)
            r = engine(toks, max_tokens, int(body.get("lookup", -1)), lambda b: send(sp.feed(b)))
        send(sp.feed(b"", final=True))
        if "error" in r:
            chunk({}, "stop", {"error": r["error"]})
        else:
            finish = "length" if len(r.get("tokens", [])) >= max_tokens else "stop"
            calls = None
            if sp.tools is not None:
                head, calls = tool_calls(sp.tools)
                if head.strip():
                    chunk({"content": head})
            if calls:
                finish = "tool_calls"
                chunk({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
            n, w, rd = len(r.get("tokens", [])), r.get("write_s") or 1e-9, r.get("read_s") or 1e-9
            pn = r.get("prompt_n", len(toks)) - r.get("cached", 0)
            chunk({}, finish, {"usage": {"prompt_tokens": len(toks), "completion_tokens": n, "total_tokens": len(toks) + n},
                               "timings": {"prompt_n": pn, "prompt_ms": rd * 1000, "prompt_per_second": pn / rd, "predicted_n": n,
                                           "predicted_ms": w * 1000, "predicted_per_second": n / w, "cache_n": r.get("cached", 0)}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    print(f"hy4_openai: :{PORT} -> {SOCK}, stop {STOP}", flush=True)
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
