#!/usr/bin/env python3
"""ink2_server.py --port N [--host H] [--alias A] --sock S --ctx C --model M.gguf --tokenizer llama-tokenize --gguf-py DIR
                 [--mmproj MMPROJ.gguf --vis ink2_vis]

An OpenAI-compatible HTTP server in front of the engine (ink2_v1 --serve S), also answering the endpoints a llama-server
client expects: /health, /props, /slots, /tokenize, /apply-template, /metrics, /v1/models and /v1/chat/completions with and
without "stream". The prompt is Inkling 2's chat template as text (tools go in as the template's tool_declare and the model's
<|content_invoke_tool_json|> calls come back as tool_calls), tokens come from llama.cpp's llama-tokenize (vocabulary only), text
back from the GGUF vocabulary (GPT-2 byte level). Pictures with --mmproj and --vis: an image_url part (a data: URL, an http(s)
URL or a local path) goes through ink2_vis (llama.cpp's mtmd projector on the CPU, ~20 s for 1600x900 = 920 rows; the rows are
cached by the picture's SHA-256 in INK2_IMG_DIR, default /tmp/ink2-img) and its rows take the template's picture placeholder. Greedy only: the engine
has no sampler, so temperature and the like are ignored. Drafts: DSpark (MTP) 3 + prompt lookup 15 unless a request says
"mtp" / "lookup". The reply's <|content_thinking|> part streams as reasoning_content, its <|content_text|> part as content.
reasoning_effort (none / minimal / low / medium / high / xhigh) or chat_template_kwargs.reasoning_effort sets the template's
"Thinking effort level". One generation at a time; the other endpoints answer meanwhile. A client that leaves stops the
generation. SIGTERM stops the engine too when INK2_ENGINE_PID is set (run.sh does). A reasoning that falls into repeating
itself word for word (greedy decoding can) is cut after its first copy and the model writes its answer (Loop, reply())."""
import argparse
import base64
import codecs
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--alias", default="inkling2-mini")
ap.add_argument("--sock", required=True)
ap.add_argument("--ctx", type=int, default=32768)
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", required=True, help="llama.cpp's llama-tokenize")
ap.add_argument("--gguf-py", required=True, help="llama.cpp's gguf-py folder (knows NVFP4)")
ap.add_argument("--mmproj", default="", help="the vision projector GGUF; none = text only")
ap.add_argument("--vis", default="", help="ink2_vis (build/ink2_vis, the image encoder)")
A, _ = ap.parse_known_args()
TOKENIZE = A.tokenizer
sys.path.insert(0, A.gguf_py)
import gguf  # noqa: E402

EFFORT = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}
ROLE = {"user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>", "tool": "<|message_tool|>",
        "developer": "<|message_system|>"}
EOS = 200006
MTP = int(os.environ.get("INK2_OA_MTP", "3"))
LOOKUP = int(os.environ.get("INK2_OA_LOOKUP", "15"))

r = gguf.GGUFReader(A.model)
f = r.fields["tokenizer.ggml.tokens"]
TOKS = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
tf = r.fields.get("tokenizer.chat_template")
TEMPLATE = bytes(tf.parts[tf.data[0]]).decode("utf-8") if tf else ""
del r
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs, n_ = bs[:], 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n_)
        n_ += 1
BYTEDEC = {chr(c): b for b, c in zip(bs, cs)}
SPECIAL = {i: t for i, t in enumerate(TOKS) if t.startswith("<|") and t.endswith("|>")}
T_THINK = next(i for i, t in SPECIAL.items() if t == "<|content_thinking|>")
T_TEXT = next(i for i, t in SPECIAL.items() if t == "<|content_text|>")
T_MODEL = next(i for i, t in SPECIAL.items() if t == "<|message_model|>")
T_TOOL = next(i for i, t in SPECIAL.items() if t == "<|content_invoke_tool_json|>")
T_END = next(i for i, t in SPECIAL.items() if t == "<|end_message|>")
T_IMGPH = next(i for i, t in SPECIAL.items() if t == "<|unused_200054|>")

LOCK = threading.Lock()
STATE = {"busy": False, "n_prompt": 0, "n_decoded": 0, "n_ctx": A.ctx, "t_start": 0.0, "requests": 0, "tokens_out": 0}


def token_bytes(i):
    t = TOKS[i]
    return t.encode() if i in SPECIAL else bytes(BYTEDEC.get(ch, ord("?")) for ch in t)


def _tojson(x, sort_keys=False, separators=None):
    # the template engines' tojson (transformers / minja): json.dumps without ASCII escapes
    return json.dumps(x, ensure_ascii=False, sort_keys=sort_keys, separators=separators)


def _args(fn):
    a = fn.get("arguments")
    if isinstance(a, str):   # OpenAI sends the arguments as a JSON string; the template wants the object
        try:
            a = json.loads(a) if a.strip() else {}
        except ValueError:
            a = {}
    return a if isinstance(a, dict) else {}


IMG_PH = "<|unused_200054|>"   # the template's picture placeholder; the server puts the picture's rows in its place
IMG_BASE = 1 << 28           # ids of picture rows (the engine's IMG_BASE): IMG_BASE + (16-bit hash << 12) + row


def _image_src(part):
    v = part.get("image_url") or part.get("url") or part.get("image") or ""
    return v.get("url", "") if isinstance(v, dict) else str(v)


def render(messages, effort, tools=None, images=None):
    """The GGUF chat template as text (Inkling 2's own, rendered by hand; tools/template_check.py compares it with the Jinja
    template): tool_declare first when there are tools, "Thinking effort level" before the first non-system message, each
    message as <|message_role|><|content_.|>...<|end_message|>, an assistant's tool calls as
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>, a tool result's header with the
    tool's name, <|content_model_end_sampling|> after an assistant turn. A picture is <|content_image|> and the placeholder; its
    source goes to images (None: pictures are dropped)."""
    num = EFFORT[effort] if isinstance(effort, str) and effort in EFFORT else float(effort if effort is not None else 0.9)
    num = max(0.0, min(0.99, num))
    eff = "0" if num == 0.0 else f"{num}"
    effort_msg = f"<|message_system|><|content_text|>Thinking effort level: {eff}<|end_message|>"
    s, emitted = "", False
    if tools:
        specs = []
        for t in tools:
            fn = t.get("function", t) if isinstance(t, dict) else {}
            specs.append({"description": fn.get("description") or "", "name": fn.get("name"), "parameters": fn.get("parameters") or {},
                          "type": (t.get("type") if isinstance(t, dict) else None) or "function"})
        s += "<|message_system|>tool_declare<|content_xml|>" + _tojson(specs, sort_keys=True, separators=(",", ":")) + "<|end_message|>"
    for m in messages:
        role = m.get("role", "user")
        if role == "developer":
            role = "system"
        if role not in ROLE:
            role = "user"
        if not emitted and role != "system":
            s += effort_msg
            emitted = True
        head = ROLE[role]
        if role == "tool":
            if m.get("name"):
                head += m["name"]
            elif m.get("tool_call_id"):
                for prev in messages:
                    if prev.get("role") == "assistant" and prev.get("tool_calls"):
                        for tc in prev["tool_calls"]:
                            if tc.get("id") == m["tool_call_id"] and (tc.get("function") or {}).get("name"):
                                head = ROLE["tool"] + tc["function"]["name"]
        elif role == "assistant" and m.get("name"):
            head += m["name"]
        calls = m.get("tool_calls") if role == "assistant" else None
        if role == "assistant" and m.get("reasoning_content"):
            s += head + "<|content_thinking|>" + m["reasoning_content"] + "<|end_message|>"
        content = m.get("content")
        if isinstance(content, str) or content is None:
            text = content or ""
            if text or not (role == "assistant" and (calls or m.get("reasoning_content"))):
                s += head + "<|content_text|>" + text + "<|end_message|>"
        else:
            for part in content:
                if isinstance(part, str):
                    s += head + "<|content_text|>" + part + "<|end_message|>"
                elif part.get("type") in (None, "text", "input_text"):
                    s += head + "<|content_text|>" + (part.get("text") if isinstance(part.get("text"), str) else "") + "<|end_message|>"
                elif part.get("type") in ("image", "input_image", "image_url") and images is not None:
                    s += head + "<|content_image|>" + IMG_PH + "<|end_message|>"
                    images.append(_image_src(part))
        for tc in calls or []:
            fn = tc.get("function") or {}
            args = _args(fn)
            s += "<|message_model|>" + str(fn.get("name")) + "<|content_invoke_tool_json|>"
            s += '{"name":' + _tojson(fn.get("name"), sort_keys=True, separators=(",", ":")) + ',"args":{'
            s += ",".join(_tojson(k) + ":" + _tojson(args[k], separators=(",", ":")) for k in sorted(args))
            s += "}}<|end_message|>"
        if role == "assistant":
            s += "<|content_model_end_sampling|>"
    if not emitted:
        s += effort_msg
    return s


def tokenize(text):
    p = subprocess.run([TOKENIZE, "-m", A.model, "--ids", "--log-disable", "--stdin"], input=text, capture_output=True, text=True,
                       env={"CUDA_VISIBLE_DEVICES": ""}, timeout=600)
    line = [x for x in p.stdout.splitlines() if x.startswith("[")]
    if not line:
        raise RuntimeError("llama-tokenize: " + p.stderr[-300:])
    return json.loads(line[-1])


VIS = {"proc": None, "lock": threading.Lock(), "cache": {}}
IMG_DIR = Path(os.environ.get("INK2_IMG_DIR", "/tmp/ink2-img"))


def _image_bytes(src):
    if src.startswith("data:"):
        return base64.b64decode(src.split(",", 1)[1])
    if src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=60) as r:
            return r.read()
    return Path(src).read_bytes()


def encode_image(src):
    """(hash, rows file, rows): the picture's projector rows from ink2_vis, cached by its SHA-256."""
    data = _image_bytes(src)
    h = hashlib.sha256(data).hexdigest()
    hit = VIS["cache"].get(h)
    if hit and Path(hit[0]).exists():
        return h, hit[0], hit[1]
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    img, rows = IMG_DIR / f"{h}.img", IMG_DIR / f"{h}.f32"
    img.write_bytes(data)
    with VIS["lock"]:
        if VIS["proc"] is None or VIS["proc"].poll() is not None:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")   # the cards are the engine's
            VIS["proc"] = subprocess.Popen([A.vis, "--mmproj", A.mmproj, "--model", A.model, "--max-tokens", "4096"],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
            first = VIS["proc"].stdout.readline().split()
            if not first or first[0] != "ready":
                raise RuntimeError("ink2_vis did not start")
        VIS["proc"].stdin.write(f"{img} {rows}\n")
        VIS["proc"].stdin.flush()
        ans = VIS["proc"].stdout.readline().split()
    if not ans or ans[0] != "ok":
        raise RuntimeError("ink2_vis: " + " ".join(ans))
    n = int(ans[1])
    VIS["cache"][h] = (str(rows), n)
    print(f"ink2_server: picture {h[:12]}: {n} rows in {ans[3]} ms", file=sys.stderr, flush=True)
    return h, str(rows), n


def place_images(ids, images):
    """ids with every placeholder replaced by its picture's rows, and the engine's image fields."""
    if not images:
        return ids, {}
    ph = [i for i, t in enumerate(ids) if t == T_IMGPH]
    if len(ph) != len(images):
        raise RuntimeError(f"{len(images)} pictures, {len(ph)} placeholders")
    if not A.mmproj or not A.vis:
        raise RuntimeError("this server reads no pictures (start it with --mmproj and --vis)")
    enc = [encode_image(src) for src in images]
    out, table, files, off, last = [], [], [], 0, 0
    for (h, f, n), at in zip(enc, ph):
        out += ids[last:at]
        id0 = IMG_BASE + (int(h[:4], 16) << 12)
        out += [id0 + r for r in range(n)]
        table += [id0, n, off]
        files.append(f)
        off += n
        last = at + 1
    out += ids[last:]
    comb = IMG_DIR / ("request-" + hashlib.sha256("".join(h for h, _, _ in enc).encode()).hexdigest()[:16] + ".f32")
    if not comb.exists():
        with open(comb, "wb") as w:
            for f in files:
                w.write(Path(f).read_bytes())
    return out, {"img": table, "embd": str(comb)}


def effort_of(body):
    kw = body.get("chat_template_kwargs") or {}
    e = kw.get("reasoning_effort", body.get("reasoning_effort"))
    if e is None and kw.get("enable_thinking") is False:
        e = "none"
    return e if e is not None else 0.9


class Parts:
    """Turns the reply's tokens into events: ("r", text) reasoning, ("c", text) content, ("tool", {"name", "arguments"}) a
    finished tool call (<|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>). The name
    after <|message_model|> is not text; other template tokens are dropped; text outside any part counts as content."""

    def __init__(self):
        self.mode = "c"
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.tool = bytearray()

    def _switch(self, out, mode):
        tail = self.dec.decode(b"", final=True)
        if tail and self.mode in ("r", "c"):
            out.append((self.mode, tail))
        self.mode = mode

    def _tool_done(self, out):
        raw = bytes(self.tool).decode("utf-8", errors="replace")
        self.tool = bytearray()
        try:
            d = json.loads(raw)
            out.append(("tool", {"name": d["name"], "arguments": json.dumps(d.get("args") or {}, ensure_ascii=False)}))
        except (ValueError, KeyError, TypeError):
            out.append(("c", raw))   # not a call after all: keep its text

    def feed(self, ids):
        out = []
        for i in ids:
            if i == T_MODEL:
                self._switch(out, "head")
            elif i == T_THINK:
                self._switch(out, "r")
            elif i == T_TEXT:
                self._switch(out, "c")
            elif i == T_TOOL:
                self._switch(out, "tool")
                self.tool = bytearray()
            elif i == T_END:
                if self.mode == "tool":
                    self._tool_done(out)
                self._switch(out, "c")
            elif i in SPECIAL or i == EOS:
                continue
            elif self.mode == "tool":
                self.tool += token_bytes(i)
            elif self.mode in ("r", "c"):
                t = self.dec.decode(token_bytes(i))
                if t:
                    out.append((self.mode, t))
        return [(k, v) for k, v in out if v]

def periodic_tail(t, min_p, max_p, min_span, min_copies):
    """(start, p) when t from start on repeats one block of p characters (min_p..max_p) back to back, at least min_copies
    times and over min_span characters, else None. Candidate periods are the distances back to the earlier places of the
    last min_p characters (the nearest first)."""
    n = len(t)
    if n < min_span:
        return None
    key = t[n - min_p:]
    lo, hi = max(0, n - max_p - min_p), n - min_p
    for _ in range(16):
        pos = t.rfind(key, lo, hi)
        if pos < 0:
            return None
        p, hi = n - min_p - pos, pos + min_p - 1
        L = p
        while L + p <= n and t[n - L - p:n - L] == t[n - L:n - L + p]:
            L += p
        while L < n and t[n - L - 1] == t[n - L - 1 + p]:
            L += 1
        if L >= max(min_span, min_copies * p):
            return n - L, p
    return None


class Loop:
    """The engine has no sampler, and a long greedy reasoning can fall into repeating itself word for word (seen on 22 Nov
    after ~20 tool calls: two lines 194 times over and a list 21 times, 84k characters, no answer). Watches
    the reply's current reasoning part; step() gives how many of the reply's tokens to keep when its tail repeats one block
    of 32 to 8,000 characters at least 3 times back to back and over 2,000 characters: the reasoning up to the end of the
    block's first copy."""
    MIN_P, MAX_P, MIN_SPAN, MIN_COPIES, EVERY = 32, 8000, 2000, 3, 128

    def __init__(self):
        self.mode, self.text, self.ends, self.checked, self.hit = None, "", [], 0, None

    def step(self, mode, evs, n_out):
        if mode != self.mode:
            self.mode, self.text, self.ends, self.checked = mode, "", [(0, n_out - 1)], 0
        if mode != "r":
            return None
        self.text += "".join(v for k, v in evs if k == "r")
        self.ends.append((len(self.text), n_out))
        if len(self.text) - self.checked < self.EVERY:
            return None
        self.checked = len(self.text)
        self.hit = periodic_tail(self.text, self.MIN_P, self.MAX_P, self.MIN_SPAN, self.MIN_COPIES)
        if self.hit is None:
            return None
        upto = self.hit[0] + self.hit[1]
        return next(k for c, k in reversed(self.ends) if c <= upto)


# what closes a looping reasoning before the answer is forced (the reasoning is in English)
NUDGE = "\n\nI am repeating the same lines. I have enough to answer, so I stop thinking here and write the answer now."
NUDGE_IDS = []


def reply(ids, mx, req, on_part=None, mode="c"):
    """The engine's reply to the prompt ids, at most mx tokens: (tokens, totals, stopped). Each token's parts go to
    on_part(kind, val) as they come ("r" reasoning, "c" content, "tool" a call). A loop in the reasoning (Loop) stops
    the engine; the reasoning is kept up to the end of the block's first copy, closed by NUDGE, and the model writes its
    answer (<|end_message|><|message_model|><|content_text|>): the engine reads the kept part again from the prompt's
    snapshot (from the start when over ~3,000 tokens were written) and goes on. A second loop ends the reply (stopped).
    mode: the part the prompt ends in ("r" when it ends inside a reasoning)."""
    parts, guard, out = Parts(), Loop(), []
    parts.mode = mode
    tot = {"cached": None, "read_s": 0.0, "write_s": 0.0, "drafted": 0, "accepted": 0, "cuts": 0}
    while True:
        t_req, t_first, final, keep = time.time(), None, None, None
        lines = engine_lines(dict(req, tokens=ids + out, max=mx - len(out), stream=1))
        try:
            for msg in lines:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                if "t" in msg:
                    t_first = t_first or time.time()
                    for t in msg["t"]:
                        out.append(t)
                        evs = parts.feed([t])
                        if on_part:
                            for kind, val in evs:
                                on_part(kind, val)
                        keep = guard.step(parts.mode, evs, len(out))
                        if keep is not None:
                            break
                    STATE["n_decoded"] = len(out)
                    if keep is not None:
                        break
                elif "tokens" in msg:
                    final = msg
        finally:
            lines.close()   # a cut: the engine stops at its next write
        if final is not None:
            if tot["cached"] is None:
                tot["cached"] = final.get("cached", 0)
            for k in ("read_s", "write_s", "drafted", "accepted"):
                tot[k] += final.get(k, 0)
        elif keep is None:
            raise RuntimeError("the engine closed without an answer")
        else:   # the cut run sends no totals: its times from here
            tot["cached"] = tot["cached"] if tot["cached"] is not None else 0
            tot["read_s"] += (t_first or t_req) - t_req
            tot["write_s"] += time.time() - (t_first or t_req)
        if keep is None:
            return out, tot, False
        tot["cuts"] += 1
        start, p = guard.hit
        print(f"ink2_server: a loop in the reasoning (copies x characters, tokens cut / written): {(guard.checked - start) / p:.1f} x {p}, "
              f"{len(out) - keep} / {len(out)}", file=sys.stderr, flush=True)
        out = out[:keep]
        if not NUDGE_IDS:
            NUDGE_IDS.extend(i for i in tokenize(NUDGE) if i not in SPECIAL)
        forced = NUDGE_IDS + [T_END, T_MODEL, T_TEXT]
        if tot["cuts"] > 1 or len(out) + len(forced) >= mx:
            return out, tot, True
        evs = parts.feed(forced)
        if on_part:
            for kind, val in evs:
                on_part(kind, val)
        out += forced
        guard = Loop()


def engine_lines(req):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(14400)   # a 500k read takes over an hour on two Tesla M40 (4227 s for 495k; the P100s 3041 s)
    s.connect(A.sock)
    s.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    try:
        while True:
            c = s.recv(1 << 16)
            if not c:
                break
            buf += c
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    yield json.loads(line)
    finally:
        s.close()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj, ctype="application/json"):
        b = obj.encode() if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            return self._send(200, {"status": "ok"})
        if path == "/props":
            return self._send(200, {"default_generation_settings": {"n_ctx": A.ctx, "params": {"temperature": 0.0}}, "total_slots": 1,
                                    "model_path": A.model, "model_alias": A.alias, "chat_template": TEMPLATE, "build_info": "ink2_v1",
                                    "modalities": {"vision": bool(A.mmproj and A.vis), "audio": False}})
        if path == "/slots":
            st = STATE
            return self._send(200, [{"id": 0, "n_ctx": A.ctx, "is_processing": st["busy"], "n_prompt_tokens": st["n_prompt"],
                                     "next_token": [{"n_decoded": st["n_decoded"], "has_next_token": st["busy"]}]}])
        if path == "/metrics":
            return self._send(200, f"# ink2_v1\nllamacpp:requests_processing {1 if STATE['busy'] else 0}\n"
                                   f"llamacpp:tokens_predicted_total {STATE['tokens_out']}\n", "text/plain")
        if path in ("/v1/models", "/models"):
            return self._send(200, {"object": "list", "data": [{"id": A.alias, "object": "model", "owned_by": "ink2_v1",
                                                                "meta": {"n_ctx_train": 1048576, "n_ctx": A.ctx}}]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            body = self._body()
        except Exception as ex:
            return self._send(400, {"error": f"bad json: {ex}"})
        if path == "/tokenize":
            try:
                ids = tokenize(body.get("content") or "")
            except Exception as ex:
                return self._send(500, {"error": str(ex)})
            return self._send(200, {"tokens": ids})
        if path == "/apply-template":
            return self._send(200, {"prompt": render(body.get("messages") or [], effort_of(body), body.get("tools"))})
        if path in ("/v1/chat/completions", "/chat/completions"):
            return self.chat(body)
        return self._send(404, {"error": "not found"})

    def chat(self, body):
        t0 = time.time()
        try:
            tools = body.get("tools") if body.get("tool_choice") != "none" else None
            images = []
            ids = tokenize(render(body.get("messages") or [], effort_of(body), tools, images))
            ids, img_req = place_images(ids, images)
        except Exception as ex:
            return self._send(500, {"error": str(ex)})
        if len(ids) >= A.ctx - 1:
            return self._send(400, {"error": {"message": f"prompt of {len(ids)} tokens exceeds the context of {A.ctx}", "type": "exceed_context_size_error"}})
        mx = int(body.get("max_tokens") or body.get("max_completion_tokens") or 0) or (A.ctx - len(ids) - 1)
        mx = max(1, min(mx, A.ctx - len(ids) - 1))
        stream = bool(body.get("stream"))
        req = {"stop": [EOS], "mtp": int(body.get("mtp", MTP)), "lookup": int(body.get("lookup", LOOKUP)), **img_req}
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(t0)
        if not LOCK.acquire(timeout=3600):
            return self._send(503, {"error": "busy"})
        try:
            STATE.update(busy=True, n_prompt=len(ids), n_decoded=0, t_start=t0)
            STATE["requests"] += 1
            calls, on_part = [], None
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def sse(obj):
                    self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()

                sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": A.alias,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]})

                def on_part(kind, val):
                    if kind == "tool":
                        call = {"index": len(calls), "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                                "function": {"name": val["name"], "arguments": val["arguments"]}}
                        calls.append(call)
                        delta = {"tool_calls": [call]}
                    else:
                        delta = {"reasoning_content" if kind == "r" else "content": val}
                    sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": A.alias,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
            try:
                out, tot, stopped = reply(ids, mx, req, on_part)
            except (BrokenPipeError, ConnectionResetError):
                return   # the client left: closing the engine's socket stops it at its next write
            n = len(out)
            STATE["tokens_out"] += n
            w = tot["write_s"] or 1e-9
            if not stream:
                think, content = [], []
                for kind, val in Parts().feed(out):
                    if kind == "tool":
                        calls.append({"index": len(calls), "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                                      "function": {"name": val["name"], "arguments": val["arguments"]}})
                    else:
                        (think if kind == "r" else content).append(val)
            finish = "tool_calls" if calls else "stop" if stopped or (out and out[-1] == EOS) else "length"
            timings = {"prompt_n": len(ids) - tot["cached"], "cache_n": tot["cached"], "prompt_ms": 1000 * tot["read_s"],
                       "prompt_per_second": (len(ids) - tot["cached"]) / max(tot["read_s"], 1e-9),
                       "predicted_n": n, "predicted_ms": 1000 * w, "predicted_per_second": max(n - 1, 0) / w,
                       "draft_n": tot["drafted"], "draft_n_accepted": tot["accepted"]}
            if tot["cuts"]:
                timings["loop_cuts"] = tot["cuts"]
            usage = {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n,
                     "prompt_tokens_details": {"cached_tokens": tot["cached"]}}
            if stream:
                sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": A.alias,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage, "timings": timings})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            message = {"role": "assistant", "content": "".join(content).strip() or (None if calls else ""),
                       "reasoning_content": "".join(think).strip()}
            if calls:
                message["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in calls]
            return self._send(200, {"id": cid, "object": "chat.completion", "created": created, "model": A.alias,
                                    "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                                    "usage": usage, "timings": timings})
        except Exception as ex:
            if stream:
                try:
                    self.wfile.write(b"data: " + json.dumps({"error": {"message": str(ex)}}).encode() + b"\n\n")
                except Exception:
                    pass
                return
            return self._send(500, {"error": {"message": str(ex)}})
        finally:
            STATE.update(busy=False)
            LOCK.release()


def _stop(*_):   # the node stops a profile with SIGTERM to this process: the engine (started by ink2-ovoj.sh) goes with it
    pid = int(os.environ.get("INK2_ENGINE_PID") or 0)
    if pid:
        try:
            os.kill(pid, 15)
        except OSError:
            pass
    os._exit(0)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    srv = ThreadingHTTPServer((A.host, A.port), H)
    srv.daemon_threads = True
    print(f"ink2_server: {A.alias} na {A.host}:{A.port}, ctx {A.ctx}, pogon {A.sock}", file=sys.stderr, flush=True)
    srv.serve_forever()
