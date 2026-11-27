#!/usr/bin/env python3
"""ink2_server.py --port N [--host H] [--alias A] --sock S --ctx C --model M.gguf --tokenizer llama-tokenize --gguf-py DIR
                 [--tok ink2_tok] [--mmproj MMPROJ.gguf --vis ink2_vis]

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
"Thinking effort level"; a request with none gets INK2_EFFORT (low = 0.2 since 23 Nov, was 0.9: on our sets low gave the best
code in the fewest tokens, medium more tokens, worse code and empty answers), and effort none without tools answers without
reasoning (the prompt ends with the answer's start). A tool call written as {"name":"X"args":{..}} (the comma before "args"
left out, the model's own pick on some prompts) is read as the call. One generation at a time; the other endpoints answer meanwhile. A client that leaves stops the
generation, also in a part that streams nothing (a tool call). SIGTERM stops the engine too when INK2_ENGINE_PID is set
(run.sh does). A reasoning that falls into repeating itself word for word (greedy decoding can) is cut after its first copy
and the model writes its answer; an answer or a tool call that does so for over 16,000 characters (INK2_LOOP_TEXT) ends the
reply (Loop, reply()). While a request runs, the log gets a line every minute (INK2_LOG_EVERY).
With --tok the tokenizer is build/ink2_tok, started once and kept (llama-tokenize takes 0.9 s to start for every request),
and a conversation's next request tokenizes only its new pieces (Tokens). The token offsets where the prompt's messages
start go to the engine with every request: it keeps checkpoints there, so a request that parts from the conversation in
the middle (a chat client moving a note from the last message to the new one, a shortened history) reads only from the
latest checkpoint before that place. A conversation that ends in the same tool call with the same result 3 times
(INK2_LOOP_TURNS) gets a system message that says so before the model's turn, and the same call written anyway is dropped
(turn_loop(), chat()). A reasoning whose copies differ only in numbers is cut too (INK2_LOOP_MASK), an answer or a call that
repeats a paragraph after 3 copies (INK2_LOOP_PAR), and the loop across turns also sees cycles of 2-3 steps and results that
differ only in numbers (turn_loop2). A looping tool call is cut and the model writes on after a system message
(INK2_LOOP_CALL); a reply that ends in a loop with nothing to show says so to the user. While a prompt is read, /slots says how
far (llama-server's n_prompt_tokens_cache / _processed and n_prompt_tokens_total, the read so far estimated)."""
import argparse
import array
import base64
import codecs
import collections
import hashlib
import json
import math
import os
import re
import select
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
ap.add_argument("--tok", default="", help="ink2_tok (the tokenizer that stays up); none = llama-tokenize once a request")
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
EFFORT_DEFAULT = os.environ.get("INK2_EFFORT", "0.2")   # p37: the effort of a request that names none (was 0.9)

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
STATE = {"busy": False, "n_prompt": 0, "n_decoded": 0, "n_ctx": A.ctx, "t_start": 0.0, "requests": 0, "tokens_out": 0, "part": "",
         "t_note": 0.0}
# /metrics' counters since the start, summed from every finished request's timings (llama-server's names): prompt tokens read
# (not the ones from the conversation cache: checkpoints, parked conversations), their seconds, the cached ones, the written
# tokens' seconds, drafted and accepted draft tokens. STATE's tokens_out is llamacpp:tokens_predicted_total
TOTALS = {"prompt_tokens": 0, "prompt_seconds": 0.0, "cached_tokens": 0, "predicted_seconds": 0.0, "drafted": 0, "accepted": 0,
          "turn_loops": 0, "loop_drops": 0}
TOTALS_MUTEX = threading.Lock()
METRICS = [   # name, type, help, value
    ("requests_processing", "gauge", "Number of requests processing.", lambda: 1 if STATE["busy"] else 0),
    ("prompt_tokens_total", "counter", "Number of prompt tokens processed.", lambda: TOTALS["prompt_tokens"]),
    ("prompt_seconds_total", "counter", "Prompt process time.", lambda: round(TOTALS["prompt_seconds"], 6)),
    ("prompt_tokens_cached_total", "counter", "Number of prompt tokens taken from the conversation cache.",
     lambda: TOTALS["cached_tokens"]),
    ("tokens_predicted_total", "counter", "Number of generation tokens processed.", lambda: STATE["tokens_out"]),
    ("tokens_predicted_seconds_total", "counter", "Predict process time.", lambda: round(TOTALS["predicted_seconds"], 6)),
    ("n_drafted_total", "counter", "Number of draft tokens.", lambda: TOTALS["drafted"]),
    ("n_drafted_accepted_total", "counter", "Number of accepted draft tokens.", lambda: TOTALS["accepted"]),
    ("loop_turns_total", "counter", "Requests that ended in the same tool call with the same result INK2_LOOP_TURNS times.",
     lambda: TOTALS["turn_loops"]),
    ("loop_calls_dropped_total", "counter", "Looping tool calls the model wrote again after the note, dropped.",
     lambda: TOTALS["loop_drops"]),
]
LOG_EVERY = float(os.environ.get("INK2_LOG_EVERY", "60"))


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
    if num == 0.0 and not tools and (not messages or messages[-1].get("role") != "assistant"):
        s += "<|message_model|><|content_text|>"   # p37: effort 0, no tools: the answer at once, without reasoning
    return s


def tokenize_once(text):
    # --no-escape: the prompt as written. Without it llama-tokenize turns a written \n, \t or \\ (two characters, as in
    # source code) into the character itself, and the model read code with broken string literals (so until p20)
    p = subprocess.run([TOKENIZE, "-m", A.model, "--ids", "--log-disable", "--no-escape", "--stdin"], input=text,
                       capture_output=True, text=True, env={"CUDA_VISIBLE_DEVICES": ""}, timeout=600)
    line = [x for x in p.stdout.splitlines() if x.startswith("[")]
    if not line:
        raise RuntimeError("llama-tokenize: " + p.stderr[-300:])
    return json.loads(line[-1])


class Tokens:
    """The prompt's tokens and where its pieces start. A piece starts at a role token (<|message_user|> and the like; special
    tokens, so the pieces tokenize one by one exactly as in the whole text) and its tokens are cached by its text: a
    conversation's next request tokenizes only its new pieces, all of them in one call. ink2_tok (--tok) stays up; without it,
    or when it fails, llama-tokenize runs once a call (0.9 s to start)."""
    ROLE = re.compile(r"(?=<\|message_(?:user|model|system|tool)\|>)")
    MAX_CHARS = 64 << 20

    def __init__(self):
        self.proc, self.lock, self.cache, self.chars, self.role_ids = None, threading.Lock(), {}, 0, None

    def raw(self, text):
        if A.tok:
            data = text.encode("utf-8")
            try:
                if self.proc is None or self.proc.poll() is not None:
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")   # the cards are the engine's
                    self.proc = subprocess.Popen([A.tok, "--model", A.model], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                 stderr=subprocess.DEVNULL, env=env)
                    if not self.proc.stdout.readline().startswith(b"ready"):
                        raise RuntimeError("ink2_tok did not start")
                self.proc.stdin.write(str(len(data)).encode() + b"\n" + data)
                self.proc.stdin.flush()
                ans = self.proc.stdout.readline().split()
                if not ans or ans[0] != b"ok":
                    raise RuntimeError("ink2_tok: " + b" ".join(ans[:8]).decode("utf-8", "replace"))
                return [int(x) for x in ans[1:]]
            except Exception as ex:   # the next call starts it again; this one goes the old way
                print(f"ink2_server: ink2_tok: {ex}", file=sys.stderr, flush=True)
                try:
                    self.proc.kill()
                except Exception:
                    pass
                self.proc = None
        return tokenize_once(text)

    def ids(self, text):
        """(tokens, token offsets of the pieces)"""
        with self.lock:
            if self.role_ids is None:
                self.role_ids = {self.raw(f"<|message_{r}|>")[-1] for r in ("user", "model", "system", "tool")}
            pieces = [x for x in self.ROLE.split(text) if x]
            if not pieces or not self.ROLE.match(pieces[0]):   # text before the first message: no pieces
                return self.raw(text), []
            new = [x for x in dict.fromkeys(pieces) if x not in self.cache]
            if new:
                got = self.raw("".join(new))
                at = [i for i, t in enumerate(got) if t in self.role_ids]
                if len(at) == len(new) and at[0] == 0:
                    parts = [got[a:b] for a, b in zip(at, at[1:] + [len(got)])]
                else:   # a BOS, or a role token the split did not see: piece by piece
                    parts = [self.raw(x) for x in new]
                    if any(not q or q[0] not in self.role_ids for q in parts):
                        return self.raw(text), []
                if self.chars > self.MAX_CHARS:
                    self.cache, self.chars = {}, 0
                for x, q in zip(new, parts):
                    self.cache[x] = q
                    self.chars += len(x)
            out, starts = [], []
            for x in pieces:
                starts.append(len(out))
                out += self.cache[x]
            return out, starts


TOKENS = Tokens()


def tokenize(text):
    return TOKENS.ids(text)[0]


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


def place_images(ids, images, starts=None):
    """ids with every placeholder replaced by its picture's rows, and the engine's image fields; starts (the pieces' offsets)
    move with the rows put in before them."""
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
    if starts:
        grow = [(at, n - 1) for (_, _, n), at in zip(enc, ph)]
        starts[:] = [x + sum(g for at, g in grow if at < x) for x in starts]
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
    if e is not None:
        return e
    return EFFORT_DEFAULT if EFFORT_DEFAULT in EFFORT else float(EFFORT_DEFAULT)   # p37


_LOOSE_HEAD = re.compile(r'\s*\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*(?:,\s*)?"?args"?\s*:\s*')


def loose_call(raw):
    """p37: (name, args) of a call json.loads refused when Inkling wrote it as {"name":"X"args":{..}} (the comma and the quote
    before "args" left out; a whole name and one JSON object of arguments, then only the closing brace), else None."""
    m = _LOOSE_HEAD.match(raw)
    if not m:
        return None
    try:
        args, end = json.JSONDecoder().raw_decode(raw[m.end():])
        name = json.loads('"' + m.group(1) + '"')
    except ValueError:
        return None
    if not isinstance(args, dict) or raw[m.end() + end:].strip() not in ("}", ""):
        return None
    return name, args


# /slots while a prompt is read (the engine says nothing until its first written token): the cached prefix expected from the
# last conversations the engine held, and the tokens read so far estimated from the engine's reading cost by depth (ms a token
# = read_scale x (READ_C0 + READ_C1 x depth); two P100 at 185 W: 4k 1,031, 30k 895, 126k 548 tok/s), the scale fitted after
# each request to the engine's own read_s (another card, another power limit)
READ_C0, READ_C1 = 0.941, 1.396e-5
RECENT = collections.deque(maxlen=8)   # (the tokens the engine held after a request as int32 bytes, that request's checkpoint hints)
STATE.update(n_cache=0, n_total=0, read_scale=1.0)


def common_prefix(a, b):
    """how many tokens two token strings (int32 bytes) share from the start: 64 KiB blocks, then halving in the first one
    that differs (bytes compare with memcmp)"""
    n, i, blk = min(len(a), len(b)), 0, 1 << 16
    while i < n and a[i:min(i + blk, n)] == b[i:min(i + blk, n)]:
        i += blk
    if i >= n:
        return n // 4
    lo, hi = i // 4, min(i + blk, n) // 4   # tokens: the first lo the same, the first hi not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[4 * lo:4 * mid] == b[4 * lo:4 * mid]:
            lo = mid
        else:
            hi = mid
    return lo


def predict_cache(ids):
    """the prefix of ids the engine is expected to take from its cache: a conversation it held continued to its end, or else
    back to its last checkpoint hint before the place where the prompt parts from it"""
    a, best = array.array("i", ids).tobytes(), 0
    for seq, hints in RECENT:
        n = common_prefix(a, seq)
        if n < len(seq) // 4:
            n = max((h for h in hints if h <= n), default=0)
        best = max(best, n)
    return min(best, max(len(ids) - 1, 0))


def read_estimate(cache, total, seconds):
    """prompt tokens read after `seconds` from the prefix `cache` (the root of the reading cost), at most total - cache - 1"""
    qa, qb = STATE["read_scale"] * READ_C1 / 2, STATE["read_scale"] * (READ_C0 + READ_C1 * cache)
    done = (-qb + math.sqrt(qb * qb + 4 * qa * 1000 * max(seconds, 0.0))) / (2 * qa)
    return int(max(0, min(done, total - cache - 1)))


def read_fit(cached, total, read_s):
    """the scale of the reading cost from a request's real read (2,048+ new tokens)"""
    if total - cached >= 2048 and read_s > 0:
        model = READ_C0 * (total - cached) + READ_C1 * (total * total - cached * cached) / 2
        STATE["read_scale"] = 0.5 * STATE["read_scale"] + 0.5 * min(max(1000 * read_s / model, 0.25), 8.0)


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
            lc = loose_call(raw)   # p37
            if lc:
                out.append(("tool", {"name": lc[0], "arguments": json.dumps(lc[1], ensure_ascii=False)}))
            else:
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


def paragraph_tail(t):
    """(start, p) when an answer's or a call's tail repeats a paragraph: a block of 200-8,000 characters 3 times or one of
    48-199 characters 8 times, back to back over 1,500 characters, p the block's own length (a long key finds a multiple of
    a shorter block); not a block of table rows (a table can repeat a row: the 16,000 rule watches those). Else None."""
    for min_p, max_p, copies in ((200, 8000, 3), (48, 199, 8)):
        hit = periodic_tail(t, min_p, max_p, 1500, copies)
        if hit is None:
            continue
        start, p = hit
        tail = t[start:]
        p = next((p // k for k in range(p // 8, 1, -1) if p % k == 0 and tail[p // k:] == tail[:-(p // k)]), p)
        rows = (tail[:p].replace("\\n", "\n") * 2).split("\n")[1:-1]   # a call's JSON has its newlines as \n
        if not rows or not all(r.lstrip().startswith("|") for r in rows if r.strip()):
            return start, p
    return None


_MASKED = re.compile(r"\d+|\s+")


def masked_tail(t, min_p, max_p, min_span, min_copies):
    """periodic_tail() on t's last 40,000 characters with every run of digits read as "#" and of white space as " ": a
    reasoning that goes over the same points again under new numbers (at 131k of context Inkling 2 Mini went over the same 13
    points of its code as items 16 to 257, 53k characters and no answer). (start, p) in t's own characters, else None."""
    off = max(0, len(t) - 40000)
    hit = periodic_tail(re.sub(r"\s+", " ", re.sub(r"\d+", "#", t[off:])), min_p, max_p, min_span, min_copies)
    if hit is None:
        return None
    pos, i = [], off   # pos[k]: where the masked text's character k is in t
    for x in _MASKED.finditer(t, off):
        pos.extend(range(i, x.start()))
        pos.append(x.start())
        i = x.end()
    pos.extend(range(i, len(t) + 1))
    return pos[hit[0]], pos[hit[0] + hit[1]] - pos[hit[0]]


class Loop:
    """The engine has no sampler, and a long greedy reasoning can fall into repeating itself word for word (seen on 22 Nov
    after ~20 tool calls: two lines 194 times over and a list 21 times, 84k characters, no answer). Watches
    the reply's current part; step() gives how many of the reply's tokens to keep when its tail repeats one block
    of 32 to 8,000 characters at least 3 times back to back and over 2,000 characters in a reasoning, at least 4 times and
    over TEXT_SPAN characters in an answer or a tool call (a table or a file can repeat a row legitimately; 0: not watched):
    the part up to the end of the block's first copy. A tool call's text (its name, its JSON) is watched as its bytes."""
    MIN_P, MAX_P, MIN_SPAN, MIN_COPIES, EVERY = 32, 8000, 2000, 3, 128
    TEXT_SPAN, TEXT_COPIES = 16000, 4
    PAR = 1   # INK2_LOOP_PAR: also a block of >= 200 characters 3 times or of 48-199 8 times, over 1,500 (answers, calls)
    MASK = 1  # INK2_LOOP_MASK: a reasoning's rule also with its digits masked (masked_tail)

    def __init__(self):
        self.mode, self.text, self.ends, self.checked, self.hit = None, "", [], 0, None

    def step(self, mode, evs, n_out, raw=b""):
        if mode != self.mode:
            self.mode, self.text, self.ends, self.checked = mode, "", [(0, n_out - 1)], 0
        if mode == "r":
            span, copies = self.MIN_SPAN, self.MIN_COPIES
        elif self.TEXT_SPAN > 0:
            span, copies = self.TEXT_SPAN, self.TEXT_COPIES
        else:
            return None
        self.text += "".join(v for k, v in evs if k == mode) if mode in ("r", "c") else raw.decode("latin-1")
        self.ends.append((len(self.text), n_out))
        if len(self.text) - self.checked < self.EVERY:
            return None
        self.checked = len(self.text)
        self.hit = None
        if mode != "r" and self.PAR:   # a repeated paragraph is a loop long before 16,000 characters
            self.hit = paragraph_tail(self.text)
        self.hit = self.hit or periodic_tail(self.text, self.MIN_P, self.MAX_P, span, copies)
        if self.hit is None and mode == "r" and self.MASK:   # a loop whose copies differ only in numbers
            self.hit = masked_tail(self.text, self.MIN_P, self.MAX_P, span, copies)
        if self.hit is None:
            return None
        upto = self.hit[0] + self.hit[1]
        if mode == "c" and self.text[upto - 1:upto] != "\n":   # an answer keeps whole lines (the first copy's last)
            nl = self.text.find("\n", upto, upto + self.hit[1])
            upto = nl + 1 if nl >= 0 else upto
        return next(k for c, k in reversed(self.ends) if c <= upto)


# what closes a looping reasoning before the answer is forced (the reasoning is in English)
NUDGE = "\n\nI am repeating the same lines. I have enough to answer, so I stop thinking here and write the answer now."
NUDGE_IDS = []
# a looping tool call: the system message the model writes on after (the call cut where it began), and the line a reply that
# ends in a loop with nothing to show gets (the client sees it, so the agent does not stop without a word)
LOOP_CALL = int(os.environ.get("INK2_LOOP_CALL", "1"))
CALL_NOTE = os.environ.get("INK2_LOOP_CALL_NOTE") or (
    "Your last tool call ({what}) started repeating the same text over and over, so it was cut off and not run. Do not write "
    "that call again. Write a shorter call instead (for example a file in smaller parts, or a command that builds the repeated "
    "part), or take a different approach.")
CALL_STOP = ("[ink2_server] I stopped here: my tool call ({what}) kept repeating the same text, so it was not run. Tell me how "
             "to go on.")
LOOP_END = "[ink2_server] I stopped here: my reply kept repeating itself. Tell me how to go on."
# p75: the reasoning's budget by effort (tokens of a reply's reasoning part): <= 0.1 / 0.2 / 0.7 / 0.9 / above; INK2_THINK_BUDGET=0 off
THINK_BUDGET = [int(x) for x in os.environ.get("INK2_THINK_BUDGET", "1024,4096,8192,16384,32768").split(",") if x.strip()]
THINK_BUDGET_TOOLS = int(os.environ.get("INK2_THINK_BUDGET_TOOLS", "8192"))   # p75: the most with tools (0: no extra limit)
if THINK_BUDGET == [0]:
    THINK_BUDGET = []
BUDGET_WAIT = 512   # p75: tokens past the budget a reasoning may take to end its sentence
BUDGET_NOTE = "\n\nI have thought about this long enough and I have what I need, so I stop thinking here and write the answer now."
BUDGET_IDS = []
_SENT_END = re.compile(r"(?:[.!?:;](?:[\"')\]]*)\s|\n)$")


def think_budget(effort, tools=False):
    """p75: the reasoning budget in tokens for a request's effort (a name of EFFORT or a number), at most THINK_BUDGET_TOOLS with
    tools; 0: no budget"""
    if not THINK_BUDGET:
        return 0
    num = EFFORT[effort] if isinstance(effort, str) and effort in EFFORT else float(effort if effort is not None else 0.9)
    k = 0 if num <= 0.1 else 1 if num <= 0.2 else 2 if num <= 0.7 else 3 if num <= 0.9 else 4
    b = THINK_BUDGET[min(k, len(THINK_BUDGET) - 1)]
    return min(b, THINK_BUDGET_TOOLS) if tools and THINK_BUDGET_TOOLS > 0 and b > 0 else b
Loop.TEXT_SPAN = int(os.environ.get("INK2_LOOP_TEXT", Loop.TEXT_SPAN))
Loop.PAR = int(os.environ.get("INK2_LOOP_PAR", Loop.PAR))
Loop.MASK = int(os.environ.get("INK2_LOOP_MASK", Loop.MASK))
PART = {"r": "the reasoning", "c": "the answer", "tool": "a tool call", "head": "a tool call"}
# a loop across turns (turn_loop()): the system message before the model's turn (LOOP_MODE s) or the note its reasoning
# starts with (r), and the text of a reply whose only call was the looping one again (dropped)
LOOP_TURNS = int(os.environ.get("INK2_LOOP_TURNS", "3"))
LOOP_CYCLE = int(os.environ.get("INK2_LOOP_CYCLE", "2"))   # a cycle of 2-3 steps repeated so many times
LOOP_NORM = int(os.environ.get("INK2_LOOP_NORM", "4"))     # the same call so many times, results the same but for digits
LOOP_SYSTEM2 = ("You have repeated the same tool calls ({what}) {n} times without getting anything new: the results were the "
                "same each time{why}. You are in a loop. Do not make these calls again. Use the results you already have, change "
                "the commands to get what is missing, or tell the user what is blocking you.")
_DIGITS = re.compile(r"\d+")
LOOP_DROP = int(os.environ.get("INK2_LOOP_DROP", "1"))
LOOP_MODE = os.environ.get("INK2_LOOP_MODE", "s")   # where the note goes: s a system message, r the reasoning, sr both
LOOP_SYSTEM = os.environ.get("INK2_LOOP_SYSTEM") or (
    "You have called {what} with exactly the same arguments {n} times in a row and got the same result each time: you are in a "
    "loop. Do not make that call again. Use the result you already have, change the command to get what is missing, or tell the "
    "user what is blocking you.")
LOOP_NOTE = os.environ.get("INK2_LOOP_NOTE") or (
    "I have already run {what} with exactly these arguments {n} times in a row, and the result was the same every time. "
    "Running it again will not tell me anything new. I will use what the result already shows, change the call to get what "
    "is missing, or tell the user what is blocking me.\n\n")
LOOP_STOP = ("[ink2_server] I stopped here: I was about to run {what} with the same arguments again ({n} times already, the same "
             "result each time). Tell me how to go on.")


def _canon_call(name, args):
    return str(name), _tojson(args, sort_keys=True, separators=(",", ":"))


def turn_loop(messages):
    """(n, calls) when the conversation ends in n >= LOOP_TURNS steps back to back that are the same: an assistant
    message's tool calls (name and arguments as canonical JSON, calls in order) and the tool messages after it (their contents,
    in order); calls = the step's (name, arguments) pairs. Else None (and always when the last message is not a tool result)."""
    if LOOP_TURNS <= 0 or not messages or messages[-1].get("role") != "tool":
        return None
    steps, i = [], len(messages)
    while i > 0:
        j = i
        while j > 0 and messages[j - 1].get("role") == "tool":
            j -= 1
        a = messages[j - 1] if 0 < j < i else {}
        if a.get("role") != "assistant" or not a.get("tool_calls"):
            break
        calls = tuple(_canon_call((tc.get("function") or {}).get("name"), _args(tc.get("function") or {})) for tc in a["tool_calls"])
        res = tuple(m.get("content") if isinstance(m.get("content"), str) else _tojson(m.get("content"), sort_keys=True)
                    for m in messages[j:i])
        if steps and (calls, res) != steps[0]:
            break
        steps.append((calls, res))
        i = j - 1
    if len(steps) >= LOOP_TURNS:
        return len(steps), steps[0][0]
    return turn_loop2(messages)


def turn_loop2(messages):
    """(n, calls, why) for a loop turn_loop() does not see: the last LOOP_CYCLE x L steps a cycle of L = 2 or 3 steps with the same
    calls and results (why ""), or the last >= LOOP_NORM steps the same call whose results differ only in digits (why " but for
    numbers such as times or ids"); calls = every call of the loop. Else None."""
    st, i = [], len(messages)
    while i > 0 and len(st) < 12:
        j = i
        while j > 0 and messages[j - 1].get("role") == "tool":
            j -= 1
        a = messages[j - 1] if 0 < j < i else {}
        if a.get("role") != "assistant" or not a.get("tool_calls"):
            break
        calls = tuple(_canon_call((tc.get("function") or {}).get("name"), _args(tc.get("function") or {})) for tc in a["tool_calls"])
        res = tuple(m.get("content") if isinstance(m.get("content"), str) else _tojson(m.get("content"), sort_keys=True)
                    for m in messages[j:i])
        st.append((calls, res))
        i = j - 1
    for L in (2, 3) if LOOP_CYCLE > 1 else ():
        need = L * LOOP_CYCLE
        if len(st) >= need and len(set(st[:L])) == L and all(st[k] == st[k - L] for k in range(L, need)):
            return need, tuple(dict.fromkeys(c for calls, _ in st[:L] for c in calls)), ""
    if LOOP_NORM > 1 and len(st) >= LOOP_NORM:
        key = (st[0][0], tuple(_DIGITS.sub("#", r) for r in st[0][1]))
        if all((x[0], tuple(_DIGITS.sub("#", r) for r in x[1])) == key for x in st[:LOOP_NORM]):
            return LOOP_NORM, st[0][0], " but for numbers such as times or ids"
    return None


def reply(ids, mx, req, on_part=None, mode="c", gone=None, budget=0):
    """The engine's reply to the prompt ids, at most mx tokens: (tokens, totals, stopped). Each token's parts go to
    on_part(kind, val) as they come ("r" reasoning, "c" content, "tool" a call). A loop in the reasoning (Loop) stops
    the engine; the reasoning is kept up to the end of the block's first copy, closed by NUDGE, and the model writes its
    answer (<|end_message|><|message_model|><|content_text|>): the engine reads the kept part again from the prompt's
    snapshot (from the start when over ~3,000 tokens were written) and goes on. A second loop ends the reply (stopped), and
    so does a loop in the answer (kept up to its first copy) or in a tool call (the call is dropped).
    mode: the part the prompt ends in ("r" when it ends inside a reasoning). gone() is asked about once a second while
    the engine writes; True (the client closed its connection) stops the engine with ConnectionResetError, also while
    the reply is in a part that streams nothing (a tool call). p75: budget > 0, a reasoning part over budget tokens is closed at
    its next sentence end by BUDGET_NOTE (once a reply) and the model writes its answer."""
    parts, guard, out = Parts(), Loop(), []
    r_n, bud = 0, False   # p75: tokens of the current reasoning part; a budget cut asked
    parts.mode = mode
    tot = {"cached": None, "read_s": 0.0, "write_s": 0.0, "drafted": 0, "accepted": 0, "cuts": 0}
    t_gone = time.time()
    hidden = []   # (start, end) in out of the system messages the engine reads and the client never gets

    def shown(stopped, why=None):   # what the client gets; tot["held"]: all the engine holds
        tot["held"], tot["loop_end"] = list(out), why if stopped else None
        if not hidden:
            return out, tot, stopped
        cut = set(i for a, b in hidden for i in range(a, b))
        return [t for i, t in enumerate(out) if i not in cut], tot, stopped
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
                        raw = b"" if parts.mode in ("r", "c") or t in SPECIAL or t >= len(TOKS) else token_bytes(t)
                        keep = guard.step(parts.mode, evs, len(out), raw)
                        if keep is not None:
                            break
                        r_n = r_n + 1 if parts.mode == "r" else 0   # p75
                        if (budget and parts.mode == "r" and r_n >= budget and not tot.get("budget")
                                and (r_n >= budget + BUDGET_WAIT or _SENT_END.search(guard.text[-8:]))):
                            keep, bud = len(out), True
                            break
                    STATE["n_decoded"], STATE["part"] = len(out), parts.mode
                    if keep is not None:
                        break
                    if gone and time.time() - t_gone >= 1.0:
                        t_gone = time.time()
                        if gone():
                            raise ConnectionResetError("the client left")
                elif "tokens" in msg:
                    final = msg
        finally:
            lines.close()   # a cut or a client that left: the engine stops at its next write
        if final is not None:
            if tot["cached"] is None:
                tot["cached"] = final.get("cached", 0)
                tot["back"], tot["ckpts"] = final.get("back"), final.get("ckpts")
            for k in ("read_s", "write_s", "drafted", "accepted"):
                tot[k] += final.get(k, 0)
        elif keep is None:
            raise RuntimeError("the engine closed without an answer")
        else:   # the cut run sends no totals: its times from here
            tot["cached"] = tot["cached"] if tot["cached"] is not None else 0
            tot["read_s"] += (t_first or t_req) - t_req
            tot["write_s"] += time.time() - (t_first or t_req)
        if keep is None:
            return shown(False)
        if bud:   # p75: the reasoning over its budget, closed at a sentence end; not a loop cut
            bud = False
            tot["budget"] = r_n
            if not BUDGET_IDS:
                BUDGET_IDS.extend(i for i in tokenize(BUDGET_NOTE) if i not in SPECIAL)
            forced = BUDGET_IDS + [T_END, T_MODEL, T_TEXT]
            print(f"ink2_server: the reasoning reached its budget ({r_n} tokens, limit {budget}): closed, the model writes its answer",
                  file=sys.stderr, flush=True)
            if len(out) + len(forced) >= mx:
                return shown(False)
            evs = parts.feed(forced)
            if on_part:
                for kind, val in evs:
                    on_part(kind, val)
            out += forced
            guard, r_n = Loop(), 0
            continue
        tot["cuts"] += 1
        start, p = guard.hit
        print(f"ink2_server: a loop in {PART[guard.mode]} (copies x characters, tokens cut / written): {(guard.checked - start) / p:.1f} x {p}, "
              f"{len(out) - keep} / {len(out)}", file=sys.stderr, flush=True)
        if guard.mode in ("tool", "head"):   # cut where the call began; once a reply, the model writes on after a note
            at = guard.ends[0][1]
            k = next((i for i in range(min(at, len(out) - 1), -1, -1) if out[i] == T_MODEL), at)
            j = next((i for i in range(k + 1, len(out)) if out[i] == T_TOOL), len(out))
            what = "".join(TOKS[t] if t not in SPECIAL and t < len(TOKS) else "" for t in out[k + 1:j]).strip()[:64] or "a tool"
            print(f"ink2_server: a loop in a tool call {what}: it began {guard.text[:240]!r}", file=sys.stderr, flush=True)
            out = out[:k]
            if not LOOP_CALL:   # as before: the call dropped
                return shown(True)
            note = tokenize("<|message_system|><|content_text|>" + CALL_NOTE.format(what=what) + "<|end_message|>")
            if tot.get("call_retry") or len(out) + len(note) + 64 >= mx:
                return shown(True, CALL_STOP.format(what=what))
            tot["call_retry"] = what
            print(f"ink2_server: the call {what} cut, the model writes on after a system message", file=sys.stderr, flush=True)
            hidden.append((len(out), len(out) + len(note)))
            out += note
            parts, guard = Parts(), Loop()
            continue
        out = out[:keep]
        if guard.mode != "r":
            return shown(True)
        if not NUDGE_IDS:
            NUDGE_IDS.extend(i for i in tokenize(NUDGE) if i not in SPECIAL)
        forced = NUDGE_IDS + [T_END, T_MODEL, T_TEXT]
        if tot["cuts"] - bool(tot.get("call_retry")) > 1 or len(out) + len(forced) >= mx:   # a cut call does not count
            return shown(True, LOOP_END)
        evs = parts.feed(forced)
        if on_part:
            for kind, val in evs:
                on_part(kind, val)
        out += forced
        guard = Loop()


def engine_lines(req):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(7200)
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

    def _gone(self):
        # the request is read, so a connection that reads as ready with nothing in it was closed by the client (as llama-server
        # checks it); a reply that streams nothing for a while (a tool call, a request without "stream") would not see it otherwise
        try:
            ready = select.select([self.connection], [], [], 0)[0]
            return bool(ready) and not self.connection.recv(1, socket.MSG_PEEK)
        except (OSError, ValueError):
            return True

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
            reading = st["busy"] and not st["n_decoded"]   # as llama-server says it while a prompt is read
            done = (read_estimate(st["n_cache"], st["n_total"], time.time() - st["t_start"]) if reading
                    else st["n_total"] - st["n_cache"] if st["busy"] else 0)
            return self._send(200, [{"id": 0, "id_task": st["requests"], "n_ctx": A.ctx, "is_processing": st["busy"],   # p42
                                     "n_prompt_tokens": st["n_cache"] + done if reading else st["n_prompt"],
                                     "n_prompt_tokens_processed": done, "n_prompt_tokens_cache": st["n_cache"],
                                     "n_prompt_tokens_total": st["n_total"],
                                     "next_token": [{"n_decoded": st["n_decoded"], "has_next_token": st["busy"]}]}])
        if path == "/metrics":   # Prometheus' text format, llama-server's names
            with TOTALS_MUTEX:
                text = "".join(f"# HELP llamacpp:{m} {h}\n# TYPE llamacpp:{m} {t}\nllamacpp:{m} {v()}\n" for m, t, h, v in METRICS)
            return self._send(200, text, "text/plain; version=0.0.4")
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
            ids, starts = TOKENS.ids(render(body.get("messages") or [], effort_of(body), tools, images))
            t_tok = time.time() - t0
            ids, img_req = place_images(ids, images, starts)
        except Exception as ex:
            return self._send(500, {"error": str(ex)})
        loop, n_ids = turn_loop(body.get("messages") or []) if tools else None, len(ids)
        if loop:
            what = " and ".join(dict.fromkeys(name for name, _ in loop[1]))
            note = LOOP_NOTE.format(what=what, n=loop[0]) if "r" in LOOP_MODE else ""
            try:
                if "s" in LOOP_MODE:
                    msg = LOOP_SYSTEM.format(what=what, n=loop[0]) if len(loop) == 2 else LOOP_SYSTEM2.format(what=what, n=loop[0], why=loop[2])
                    ids = ids + tokenize("<|message_system|><|content_text|>" + msg + "<|end_message|>")
                if note:
                    ids = ids + [T_MODEL, T_THINK] + [i for i in tokenize(note) if i not in SPECIAL]
            except Exception as ex:
                print(f"ink2_server: a loop across turns: the note could not be tokenized ({ex})", file=sys.stderr, flush=True)
                loop = None
        if len(ids) >= A.ctx - 1:
            return self._send(400, {"error": {"message": f"prompt of {len(ids)} tokens exceeds the context of {A.ctx}", "type": "exceed_context_size_error"}})
        mx = int(body.get("max_tokens") or body.get("max_completion_tokens") or 0) or (A.ctx - len(ids) - 1)
        mx = max(1, min(mx, A.ctx - len(ids) - 1))
        stream = bool(body.get("stream"))
        req = {"stop": [EOS], "mtp": int(body.get("mtp", MTP)), "lookup": int(body.get("lookup", LOOKUP)), **img_req,
               "ckpt": [x for x in starts if x < n_ids - 8][-4096:]}   # p42: not the answer's own head (n - 1 is checkpointed)
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(t0)
        if not LOCK.acquire(blocking=False):
            print(f"ink2_server: a request of {len(ids)} prompt tokens waits for the one before it", file=sys.stderr, flush=True)
            if not LOCK.acquire(timeout=3600):
                return self._send(503, {"error": "busy"})
        try:
            STATE.update(busy=True, n_prompt=len(ids), n_decoded=0, part="", n_total=len(ids), n_cache=predict_cache(ids))
            STATE.update(t_start=time.time(), t_note=time.time())
            STATE["requests"] += 1
            calls, on_part, said, dropped = [], None, [], [0]
            if loop:
                where = {"s": "as a system message", "r": "in the reasoning"}
                print(f"ink2_server: a loop across turns: {loop[0]} x {what} (the same call, the same result): the note "
                      f"{' and '.join(where[m] for m in LOOP_MODE if m in where)}", file=sys.stderr, flush=True)
                with TOTALS_MUTEX:
                    TOTALS["turn_loops"] += 1

            def looping(val):   # the loop's call written again after the note
                if not (loop and LOOP_DROP):
                    return False
                try:
                    if _canon_call(val["name"], json.loads(val["arguments"])) not in loop[1]:
                        return False
                except ValueError:
                    return False
                dropped[0] += 1
                print(f"ink2_server: a loop across turns: the same {val['name']} call after the note, dropped", file=sys.stderr, flush=True)
                return True
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
                    if kind == "tool" and looping(val):
                        return
                    if kind == "c" and val.strip():
                        said.append(1)
                    if kind == "tool":
                        call = {"index": len(calls), "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                                "function": {"name": val["name"], "arguments": val["arguments"]}}
                        calls.append(call)
                        delta = {"tool_calls": [call]}
                    else:
                        delta = {"reasoning_content" if kind == "r" else "content": val}
                    sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": A.alias,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
                if loop and note:
                    on_part("r", note)
            try:
                out, tot, stopped = reply(ids, mx, req, on_part, mode="r" if loop and note else "c", gone=self._gone,
                                          budget=think_budget(effort_of(body), bool(tools)))   # p75
            except (BrokenPipeError, ConnectionResetError):   # closing the engine's socket stopped it at its next write
                print(f"ink2_server: request: the client left after {STATE['n_decoded']} written tokens ({len(ids)} prompt tokens); "
                      f"the engine stopped", file=sys.stderr, flush=True)
                return
            n = len(out)
            STATE["tokens_out"] += n
            if not tot["cuts"] and not tot.get("budget"):   # a cut reads its kept part again, so its read_s is no measure (p75 too)
                read_fit(tot["cached"], len(ids), tot["read_s"])
            STATE["n_cache"] = tot["cached"]   # the engine's own count from here on
            hints = req["ckpt"] + [len(ids) - 1] if "ckpt" in req else []   # the engine checkpoints n - 1 too (none without "ckpt")
            RECENT.append((array.array("i", ids + tot.get("held", out)).tobytes(), hints))
            w = tot["write_s"] or 1e-9
            print(f"ink2_server: request: prompt tokens / cached / back to checkpoint / checkpoints, tokenize / read / written / write {len(ids)} / {tot['cached']} / {tot.get('back')} / {tot.get('ckpts')}, "
                  f"{1000 * t_tok:.0f} ms / {tot['read_s']:.2f} s / {n} / {w:.2f} s", file=sys.stderr, flush=True)
            if not stream:
                think, content, pp = [note] if loop and note else [], [], Parts()
                pp.mode = "r" if loop and note else "c"
                for kind, val in pp.feed(out):
                    if kind == "tool" and looping(val):
                        continue
                    if kind == "tool":
                        calls.append({"index": len(calls), "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                                      "function": {"name": val["name"], "arguments": val["arguments"]}})
                    else:
                        (think if kind == "r" else content).append(val)
            if tot.get("loop_end") and not calls and not dropped[0] and not (said if stream else "".join(content).strip()):
                if stream:   # a reply that ended in a loop with nothing to show says so
                    on_part("c", tot["loop_end"])
                else:
                    content.append(tot["loop_end"])
            if dropped[0] and not calls:   # the looping call was the reply's only call
                stop = ("\n\n" if (said if stream else "".join(content).strip()) else "") + LOOP_STOP.format(what=what, n=loop[0])
                if stream:
                    on_part("c", stop)
                else:
                    content.append(stop)
            finish = "tool_calls" if calls else "stop" if stopped or (out and out[-1] == EOS) else "length"
            timings = {"prompt_n": len(ids) - tot["cached"], "cache_n": tot["cached"], "prompt_ms": 1000 * tot["read_s"],
                       "prompt_per_second": (len(ids) - tot["cached"]) / max(tot["read_s"], 1e-9),
                       "predicted_n": n, "predicted_ms": 1000 * w, "predicted_per_second": max(n - 1, 0) / w,
                       "draft_n": tot["drafted"], "draft_n_accepted": tot["accepted"]}
            if tot["cuts"]:
                timings["loop_cuts"] = tot["cuts"]
            if tot.get("call_retry"):
                timings["loop_call_retry"] = 1
            if loop:
                timings["loop_turns"], timings["loop_dropped"] = loop[0], dropped[0]
            with TOTALS_MUTEX:   # /metrics
                TOTALS["prompt_tokens"] += timings["prompt_n"]
                TOTALS["prompt_seconds"] += timings["prompt_ms"] / 1000
                TOTALS["cached_tokens"] += timings["cache_n"]
                TOTALS["predicted_seconds"] += timings["predicted_ms"] / 1000
                TOTALS["drafted"] += timings["draft_n"]
                TOTALS["accepted"] += timings["draft_n_accepted"]
                TOTALS["loop_drops"] += dropped[0]
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


def ticker():
    """A line every LOG_EVERY seconds (INK2_LOG_EVERY, 0: none) while a request runs: a request prints its line only at its
    end, and a tool call streams nothing until it is whole, so a long read or reply would leave the log silent."""
    while True:
        time.sleep(min(5.0, LOG_EVERY))
        st = STATE
        if st["busy"] and time.time() - st["t_note"] >= LOG_EVERY:
            st["t_note"] = time.time()
            now = f"writing {PART.get(st['part'], 'the reply')}" if st["n_decoded"] else "reading the prompt"
            print(f"ink2_server: busy for {time.time() - st['t_start']:.0f} s: {st['n_prompt']} prompt tokens, {st['n_decoded']} written, "
                  f"now {now}", file=sys.stderr, flush=True)


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
    if LOG_EVERY > 0:
        threading.Thread(target=ticker, daemon=True).start()
    print(f"ink2_server: {A.alias} on {A.host}:{A.port}, ctx {A.ctx}, engine {A.sock}", file=sys.stderr, flush=True)
    srv.serve_forever()
