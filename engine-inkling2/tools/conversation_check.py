#!/usr/bin/env python3
"""conversation_check.py [URL=http://127.0.0.1:8080]: a conversation through the running server (run.sh), the way a chat client
sends it: the whole history each turn, with a note about the environment at the end of the last user message that moves to the
new one, so a turn's prompt parts from the previous one a few tokens before its end. With the engine's checkpoints each request
should read only what is new:
  1a  first turn: everything is read once (a ~16k-token system message)
  1b  its tool round: only the call and the tool's result
  2   second turn: back to the start of the first user message; the two short messages are read
  3a  a long answer (over 3,056 tokens were the old snapshot's limit), 3b the turn after it: only that answer and the new message
  4   /tokenize: a written backslash-n (two characters, as in source code) stays two characters
Prints tokens read / cached and the time to the first byte of every request; OK when a request reads no more than its limit."""
import json
import sys
import time
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
TOOLS = [{"type": "function", "function": {"name": "web_search", "description": "Search the web.", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
SYSTEM = "You are a helpful assistant. Answer briefly.\n\n" + "\n".join(
    f"Rule {i}: for questions about topic {i} check source {i % 17} first, then answer in at most {2 + i % 5} sentences; write "
    f"numbers with a decimal point and dates as 2026-{1 + i % 12:02d}-{1 + i % 28:02d}." for i in range(1, 300))


def env(t):
    return f"\n\n[Environment: Sunday 22 Nov 2026, {t}; device: phone]"


def ask(name, messages, limit=None, max_tokens=400, effort="low"):
    body = {"messages": messages, "tools": TOOLS, "stream": True, "max_tokens": max_tokens, "reasoning_effort": effort}
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, out, last = time.time(), None, {"content": "", "tools": []}, None
    with urllib.request.urlopen(r, timeout=1800) as f:
        for line in f:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            ch = d["choices"][0]
            dl = ch.get("delta") or {}
            if first is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("tool_calls")):
                first = time.time() - t0
            out["content"] += dl.get("content") or ""
            out["tools"] += dl.get("tool_calls") or []
            if ch.get("finish_reason"):
                last = d
    t = last["timings"]
    ok = limit is None or t["prompt_n"] <= limit
    print(f"{name}: {'OK' if ok else 'TOO MUCH READ'}  prompt {last['usage']['prompt_tokens']}, read {t['prompt_n']}, cached {t['cache_n']}, "
          f"reading {t['prompt_ms'] / 1000:.2f} s, first byte after {first:.2f} s, written {t['predicted_n']} "
          f"({t['predicted_per_second']:.1f} tok/s)", flush=True)
    out["n"], out["ok"] = t["predicted_n"], ok
    return out


sys_msg = {"role": "system", "content": SYSTEM}
q1 = "What is the capital of Slovenia? Use the search."
m = [sys_msg, {"role": "user", "content": q1 + env("16:05")}]
a = ask("1a first turn", m)
call = (a["tools"] or [{"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{\"query\": \"capital of Slovenia\"}"}}])[0]
call = {"id": call["id"], "type": "function", "function": call["function"]}
m2 = m + [{"role": "assistant", "content": a["content"] or None, "tool_calls": [call]},
          {"role": "tool", "tool_call_id": call["id"], "content": "Ljubljana is the capital of Slovenia; it had 297,575 inhabitants on 1 Jan 2026. " * 12}]
a2 = ask("1b tool round", m2, limit=900)
hist = [sys_msg, {"role": "user", "content": q1}, {"role": "assistant", "content": a2["content"] or "Ljubljana."}]
q2 = "And how many people live there? Only the number."
a3 = ask("2 second turn", hist + [{"role": "user", "content": q2 + env("16:06")}], limit=600)
hist += [{"role": "user", "content": q2}, {"role": "assistant", "content": a3["content"] or "297,575"}]
q3 = "Write a long essay (at least 2,500 words) on the history of Ljubljana."
a4 = ask("3a long answer", hist + [{"role": "user", "content": q3 + env("16:07")}], limit=600, max_tokens=3600, effort="none")
hist += [{"role": "user", "content": q3}, {"role": "assistant", "content": a4["content"]}]
a5 = ask("3b the turn after it", hist + [{"role": "user", "content": "Thanks. In one sentence: which century mattered most?" + env("16:09")}],
         limit=600 + a4["n"])


def tokens(text):
    r = urllib.request.Request(URL + "/tokenize", json.dumps({"content": text}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as f:
        return json.loads(f.read())["tokens"]


written, real = tokens('print("a\\nb")'), tokens('print("a\nb")')
esc_ok = len(written) > 0 and written != real
print(f"4 a written backslash-n: {'OK' if esc_ok else 'WRONG, it became a newline'}  {len(written)} tokens as written, {len(real)} with a real "
      "newline", flush=True)
bad = sum(not x["ok"] for x in (a2, a3, a4, a5)) + (not esc_ok)
print("all OK" if not bad else f"{bad} checks failed")
sys.exit(1 if bad else 0)
