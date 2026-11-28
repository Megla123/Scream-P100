#!/usr/bin/env python3
"""bench.py [URL=http://127.0.0.1:8080] [LONG_TEXT_FILE]: speed of the running server (run.sh) as its `timings` report them.
fresh: four prompts of prose and code, 300 tokens each, the tokens / the writing time over all; edit: a file of this repo
repeated with one rename (where prompt lookup and the drafter pay), runs 2-3 counted; read: one cold prompt of LONG_TEXT_FILE
(default: docs/GUIDE.md of this repo, ~40k tokens), its reading speed and the writing after it. Greedy, thinking effort 0.
Our numbers in docs/INKLING2.md come from the shared test set of our Flash-Next builds (same idea, other prompts)."""
import json
import sys
import time
import urllib.request
from pathlib import Path

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
ROOT = Path(__file__).resolve().parents[2]
LONG = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs/GUIDE.md"


def chat(text, max_tokens, effort="none"):
    body = {"model": "inkling2-mini", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": 0, "reasoning_effort": effort}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())["timings"]


fresh = ["Explain in about 200 words how a heat pump works and when it pays off.",
         "Write a Python function that merges overlapping intervals, with an example.",
         "Razloži v približno 200 besedah, kako nastaneta plima in oseka.",
         "Write a C function that parses a key=value config file into a linked list, with careful error handling."]
n = ms = 0
for p in fresh:
    t = chat(p, 300, "low")
    n, ms = n + t["predicted_n"], ms + t["predicted_ms"]
print(f"fresh: {1000 * n / ms:.1f} tok/s ({n} tokens)")
src = (ROOT / "engine-inkling2/serve/ink2_server.py").read_text(encoding="utf-8")
ask = "Rename the variable `parts` to `pieces` everywhere in this file and change nothing else. Reply with the whole file only.\n\n```python\n" + src + "\n```"
runs = [chat(ask, 1600)["predicted_per_second"] for _ in range(3)]
print(f"edit: {sum(runs[1:]) / 2:.1f} tok/s (runs {', '.join(f'{x:.1f}' for x in runs)})")
text = LONG.read_text(encoding="utf-8", errors="replace")
t0 = time.time()
t = chat(text + "\n\nSummarize the text above in five bullet points.", 200)
print(f"read: {t['prompt_n']} tokens at {t['prompt_per_second']:.1f} tok/s, writing after it {t['predicted_per_second']:.1f} tok/s "
      f"({time.time() - t0:.0f} s)")
