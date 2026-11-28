#!/usr/bin/env python3
"""multi_bench.py [URL=http://127.0.0.1:8080] [COUNTS=1,2,4]: several conversations at once on a server started with
run.sh INK2_MULTI=tpb|pp (COUNTS up to its INK2_SLOTS). Every prompt is first sent alone; then for each count N > 1, N
different prompts go together. Fresh writing of 256 tokens each, greedy, no reasoning, the server's default drafts. Prints
each count's writing speed as the server's timings report it: a conversation's and the sum over the N (the tokens the two
cards write in a second). Then every answer sent together is compared with the same prompt sent alone: a conversation's
output must not depend on the others. Ends "same alone and together: yes" or lists the differences (exit code 1)."""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
COUNTS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4").split(",")]
PROMPTS = ["Write an essay of about 600 words on the history of the bicycle.",
           "Explain step by step, in about 600 words, how a compiler turns source code into machine code.",
           "Napiši zgodbo v približno 600 besedah o svetilničarju na majhnem otoku.",
           "Write a Python module with an LRU cache class, its unit tests, and a short explanation of the design.",
           "Describe the water cycle in detail for a school textbook, in about 600 words.",
           "Write a C program that reads a CSV file and prints the minimum, maximum and mean of every numeric column."]


def chat(text):
    body = {"model": "inkling2-mini", "messages": [{"role": "user", "content": text}], "max_tokens": 256,
            "temperature": 0, "reasoning_effort": "none"}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or ""), d["timings"]


n_max = max(COUNTS)
if n_max > len(PROMPTS):
    sys.exit(f"multi_bench.py: at most {len(PROMPTS)} at once")
alone = [chat(p) for p in PROMPTS[:n_max]]
speeds = [t["predicted_per_second"] for _, t in alone]
print(f"1 at once: {sum(speeds) / len(speeds):.1f} tok/s (each prompt alone: {', '.join(f'{x:.1f}' for x in speeds)})")
diff = []
for n in COUNTS:
    if n < 2:
        continue
    t0 = time.time()
    with ThreadPoolExecutor(n) as ex:
        res = list(ex.map(chat, PROMPTS[:n]))
    wall = time.time() - t0
    each = [t["predicted_per_second"] for _, t in res]
    print(f"{n} at once: {sum(each):.1f} tok/s in all, {sum(each) / n:.1f} a conversation "
          f"({', '.join(f'{x:.1f}' for x in each)}; {sum(t['predicted_n'] for _, t in res)} tokens in {wall:.1f} s)")
    diff += [(n, i) for i, (text, _) in enumerate(res) if text != alone[i][0]]
if diff:
    print("NOT the same alone and together:", ", ".join(f"prompt {i + 1} with {n} at once" for n, i in diff))
    sys.exit(1)
print("same alone and together: yes")
