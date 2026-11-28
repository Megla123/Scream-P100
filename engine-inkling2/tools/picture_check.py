#!/usr/bin/env python3
"""picture_check.py [URL=http://127.0.0.1:8080] [IMAGE]: a picture and two questions through the running server (run.sh with
the projector): its big title, then a short description. IMAGE defaults to this branch's card picture,
docs/inkling2-scream-p100.jpg, whose title is "Inkling 2 Mini on two P100s. An engine of its own." Greedy, thinking effort 0.
The first question also waits for ink2_vis (~20 s on the CPU); the second one finds the picture's rows cached."""
import base64
import json
import mimetypes
import sys
import time
import urllib.request
from pathlib import Path

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
IMG = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "docs/inkling2-scream-p100.jpg"
mime = mimetypes.guess_type(IMG.name)[0] or "image/jpeg"
data = f"data:{mime};base64," + base64.b64encode(IMG.read_bytes()).decode()
for q, mx in (("What does the big title in the picture say? Answer with the title only.", 600),
              ("Describe the picture in two sentences.", 1200)):
    body = {"model": "inkling2-mini", "temperature": 0, "max_tokens": mx, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data}}, {"type": "text", "text": q}]}]}
    t0 = time.time()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    m, t = out["choices"][0]["message"], out.get("timings") or {}
    print(f"Q: {q}\nA: {(m.get('content') or '').strip()}\n   ({len(m.get('reasoning_content') or '')} characters of reasoning, "
          f"prompt {t.get('prompt_n')} tokens in {t.get('prompt_ms', 0) / 1000:.1f} s, writing {t.get('predicted_per_second', 0):.1f} "
          f"tok/s, {time.time() - t0:.1f} s in all)", flush=True)
