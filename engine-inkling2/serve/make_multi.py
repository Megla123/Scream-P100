#!/usr/bin/env python3
"""make_multi.py SRC DST NAME: serve/ink2_server.py (one generation at a time, a lock) -> DST for the builds that run several
conversations at once (build/ink2_tpb, build/ink2_pp): up to --slots generations at once (a semaphore; each opens its own
connection to the engine, which runs them side by side), /props and /slots say so and name the build NAME. Nothing else
changes (the template, tools, the loop guard, streaming). run.sh makes DST at every start with INK2_MULTI, so it follows the
server."""
import sys
from pathlib import Path

src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
s = Path(src).read_text(encoding="utf-8")


def rep(a, b):
    global s
    if s.count(a) != 1:
        sys.exit(f"make_multi.py: {src} has changed ({s.count(a)} x {a[:60]!r}), update make_multi.py")
    s = s.replace(a, b)


rep("A, _ = ap.parse_known_args()",
    'ap.add_argument("--slots", type=int, default=4, help="generations at once (the engine\'s conversation slots)")\n'
    "A, _ = ap.parse_known_args()")
rep("LOCK = threading.Lock()", "LOCK = threading.BoundedSemaphore(A.slots)   # make_multi.py: up to --slots conversations at once")
rep('"total_slots": 1,', '"total_slots": A.slots,')
rep('"build_info": "ink2_v1"', f'"build_info": "{name}"')
Path(dst).write_text(s, encoding="utf-8", newline="\n")
print(f"make_multi.py: {dst} ({name}, up to --slots generations at once)")
