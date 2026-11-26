#!/usr/bin/env python3
"""loop_check.py: the server's loop guard (serve/ink2_server.py's periodic_tail and Loop) on texts, without the model. A
reasoning that falls into repeating two lines (as Inkling's did on 22 Nov after ~20 tool calls) must be cut after the
lines' first copy, and so must a reasoning that goes over the same points again under new numbers (26 Nov, at 131k of
context), an answer or a tool call that repeats a paragraph; ordinary long texts (this branch's docs and the server's own
source, read as a reasoning and as an answer) and a table of 200 equal rows must never be. Texts go in as ~4-character
tokens, as the server sees a reply."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "serve/ink2_server.py").read_text(encoding="utf-8")
ns = {"re": re}   # the guard reads a reasoning's numbers masked (re)
exec(src[src.index("def periodic_tail("):src.index("# what closes a looping reasoning")], ns)
Loop = ns["Loop"]


def cut(text, mode="r"):
    """(characters read when the guard cut, characters kept) or None. mode: "r" reasoning, "c" answer, "tool" a call's JSON."""
    g, toks = Loop(), [text[i:i + 4] for i in range(0, len(text), 4)]
    for n, t in enumerate(toks, 1):
        k = g.step(mode, [(mode, t)], n) if mode in ("r", "c") else g.step(mode, [], n, t.encode("latin-1"))
        if k is not None:
            return sum(map(len, toks[:n])), sum(map(len, toks[:k]))
    return None


def loops(name, head, block, copies, mode):
    """1 when the guard does not cut head + block * copies after the block's first copy."""
    t = head + block * copies
    r = cut(t, mode)
    print(f"{name} ({len(t)} characters): " + (f"cut after {r[0]}, {r[1]} kept" if r else "NOT CUT"))
    # one copy kept (its start can slip back a few characters, into the end of head)
    return not r or not (len(head) + len(block) - 16 <= r[1] <= len(head) + len(block))


head = ("The user wants server GPUs better than the P100 in Europe. From the searches: A100 80GB refurbished 7,479 EUR, a DGX "
        "A100 with 8x A100 40GB 50,000 EUR, V100 32GB 425-1,099 USD on eBay, the P100 itself 180-289 EUR.\n\n")
block = ("Let me also check the HydraGPU page via search snippet - it says \"NVIDIA A100 80GB - Refurbished - 7,479 EUR\". "
         "That's a good price.\n\nLet me also check the server-parts.eu DGX A100 offer - it's 50,000 EUR for 8x A100 40GB.\n\n")
bad = loops("looping reasoning", head, block, 40, "r")
bad += loops("looping answer", head, block, 100, "c")
bad += loops("looping tool call", '{"name":"write_file","args":{"path":"game.py","content":"import random\\n\\n',
             "def check(board, x, y):\\n    if board[y][x] == 0:\\n        return False\\n    # check the cell again\\n", 300, "tool")
# the same four points of a review again and again, numbered on (57-60, 61-64, ...): the plain rule never sees a copy
points = ("in `renderBoard`, the `last-from` and `last-to` classes are applied, and `lastMove` is set in `doMove`. Good.",
          "in `san`, for castling, `move.to.c === 6 ? 'O-O' : 'O-O-O'`. Correct.",
          "in `doMove`, `const color = state.turn;` is taken before `applyMove`. Correct.",
          "in `aiMove`, when the AI has no moves, `AI.pickMove` returns null and `doMove` is not called. Handled.")
review = "".join(f"{k + 1}. One issue: {points[k % 4]}\n\n" for k in range(56, 56 + 4 * 40))
first = len("".join(f"{k + 1}. One issue: {points[k % 4]}\n\n" for k in range(56, 60)))
r = cut(head + review, "r")
print(f"a reasoning going over the same points under new numbers ({len(head + review)} characters): "
      + (f"cut after {r[0]}, {r[1]} kept" if r else "NOT CUT"))
bad += not r or not (len(head) + first - 16 <= r[1] <= len(head) + first + 4)
for f in sorted((ROOT.parent / "docs").glob("*.md")) + [ROOT / "serve/ink2_server.py", ROOT / "README.md"]:
    t = f.read_text(encoding="utf-8")
    for mode in ("r", "c"):
        r = cut(t, mode)
        print(f"{f.relative_to(ROOT.parent)} ({len(t)} characters) as {'a reasoning' if mode == 'r' else 'an answer'}: "
              + (f"CUT after {r[0]}" if r else "not cut"))
        bad += bool(r)
t = "| x | y | z |\n|---|---|---|\n" + "| 0 | 0 | 0 |\n" * 200
r = cut(t, "c")
print(f"a table of 200 equal rows as an answer ({len(t)} characters): " + (f"CUT after {r[0]}" if r else "not cut"))
bad += bool(r)
print("OK" if not bad else "FAILED")
sys.exit(1 if bad else 0)
