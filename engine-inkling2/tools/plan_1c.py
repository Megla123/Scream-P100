#!/usr/bin/env python3
"""plan_1c.py OUT FILE... (one-card engine ink2_1c): the order in which build/ink2_1c keeps experts in VRAM (INK2_1C_PLAN, the
repository's data/plan_1c.txt): lines "layer expert weight", by weight a byte, the most used first. FILE: a writing-path routing
trace (INK2_RTRACE=FILE on ink2_1c appends "slot T e e e ..." a layer of every forward; name it sled-*.txt) or a reading-path count
file (48 lines of 128 counts, one a slot). Each file counts the same (its picks divided by its total); the traces together weigh
RSTAT_SLED (default 1.0) times the count files together. The weight is the expert's share of all picks (the engine splits its cache
of RAM experts by the layers' weight left in RAM). data/plan_1c.txt came from 5 x 30k tokens of reading (Slovenian, English, code,
a chat with tools, answers) and 5 x 400 written tokens."""
import os
import sys
from pathlib import Path

NPHYS, LS, NCORE, NE = 24, 6, 12, 128
OUT = Path(sys.argv[1])
files = [Path(f) for f in sys.argv[2:]]
size = [8.46 if p >= NPHYS - 4 else 4.48 for p in range(NPHYS)]


def phys(s):
    return s if s < NPHYS else LS + (s - NPHYS) % NCORE


def counts(f):
    c = [[0] * NE for _ in range(NPHYS)]
    if f.name.startswith("sled-"):
        for line in f.read_text().splitlines():
            x = list(map(int, line.split()))
            if x[1] == 1:
                for e in x[2:10]:
                    c[phys(x[0])][e] += 1
    else:
        for s, r in enumerate(l for l in f.read_text().splitlines() if l.strip()):
            for e, v in enumerate(map(int, r.split())):
                c[phys(s)][e] += v
    return c


sled = [f for f in files if f.name.startswith("sled-")]
pf = [f for f in files if not f.name.startswith("sled-")]
ws = float(os.environ.get("RSTAT_SLED", "1.0"))
w = [[0.0] * NE for _ in range(NPHYS)]
for group, gw in ((pf, 1.0), (sled, ws)):
    for f in group:
        c = counts(f)
        tot = sum(map(sum, c)) or 1
        for p in range(NPHYS):
            for e in range(NE):
                w[p][e] += gw / len(group) * c[p][e] / tot
s = sum(map(sum, w))
order = sorted(((w[p][e] / s / size[p], p, e) for p in range(NPHYS) for e in range(NE)), reverse=True)
OUT.write_text("".join(f"{p} {e} {w[p][e] / s:.8f}\n" for _, p, e in order))
print(OUT, len(order), "vrstic;", len(pf), "datotek branja,", len(sled), "sledi pisanja (utez", ws, ")")
