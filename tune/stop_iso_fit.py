#!/usr/bin/env python3
"""stop_iso_fit.py: fit the learned MTP stop rule's tables (engine/src/stop_iso.hpp, FNX_DRAFT_STOP_ISO) from an FNX_DRAFT_TRACE trace.

The rule (Keoian's LLAMA_SPEC_STOP_ISO, ported): draft d is verified while
    S(d) = prod_{i<d} iso_q(q_i) * iso_p1(p1_d) >= theta,
iso_q / iso_p1 isotonic maps of the target's acceptance of a draft against the drawn token's q and against the top
probability p1 of the step's distribution.

TRACE (one line per verify round of an MTP chain, tab-separated; '#' lines are skipped):
    req  temp  n  acc  q  p1  a
  req   the request's number (dev / test split by request, never by round)
  temp  the request's sampling temperature (picks the set)
  n     drafts verified in the round; acc  how many stood (a prefix)
  q, p1, a   n comma-separated values per draft: the drawn token's probability under the drafter, the top probability
        of the step's distribution, and the target's acceptance of that draft GIVEN the drafts before it:
        min(1, p/q) under speculative sampling, p(d) under exact-match sampling, 1/0 at temperature 0. The verify computes
        every row of a window, so a is known past the first rejection too; a '-' (unknown) falls back to the outcome
        (1 before acc, 0 at acc, the rows after it unused).
For the fit and the simulation the chain must NOT have been cut by a stop rule while tracing (draft the full K): a
rule-cut trace only shows the drafts the old rule liked. FNX_DRAFT_TRACE's other lines are used too: "#R req temp top_p
top_k presence" gives each request's sampling (for sets with top_p / top_k / presence ranges), "#L req temp rows acc
verify_ms draft_ms" a lookup round's time (for --cost auto); its MTP lines carry "M rows verify_ms draft_ms" after a.

COST: --cost "1:22.2,2:29.7,..." = a round's time (ms) by window rows (1 + drafts), this machine's; the simulation
prices a round that keeps L drafts at cost[L + 1]. --cost auto: from the trace's own verify ms: the mean
of 1-row rounds and of full chains, the widths between on the line through them, plus the mean ms of a draft call a draft
(FNX_DRAFT_TRACE's columns 8-10).

SETS: --set name:lo:hi[,top_p=lo:hi][,top_k=lo:hi][,presence=lo:hi], in priority order: a round goes to the first set
whose ranges hold its request's sampling (top_k 0 = no cut).

    stop_iso_fit.py TRACE... --cost 1:22.2,2:29.7,3:34.5,4:41.1,5:51.1 --set think:0.95:1.05 --set chat:0.001:1e9 \
        --out model.stop_iso [--theta 0.05:0.40:0.025] [--report report.md]
Sets are matched in the order given (the first window holding a round's temperature takes it), as the engine does.
"""
import argparse
import math
import random
import sys
import zlib

KNOTS = [0.000, 0.020, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500,
         0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.980, 0.995, 1.000]   # the owner's


def pava(xs, ys):
    """Isotonic (non-decreasing) least squares of ys on xs: [(x, fitted)] at the sorted distinct xs."""
    pts = sorted(zip(xs, ys))
    blocks = []   # [sum_y, n, x_sum]
    i = 0
    while i < len(pts):   # one block per distinct x first
        x = pts[i][0]
        s = n = 0.0
        while i < len(pts) and pts[i][0] == x:
            s += pts[i][1]
            n += 1
            i += 1
        blocks.append([s, n, [x]])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, n2, x2 = blocks.pop()
            blocks[-1][0] += s2
            blocks[-1][1] += n2
            blocks[-1][2] += x2
    out = []
    for s, n, xl in blocks:
        for x in xl:
            out.append((x, s / n))
    return out


def interp(fit, x):
    if not fit:
        return 0.0
    if x <= fit[0][0]:
        return fit[0][1]
    for (x0, y0), (x1, y1) in zip(fit, fit[1:]):
        if x <= x1:
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0) if x1 > x0 else y1
    return fit[-1][1]


def table(fit):
    v = [min(1.0, max(0.0, interp(fit, k))) for k in KNOTS]
    for j in range(1, len(v)):   # interpolation of a non-decreasing fit is non-decreasing; rounding is not
        v[j] = max(v[j], v[j - 1])
    return v


def lookup(tab, x):   # the engine's Map (stop_iso.hpp), in double
    if not x > KNOTS[0]:
        return tab[0]
    for j in range(1, len(KNOTS)):
        if x <= KNOTS[j]:
            t = (x - KNOTS[j - 1]) / (KNOTS[j] - KNOTS[j - 1])
            return tab[j - 1] + t * (tab[j] - tab[j - 1])
    return tab[-1]


def read(paths, timing=None):
    """the MTP rounds; `timing` (a list) also gets (rows, verify ms, draft ms, drafts) of every traced round"""
    rounds = []
    for path in paths:
        samp = {}
        with open(path, errors="replace") as f:
            for ln, line in enumerate(f, 1):
                if line.startswith("#R\t"):
                    c = line.rstrip("\n").split("\t")
                    samp[c[1]] = (float(c[2]), float(c[3]), float(c[4]), float(c[5]))
                    continue
                if line.startswith("#L\t") and timing is not None:
                    c = line.rstrip("\n").split("\t")
                    timing.append((int(c[3]), float(c[5]), float(c[6]), 0))
                    continue
                if not line.strip() or line.startswith("#"):
                    continue
                c = line.rstrip("\n").split("\t")
                if len(c) < 7:
                    sys.exit(f"{path}:{ln}: expected 7 tab-separated fields")
                n, acc = int(c[2]), int(c[3])
                if n < 1:
                    continue
                lists = [c[k].split(",") for k in (4, 5, 6)]
                if any(len(l) != n for l in lists):
                    sys.exit(f"{path}:{ln}: q, p1 and a need n = {n} values each")
                q = [float(v) for v in lists[0]]
                p1 = [float(v) for v in lists[1]]
                a = [(1.0 if j < acc else 0.0) if v == "-" else float(v) for j, v in enumerate(lists[2])]
                known = [v != "-" for v in lists[2]]
                t, tp, tk, pr = samp.get(c[0], (float(c[1]), 1.0, 0.0, 0.0))
                rounds.append(dict(req=f"{path}:{c[0]}", temp=t, top_p=tp, top_k=tk, presence=pr, n=n, acc=acc, q=q, p1=p1, a=a,
                                   known=known))
                if timing is not None and len(c) >= 11 and c[7] == "M":
                    timing.append((int(c[8]), float(c[9]), float(c[10]), n))
    return rounds


def auto_cost(timing, top):
    """rows -> ms from the trace's own rounds: the mean verify ms of 1-row rounds (each request's first) and of full
    chains (top rows), the rows between on the line through those two (a line through every round, lookup windows of up
    to 16 rows included, bends with their cost and prices short windows far too low), plus a draft call per draft"""
    by = {}
    for r, v, _, _ in timing:
        if r <= top:
            by.setdefault(r, []).append(v)
    drafts = sum(d for _, _, _, d in timing)
    call = sum(dm for _, _, dm, d in timing if d) / drafts if drafts else 0.0
    if len(by.get(1, [])) >= 10 and len(by.get(top, [])) >= 10:
        c1, ct = sum(by[1]) / len(by[1]), sum(by[top]) / len(by[top])
        b = (ct - c1) / (top - 1)
        a0 = c1 - b
    else:
        pts = [(r, v) for r, vs in by.items() for v in vs]
        if len(pts) < 20 or len(by) < 2:
            sys.exit("--cost auto: the trace needs rounds of 1 row and of full chains (or 20 rounds of two widths)")
        n = len(pts)
        mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
        b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
        a0 = my - b * mx
    return {r: a0 + b * r + (r - 1) * call for r in range(1, top + 1)}, (a0, b, call)


def parse_set(spec):
    head, *extra = spec.split(",")
    name, lo, hi = head.split(":")
    rng = {"temp": (float(lo), float(hi))}
    for e in extra:
        k, v = e.split("=")
        a, b = v.split(":")
        if k not in ("top_p", "top_k", "presence"):
            sys.exit(f"--set {spec}: unknown range {k}")
        rng[k] = (float(a), float(b))
    return name, rng


def holds(rng, r):
    return all(lo <= r[k] <= hi for k, (lo, hi) in rng.items())


def survival_kept(r, tq, tp, theta):
    """Drafts the rule keeps of a traced full chain."""
    s = 1.0
    for d in range(r["n"]):
        if s * lookup(tp, r["p1"][d]) < theta:
            return d
        s *= lookup(tq, r["q"][d])
    return r["n"]


def tokens(r, L):
    """Expected tokens of a round that verifies the first L drafts: 1 + sum_j prod_{i<=j} a_i (all rows known), else the
    observed prefix (1 + min(acc, L))."""
    if all(r["known"][:L]):
        e, s = 1.0, 1.0
        for j in range(L):
            s *= r["a"][j]
            e += s
        return e
    return 1.0 + min(r["acc"], L)


def simulate(rounds, cost, keep):
    tok = ms = drafts = 0.0
    for r in rounds:
        L = keep(r)
        if L + 1 not in cost:
            sys.exit(f"--cost has no window of {L + 1} rows")
        tok += tokens(r, L)
        ms += cost[L + 1]
        drafts += L
    n = max(1, len(rounds))
    return dict(tps=1000.0 * tok / ms if ms else 0.0, tok=tok / n, drafts=drafts / n, ms=ms / n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="+")
    ap.add_argument("--cost", required=True, help="rows:ms,... a round's time by window rows (1 + drafts), or auto")
    ap.add_argument("--set", action="append", required=True,
                    help="name:lo:hi[,top_p=lo:hi][,top_k=lo:hi][,presence=lo:hi] (temperature and other ranges), in priority order")
    ap.add_argument("--out", required=True, help="the tables file to write (stop_iso 1)")
    ap.add_argument("--theta", default="0.025:0.95:0.025", help="lo:hi:step of the theta grid (a card that copies its missing experts in, at 15 ms or more a row, wants a high theta)")
    ap.add_argument("--min-p", default="0.3,0.5,0.7", help="p1 cut-offs to compare against (keep while p1 >= p)")
    ap.add_argument("--test-share", type=float, default=0.25, help="share of requests held out for the report")
    ap.add_argument("--report", help="write the comparison as markdown here too")
    a = ap.parse_args()
    sets = [parse_set(s) for s in a.set]
    lo, hi, st = (float(v) for v in a.theta.split(":"))
    grid = [round(lo + i * st, 6) for i in range(int(round((hi - lo) / st)) + 1)]
    timing = []
    rounds = read(a.trace, timing)
    if a.cost == "auto":
        K = max((r["n"] for r in rounds), default=4)
        cost, (c0, c1, call) = auto_cost(timing, K + 1)
        cost_txt = f"auto: verify {c0:.1f} + {c1:.2f} ms a row, {call:.2f} ms a draft call -> " + ",".join(f"{k}:{v:.1f}" for k, v in cost.items())
    else:
        cost = {int(k): float(v) for k, v in (kv.split(":") for kv in a.cost.split(","))}
        cost_txt = a.cost
    lines = [f"# stop_iso_fit.py: {len(rounds)} rounds from {', '.join(a.trace)}", f"# cost (ms by rows): {cost_txt}"]
    out = ["stop_iso 1", f"# fit by tune/stop_iso_fit.py on {len(rounds)} traced rounds; cost {cost_txt}"]
    for name, rng in sets:
        tlo, thi = rng["temp"]
        rs = [r for r in rounds if next((n for n, g in sets if holds(g, r)), None) == name]
        if not rs:
            lines.append(f"\n## set {name} ({tlo}..{thi}): no rounds, not written")
            continue
        test = lambda r: (zlib.crc32(r["req"].encode()) % 1000) < 1000 * a.test_share
        dev = [r for r in rs if not test(r)]
        held = [r for r in rs if test(r)]
        # the fit: each draft against its acceptance given the drafts before it. A known a holds for every verified row
        # (the p/q test of row j uses its own uniform, so "the drafts before stood" does not change it); an outcome only
        # up to the first rejection
        xq, xp, ya = [], [], []
        for r in dev:
            for j in range(r["n"]):
                if not r["known"][j] and j > r["acc"]:
                    break
                xq.append(r["q"][j]); xp.append(r["p1"][j]); ya.append(r["a"][j])
        tq, tp = table(pava(xq, ya)), table(pava(xp, ya))
        K = max(r["n"] for r in rs)
        best = max(grid, key=lambda th: simulate(dev, cost, lambda r: survival_kept(r, tq, tp, th))["tps"])
        lines.append(f"\n## set {name}: temperature {tlo}..{thi}, {len(rs)} rounds ({len(dev)} dev, {len(held)} held out), "
                     f"{len(ya)} drafts fit, K = {K}")
        lines.append("\n| rule | dev tok/s | held-out tok/s | held-out vs full K | drafts a round | tokens a round | ms a round |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        full_h = simulate(held, cost, lambda r: r["n"])
        arms = [(f"full K ({K})", lambda r: r["n"])]
        for mp in (float(v) for v in a.min_p.split(",") if v):
            arms.append((f"p1 >= {mp}", lambda r, mp=mp: next((d for d in range(r["n"]) if r["p1"][d] < mp), r["n"])))
        for th in grid:
            arms.append((f"rule, theta {th}" + (" (best on dev)" if th == best else ""),
                         lambda r, th=th: survival_kept(r, tq, tp, th)))
        for label, keep in arms:
            d, h = simulate(dev, cost, keep), simulate(held, cost, keep)
            gain = 100.0 * (h["tps"] / full_h["tps"] - 1.0) if full_h["tps"] else 0.0
            lines.append(f"| {label} | {d['tps']:.2f} | {h['tps']:.2f} | {gain:+.1f}% | {h['drafts']:.2f} | {h['tok']:.2f} | {h['ms']:.1f} |")
        lines.append("\n| knot | " + " | ".join(f"{k:g}" for k in KNOTS) + " |")
        lines.append("|---|" + "---:|" * len(KNOTS))
        lines.append("| iso_q | " + " | ".join(f"{v:.3f}" for v in tq) + " |")
        lines.append("| iso_p1 | " + " | ".join(f"{v:.3f}" for v in tp) + " |")
        keys = "".join(f" {k} {lo_:g} {hi_:g}" for k, (lo_, hi_) in rng.items() if k != "temp")
        out += [f"set {name} theta {best:g} temp {tlo:g} {thi:g}{keys}   # best on dev; {len(ya)} drafts",
                "x " + " ".join(f"{k:g}" for k in KNOTS),
                "q " + " ".join(f"{v:.4f}" for v in tq),
                "p1 " + " ".join(f"{v:.4f}" for v in tp)]
    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.report:
        with open(a.report, "w") as f:
            f.write(text)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
