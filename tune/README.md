# Tuning for your machine

We have one machine (README, "The machine"), and the profiles in `profile/` are tuned on it. Two tools look at yours:

| Tool | Loads the model | Takes | What it gives |
|---|---|---|---|
| `tune/check.py` | no | about a minute | the hardware around the cards, measured copies and disk reads, the profile to use, the bottlenecks with advice, a report for a GitHub issue |
| `tune/tune.py` | yes, once per candidate | 4-5 hours for every group, about 2.5 for `--only cpu_exact` (about 10 minutes a load) | the exact switches measured on your machine, the exact CPU path among them; writes `DATA/profile.tuned.json` |

Both need Python 3.10 and nothing beyond what `setup.sh` installs; `check.py` reads the driver's `libcuda.so.1` directly
(no CUDA toolkit). Its bench of the exact CPU kernels builds one test program with the host's `g++` when there is one
(`--no-exact` skips it; `tune.py --no-kernel-bench` likewise).

## check.py

```bash
python3 tune/check.py --data /data/flash-next --out report.md
python3 tune/check.py --data /data/flash-next --gpu 1,2 --out report-1-2.md   # with more than two P100: the pair to look at
```

Run it with the cards free: the copy test takes about 600 MiB of each card for a few seconds and is skipped on a card
that has more than 1 GiB in use (`--force-copy` runs it anyway). With more than two P100 it looks at the pair in
`DATA/profile.json` (else the pair from two CPUs with the least VRAM in use; `--gpu A,B` chooses), and says so when
another pair hangs from two CPUs or two switches and this one does not. It reads and measures:

- **The cards**: two Tesla P100 16 GB (cc 6.0), the driver (the 580 branch is NVIDIA's last for Pascal), power limit,
  clocks, the NUMA node, the IOMMU group and BAR1 of each card (a P100 needs a 16 GiB BAR1: Above 4G Decoding).
- **The processor**: x86-64, AVX2, and the platform. The engine and Strata's server need no AVX2 (`build.sh` sets no
  `-march`), but Strata's picture encoder is built for an AVX2 baseline by default and stops on an older processor:
  there use `setup.sh --no-pictures` or build `strata-vision` with `-DSTRATA_PORTABLE=OFF`. On Sandy Bridge-EP (Xeon
  E5 v1, Core i7 39xx) NVIDIA's driver runs the cards at PCIe Gen2 unless the module option `NVreg_EnablePCIeGen3=1` is
  set; Ivy Bridge-EP (E5 v2) runs Gen3 on the same boards.
- **Each card's PCIe path**: the card's link and every bridge above it up to the CPU, as they run and at most. An idle
  P100 drops its link to Gen1, so the link is also read while the card copies.
- **Copies** from pinned host memory, as the engine copies missing experts: to each card in 64 MiB and 1 MiB pieces,
  back to the host, from the other CPU's memory, and to both cards at once (two cards behind one PCIe switch share its
  uplink, which a single card's test does not show).
- **RAM**, swap, NUMA nodes and their memory, the IOMMU (`/proc/cmdline`, `/sys/class/iommu`, the cards' IOMMU groups),
  transparent huge pages.
- **The exact CPU kernels** (`FNX_CPU_EXACT`, [GUIDE.md](../docs/GUIDE.md#the-exact-cpu-path)): it builds
  `engine/tools/cpu_exact_test.cpp` with the host compiler, takes one card's half of two routed experts out of the model's
  files (`tune/sample_expert.py`, which reads the GGUF headers itself), and times one on a single core and on the pool
  the engine would take at a card's CPU (that CPU's cores less two, 16 at most), for windows of 1 and 2 rows, beside the
  fp32 kernels. The verdict holds the pool's time against the copy of the same bytes over each card's link: that pair
  of numbers is what decides whether the CPU can help, and where `tune.py` starts its search.
- **The disk with the model**: reads in a row (4 readers of 16 MiB, `O_DIRECT`; the load reads 48 GB) and random
  128 KiB reads (a token's row of the 26.8 GiB lookup table, which the 64 GB mode and the UD-Q4_K_XL profiles read from
  the disk). If other work waits for I/O meanwhile, it says so.

Then it picks the profile by the RAM (96 GB: the default IQ3_S profile; 64 GB: `--ram 64`; less: none) and lists what
limits your machine against ours. `--write-profile OUT` writes `DATA/profile.json` with its suggested switches.

Our machine, for comparison (`check.py` in a test window, 13 Nov 2026): 12.3 GB/s to each card (Gen3 x16, `iommu=pt`;
1 MiB copies 10.8, back to the host 13.1), the same from the other CPU's memory and with both cards at once; the disk
0.50-0.53 GB/s in a row and 1,350-1,700 random 128 KiB reads a second (a logical drive of an HPE Smart Array controller;
a run while another job wrote to the disk gave 34, which is why the report notes I/O pressure).

### What the rules rest on

Each piece of advice is marked **[measured]** when it rests on a measurement on our machine and **[unmeasured]** when it
is what the design implies. The second kind has not been tried on another machine:

| Finding | Advice | Basis |
|---|---|---|
| no AVX2 | pictures off, or the encoder built for this processor | **[unmeasured]** on such a processor; read from Strata v0.2.9's `tools/vision/CMakeLists.txt` |
| Sandy Bridge-EP | `NVreg_EnablePCIeGen3=1`, then check the link while copying | **[unmeasured]** here; NVIDIA's driver documentation |
| a virtual machine | run on the host, or `iommu=pt` there | **[unmeasured]** in a VM; the copy test shows what it costs |
| fewer than 8 CPU threads | a CPU with more | **[unmeasured]**: a spinning host thread per card, the server and 12 embedding threads; ours has 16 at 1.7 GHz and writes 66.6 tok/s |
| governor `powersave` outside intel_pstate / amd-pstate | ondemand, schedutil or performance | **[unmeasured]** |
| BAR1 without an address | Above 4G Decoding in the BIOS | the engine refuses such a card (`card_check.hpp`) |
| the IOMMU translates the cards' DMA | `iommu=pt` (or VT-d off) | [measured] 7.6-8 GB/s instead of 11.9 on ours |
| a card's link below Gen3 x16 while copying, or a narrower bridge above it | slot wiring, riser, BIOS slot speed | [measured] ours run Gen3 x16 |
| below about 10 GB/s to a card (x8, Gen2, a shared switch uplink) | start `tune.py` from fewer guessed next-layer copies: `FNX_PREFETCH_WANT` 2-3, `FNX_PREFETCH_MIN_T` 2 | **[unmeasured]** rule. It comes from UD-Q4_K_XL on ours, where PCIe was the limit: 6 guesses a row → 2 gave edits +15-20 % |
| a card below about 10 GB/s or on a link of x8 or x4, with UD-Q4_K_XL in the 96 GB mode, and CPUs with AVX2, FMA, F16C and 10 or more cores at each card | measure `FNX_CPU_EXPERTS=1` with `tune.py --allow-cpu-experts`: the experts that are not in VRAM computed on the CPU instead of copied over that link ([GUIDE.md](../docs/GUIDE.md#missing-experts-on-the-cpu)). NOT bit-exact and not repeatable run to run, so only a note, and never tried without the option | [measured] on one machine: one card on Gen3 x8, 2 x 18 cores at 2.3 GHz with four DDR3-1600 channels each: UD-Q4_K_XL fresh writing +41 %, code edits 2.5x, reads unchanged. **[unmeasured]** anywhere else; the 10 cores are a guess (the pool leaves two free and ran with 16 threads a card there). With fewer cores the note says it may lose |
| the exact kernels' time for an expert on a card's cores against the copy of its bytes to each card | the `cpu_exact` group of `tune.py` (exact: it needs no option) finds the CPU's share of a window's missing experts and keeps it only if it gains | [measured] ours (136 us on 16 cores against 236 us to the card on x8 and 122 us to the one on x16): kept, fresh +19 %, edits +88 % and +92 % with bitlamas' Q4_K_XL-IQ4NL, +37 % and +119 % / +125 % with UD-Q4_K_XL; [unmeasured] with both cards on x16 |
| both cards at once much slower than each alone | slots on different root ports or CPUs | [unmeasured] threshold; a user's machine with two cards behind one switch wrote 45 tok/s there, 58 with one card per CPU |
| more than two P100, the pair from one CPU or behind one switch while another pair is not | `setup.sh --gpu A,B` with that pair | [measured] on that user's machine (above) |
| the other CPU's memory slower than the card's own | nothing: the engine pins each card's thread to its CPU; `FNX_NUMA_EXPERTS=1` also places the RAM copies (same bits) | [measured] the same on ours, `FNX_NUMA_EXPERTS=1` in the noise |
| power limit under 185 W | raise it (`nvidia-smi -pl`) | [unmeasured]; 250 W gave reads at most 1-1.5 % over 185 W on ours [measured] |
| slow disk in a row | only the load is slower | [measured] 0.5 GB/s, about 100 s of the load on ours |
| few random reads, with the lookup table on the disk | an SSD or NVMe | [measured] on ours the first long read after a load ran at 765 tok/s, later ones at about 900 (UD-Q4_K_XL) |

## tune.py

```bash
python3 tune/tune.py --data /data/flash-next --dry-run                 # the check and the plan, nothing loaded
python3 tune/tune.py --data /data/flash-next                           # about 2-3 hours
python3 tune/tune.py --data /data/flash-next --only prefetch,dma --screen
setsid nohup python3 tune/tune.py --data /data/flash-next > /dev/null 2>&1 < /dev/null &   # detached: DATA/tune-*/progress.log
```

It starts from `DATA/profile.json` (the profile `setup.sh` wrote, at `FNX_ROUTE_PRIOR=0`) and:

1. runs `check.py` and prints its verdict; a suggested `FNX_PREFETCH_WANT` is tried first;
2. loads the starting profile twice: the spread between the two runs is your machine's noise;
3. tries one group of switches at a time, each candidate on a freshly loaded engine with `bench.py`'s fresh writing,
   edits and long read (about 4 minutes after the load; `--tests` to change). Each test's threshold is the larger of
   `--min-gain` (2 %; 4 % for fresh writing and writing after the long read, as the cards' clock follows recent load)
   and its noise. A candidate is kept when one test is faster by more than its threshold, the mean of all by at least
   half `--min-gain`, none slower by more than its threshold, and its first-token logits are the starting profile's bit
   for bit. The fastest kept candidate of a group is the bar for the next group;
4. runs the best profile once more, compares its logits with the starting profile's (`FNX_DUMP_FIRST_LOGITS`), and
   writes it to `DATA/profile.tuned.json` only if this run, too, beats the starting profile by the same rule (else it
   writes the starting profile and says so: `END ok-no-gain`). Every arm is in a table in `DATA/tune-*/summary.md`.

| Group (`--only`) | Switches | Candidates |
|---|---|---|
| `prefetch` | `FNX_PREFETCH_WANT`, `FNX_PREFETCH_MIN_T`, `FNX_PREFETCH` | 2, 3, 4, 6, 9 guesses a row and off; then from 2 or 4 rows. With `--screen` the guesses are first compared in one load through `FNX_PREFETCH_FILE` (edits, forth and back), and only those ahead there get a load of their own |
| `dma` | `FNX_DMA_SPLIT` | 0, 1, 2 |
| `experts` | `FNX_EXPERT_RAM_GIB`, `FNX_EXPERT_DYN_GIB` | ±1 GiB in RAM only (less: more experts in VRAM, until the load refuses); ±2 GiB dynamic where the profile sets it, from its whole GiB (UD-Q4_K_XL's 18.430: 20 and 16) |
| `ple` | `FNX_PLE_RAM` | the other setting, when the RAM allows |
| `convcache` | `--conversation-cache-mib` | half, only when an arm came within 4 GiB of the RAM's end (RAM, not speed) |
| `cpu_exact` | `FNX_CPU_EXPERTS=1` with `FNX_CPU_EXACT=1`, `FNX_CPU_SHARE=A,B` | the exact CPU path on: A (the CPU's per cent of a window's missing experts at 1 row) 50, 30, 70 with `check.py`'s guess for B (at 16 rows); then B ±15; then A ±10; then `FNX_CPU_WARM=1`, `FNX_CPU_MIN_T=3`, and `FNX_CPU_THREADS` on both hyperthreads (alone, and with B +15: more threads pay only when the CPU has more to do), each a step of its own. The later steps run only if the first kept the path; a second run from the tuned profile goes on from where the first stopped |
| `draft` | `FNX_DRAFT_STOP_ISO` (the learned stop rule) | one load first traces full MTP chains (`FNX_DRAFT_ADAPT=0 FNX_DRAFT_MIN_P=0 FNX_DRAFT_TRACE`) over [stop_iso_prompts.json](stop_iso_prompts.json) (36 prompts in English, Slovenian and code, none of bench.py's) at greedy and the model card's thinking (1.0 / 0.95 / 20) and instruct (0.7 / 0.8 / 20, presence 1.5) settings; [stop_iso_fit.py](stop_iso_fit.py) fits the tables at the trace's own round costs (`--cost auto`); the next load measures them against the length planner. Kept, they are written beside the profile as `stop_iso.txt`. Profiles without an MTP chain skip it. Last in the default set, after `cpu_exact`: the tables follow the round costs, which the CPU path changes |
| `stage` (not by default) | `FNX_PF_STAGE_MIB`, `FNX_PF_TAIL_MIB` | half the staging buffers; 512 MiB more lent slots. Ours are at the size of the largest layer and smaller ones were slower |

All of these give the same bits ("none" under output effect in [docs/SWITCHES.md](../docs/SWITCHES.md)).
`FNX_ROUTE_PRIOR` (lossy) is tried only with `--allow-lossy`.

**The exact CPU path (`cpu_exact`).** With `FNX_CPU_EXACT=1` the CPU computes a missing expert with the cards' own
arithmetic, and with `FNX_CPU_SHARE` it takes only a part of a window's missing experts while the copy engine brings the
others ([GUIDE.md](../docs/GUIDE.md#the-exact-cpu-path)). The answers are the starting profile's bit for bit,
so the group is in the default set (before `draft`) and needs no option. `tune.py` holds it to that itself: every arm's
first-token logits must be the baseline's, the engine's log must show the exact path on both cards, and a profile that
keeps the path is loaded once more with `FNX_CPU_CHECK=1` (the copy engine and the cards compute every window as without
the path, and each row the CPU computed is compared with the card's own). Only if that run counted rows on both cards
and not one value differed in a bit is the profile written; otherwise the starting profile is, and the run ends
`error-cpu-check`. It is skipped, with the reason, where `FNX_CPU_EXPERTS` cannot run (no copy engine, a low-RAM mode,
Edge0, no AVX2/FMA/F16C).

Measured on our server on 24 Nov 2026 (two Tesla P100 at a 150 W cap, card 0 on a x8 link, a card at each of 2 x 18
cores), `tune.py --only cpu_exact` from a profile without a CPU switch: 15 loads in 2 h 10 min on bitlamas'
Q4_K_XL-IQ4NL (256k), 15 loads in 2 h on UD-Q4_K_XL (the 512k profile). One load an arm, tok/s:

| File | `check.py`: an expert on 16 cores, its copy to card 0 / card 1 | Its guess for B | Written | Fresh | Edits | After the long read | The bit check |
|---|---|---:|---|---:|---:|---:|---|
| Q4_K_XL-IQ4NL | 136 us; 236 / 122 us | 60 | `FNX_CPU_SHARE=70,75 FNX_CPU_THREADS=32` | 47.6 -> 56.5 | 53.9 / 54.9 -> 101.3 / 105.2 | 42.1 -> 45.4 | 792,199 rows a card, 0 values |
| UD-Q4_K_XL | 139 us; 284 / 149 us | 70 | `FNX_CPU_SHARE=70,70 FNX_CPU_THREADS=32` | 36.8 -> 51.2 | 37.5 / 37.6 -> 82.2 / 84.5 | 33.5 -> 42.7 | 1,096,914 rows a card, 0 values |

In both runs every arm's first-token logits were the baseline's bit for bit (13 of 13). What the steps found there: the
1-row share is flat between 50 and 70 and loses at 30 (fresh writing) and at 80 (writing after a read); the 16-row
share has a sharp best value (15 points lower loses 14 % of the edits); `FNX_CPU_WARM=1` and `FNX_CPU_MIN_T=3` stay in
the noise; the second hyperthreads gain 3-9 % on the edits, on the first file only together with a larger 16-row share.

What `tune.py` does not do, and how to do it by hand:

1. **The card's side of the claim.** That a P100's fp16 FMA and row kernels are the steps the CPU reproduces is tested
   by a program of its own. Build with `-DSTRATA_BUILD_TESTS=ON` (as `build.sh` configures, plus that option), then with
   the cards free run `split_cpu_exact_test` (no card: the CPU's step against an integer oracle, the kernels against
   the card's steps) and `split_cpu_exact_gpu_test 0` and `... 1` (each card's HFMA2, HMUL2 and float-to-half against the
   oracle, its row kernels and SwiGLU against the CPU's). Each must end `all passed` / `all bit for bit`. Do this once
   for a new card model, driver or CUDA toolkit.
2. **Where the time goes.** `"FNX_PROF": "1"` in a profile's `env` prints the windows by rows (`fnx prof T=...`) and, with
   the CPU path on, each card's lists by rows with their stages (`fnx: cpu experts, card N: ... T16: 3349 x 12.4, 1558 us
   (gate|up 1036, the card's SwiGLU 73, down 449; ...)`). A share is right when the CPU's list and the copies of the
   others end together; if the CPU's list is the longer side at 16 rows, lower B.
3. **Whole answers.** `tune.py` compares first-token logits and runs the bit check. Whole greedy answers (a HumanEval
   run with and without the path, text for text) are a longer test we ran by hand; see the PR's description for ours.

`FNX_CPU_EXPERTS=1` is tried only with `--allow-cpu-experts`, as the last group and on top of what the exact groups
kept: once as the profile is and once with the prefetch it was measured with (`FNX_PREFETCH=1`, 2 guesses a row; more
made it slower where it was measured). It is for a machine whose cards receive the experts that are not in VRAM slowly (a link of x8
or x4) and whose CPUs have cores to spare; `check.py` says when a machine looks like that, and `tune.py` then tells you
the option exists. It is not bit-exact and not repeatable from run to run, so its arms' logits differ from the
baseline's by design, the engine's log is read to see that the path really ran on both cards, and the summary says in
so many words when the written profile has it. It is skipped, with the reason, for a profile it cannot run with (no
copy engine, a low-RAM mode, Edge0) or a processor without AVX2, FMA and F16C. `tune.py` refuses a starting profile
that already has it: start from the exact one.

On the one machine it has run on (22 Nov 2026: one card on Gen3 x8 at 6.2 GB/s, the other on x16 at 12.4, 2 x 18
cores; UD-Q4_K_XL, the 512k profile; `--only convcache --allow-cpu-experts`): four loads in 34 minutes. The two
baselines agreed within 0.4 % in writing (11 % in the long read: the first load of the file reads slower); the arm
was kept at fresh +43 %, edits +155 % and +166 %, writing after the long read +68 %; the final run confirmed fresh
+38 %, edits +145 % and +159 %, +31 % after the long read, and the summary carried the notice above.

**Safety.** It refuses to start beside another engine, with the cards in use, or with something answering on its port
(18097, `--port`). Before each load it compares the RAM available with what the profile holds (README's numbers) and
skips a candidate that would not fit. It stops the whole run when swap use grows by more than 512 MiB while an arm's
tests run (`--swap-mib`) or by more than 2 GiB during its load (`--swap-load-mib`): a full machine pages out other
programs' idle memory while the engine pins its own (530 MiB in one of our loads), but a profile that does not fit keeps
swapping while it works. It stops only the server it started and the processes under
it, by PID (Strata runs the engine in a process group of its own). `DATA/tune-*/STOP` ends it after the running arm;
`progress.log` ends with a line `END <result>`.

**A smaller machine on a bigger one.** `--memory-max 58G` runs every arm in a memory cgroup of that size without swap
(`systemd-run --user --scope -p MemoryMax=58G -p MemorySwapMax=0`), as we ran the 64 GB mode (README, "64 GB of RAM").
The cgroup counts the pinned copies, the engine's memory and the page cache.

**On ours** (13 Nov 2026, the IQ3_S 512k profile at `FNX_ROUTE_PRIOR=0`, engine `c213636c`), `--screen --only
prefetch,dma` took 7 loads, about 70 minutes, and kept the profile as it is:

| Arm | Fresh | Edit py | Edit md | Read 38,862 | Write after it | Verdict |
|---|---:|---:|---:|---:|---:|---|
| base1 / base2 | 67.0 / 66.7 | 111.0 / 111.0 | 116.1 / 115.9 | 961 / 962 | 62.0 / 60.7 | noise 0.5 % or less (writing after the read 2.1 %) |
| screen, one load: `FNX_PREFETCH_WANT` 6 / 2 / 3 / 4 / 9 | | edits (mean of py and md): 113.5 / 114.3 / 114.3 / 114.8 / 107.8 | | | | none ahead of 6 by 2 % |
| `FNX_PREFETCH=0` | 66.8 | 111.8 | 114.5 | 961 | 61.7 | in the noise |
| `FNX_PREFETCH_MIN_T=2` | 66.0 | 111.7 | 115.8 | 961 | 64.3 | in the noise |
| `FNX_DMA_SPLIT=0` | 65.6 | 106.5 | 111.2 | 961 | 60.1 | edit md −4.2 % |
| `FNX_DMA_SPLIT=1` | 66.0 | 109.8 | 114.7 | 961 | 61.1 | in the noise |

Every arm's first-token logits were the baseline's bit for bit (19 of 19). An earlier run stopped at its fourth load,
when swap grew by 530 MiB while the engine pinned its memory; that is why the limit during a load is now 2 GiB.

**Keep in mind.** The benchmark's numbers depend on the expert cache, which learns what it reads, so every candidate
gets a fresh load and the order of the arms does not favour any. A change that wins by less than the noise is reported
but not kept; run the group again (`--only`) if a result looks close.
