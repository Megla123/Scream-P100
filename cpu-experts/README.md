# CPU experts (`FNX_CPU_EXPERTS`): an experiment, not in `main`

This branch is `main` (`b2197f8`) plus the patch we wrote on 15-16 Nov 2026: each card's own CPU socket computes some of
a decode layer's missing experts instead of the card waiting for their copy over PCIe. We measured it, do not use it,
and had not published it. It is here as a starting point for
[issue #6](https://github.com/Megla123/Scream-P100/issues/6), not as a feature of the release.

## Status

- **Off unless asked for.** Without `FNX_CPU_EXPERTS=1`, or in a build without ggml's CPU libraries (the default
  `build.sh` build), none of it runs. A build with the patch and the switch unset gave the same bits as the build without
  the patch in our rounds.
- **Not bit-exact when on.** The CPU computes with ggml's CPU dot products, whose activations are 8-bit (q8_K for Q4_K
  and Q5_K weights, q8_1 for Q5_1); the cards' kernels read f16 activations. First-token KL against the switch off:
  0.003-0.006 mean, 0.045 at most, same first token in 18 of 18 prompts (table below).
- **What was built.** The seven files of `engine/` this branch changes and the two it adds (`src/cpu_moe.cu`,
  `src/cpu_moe.hpp`) are byte for byte the sources of the binary `fa2b8f86ab7e6cd2…` (docs/PROVENANCE.md), which ran
  the fast profile's rounds of 18 Nov; `engine/CMakeLists.txt` differs from that build only in `FNX_DIR` (`${_sp}` as in
  `main`, where our build had our server's path). The table of 16 Nov ran on that day's development binary
  (`31b05cc5…`: the engine as it was then, prompt-path jobs of 64 rows, with the same `cpu_moe.cu`). **This branch itself
  has not been built or run in this repository's layout.** docs/PROVENANCE.md's hashes describe `main`, not these files.
- **Run only on our box**: 2 x Xeon Bronze 3106 (8 cores each, 1.7 GHz, no turbo), 99 GiB RAM, both P100 on x16 Gen3,
  one card per socket, Unsloth's UD-Q4_K_XL (gate|up Q4_K, down Q5_1 in most layers). Other expert types go through the
  same ggml traits (`vec_dot`, `vec_dot_type`), but none was run.

## How it works

A card holds half of every expert (its rows of gate|up, its columns of down) and so does its copy in host memory. With
the switch on, in a decode window of a routed layer:

1. `cache_plan` (`src/moe.cuh`) gives the CPU the misses with the fewest rows of the window: first those of 1 row, then
   2, up to `FNX_CPU_NMAX`, at most `FNX_CPU_MAX` a layer. They take no cache slot and are not copied; everything else
   goes on as before.
2. `k_cpu_pub` (`src/cpu_moe.cu`) writes those jobs and the window's FFN input (f16) to a mailbox in mapped host memory
   and sets the jobs' rows to 0, so the row kernels skip them (`src/p100_moe_rows.cuh`).
3. The card's workers (`FNX_CPU_THREADS`, pinned to the cores in the card's
   `/sys/bus/pci/devices/<id>/local_cpulist`) compute the card's half of each of those experts from the host copy:
   activations to the dot type by that type's `from_float`, gate|up with `vec_dot`, SiLU(gate) x up, down with `vec_dot`.
4. `k_cpu_land`, just before the two cards exchange their sums, waits for the workers (3 s, then an error bit) and copies
   the pairs' rows into `part`, where the card's own results are.

It runs only where the expert cache plans in the router with the copy engine (`over && dma` in `src/engine.cu`, the
release's UD-Q4_K_XL profiles with `FNX_SH_IN=0`), never on the prompt path and never on Edge0's latent layers.

| Switch | Default | |
|---|---|---|
| `FNX_CPU_EXPERTS` | 0 | 1 turns it on; a build without the libraries prints a warning and stays off |
| `FNX_CPU_MAX` | 8 | most missing experts a layer for the CPU (0 to 64) |
| `FNX_CPU_NMAX` | 2 | only misses with at most this many rows |
| `FNX_CPU_MIN_T` | 1 | only in windows of at least this many rows (**never measured**) |
| `FNX_CPU_THREADS` | 6 | workers a card |
| `FNX_CPU_STATS` | 0 | 1 prints, per card every 20,000 layers with jobs, the jobs, pairs and µs a layer |
| `FNX_CPU_DBG` | 0 | 1: the selection only, the card computes those jobs itself; 2: results are not copied back |

## Build

What we did on our server, written for this repository's layout (not run here). The engine has no OpenMP and brings its
own threads, so ggml's CPU libraries are built without it:

```bash
./build.sh        # the usual build; it also fetches llama.cpp at the pinned commit into third_party/llama.cpp
cmake -S third_party/llama.cpp/ggml -B build-ggml-cpu -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_OPENMP=OFF \
  -DBUILD_SHARED_LIBS=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build-ggml-cpu -j8 --target ggml-cpu ggml-base
cmake -DFNX_CPU_GGML_LIBS="$PWD/build-ggml-cpu/src/libggml-cpu.a;$PWD/build-ggml-cpu/src/libggml-base.a" build
cmake --build build --target strata-split-2 -j8        # compiles the engine again: the option adds -DFNX_CPU_GGML=1
```

`cpu_moe_bench.c` here is the micro-benchmark we ran first: how many experts a socket computes with ggml's dots, without
the engine (sizes and types of UD-Q4_K_XL are constants at its top). It uses OpenMP, so it links ggml's default build:

```bash
gcc -O3 -march=native -fopenmp cpu_moe_bench.c -I$G/include -I$G/src $B/src/libggml-cpu.a $B/src/libggml-base.a \
  -lstdc++ -lm -lpthread -o cpu_moe_bench      # G = third_party/llama.cpp/ggml, B = a build of it with OpenMP on
numactl --cpunodebind=0 --membind=0 ./cpu_moe_bench 64 1 8 1     # experts, rows, threads, mode 1: threads take whole experts
```

## What we measured (16 Nov 2026, UD-Q4_K_XL at 512k, 27 % of the experts on the cards)

Round `n16c1`, tok/s, one load per arm; edits are a Python file and a Markdown file returned with small changes.

| `FNX_CPU_MAX` / `FNX_CPU_NMAX` | fresh | edit py | edit md | first-token KL, mean / max | jobs, pairs, µs a layer (a card) |
|---|---|---|---|---|---|
| off | 50.39 | 60.65 | 64.1 | | |
| 2 / 2 | 45.59 | 62.05 | 65.0 | 0.00403 / 0.01469 | 1.83, 1.90, 432 |
| 4 / 2 | 48.42 | 67.65 | 69.5 | 0.00261 / 0.01121 | 3.19, 3.36, 529 |
| 8 / 3 | 44.05 | 75.2 | 80.05 | 0.00576 / 0.04497 | 5.01, 5.56, 594 |
| 24 / 16 | 41.24 | 77.95 | 81.25 | 0.00344 / 0.01810 | 8.05, 10.28, 894 |

Edits up to +28 %, fresh writing 4-18 % slower: in windows of few rows the CPU costs more than the copies it saves.
`FNX_CPU_MIN_T` was written for exactly that (CPU only in wide windows, fresh writing left alone and exact), and the
round that would have measured it was stopped: we decided to keep the UD-Q4_K_XL profile bit-exact.

Where a window's time went on one card that day, in ms (expert time includes waiting for copies):

| | experts gate\|up | experts down | everything else | window |
|---|---|---|---|---|
| fresh (3.3 rows a window) | 15.5 | 12.4 | 21.4 | 49.3 |
| edits (11.7 rows a window) | 67.2 | 58.4 | 49.6 | 175.2 |

The micro-benchmark (one socket, one UD-Q4_K_XL expert = 2.93 MiB, threads taking whole experts): 8 threads get through
an expert every 104 µs at 1 row and every 188 µs at 2 rows; with both cards' copy engines running beside it and 6
threads, 137 and 251 µs. One expert split over 8 threads took 251 µs at 1 row. A copied expert (both halves in
parallel) arrives in about 125 µs. The benchmark computes whole experts; in the engine a socket computes only its card's
half of each.

With the fast profile (Coder mask + route prior, docs/GUIDE.md "Fast UD-Q4_K_XL", 18 Nov) the CPU arms (8 / 3 and
4 / 2) were 1-2 % slower in every test and 21 % slower writing after a long read: there almost every expert a token
needs is on the cards already.

## Bugs it had, so you need not find them again

- **The activations' type.** They must go through the *dot type's* `from_float` (q8_K, q8_1), not the weight type's
  (Q4_K, Q5_1). The weight type's gives garbage, the NaN reached the router through the residual, and the router wrote
  past its table: the symptom was `illegal memory access` in the warm-up windows, far from the cause.
- **Reading mapped host memory in a kernel.** `__ldcv` (`ld.global.cv`) reads of the mailbox in `k_cpu_land` were an
  illegal access; plain `volatile` reads work.
- Not a bug we hit, but easy to miss: after planning, every job of the layer sets its slot's reference bit, and the CPU's
  jobs have no slot (`src/moe.cuh`, the last hunk).

## Not done

- `FNX_CPU_MIN_T` and more than 6 threads a card were never measured.
- The CPU's experts never enter the cache, so an expert the CPU computes in window after window is never promoted.
- No per-layer view of the misses, and no handling of a worker that falls behind beyond the 3 s timeout.
