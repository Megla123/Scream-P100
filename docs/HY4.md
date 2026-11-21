# 🧪 Hy4 flash on two Tesla P100 (test build)

> ⚠️ **Test build.** Written in two days and measured on one machine only (ours, 21 Nov 2026); nobody else has run it
> yet. It passed our chat, code, tool-call and 64k-context tests, but expect rough edges. Results, good or bad, are
> welcome as GitHub issues.

[Tencent's Hy4 flash](https://huggingface.co/tencent/Hy4-flash) on two **Tesla P100 16 GB**, with an engine of our own in
[`engine-hy4/`](../engine-hy4/). It is not Strata's split engine: Hy4 has an architecture that engine does not know (Engram
tables, sparse attention with an indexer, latent experts, four residual streams), so we wrote one for it, about 5,000 lines of
CUDA and C++. It keeps the idea of the main engine: **both cards work on every layer at once**, the experts that do not fit in
VRAM stream in from RAM over PCIe.

On the same machine, llama.cpp with Hy4 support (PR #32517) writes at 6-9 tok/s, with the CPU computing about two thirds of
the experts. This engine writes at **27 tok/s** and reads at **337 tok/s** (442 with f16 sums).

![Hy4 flash on two Tesla P100](hy4-scream-p100.jpg)

## Quick start

You need Linux, two Tesla P100 16 GB, **96 GB of RAM**, CUDA 12, and ~101 GB of disk for the model ([requirements](#requirements)).

```bash
git clone -b hy4-test https://github.com/Megla123/Scream-P100.git && cd Scream-P100
bash engine-hy4/build.sh                 # the engine, about a minute
bash engine-hy4/setup.sh /data/hy4       # the model (100.5 GB), Python venv, chat template, llama-tokenize
bash engine-hy4/run.sh /data/hy4         # OpenAI-compatible API on http://127.0.0.1:8080/v1 after ~6 minutes
python3 engine-hy4/bench.py              # on a freshly started engine: our numbers below
```

With more than two P100, `run.sh` takes two from different CPUs by itself; `--gpu 1,2` names them (as nvidia-smi numbers
them), and `run.sh /data/hy4 --check` prints the cards and slots it would use without starting anything.

If you already have the files: `HY4_TRUNK=/path/Hy4-flash-IQ3_XXS-00001-of-00002.gguf
HY4_ENGRAM=/path/Hy4-flash-Q3_K_M-EIQ4_NL-00002-of-00002.gguf bash engine-hy4/setup.sh /data/hy4` links them instead of
downloading (and still checks their SHA-256).

## Speed

Our machine: HPE DL380 Gen10, two Xeon sockets with one P100 on each (PCIe 3.0 x16, no P2P), 99 GiB of RAM, CUDA 12.4.
`bench.py` on a freshly started engine, 64k context (`run.sh` defaults), no thinking (21 Nov 2026, 10:00 and 10:15; the first
build's of 08:03 for comparison):

| Test | default (f32) | `HY4_PF_F16=7` | first build |
|---|---|---|---|
| Fresh writing, 12 prompts of 300 tokens | **27.0** | **28.1** | 27.0 |
| Edit, the Python file back with one name changed (2,016 tokens) | **28.1** | **28.6** | 28.2 |
| Edit, the Markdown text back with one word changed (1,623 tokens) | **29.9** | **30.0** | 29.7 |
| Reading 14,302 tokens | **337** | **442** | 294 |

Writing is the same in all three (the f16 switch changes only reading); the 28.1 is run-to-run spread. The edit answers were
the exact expected text. Prompt lookup does the most for edits; on our own edit tasks (other files) the engine wrote 32.1
(Python) and 34.7 (Markdown).

**Reading with f16 sums** (`HY4_PF_F16=7`) is off by default because it is not bit-equal to the f32 path. On an 11,048-token
prompt its last-row logits differ from the f32 path's by KL 0.024, the first token the same; for scale, the first build
(which rounded the sparse attention's Q and P to f16) differs from today's f32 path by KL 0.018, and even another order of
the f32 sums in that attention gives 0.006. On our test sets it scored the same as the first build (Quality, below).

Long prompts (our own runs, 21 Nov): 29,273 tokens read at 277 tok/s, then writing at 26.9 tok/s; at `--ctx 65536`
61,829 tokens at 252 tok/s, then 21.5 tok/s. For comparison, Qwen3.8-Flash-Next UD-Q4_K_XL on the main engine and the same
cards: 55-56 tok/s fresh, 63 edits, 785-880 tok/s reading.

Where the time goes when writing (~36 ms a token): about 12 ms waiting for experts that missed the VRAM cache (~90 a token,
0.13 ms each over PCIe), 8 ms the expert math, 8 ms the other weights, 3-4 ms waiting for the other card, the rest attention
and launches. The cards hold ~37 % of the experts; more VRAM is the biggest lever.

## Quality

Our own test sets (Slovenian questions, language and open answers; HumanEval-style functions; code edits; English), greedy,
against Flash-Next UD-Q4_K_XL on the same machine:

| Set | Hy4 flash, no thinking | Hy4 flash, no thinking, `HY4_PF_F16=7` | Hy4 flash, `reasoning_effort: high` | Flash-Next Q4 |
|---|---|---|---|---|
| Slovenian questions | 13/15 | 12/15 | 15/15 | 15/15 |
| Answers not in Slovenian | 0/18 | 0/18 | | 0/18 |
| Poor open answers | 0/6 | 0/6 | | 1/6 |
| Functions (tests pass) | 44/45 | 44/45 | | 44/45 |
| Code edits | 26/26 | 26/26 | | 26/26 |
| English | 14/16 | 15/16 | 16/16 | 16/16 |

Short questions (under 80 tokens) go through the writing path, which the f16 switch does not touch; the functions and the
code edits read longer prompts.

Without a system prompt, Hy4 flash sometimes answers a Slovenian question in Croatian or Serbian (so does llama.cpp); with a
system prompt that names the language it answers in Slovenian.

Correctness of the engine: the single-card first version matched a numpy reference of the layers to 1e-7 relative (f32); its
first-token logits against llama.cpp's: same top 5, KL 0.0316 / 0.0002 / 0.0000 on an English, a Slovenian and a code prompt
(llama.cpp computes in f16 on the P100). The two-card engine gave the same output; each later kernel was checked against
the one before it (`bench_dec` on random data, the same answers on our prompts).

## How it works

```
                 card 0                                 card 1
 attention   heads 0-15 / 0-31 ──┐                 ┌── heads 16-31 / 32-63      (whole latent cache, indexer and Engram on both)
                                 ├─ sum in mapped ─┤
 FFN         every expert's      │   host memory   │    every expert's
             FF columns 0-767  ──┘                 └──  FF columns 768-1535      (experts: pinned RAM on the card's NUMA node
                                                                                   → VRAM slots, one LRU for both cards)
```

- **Model** (from the GGUF): 42 layers (layer 0 dense, 41 MoE), hidden 3072, 4 residual streams (iHC), vocabulary 120,832.
  MLA attention with one 576-value latent row a token; 30 sliding-window layers (512, a 1024-row ring) and 12 sparse layers
  whose indexer picks 2048 positions; Engram n-gram tables on layers 1, 11, 21 and 31 (read from RAM); 512 latent experts
  (1536 wide), 16 a token, plus a shared expert.
- **Split:** attention heads, every expert's FF width, layer 0's FFN and the vocabulary are halved. Two sums a layer through
  page-locked mapped memory; both cards add the same two parts, so they hold the same bits.
- **Experts:** IQ3_XXS halves of 1.35 MB a card in page-locked RAM, 7700 VRAM slots a card at 64k context (8100 at 32k).
  A miss copies over each card's own PCIe link while the hits compute. The host keeps one LRU for both cards.
- **Writing:** attention over position ranges for all heads at once, grouped GEMVs, an exact bitonic router, a radix top-k
  for the indexer at any context, and prompt lookup (up to 8 drafted tokens, adaptive). Hy4 flash has no MTP layer.
- **Reading:** 2048-token chunks; experts as a tiled IQ3 GEMM whose warps skip tiles' empty token columns, window attention
  through cuBLAS, the sparse layers' attention in one pass a token straight from the cache, the chunk's scratch borrowed from
  the top cache slots, the next chunk's token rows and Engram rows made on host threads while this one runs. A prompt that
  starts with the previous one reuses its cache (the window ring permitting).
- **Reading in f16 (optional, `HY4_PF_F16=7`):** f16 sums where the P100 is twice as fast: the dense matrices as HGEMM
  (13-15 TFLOPS against SGEMM's 7-8; f16 inputs with f32 sums are no faster on this card), the experts in half2 (each
  32-column sub-block summed in f16, then scaled and added in f32: 8.2 TFLOPS against 4.6), the window attention as HGEMM on
  its f16 rows. The router, Engram, the indexer and the sparse attention stay f32. This is the arithmetic class of
  llama.cpp's prompt path on the P100.

## API

`run.sh` starts `hy4_v1 --serve` on a Unix socket and [`serve/hy4_openai.py`](../engine-hy4/serve/hy4_openai.py) in front of it:
`POST /v1/chat/completions` (stream, `tools` and `tool_calls` in Hy4's own format, `reasoning_content`), `GET /v1/models`,
`GET /health` (503 while loading). Model id `hy4-flash-iq3xxs`.

- `reasoning_effort` `high` or `max` (top level or in `chat_template_kwargs`) turns thinking on; anything else is no_think.
- **Greedy only** (temperature and sampling are ignored), **one request at a time**, `usage` and llama-server-style `timings`.
  The last stream chunk's `timings` (`prompt_per_second`, `predicted_per_second`) are the engine's own clocks. The time to
  the first chunk also holds `llama-tokenize` and the template.
- **The first stream chunk carries the role together with the first text** (since 21 Nov). Before, a chunk with only the
  role went out before the prompt was read, so a benchmark that times the first chunk (a user's llama-benchy) got a
  meaningless prompt speed.
- `"lookup": N` in the request sets the drafted tokens for that request (0 = off).
- Tokenizing goes through llama.cpp's `llama-tokenize` (one process a request); the chat template is the GGUF's.

## Switches

Environment variables of the engine; all are off or at the measured default.

| Switch | Default | What |
|---|---|---|
| `HY4_LOOKUP` | 8 | most tokens drafted by prompt lookup a step (0 = off) |
| `HY4_LOOKUP_MIN` | 3 | shortest n-gram match that drafts |
| `HY4_PF_CHUNK` | 2048 | prompt-path chunk (tokens) |
| `HY4_PF_MIN` | 80 | shorter prompt rests go through the writing path |
| `HY4_PF_OFF=1` | | no prompt path (16-token windows, slow) |
| `HY4_PF_F16` | 0 | **reading with f16 sums** (bits: 1 dense matrices, 2 experts, 4 window attention; 7 all): ~+30 % reading, not bit-equal (see Speed) |
| `HY4_PF_SKIP=0` | | expert kernels without skipping empty token columns (the same bits, slower) |
| `HY4_PF_SELK` | 1 | sparse attention when reading: 1 one pass a token from the cache (f32), 0 the earlier gathered copy with Q and P in f16 |
| `HY4_PF_HOST=0` | | the earlier host side of reading (token rows and Engram rows made on the host thread, the same bits) |
| `HY4_PF_XS=1` | | the cross-card sum in one kernel (the same bits, measured no faster) |
| `HY4_PF_LOGITS=FILE` | | each chunk's last-row logits into FILE.c0 / FILE.c1 (for comparisons) |
| `HY4_PF_CHECK=1` | | count non-finite values after every layer when reading |
| `HY4_PFIQ` | 5 | prompt-path expert kernel (5 = tiles of 128 rows × 64 tokens; 4 = 128 × 32; 1-3 the first ones) |
| `HY4_PF_STAGE=0` | | prompt path: experts through the cache instead of staged a layer ahead |
| `HY4_PF_SWAG=0`, `HY4_PF_SPG=0`, `HY4_PF_SELH=0` | 1 | window / sparse attention of the prompt path without cuBLAS / f32 instead of f16 |
| `HY4_NUMA_OFF=1` | | do not move threads and pinned memory to each card's NUMA node |
| `HY4_DEC_OFF=1` | | the first version's writing kernels (slower, for comparison) |
| `HY4_MT=1` | | each distinct expert once for all tokens of a drafted window (measured slower) |
| `HY4_PREFETCH` | 0 | 1 / 2: guess next experts and copy them early (measured slower) |
| `HY4_PROF=1` | | ms a token by part, a request |
| `HY4_TRACE=FILE` | | append every routing pick (for `tools/sim_cache.py`) |
| `HY4_DEBUG=1` | | prompt-path chunk timings |
| `HY4_LASTLOG=FILE` | | the last request's speeds as two text lines |

## Tools

- `build/bench_dec`: the writing and prompt-path kernels against the first version's on random data (same results, time a call).
- `SAMO_PF4=1 build/bench_dec`: the prompt path's expert kernels (f32, with and without skipping, half2) at a chunk's token counts.
- `build/gemm_bench`: the prompt path's GEMM shapes with SGEMM, f16 inputs with f32 sums, and HGEMM: TFLOPS and error.
- `build/bench_bw`: host → card copy speed of one expert half, from each NUMA node, one card and both at once.
- `tools/sim_cache.py TRACE [SLOTS ...]`: the expert cache under LRU, LFU, 2-segment LRU and Belady's bound on a
  `HY4_TRACE` file. On ours: LRU 122 misses a token, Belady 56, other policies no better than LRU; 9000 slots −27 %.

## Requirements

- **Linux x86-64**, **two Tesla P100 16 GB (PCIe)**, ideally each on a Gen3 x16 link. Other cards are untested (the build is
  sm_60 and sizes its cache for 16 GB).
- **Both cards free of other work.** `--ctx 65536 --slots 7700` and `--ctx 32768 --slots 8100` fill a 16 GB P100. On a user's
  machine a desktop held about 1 GiB of the first card, and the engine stopped with "out of memory" after a minute of
  loading. Since 21 Nov, `run.sh` takes VRAM that other processes hold off the slots and says so; 1 GiB costs about 770
  slots, and fewer slots mean more copies while writing.
- **With more than two cards, one card per CPU.** Two cards behind one PCIe switch share its link to memory, and writing
  waits on expert copies (see Speed). On that user's machine with four P100:
  - Hy4 wrote 21.6 tok/s on two cards behind one switch.
  - On the same machine, Flash-Next on the main engine went from 45 tok/s on such a pair to 58 tok/s with one card per CPU.
- **96 GB of RAM:** 53 GiB of experts are pinned, and the Engram table (33 GiB) is read once into the page cache after
  loading (`HY4_PREREAD=0` skips that). We ran it with the table in `/dev/shm`; from an SSD with less RAM it has not been
  measured.
- **CUDA 12** (we use 12.4 with g++-13), Python 3 with venv, git, cmake and a C++ compiler (for `llama-tokenize`).
- **Disk:** 100.5 GB for the two model files, ~1 GB for llama.cpp and the venv.
- Only these files are supported: the IQ3_XXS trunk and the IQ4_NL Engram table of
  [Xylolabs/Hy4-flash-GGUF](https://huggingface.co/Xylolabs/Hy4-flash-GGUF) (revision `dc1e22e7`). The engine reads IQ3_XXS
  experts only. No pictures (the vision tower is not ported).

## Licenses

The engine is ours (MIT, as the rest of this repository). `src/hy4_tables.h` holds ggml's IQ3_XXS and IQ4_NL tables (MIT,
the ggml authors). Not included: the model (Tencent Hy license, see the model's repository; `setup.sh` downloads it from
Xylolabs's), and llama.cpp (`setup.sh` fetches it at commit `0a054d0b`, PR #32517, for its tokenizer; we used the same
build's `src/models/hy-v4-flash.cpp` as the reference for the model graph).
