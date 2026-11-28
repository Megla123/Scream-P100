# engine-inkling2: `ink2`, Inkling 2 Mini on two Tesla P100

**Not a Strata fork:** an engine written from scratch for Thinking Machines' Inkling 2 Mini (no Strata code; llama.cpp only as a library for the tokenizer and the picture encoder). The Strata-derived engine of Scream P100 is the Flash-Next one, `engine/` on `main`. `ink2` runs Inkling 2 Mini (26.4B looped MoE, NVFP4 experts) on two Tesla P100 16 GB: both
cards on every layer at once, the whole model in VRAM, two drafters (DFlash 2 by default since 27 Nov, 8 % faster writing than DSpark, 15 % in English; DSpark) and prompt-lookup drafts, an OpenAI-compatible server
with tool calls, pictures and a reasoning budget by effort (27 Nov). Fresh writing 72 tok/s (97 with drafts), edits ~170,
reading ~1,030 tok/s at 4k and ~800-825 at 35-38k, writing at a depth of 240k 42 tok/s with drafts, 43.5 without (27 Nov, 185 W a
card); context
**256k** (KV f16) or **512k** (KV int8 past 32k, the first 32k bit for bit as 256k, same speed); llama.cpp on the same cards:
13.0 / 143. A conversation keeps its place: checkpoints and parked conversations in RAM, so a chat's next turn reads only
its new messages, a 128-token piece in 336 ms (23 Nov). **One P100** (`build/ink2_1c`, `INK2_CARDS=1`): the most used experts in
VRAM, the rest in RAM with an on-card cache, 49 tok/s fresh writing since 26 Nov (exact, 40 before; 4.8 GiB of RAM for experts)
or 48 within 6 GB on 23 Nov (`INK2_LOSSY=1`, output changes).
**Several chats at once** (test builds `build/ink2_tpb` and `build/ink2_pp`, `INK2_MULTI=tpb|pp`): up to 4-6 conversations
write together, 121 / 136 tok/s in all with 2 / 4 (`ink2_tpb`, alone bit for bit `ink2_v1`) or 148 / 162-166 with 4 / 6
(`ink2_pp`, a pipeline), each answer the same as alone.
The full write-up: [docs/INKLING2.md](../docs/INKLING2.md).

```bash
bash engine-inkling2/build.sh                 # build/ink2_v1, ink2_1c, ink2_tpb, ink2_pp, ink2_vis, ink2_tok + llama-tokenize (CPU)
bash engine-inkling2/setup.sh /data/inkling2  # model 17.6 GB + drafters 0.99 + 0.4 GB + projector 0.64 GB (--no-pictures: without), a venv
bash engine-inkling2/run.sh /data/inkling2 [CTX=32768|256k|512k] [PORT=8080]   # log also in /data/inkling2/ink2-PORT.log
INK2_CARDS=1 bash engine-inkling2/run.sh /data/inkling2   # one P100 (INK2_LOSSY=1: faster, ~2 GB less RAM, output changes)
INK2_MULTI=tpb bash engine-inkling2/run.sh /data/inkling2  # up to 4 chats at once (test build; INK2_MULTI=pp: the pipeline)
python3 engine-inkling2/tools/bench.py http://127.0.0.1:8080
python3 engine-inkling2/tools/picture_check.py http://127.0.0.1:8080
python3 engine-inkling2/tools/conversation_check.py http://127.0.0.1:8080
python3 engine-inkling2/tools/multi_bench.py http://127.0.0.1:8080 1,2,4   # with INK2_MULTI: speed at once, alone = together
```

Streaming responses send SSE comments after 15 seconds without output, including while queued or reading the prompt.
Set `INK2_STREAM_TOOL_ARGUMENTS=1` on the server to enable incremental arguments for declared functions in the standard
Inkling tool format. The default sends whole tool calls. The native engine skips disconnected socket requests and checks
for cancellation between prefill chunks. These checks require rebuilding `ink2_v1`. Run
`python3 engine-inkling2/tools/stream_check.py` to verify the HTTP protocol and disconnect handling with a simulated engine.

| Path | What |
|---|---|
| `src/ink2_v1.cu` | the engine: loading and splitting, the writing path, the reading path (2,048-token chunks), drafts, checkpoints and parked conversations, the socket server; since 24 Nov the one-card code (`src-1c/`) built for two cards with p38-p41 (the reading exchange's buffers on each card's NUMA node, `INK2_NUMA`); since 26 Nov also `build/ink2_1c` (`-DINK2_NC=1`), with p63 (one card: cache slots in the reading path's buffers, `INK2_1C_SHARE`) |
| `src/ink2_fuse.cuh` | merged launches of the writing path (the loop controllers, the norms and Q8 matrices), the same bits |
| `src/ink2_kern.cuh` | writing kernels: Q8 GEMVs, NVFP4 expert rows, attention by KV head, the fused exchange / convolution kernels |
| `src/ink2_pf.cuh` | reading kernels: NVFP4 and Q8 expert tile GEMMs, convolutions over a chunk, softmax with the relative bias and sink |
| `src/ink2_dspark.cuh` | the DSpark drafter's kernels (NeoX RoPE, block attention, the Markov head) |
| `src/ink2_dflash2.cuh` | the DFlash 2 drafter (27 Nov): the block of 8 rows with its two-tap convolutions, its K/V rings, the two-pass top-16, own GEMVs for 8 rows; `src-tpb/` has a copy |
| `tools/dflash2_gguf.py` | the DFlash 2 checkpoint (safetensors) to the engine's GGUF, numpy and gguf-py only; `run.sh` runs it at the first start |
| `src/ink2_vis.cpp` | the picture encoder: llama.cpp's mtmd (the vision projector) on the CPU, rows to a file |
| `src/ink2_tok.cpp` | llama.cpp's tokenizer as a process that stays up (the server's `--tok`) |
| `src-1c/` | the one-card build of 23 Nov (`-DINK2_NC=1 -DINK2_RING=2048`, `gguf_min.*` from `src/`): experts placed by a plan, staged reading, the on-card cache of RAM experts, the lossy requantizing; kept for reference, `build.sh` builds `ink2_1c` from `src/` since 26 Nov |
| `src-tpb/` | several chats at once in production's layout (`ink2_tpb.cu` over a copy of the engine with `INK2_PP`): conversation slots, batches of up to 4 conversations' writing pieces, reading chunks interleaved with the waiting writers |
| `src-pp/` | several chats at once as a pipeline (`ink2_pp.cu`, one-card kernels, `-DINK2_RING=2048`): half of the layers on each card, 8 segments a token, a thread and a queue a card, the drafter on card 1 |
| `src/gguf_min.*` | a small GGUF reader (NVFP4 = type 40) |
| `src/bench_dec.cu`, `src/ink2_bench.cuh` | a one-card microbenchmark of the decode kernels on random weights |
| `serve/ink2_server.py` | the HTTP server (OpenAI API + llama-server endpoints, tools, pictures, the loop guard and the reasoning budget), greedy by default (`INK2_SAMPLING=1`: temperature, min-p and seed by request) |
| `data/draft_vocab_32768.bin`, `tools/draft_vocab.py` | the drafter's reduced vocabulary and how it was made |
| `data/plan_1c.txt`, `tools/plan_1c.py` | the one-card build's order of experts for VRAM and how to make one from routing traces |
| `serve/make_multi.py` | the server for `INK2_MULTI` (`build/ink2_server_multi.py`): up to `--slots` requests at once |
| `tools/bench.py` | fresh writing, an edit and a long read through the API |
| `tools/picture_check.py` | a picture and two questions through the API |
| `tools/multi_bench.py` | 1, 2, 4 … conversations at once through the API, and every answer the same as alone |
| `tools/conversation_check.py` | a chat-style conversation with a tool: tokens read a request, first byte; a written backslash-n |

Measured on one machine only (2× P100 16 GB, PCIe 3.0 x16 each, 99 GiB RAM, CUDA 12.4). MIT, like the rest of the repository;
the model's own license applies to its weights.

### Vision projector device and residency

The projector uses the CPU and loads on the first image by default. To use a GPU, run `build.sh` first, then run
`bash engine-inkling2/build-vision-cuda.sh` to build the CUDA helper. The helper builds for Pascal compute capabilities
6.0 and 6.1 by default. Set `INK2_VISION_CUDA_ARCHITECTURES` to select different targets.

Run `nvidia-smi -L` to find the GPU UUID and pass it as `--vision-device GPU-…` to `run.sh`. The same option can be
set with `INK2_VISION_DEVICE`. The projector's device selection applies to the helper process. The engine retains its
own CUDA device selection.

Use `--vision-residency` or `INK2_VISION_RESIDENCY` to choose `auto` (load the GPU projector at startup and retain it,
load the CPU projector on first use), `resident` (preload and retain either device), or `on-demand` (release after image
preparation and reuse recent embeddings). A GPU selection requires the CUDA helper and projector weights. The server
reports a startup or encoding failure if the selected device is unavailable. Requests can contain up to four images,
each up to 16 MiB encoded. The experimental multi-conversation builds continue to serve text only.

Run `python3 engine-inkling2/tools/vision_device_check.py` to check device selection and lifecycle with a simulated helper.
