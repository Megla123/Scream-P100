#!/bin/bash
# engine-inkling2/run.sh DATA [CTX=32768] [PORT=8080]: the engine on both P100 (the model, the DSpark drafter and its draft
# vocabulary), then the OpenAI-compatible server on http://127.0.0.1:PORT/v1. CTX up to 524288 with the drafter, two choices
# for a long context (names accepted):
#   256k = 262144  the KV cache all in f16, 15.2 GiB a card
#   512k = 524288  the full layers' KV from row 32,768 on in int8 (the rows below in f16, bit for bit as 256k), 15.24 GiB a card
# Any CTX above 262144 takes the int8 mode by itself (INK2_KV8=0/1 forces it). 32768: 11.8 GiB a card. Pictures when DATA holds
# the vision projector (setup.sh) and build/ink2_vis exists, INK2_PICTURES=0 turns them off. Ready in ~30-70 s. Ctrl-C or
# SIGTERM stops both. The log goes to the terminal and, with each line's time, to DATA/ink2-PORT.log (INK2_LOG: another file,
# 0 = none). INK2_* switches pass through to the engine (docs/INKLING2.md); INK2_OA_MTP / INK2_OA_LOOKUP set the
# server's default drafts (3 / 15), INK2_LOOP_TEXT / INK2_LOG_EVERY its loop limit for answers and tool calls (16000
# characters) and its progress line while a request runs (60 s), INK2_EFFORT the reasoning effort of a request that names
# none (low = 0.2; clients such as Open WebUI send none). The engine keeps conversations' checkpoints and parked
# conversations in RAM (up to INK2_CKPT_MB = 24576 and INK2_PARK_MB = 32768 MiB, taken as they are needed; set them lower on
# a machine with less RAM).
# One P100 (docs/INKLING2.md "One P100"): INK2_CARDS=1 runs build/ink2_1c on CUDA_VISIBLE_DEVICES (default 0) without the drafter:
# the experts that fit stay in VRAM in data/plan_1c.txt's order, the rest in RAM (~5 GiB at 32k; checkpoints up to 2048 MiB, no
# parked conversations), with an on-card cache of RAM experts (INK2_1C_CACHE_MB, 1536; since 26 Nov also slots in the reading
# path's buffers, INK2_1C_SHARE=1, and no p34 scratch, INK2_1C_SCRATCH_MB=0: the same bits). INK2_LOSSY=1 (one card only) requantizes
# the last 4 layers' Q8 experts to NVFP4 at load: ~18 % faster writing and ~2 GiB less RAM, but the output changes (see the docs).
# The context costs the card's room for experts: 32768 is the default, 65536 still runs well; above that writing slows.
# Several conversations at once (test builds, docs/INKLING2.md "Several conversations at once"): INK2_MULTI=tpb runs build/ink2_tpb
# (production's layout with conversation slots and batches; a conversation alone is bit for bit build/ink2_v1), INK2_MULTI=pp
# build/ink2_pp (a pipeline: half of the layers on each card). Up to INK2_SLOTS (4) requests run side by side, each in its own
# slot of CTX tokens (default tpb 24576, pp 65536; tpb: 4 x 24k or 3 x 40k, pp: 4 x 64k or 6 x 32k fit). No pictures, no
# checkpoints or parked conversations (a conversation stays in its slot until another needs it).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); B=$H/build
D=${1:?usage: run.sh DATA [CTX|256k|512k] [PORT]}; CTX=${2:-32768}; CTX_SET=${2:+1}; PORT=${3:-8080}
case "$CTX" in 256k|256K) CTX=262144 ;; 512k|512K) CTX=524288 ;; esac
[[ "$CTX" =~ ^[0-9]+$ ]] && [ "$CTX" -le 524288 ] || { echo "CTX: a number up to 524288, 256k or 512k" >&2; exit 2; }
# everything below (this script's, the engine's and the server's lines) also goes to a log file, each line with its time:
# INK2_LOG (default DATA/ink2-PORT.log, appended; 0 = the terminal only). The writer ignores Ctrl-C, so the last lines get in
LOG=${INK2_LOG:-$D/ink2-$PORT.log}
if [ "$LOG" != 0 ] && touch "$LOG" 2>/dev/null; then
  exec > >(set +e; trap '' INT TERM; while IFS= read -r l || [ -n "$l" ]; do printf '%s\n' "$l"; printf '%(%F %T)T %s\n' -1 "$l" >> "$LOG" 2>/dev/null; done) 2>&1
  echo "run.sh: the log also goes to $LOG"
elif [ "$LOG" != 0 ]; then
  echo "run.sh: cannot write $LOG, the log goes to the terminal only" >&2
fi
ENG=$B/ink2_v1; DRAFT=(--mtp "$D/dspark-Inkling-2-Mini-Q8_0.gguf" --draft-vocab "$H/data/draft_vocab_32768.bin"); MODE="both cards"
if [ "${INK2_CARDS:-2}" = 1 ]; then   # one P100: build/ink2_1c, no drafter (on one card its rejected drafts cost PCIe transfers too)
  ENG=$B/ink2_1c; DRAFT=(); MODE="one card"
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} INK2_1C_PLAN=${INK2_1C_PLAN:-$H/data/plan_1c.txt} \
    INK2_1C_CACHE_MB=${INK2_1C_CACHE_MB:-1536} INK2_1C_MARGIN_MB=${INK2_1C_MARGIN_MB:-150} \
    INK2_1C_SHARE=${INK2_1C_SHARE:-1} INK2_1C_SCRATCH_MB=${INK2_1C_SCRATCH_MB:-0}
  if [ "${INK2_LOSSY:-0}" = 1 ]; then export INK2_1C_Q8_FP4=1; MODE="one card, lossy (last 4 layers NVFP4)"; fi
  [ -x "$ENG" ] || { echo "no $ENG: run build.sh again" >&2; exit 1; }
fi
SERVER=$H/serve/ink2_server.py; SL=()
if [ -n "${INK2_MULTI:-}" ]; then   # several conversations at once (test builds): one slot a request, up to INK2_SLOTS side by side
  case "$INK2_MULTI" in tpb|pp) ;; *) echo "INK2_MULTI: tpb or pp" >&2; exit 2 ;; esac
  ENG=$B/ink2_$INK2_MULTI; SLOTS=${INK2_SLOTS:-4}
  [ -n "$CTX_SET" ] || CTX=$([ "$INK2_MULTI" = tpb ] && echo 24576 || echo 65536)
  [ -x "$ENG" ] || { echo "no $ENG: run build.sh again" >&2; exit 1; }
  MODE="both cards, up to $SLOTS conversations at once (ink2_$INK2_MULTI, a slot of $CTX each)"; INK2_PICTURES=0
  SERVER=$B/ink2_server_multi.py; SL=(--slots "$SLOTS")
  "$D/venv/bin/python" "$H/serve/make_multi.py" "$H/serve/ink2_server.py" "$SERVER" "ink2_$INK2_MULTI" >&2 || exit 1
fi
MMPROJ=$D/mmproj-Inkling-2-Mini-BF16.gguf; VIS=()
if [ "${INK2_PICTURES:-1}" != 0 ] && [ -f "$MMPROJ" ] && [ -x "$B/ink2_vis" ]; then VIS=(--mmproj "$MMPROJ" --vis "$B/ink2_vis"); fi
echo "run.sh: $MODE, context $CTX ($([ "$CTX" -gt 262144 ] && echo "KV int8 from row 32768" || echo "KV f16")), pictures $([ ${#VIS[@]} -gt 0 ] && echo on || echo off), port $PORT, commit $(git -C "$H" log -1 --format=%h 2>/dev/null || echo ?)" >&2
SOCK=$D/ink2-$PORT.sock
rm -f "$SOCK"
"$ENG" --model "$D/Inkling-2-Mini-NVFP4-Q8_0.gguf" ${DRAFT[@]+"${DRAFT[@]}"} --serve "$SOCK" --ctx "$CTX" ${SL[@]+"${SL[@]}"} &
EP=$!
trap 'kill $EP 2>/dev/null' EXIT
for i in $(seq 180); do [ -S "$SOCK" ] && break; kill -0 $EP 2>/dev/null || { echo "the engine ended while loading" >&2; exit 1; }; sleep 1; done
[ -S "$SOCK" ] || { echo "no engine socket after 180 s" >&2; exit 1; }
export INK2_ENGINE_PID=$EP
TOK=(); [ -x "$B/ink2_tok" ] && TOK=(--tok "$B/ink2_tok")   # the tokenizer that stays up (a build from before it: llama-tokenize a request)
"$D/venv/bin/python" "$SERVER" --port "$PORT" --sock "$SOCK" --ctx "$CTX" --model "$D/Inkling-2-Mini-NVFP4-Q8_0.gguf" \
  --tokenizer "$B/llama.cpp/build/bin/llama-tokenize" --gguf-py "$B/llama.cpp/gguf-py" ${VIS[@]+"${VIS[@]}"} ${TOK[@]+"${TOK[@]}"} ${SL[@]+"${SL[@]}"}
