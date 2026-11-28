#!/bin/bash
# engine-inkling2/run.sh DATA [CTX=32768] [PORT=8080]: the engine on both P100 (the model and its drafter: DFlash 2, or DSpark with
# INK2_DRAFTER=dspark), then the OpenAI-compatible server on http://127.0.0.1:PORT/v1. CTX up to 524288 with the drafter, two choices
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
# One P100 (docs/INKLING2.md "One P100"): INK2_CARDS=1 runs build/ink2_1c on CUDA_VISIBLE_DEVICES (default 0) without a drafter
# (INK2_DRAFTER=dflash2 turns DFlash 2 on: fresh writing +5 %, edits 9 % slower, so it stays off by default):
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
# Sampling (docs/INKLING2.md "Sampling"): greedy by default, a request's temperature is ignored; INK2_SAMPLING=1 lets a request's
# temperature, min_p (INK2_MIN_P, 0.05, when it names none) and seed through (both cards and one card; the INK2_MULTI builds
# stay greedy). The drafts stay exact with it. INK2_THINK_BUDGET (1024,4096,8192,16384,32768 tokens by effort; 0 = off) is the
# reasoning's budget.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); B=$H/build
D=${1:?usage: run.sh DATA [CTX|256k|512k] [PORT] [--vision-device cpu|GPU-UUID] [--vision-residency auto|resident|on-demand]}
shift
CTX=32768; CTX_SET=; PORT=8080
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then CTX=${1:-32768}; CTX_SET=${1:+1}; shift; fi
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then PORT=${1:-8080}; shift; fi
DEVICE=${INK2_VISION_DEVICE-cpu}
RESIDENCY=${INK2_VISION_RESIDENCY-auto}
while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || { echo "missing value for $1" >&2; exit 2; }
  case "$1" in
    --vision-device) DEVICE=$2 ;;
    --vision-residency) RESIDENCY=$2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift 2
done
case "$RESIDENCY" in
  auto|resident|on-demand) ;;
  *) echo "vision residency must be auto, resident or on-demand" >&2; exit 2 ;;
esac
[[ "$DEVICE" == cpu || "$DEVICE" =~ ^GPU-[[:xdigit:]]{8}(-[[:xdigit:]]{4}){3}-[[:xdigit:]]{12}$ ]] ||
  { echo "vision device must be cpu or a full NVIDIA GPU UUID" >&2; exit 2; }
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
# the drafter (docs/INKLING2.md "DFlash 2"): INK2_DRAFTER=dflash2 (the default since 27 Nov: Inco AI's and z-lab's DFlash 2,
# drafts at q >= INK2_D2_QMIN 0.7, the server's default INK2_OA_MTP 7) or dspark (Thinking Machines' DSpark through Unsloth's GGUF
# with its draft vocabulary, the server's default 3 drafts, as before). The DFlash 2 GGUF is made from setup.sh's checkpoint at the
# first start (tools/dflash2_gguf.py, ~2 min, needs build.sh's llama.cpp/gguf-py); without the checkpoint DSpark runs
DRAFTER=${INK2_DRAFTER:-dflash2}; DF2=$D/dflash2-Inkling-2-Mini-Q8_0.gguf
if [ "${INK2_MULTI:-}" = pp ]; then DRAFTER=dspark; fi   # the pipeline knows DSpark only (ink2_tpb has both since 27 Nov)
if [ "$DRAFTER" = dflash2 ] && [ ! -f "$DF2" ]; then
  if [ -f "$D/dflash2-Inkling-2-Mini.safetensors" ] && [ -d "$B/llama.cpp/gguf-py" ]; then
    echo "run.sh: making $DF2 from the DFlash 2 checkpoint (once, ~2 min)"
    "$D/venv/bin/python" "$H/tools/dflash2_gguf.py" "$D/dflash2-Inkling-2-Mini.safetensors" "$DF2.tmp" "$B/llama.cpp/gguf-py" && mv "$DF2.tmp" "$DF2"
  else
    echo "run.sh: no DFlash 2 checkpoint in $D (run setup.sh again), DSpark drafts instead" >&2; DRAFTER=dspark
  fi
fi
case "$DRAFTER" in
  dflash2) DRAFT=(--mtp "$DF2"); export INK2_D2_QMIN=${INK2_D2_QMIN:-0.7} INK2_OA_MTP=${INK2_OA_MTP:-7} ;;
  dspark) DRAFT=(--mtp "$D/dspark-Inkling-2-Mini-Q8_0.gguf" --draft-vocab "$H/data/draft_vocab_32768.bin") ;;
  *) echo "INK2_DRAFTER: dflash2 or dspark" >&2; exit 2 ;;
esac
ENG=$B/ink2_v1; MODE="both cards, drafter $DRAFTER"
if [ "${INK2_CARDS:-2}" = 1 ]; then   # one P100: build/ink2_1c, no drafter unless INK2_DRAFTER names one (rejected drafts cost PCIe transfers too)
  ENG=$B/ink2_1c; MODE="one card"
  if [ -n "${INK2_DRAFTER:-}" ]; then MODE="one card, drafter $DRAFTER"; else DRAFT=(); fi
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
  MODE="both cards, up to $SLOTS conversations at once (ink2_$INK2_MULTI, a slot of $CTX each, drafter $DRAFTER)"; INK2_PICTURES=0
  export INK2_SAMPLING=0   # the multi builds write greedy (no sampling in ink2_tpb, ink2_pp)
  SERVER=$B/ink2_server_multi.py; SL=(--slots "$SLOTS")
  "$D/venv/bin/python" "$H/serve/make_multi.py" "$H/serve/ink2_server.py" "$SERVER" "ink2_$INK2_MULTI" >&2 || exit 1
fi
MMPROJ=$D/mmproj-Inkling-2-Mini-BF16.gguf; VIS=()
ENCODER=$B/ink2_vis
if [ "$DEVICE" != cpu ]; then ENCODER=$B/ink2_vis_cuda; fi
if [ "${INK2_PICTURES:-1}" != 0 ] && [ -f "$MMPROJ" ] && [ -x "$ENCODER" ]; then
  VIS=(--mmproj "$MMPROJ" --vis "$ENCODER")
elif [ "$DEVICE" != cpu ] || [ "$RESIDENCY" = resident ]; then
  echo "requested vision configuration requires pictures, the projector, and $ENCODER" >&2
  exit 2
fi
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
  --tokenizer "$B/llama.cpp/build/bin/llama-tokenize" --gguf-py "$B/llama.cpp/gguf-py" --vision-device "$DEVICE" --vision-residency "$RESIDENCY" ${VIS[@]+"${VIS[@]}"} ${TOK[@]+"${TOK[@]}"} ${SL[@]+"${SL[@]}"}
