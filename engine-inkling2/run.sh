#!/bin/bash
# engine-inkling2/run.sh DATA [CTX=32768] [PORT=8080]: the engine on both P100 (the model, the DSpark drafter and its draft
# vocabulary), then the OpenAI-compatible server on http://127.0.0.1:PORT/v1. CTX up to 524288 with the drafter, two choices
# for a long context (names accepted):
#   256k = 262144  the KV cache all in f16, 15.2 GiB a card
#   512k = 524288  the full layers' KV from row 32,768 on in int8 (the rows below in f16, bit for bit as 256k), 15.24 GiB a card
#   512k-f16       524288 with every KV row in f16 (INK2_KV8=0): needs 24 GB cards (two Tesla M40: 19.14 GiB a card), bit for
#                  bit the 256k mode, the same speed (docs/INKLING2-M40.md)
# Any CTX above 262144 takes the int8 mode by itself (INK2_KV8=0/1 forces it). 32768: 11.8 GiB a card. Pictures when DATA holds
# the vision projector (setup.sh) and build/ink2_vis exists, INK2_PICTURES=0 turns them off. Ready in ~30-70 s. Ctrl-C or
# SIGTERM stops both. INK2_* switches pass through to the engine (docs/INKLING2.md); INK2_OA_MTP / INK2_OA_LOOKUP set the
# server's default drafts (3 / 15).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); B=$H/build
D=${1:?usage: run.sh DATA [CTX|256k|512k|512k-f16] [PORT]}; CTX=${2:-32768}; PORT=${3:-8080}
case "$CTX" in 256k|256K) CTX=262144 ;; 512k|512K) CTX=524288 ;; 512k-f16|512K-F16) CTX=524288; export INK2_KV8=0 ;; esac
[[ "$CTX" =~ ^[0-9]+$ ]] && [ "$CTX" -le 524288 ] || { echo "CTX: a number up to 524288, 256k, 512k or 512k-f16" >&2; exit 2; }
MMPROJ=$D/mmproj-Inkling-2-Mini-BF16.gguf; VIS=()
if [ "${INK2_PICTURES:-1}" != 0 ] && [ -f "$MMPROJ" ] && [ -x "$B/ink2_vis" ]; then VIS=(--mmproj "$MMPROJ" --vis "$B/ink2_vis"); fi
echo "run.sh: context $CTX ($([ "$CTX" -gt 262144 ] && [ "${INK2_KV8:-}" != 0 ] && echo "KV int8 from row 32768" || echo "KV f16")), pictures $([ ${#VIS[@]} -gt 0 ] && echo on || echo off)" >&2
SOCK=$D/ink2-$PORT.sock
rm -f "$SOCK"
"$B/ink2_v1" --model "$D/Inkling-2-Mini-NVFP4-Q8_0.gguf" --mtp "$D/dspark-Inkling-2-Mini-Q8_0.gguf" \
  --draft-vocab "$H/data/draft_vocab_32768.bin" --serve "$SOCK" --ctx "$CTX" &
EP=$!
trap 'kill $EP 2>/dev/null' EXIT
for i in $(seq 180); do [ -S "$SOCK" ] && break; kill -0 $EP 2>/dev/null || { echo "the engine ended while loading" >&2; exit 1; }; sleep 1; done
[ -S "$SOCK" ] || { echo "no engine socket after 180 s" >&2; exit 1; }
export INK2_ENGINE_PID=$EP
"$D/venv/bin/python" "$H/serve/ink2_server.py" --port "$PORT" --sock "$SOCK" --ctx "$CTX" --model "$D/Inkling-2-Mini-NVFP4-Q8_0.gguf" \
  --tokenizer "$B/llama.cpp/build/bin/llama-tokenize" --gguf-py "$B/llama.cpp/gguf-py" ${VIS[@]+"${VIS[@]}"}
