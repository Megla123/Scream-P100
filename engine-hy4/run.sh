#!/bin/bash
# engine-hy4/run.sh DATA [--port 8080] [--host 127.0.0.1] [--ctx 65536] [--slots 7700] [--gpu A,B] [--check]: the Hy4 flash engine on
# a Unix socket in DATA and the OpenAI-compatible wrapper on http://HOST:PORT/v1 (chat completions with stream, tools and
# reasoning_content, /health, /v1/models). Ctrl+C or SIGTERM stops both. Loading takes ~4 minutes (53 GiB of experts into page-locked
# RAM), then the Engram table is read once into the page cache (~2 minutes; HY4_PREREAD=0 skips it). Tested: --ctx 65536 with --slots
# 7700 and --ctx 32768 with --slots 8100 (more slots, fewer copies over PCIe); both fill a 16 GB P100 that nothing else uses.
# --gpu: the two P100 as nvidia-smi numbers them (else CUDA_VISIBLE_DEVICES, else the two there are, or with more the pair from two
# CPUs with the least VRAM in use). VRAM that other processes hold on a card comes off the slots (HY4_VRAM_CHECK=0 skips that);
# --check prints the cards and the slots and starts nothing. Engine switches (HY4_*) are in docs/HY4.md.
set -euo pipefail
usage="usage: bash engine-hy4/run.sh DATA [--port N] [--host H] [--ctx C] [--slots S] [--gpu A,B] [--check]"
[ $# -ge 1 ] || { echo "$usage" >&2; exit 1; }
DATA=$(cd "$1" && pwd)
shift
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=8080 HOST=127.0.0.1 CTX=65536 SLOTS=7700 GPU='' CHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT=$2; shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --ctx) CTX=$2; shift 2 ;;
    --slots) SLOTS=$2; shift 2 ;;
    --gpu) GPU=$2; echo "$GPU" | grep -Eq '^[0-9]+,[0-9]+$' && [ "${GPU%,*}" != "${GPU#*,}" ] ||
             { echo "run.sh: --gpu: two different card numbers as nvidia-smi shows them, e.g. 1,2" >&2; exit 1; }; shift 2 ;;
    --check) CHECK=1; shift ;;
    *) echo "run.sh: unknown argument $1" >&2; echo "$usage" >&2; exit 1 ;;
  esac
done
[ -f "$DATA/hy4.env" ] || { echo "run.sh: no $DATA/hy4.env, run setup.sh $DATA first" >&2; exit 1; }
set -a
. "$DATA/hy4.env"
set +a
BIN=$HERE/build/hy4_v1
[ -x "$BIN" ] || { echo "run.sh: no $BIN, run engine-hy4/build.sh first" >&2; exit 1; }
SHARD2=$DATA/Hy4-flash-IQ3_XXS-00002-of-00002.gguf
[ -e "$(readlink -f "$SHARD2")" ] || { echo "run.sh: the Engram table $SHARD2 is not there" >&2; exit 1; }

# the cards: the engine takes the first two CUDA devices; with --gpu or CUDA_VISIBLE_DEVICES they are numbered as nvidia-smi
# numbers them (PCI bus order). A user's 4-card machine wrote 21.6 tok/s on two cards behind one PCIe switch; one card per CPU
# gives each card its own link to memory (Flash-Next on that machine: 45 -> 58 tok/s)
CARDS=
if [ -n "$GPU" ]; then
  export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID} CUDA_VISIBLE_DEVICES=$GPU
  CARDS=$GPU
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
  CARDS=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1-2)
elif command -v nvidia-smi >/dev/null; then
  p100=$( (nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used --format=csv,noheader,nounits || true) | awk -F', ' '/P100/ {print $1, $3, $4}' |
    while read -r i bus used; do
      bdf=$(echo "$bus" | tr 'A-F' 'a-f'); bdf=${bdf: -12}
      echo "$i $used $(cat "/sys/bus/pci/devices/$bdf/numa_node" 2>/dev/null || echo -1)"
    done)
  if [ "$(echo "$p100" | grep -c . || true)" -gt 2 ]; then
    CARDS=$(echo "$p100" | awk 'BEGIN {n = 0}
      {ix[n] = $1; used[n] = $2; node[n] = $3; n++}
      END {best = -1
           for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) {
             s = (node[i] == node[j] || node[i] < 0 ? 1000000 : 0) + (used[i] > used[j] ? used[i] : used[j])
             if (best < 0 || s < best) {best = s; pair = ix[i] "," ix[j]}
           }
           print pair}')
    export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID} CUDA_VISIBLE_DEVICES=$CARDS
    echo "run.sh: $(echo "$p100" | grep -c .) P100 ($(echo "$p100" | awk '{printf "%s%s: NUMA node %s, %s MiB in use", (NR > 1 ? "; " : ""), $1, $3, $2}')); cards $CARDS (--gpu A,B chooses)"
  elif [ -n "$p100" ]; then
    CARDS=$(echo "$p100" | awk '{print $1}' | paste -sd, -)
  fi
fi

# VRAM a card: the engine takes 4.09 GiB + 15,400 bytes a context position + 1.29 MiB a slot (one card's half of an expert), the
# driver and cuBLAS about 1.1 GiB more, so 8100 slots at --ctx 32768 (7700 at 65536) fill a free P100. Another process's VRAM on a
# card (a desktop takes ~1 GiB) comes off the slots; else the engine stops with "out of memory" after a minute of loading
if [ -n "$CARDS" ] && [ "${HY4_VRAM_CHECK:-1}" != 0 ] && command -v nvidia-smi >/dev/null; then
  used=$( (nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$CARDS" || true) | sort -t, -k2 -n | tail -1)
  if [ -n "$used" ]; then
    card=${used%%,*} mib=$(echo "${used#*,}" | tr -d ' ')
    fit=$(( 8100 + ((32768 - CTX) * 15400 - (mib > 32 ? mib - 32 : 0) * 1048576) / 1354752 ))
    if [ "$fit" -lt 3300 ]; then
      echo "run.sh: card $card has $mib MiB in use by other processes (nvidia-smi): room for $(( fit > 0 ? fit : 0 )) slots at" \
           "--ctx $CTX, and the engine needs about 3,300 (its prompt path's scratch lives in them); free the card, or other cards (--gpu)" >&2
      exit 1
    elif [ "$fit" -lt "$SLOTS" ]; then
      echo "run.sh: card $card has $mib MiB in use by other processes (nvidia-smi), so $fit slots instead of $SLOTS at --ctx $CTX" \
           "(a slot holds one card's half of an expert, 1.29 MiB; fewer slots copy more over PCIe while writing; HY4_VRAM_CHECK=0 skips this)"
      SLOTS=$fit
    fi
  fi
fi
if [ "$CHECK" = 1 ]; then
  echo "run.sh: --check: cards ${CARDS:-0,1 of CUDA}, context $CTX, $SLOTS slots a card; nothing started"
  exit 0
fi

SOCK=$DATA/hy4-$PORT.sock
rm -f "$SOCK"   # an old socket would let the wrapper start before the engine is loaded
echo "run.sh: $(date '+%F %T') engine $BIN, cards ${CARDS:-0,1 of CUDA}, context $CTX, $SLOTS slots a card"
"$BIN" --model "$HY4_GGUF" --slots "$SLOTS" --ctx "$CTX" --serve "$SOCK" &
ENGINE=$!
WRAP=
stop() {
  [ -n "$WRAP" ] && kill "$WRAP" 2>/dev/null
  kill "$ENGINE" 2>/dev/null
  wait 2>/dev/null
  exit 0
}
trap stop TERM INT
for _ in $(seq 900); do
  [ -S "$SOCK" ] && break
  kill -0 "$ENGINE" 2>/dev/null || { echo "run.sh: the engine ended while loading (its last lines above; \"out of memory\":" \
    "something else holds VRAM on the cards, see nvidia-smi; use fewer --slots or other cards with --gpu)" >&2; exit 1; }
  sleep 1
done
[ -S "$SOCK" ] || { echo "run.sh: the engine did not open $SOCK in 15 minutes" >&2; stop; }
if [ "${HY4_PREREAD:-1}" != 0 ]; then
  t0=$(date +%s)
  cat "$(readlink -f "$SHARD2")" > /dev/null
  echo "run.sh: Engram table read into the page cache in $(( $(date +%s) - t0 )) s"
fi
"$HY4_PYTHON" "$HERE/serve/hy4_openai.py" "$PORT" "$SOCK" "$HOST" &
WRAP=$!
echo "run.sh: OpenAI-compatible API on http://$HOST:$PORT/v1 (model hy4-flash-iq3xxs)"
wait -n "$ENGINE" "$WRAP" || true
stop
