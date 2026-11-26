#!/bin/bash
# engine-inkling2/setup.sh DATA [--no-pictures]: downloads Inkling 2 Mini (Xylolabs's NVFP4-Q8_0 GGUF, 17.6 GB), its DSpark
# drafter (Unsloth's Q8_0 GGUF, 0.4 GB) and, for pictures, Xylolabs's vision projector (BF16 GGUF, 0.64 GB; not with
# --no-pictures) into DATA, checks each by SHA-256, and makes a Python venv for the server (numpy and what llama.cpp's gguf-py
# imports: pyyaml, tqdm, requests; DATA/venv). Resumes a broken download. About 19 GB free in DATA.
set -euo pipefail
D=${1:?usage: setup.sh DATA [--no-pictures]}; PICTURES=1
[ "${2:-}" = --no-pictures ] && PICTURES=0
mkdir -p "$D"; cd "$D"
get() {   # url file sha256
  if [ -f "$2" ] && [ "$(sha256sum "$2" | cut -c1-64)" = "$3" ]; then echo "ok   $2"; return; fi
  echo "get  $2"
  curl -fL -C - -o "$2" "$1"
  [ "$(sha256sum "$2" | cut -c1-64)" = "$3" ] || { echo "SHA-256 of $2 does not match" >&2; exit 1; }
}
# Both files were renamed on Hugging Face on 22-23 Nov 2026 (the same bytes: the SHA-256 below are the ones of 21 Nov). They
# keep their first names in DATA, which run.sh uses. Unsloth's dspark-Inkling-2-Mini-Q8_0.gguf of today is another file
get https://huggingface.co/Xylolabs/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-NVFP4-Lite.gguf \
    Inkling-2-Mini-NVFP4-Q8_0.gguf d4fcaf47c7f3efbbe057bcf2e516076b7a517017308284b2032c92851edfbd37
get https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/dspark-Inkling-2-Mini-Q8_0_Old.gguf \
    dspark-Inkling-2-Mini-Q8_0.gguf 2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773
if [ $PICTURES = 1 ]; then
  get https://huggingface.co/Xylolabs/Inkling-2-Mini-GGUF/resolve/main/mmproj-Inkling-2-Mini-BF16.gguf \
      mmproj-Inkling-2-Mini-BF16.gguf 73b64fc98003b6fd5dca85dae26da3fe292d6075978138d319bdcbe4528f5b15
fi
if [ ! -x venv/bin/python ] || ! venv/bin/python -c "import numpy, yaml, tqdm, requests" 2>/dev/null; then
  [ -x venv/bin/python ] || python3 -m venv venv
  venv/bin/pip install -q numpy pyyaml tqdm requests   # what llama.cpp's gguf-py imports
fi
echo "done: $D (run: bash engine-inkling2/run.sh $D)"
