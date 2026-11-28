#!/bin/bash
# engine-inkling2/build.sh: builds the Inkling 2 Mini engine (build/ink2_v1, one CUDA program for two Tesla P100, sm_60), its
# one-card build (build/ink2_1c from src/ with -DINK2_NC=1 since 26 Nov, src-1c/ is the older one-card source: one P100, the
# experts that do not fit in VRAM in RAM; run.sh INK2_CARDS=1), the
# two test builds for several conversations at once (build/ink2_tpb from src-tpb/: production's layout with conversation slots
# and batches; build/ink2_pp from src-pp/: a pipeline, half of the layers on each card; run.sh INK2_MULTI=tpb|pp) and,
# from llama.cpp at commit 690b87d on the CPU (no CUDA in that build): llama-tokenize and gguf-py for the server's tokenizer,
# libmtmd for build/ink2_vis, the picture encoder (the vision projector runs on the CPU; the cards are full of the engine),
# and build/ink2_tok, the same tokenizer as a process that stays up (llama-tokenize takes 0.9 s to start for every request).
# Needs CUDA 12.x (nvcc, cuBLAS), a host compiler nvcc accepts (CUDA 12.4: g++ <= 13; set HOST_CXX=g++-13 if your default is
# newer), cmake, curl. About 1 minute for each engine and 3-5 for the llama.cpp part. A build from before pictures (no
# libmtmd yet) gets only the missing part.
set -euo pipefail
cd "$(dirname "$0")"
H=$(pwd); B=$H/build; mkdir -p "$B"
HOST_CXX=${HOST_CXX:-g++}
echo "== ink2_v1 (nvcc, host compiler $HOST_CXX)"
(cd src && nvcc -O2 -std=c++17 -arch=sm_60 -ccbin "$HOST_CXX" -o "$B/ink2_v1" ink2_v1.cu gguf_min.cpp -lcublas)
echo "== ink2_1c (one P100, nvcc, host compiler $HOST_CXX)"
(cd src && nvcc -O2 -std=c++17 -arch=sm_60 -ccbin "$HOST_CXX" -DINK2_NC=1 -DINK2_RING=2048 -o "$B/ink2_1c" ink2_v1.cu gguf_min.cpp -lcublas)
echo "== ink2_tpb (several conversations at once: slots and batches, nvcc, host compiler $HOST_CXX)"
(cd src-tpb && nvcc -O2 -std=c++17 -arch=sm_60 -ccbin "$HOST_CXX" -I../src -o "$B/ink2_tpb" ink2_tpb.cu ../src/gguf_min.cpp -lcublas -lpthread)
echo "== ink2_pp (several conversations at once: a pipeline over the two cards, nvcc, host compiler $HOST_CXX)"
(cd src-pp && nvcc -O2 -std=c++17 -arch=sm_60 -ccbin "$HOST_CXX" -DINK2_RING=2048 -I../src -o "$B/ink2_pp" ink2_pp.cu ../src/gguf_min.cpp   -lcublas -lpthread)
LC=690b87d; LB=$B/llama.cpp
if [ ! -f "$LB/CMakeLists.txt" ]; then
  echo "== llama.cpp $LC: sources"
  rm -rf "$LB"; mkdir -p "$LB"
  curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/$LC.tar.gz" | tar xz -C "$LB" --strip-components=1
fi
if [ ! -x "$LB/build/bin/llama-tokenize" ] || [ ! -e "$LB/build/bin/libmtmd.so" ]; then
  echo "== llama.cpp $LC: llama-tokenize and libmtmd (CPU)"
  cmake -S "$LB" -B "$LB/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > "$B/llama.cpp.cmake.log"
  cmake --build "$LB/build" --target llama-tokenize mtmd -j "$(nproc)" > "$B/llama.cpp.build.log"
fi
echo "== ink2_vis (pictures, $HOST_CXX)"
"$HOST_CXX" -O2 -std=c++17 -I"$LB/include" -I"$LB/ggml/include" -I"$LB/tools/mtmd" -o "$B/ink2_vis" src/ink2_vis.cpp \
  -L"$LB/build/bin" -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,"$LB/build/bin"
echo "== ink2_tok (the tokenizer that stays up, $HOST_CXX)"
"$HOST_CXX" -O2 -std=c++17 -I"$LB/include" -I"$LB/ggml/include" -o "$B/ink2_tok" src/ink2_tok.cpp \
  -L"$LB/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LB/build/bin"
echo "done: $B/ink2_v1, $B/ink2_1c, $B/ink2_tpb, $B/ink2_pp, $B/ink2_vis, $B/ink2_tok, $LB/build/bin/llama-tokenize"
