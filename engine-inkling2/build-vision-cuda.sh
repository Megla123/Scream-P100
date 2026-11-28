#!/bin/bash
# Build only the CUDA projector helper from the llama.cpp source prepared by build.sh.
set -euo pipefail
cd "$(dirname "$0")"
H=$(pwd)
LB="$H/build/llama.cpp"
OUT="$H/build/llama-vision-cuda"
HOST_CXX=${HOST_CXX:-g++}

[ -f "$LB/CMakeLists.txt" ] || { echo "Run build.sh --helpers-only first to prepare the pinned llama.cpp source." >&2; exit 2; }
cmake -S "$LB" -B "$OUT" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$HOST_CXX" -DCMAKE_CUDA_HOST_COMPILER="$HOST_CXX" \
  -DCMAKE_CUDA_ARCHITECTURES="${INK2_VISION_CUDA_ARCHITECTURES:-60;61}" \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DGGML_CUDA=ON -DGGML_BACKEND_DL=OFF -DLLAMA_CURL=OFF
cmake --build "$OUT" --target mtmd -j "${INK2_VISION_BUILD_JOBS:-1}"
"$HOST_CXX" -O2 -std=c++17 -I"$LB/include" -I"$LB/ggml/include" -I"$LB/tools/mtmd" \
  -o "$H/build/ink2_vis_cuda" "$H/src/ink2_vis.cpp" \
  -L"$OUT/bin" -lmtmd -lllama -lggml -lggml-base '-Wl,-rpath,$ORIGIN/llama-vision-cuda/bin'
