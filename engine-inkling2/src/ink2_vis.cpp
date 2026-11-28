// ink2_vis.cpp (2026-11-21, p17): Inkling 2's image encoder for ink2_server.py, through llama.cpp's mtmd (the mmproj GGUF, its
// preprocessing and ViT, the same code llama-server runs). Loads the text model's vocabulary only (mtmd wants the model for
// its input width) and the projector, then reads lines "IMAGE_PATH OUT_PATH" on stdin; for each it writes the image's rows
// (n_tokens x n_embd f32) to OUT_PATH and prints "ok N_TOKENS N_EMBD MS", or "err MESSAGE". Runs on the CPU unless --gpu
// with a CUDA backend available. CUDA_VISIBLE_DEVICES selects the card for this process.
#include "ggml-backend.h"
//   ink2_vis --mmproj M.gguf --model Inkling.gguf [--threads N] [--gpu] [--max-tokens N]
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

int main(int argc, char ** argv) {
    std::string mmproj, model;
    int threads = (int) std::thread::hardware_concurrency(), max_tokens = 0;
    bool gpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--mmproj") mmproj = nx();
        else if (a == "--model") model = nx();
        else if (a == "--threads") threads = std::stoi(nx());
        else if (a == "--max-tokens") max_tokens = std::stoi(nx());
        else if (a == "--gpu") gpu = true;
    }
    if (mmproj.empty() || model.empty()) {
        std::fprintf(stderr, "usage: ink2_vis --mmproj M.gguf --model Inkling.gguf [--threads N] [--gpu] [--max-tokens N]\n");
        return 2;
    }
    llama_backend_init();
    ggml_backend_dev_t device = gpu ? ggml_backend_dev_by_name("CUDA0") : nullptr;
    if (gpu && !device) {
        std::fprintf(stderr, "ink2_vis: --gpu requires a visible CUDA device and a CUDA-enabled helper build\n");
        return 1;
    }
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    llama_model * lm = llama_model_load_from_file(model.c_str(), mp);
    if (!lm) {
        std::fprintf(stderr, "ink2_vis: cannot load %s\n", model.c_str());
        return 1;
    }
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = gpu;
    cp.device = device;
    cp.n_threads = threads;
    cp.print_timings = false;
    cp.warmup = false;
    if (max_tokens > 0) cp.image_max_tokens = max_tokens;
    mtmd_context * ctx = mtmd_init_from_file(mmproj.c_str(), lm, cp);
    if (!ctx || !mtmd_support_vision(ctx)) {
        std::fprintf(stderr, "ink2_vis: cannot load the projector %s\n", mmproj.c_str());
        return 1;
    }
    int n_embd = llama_model_n_embd_inp(lm);
    if (n_embd <= 0) n_embd = 2304;   // the vocabulary-only load has no hparams width; Inkling 2 Mini's embedding_length
    std::printf("ready %d\n", n_embd);
    std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string img, out;
        in >> img >> out;
        if (img.empty() || out.empty()) {
            std::printf("err bad line\n");
            std::fflush(stdout);
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, img.c_str(), false, mtmd_helper_init_opt_default());
        if (!bw.bitmap) {
            std::printf("err cannot read image %s\n", img.c_str());
            std::fflush(stdout);
            continue;
        }
        const char * marker = mtmd_default_marker();
        mtmd_input_text text{marker, std::strlen(marker), false, true};
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        const mtmd_bitmap * bm = bw.bitmap;
        int32_t rc = mtmd_tokenize(ctx, chunks, &text, &bm, 1);
        size_t n_tok = 0;
        bool done = false;
        for (size_t c = 0; rc == 0 && c < mtmd_input_chunks_size(chunks) && !done; ++c) {
            const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, c);
            if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
            n_tok = mtmd_input_chunk_get_n_tokens(ch);
            rc = mtmd_encode_chunk(ctx, ch);
            if (rc != 0) break;
            const float * e = mtmd_get_output_embd(ctx);
            FILE * f = std::fopen(out.c_str(), "wb");
            if (!f || std::fwrite(e, sizeof(float), n_tok * (size_t) n_embd, f) != n_tok * (size_t) n_embd) rc = 3;
            if (f) std::fclose(f);
            done = true;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (rc == 0 && done) std::printf("ok %zu %d %.0f\n", n_tok, n_embd, ms);
        else std::printf("err encode rc %d\n", rc);
        std::fflush(stdout);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
    }
    mtmd_free(ctx);
    llama_model_free(lm);
    return 0;
}
