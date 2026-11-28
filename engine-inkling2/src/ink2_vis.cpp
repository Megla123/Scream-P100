// ink2_vis.cpp (2026-11-21, p17): Inkling 2's image encoder for ink2_server.py, through llama.cpp's mtmd (the mmproj GGUF, its
// preprocessing and ViT, the same code llama-server runs). Loads the text model's vocabulary only (mtmd wants the model for
// its input width) and the projector, then reads lines "IMAGE_PATH OUT_PATH" on stdin; for each it writes the image's rows
// (n_tokens x n_embd f32) to OUT_PATH and prints "ok N_TOKENS N_EMBD MS", or "err MESSAGE". Runs on the CPU unless --gpu
// with a CUDA backend available. CUDA_VISIBLE_DEVICES selects the card for this process.
//   ink2_vis --mmproj M.gguf --model Inkling.gguf [--threads N] [--gpu] [--max-tokens N]
// --strata-protocol serves ENC commands and writes Strata's SVE1 image records.
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

static int inkling_projector_width(const std::string & path) {
    ggml_context * tensors = nullptr;
    gguf_context * metadata = gguf_init_from_file(path.c_str(), {true, &tensors});
    if (!metadata) return 0;
    int64_t key = gguf_find_key(metadata, "clip.projector_type");
    if (key < 0) key = gguf_find_key(metadata, "clip.vision.projector_type");
    int width = 0;
    if (key >= 0 && gguf_get_kv_type(metadata, key) == GGUF_TYPE_STRING &&
        std::strcmp(gguf_get_val_str(metadata, key), "inkling2v") == 0) {
        const ggml_tensor * output = ggml_get_tensor(tensors, "mm.2.weight");
        if (output && output->ne[1] > 0 && output->ne[1] <= INT32_MAX) width = (int) output->ne[1];
    }
    ggml_free(tensors);
    gguf_free(metadata);
    return width;
}

int main(int argc, char ** argv) {
    std::string mmproj, model;
    int threads = (int) std::thread::hardware_concurrency(), max_tokens = 0, min_tokens = 0;
    bool gpu = false, strata = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--mmproj") mmproj = nx();
        else if (a == "--model") model = nx();
        else if (a == "--threads") threads = std::stoi(nx());
        else if (a == "--max-tokens") max_tokens = std::stoi(nx());
        else if (a == "--min-tokens") min_tokens = std::stoi(nx());
        else if (a == "--gpu") gpu = true;
        else if (a == "--strata-protocol") strata = true;
    }
    if (mmproj.empty() || model.empty()) {
        std::fprintf(stderr, "usage: ink2_vis --mmproj M.gguf --model Inkling.gguf [--threads N] [--gpu] [--max-tokens N] [--min-tokens N] [--strata-protocol]\n");
        return 2;
    }
    // The vocab-only loader skips the text dimensions. Read the actual Inkling projection width
    // before copying output rows so a mismatched projector cannot cause an out-of-bounds read.
    const int n_embd = inkling_projector_width(mmproj);
    if (n_embd <= 0 || (!strata && n_embd != 2304)) {
        std::fprintf(stderr, "ink2_vis: incompatible Inkling projector %s\n", mmproj.c_str());
        return 1;
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
    if (min_tokens > 0) cp.image_min_tokens = min_tokens;
    mtmd_context * ctx = mtmd_init_from_file(mmproj.c_str(), lm, cp);
    if (!ctx || !mtmd_support_vision(ctx)) {
        std::fprintf(stderr, "ink2_vis: cannot load the projector %s\n", mmproj.c_str());
        return 1;
    }
    std::printf(strata ? "READY %d mtmd\n" : "ready %d\n", n_embd);
    std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string img, out;
        if (strata) {
            std::string cmd;
            in >> cmd;
            if (cmd == "QUIT") break;
            if (cmd != "ENC") {
                std::puts("ERR expected ENC or QUIT");
                std::fflush(stdout);
                continue;
            }
        }
        in >> img >> out;
        if (img.empty() || out.empty()) {
            std::puts(strata ? "ERR bad line" : "err bad line");
            std::fflush(stdout);
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, img.c_str(), false, mtmd_helper_init_opt_default());
        if (!bw.bitmap) {
            std::printf(strata ? "ERR cannot read the image %s\n" : "err cannot read image %s\n", img.c_str());
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
        size_t grid_x = 0, grid_y = 0;
        for (size_t c = 0; rc == 0 && c < mtmd_input_chunks_size(chunks) && !done; ++c) {
            const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, c);
            if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
            n_tok = mtmd_input_chunk_get_n_tokens(ch);
            if (strata) {
                const auto * tokens = mtmd_input_chunk_get_tokens_image(ch);
                grid_x = mtmd_image_tokens_get_nx(tokens);
                grid_y = mtmd_image_tokens_get_ny(tokens);
                if (!grid_x || !grid_y || n_tok > INT32_MAX || grid_x * grid_y != n_tok ||
                    (max_tokens > 0 && n_tok > (size_t) max_tokens)) {
                    rc = 4;
                    break;
                }
                for (size_t rest = c + 1; rest < mtmd_input_chunks_size(chunks); ++rest) {
                    if (mtmd_input_chunk_get_type(mtmd_input_chunks_get(chunks, rest)) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                        rc = 4;
                        break;
                    }
                }
                if (rc != 0) break;
            }
            rc = mtmd_encode_chunk(ctx, ch);
            if (rc != 0) break;
            const float * e = mtmd_get_output_embd(ctx);
            FILE * f = std::fopen(out.c_str(), "wb");
            if (f && strata) {
                const int32_t header[] = {0x31455653, (int32_t) n_tok, (int32_t) grid_x, (int32_t) grid_y, n_embd};
                if (std::fwrite(header, sizeof(header), 1, f) != 1) rc = 3;
            }
            if (!f || (rc == 0 && std::fwrite(e, sizeof(float), n_tok * (size_t) n_embd, f) != n_tok * (size_t) n_embd)) rc = 3;
            if (f && std::fclose(f) != 0) rc = 3;
            done = true;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (rc == 0 && done) {
            if (strata) std::printf("OK %zu %zu %zu %.0f\n", n_tok, grid_x, grid_y, ms);
            else std::printf("ok %zu %d %.0f\n", n_tok, n_embd, ms);
        } else {
            std::printf(strata ? "ERR encode rc %d\n" : "err encode rc %d\n", rc);
        }
        std::fflush(stdout);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
    }
    mtmd_free(ctx);
    llama_model_free(lm);
    return 0;
}
