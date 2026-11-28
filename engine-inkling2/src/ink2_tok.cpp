// ink2_tok.cpp (2026-11-22, p21): Inkling 2's tokenizer as a process that stays up, for ink2_server.py. llama-tokenize, which
// the server ran once a request, takes 0.9 s to start and read the GGUF's vocabulary even for 28 tokens (measured 22. 11.; 1.5 s
// for 122k tokens). The same llama.cpp code and flags as llama-tokenize --ids --no-escape: the vocabulary only, special tokens
// parsed, BOS as the vocabulary says, and the text AS WRITTEN (llama-tokenize by itself turns a written \n, \t or \\ into the
// character, which broke string literals in source code a prompt carried; the server's calls before p21 did that).
// Protocol: a line with the text's length in bytes, then the bytes; the answer is one line
// "ok ID ID ..." or "err MESSAGE". Start it with CUDA_VISIBLE_DEVICES= (no CUDA context beside the engine).
//   ink2_tok --model Inkling.gguf
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    if (model.empty()) {
        std::fprintf(stderr, "usage: ink2_tok --model Inkling.gguf\n");
        return 2;
    }
    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    llama_model * lm = llama_model_load_from_file(model.c_str(), mp);
    if (!lm) {
        std::printf("err cannot load %s\n", model.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(lm);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::printf("ready %d\n", llama_vocab_n_tokens(vocab));
    std::fflush(stdout);
    std::string line, text;
    std::vector<llama_token> ids;
    while (std::getline(std::cin, line)) {
        size_t n = 0;
        try {
            n = std::stoull(line);
        } catch (...) {
            std::printf("err bad length\n");
            std::fflush(stdout);
            continue;
        }
        text.resize(n);
        if (n && !std::cin.read(&text[0], (std::streamsize) n)) break;
        ids.resize(text.size() + 16);
        int got = llama_tokenize(vocab, text.data(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), add_bos, true);
        if (got < 0) {   // the buffer was too small: -got tokens are needed
            ids.resize((size_t) -got);
            got = llama_tokenize(vocab, text.data(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), add_bos, true);
        }
        if (got < 0) {
            std::printf("err tokenize %d\n", got);
            std::fflush(stdout);
            continue;
        }
        std::string out = "ok";
        out.reserve((size_t) got * 7 + 8);
        for (int i = 0; i < got; ++i) {
            out += ' ';
            out += std::to_string(ids[i]);
        }
        out += '\n';
        std::fwrite(out.data(), 1, out.size(), stdout);
        std::fflush(stdout);
    }
    llama_model_free(lm);
    llama_backend_free();
    return 0;
}
