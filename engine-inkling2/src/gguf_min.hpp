// gguf_min (from Hy4 v0, 2026-11-20; Inkling 2: NVFP4 added 2026-11-21): a small read-only GGUF reader. Maps every "-0000N-of-0000M" shard beside the first,
// indexes tensors by name and keeps the first shard's metadata (integers, floats, strings, number arrays). Nothing is
// converted: the split engine's own reader re-quantizes tensors by Flash-Next names, which on Hy4 would touch the
// 35 GB Engram table.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ink2 {

enum GType : uint32_t {
    G_F32 = 0, G_F16 = 1, G_Q4_0 = 2, G_Q8_0 = 8, G_IQ3_XXS = 18, G_IQ4_NL = 20, G_BF16 = 30, G_I32 = 26, G_NVFP4 = 40,
};
struct GTypeInfo { int block, bytes; };
GTypeInfo gtype_info(uint32_t t);   // throws on a type this reader does not know

struct GTensor {
    std::string name;
    uint32_t type = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    int n_dims = 0;
    const uint8_t* data = nullptr;   // in the mapping
    size_t nbytes = 0;
    size_t row_bytes() const;        // bytes of one ne0 row
};

class GgufMin {
public:
    explicit GgufMin(const std::string& first_shard);
    ~GgufMin();
    GgufMin(const GgufMin&) = delete;
    GgufMin& operator=(const GgufMin&) = delete;
    const GTensor& t(const std::string& name) const;   // throws when missing
    const GTensor* find(const std::string& name) const;
    int64_t i(const std::string& key) const;
    double f(const std::string& key) const;
    const std::string& s(const std::string& key) const;
    std::vector<int64_t> ia(const std::string& key) const;   // integer or bool arrays (u64 values above 2^63 wrap)
    std::vector<double> fa(const std::string& key) const;
    std::vector<uint64_t> ua(const std::string& key) const;  // the same array as unsigned
    bool has(const std::string& key) const;
    const std::vector<std::string>& sa(const std::string& key) const;   // string arrays (the tokenizer's tokens)
private:
    struct Map { void* base = nullptr; size_t size = 0; int fd = -1; };
    std::vector<Map> maps_;
    std::map<std::string, GTensor> tensors_;
    std::map<std::string, int64_t> ints_;
    std::map<std::string, double> floats_;
    std::map<std::string, std::string> strs_;
    std::map<std::string, std::vector<uint64_t>> uarr_;
    std::map<std::string, std::vector<double>> farr_;
    std::map<std::string, std::vector<std::string>> sarr_;
    void load(const std::string& path, bool meta);
};

}  // namespace ink2
