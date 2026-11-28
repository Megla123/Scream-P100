#include "gguf_min.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace ink2 {

GTypeInfo gtype_info(uint32_t t) {
    switch (t) {
        case G_F32: return {1, 4};
        case G_F16: return {1, 2};
        case G_BF16: return {1, 2};
        case G_I32: return {1, 4};
        case G_Q4_0: return {32, 18};
        case G_Q8_0: return {32, 34};
        case G_IQ4_NL: return {32, 18};
        case G_IQ3_XXS: return {256, 98};
        case G_NVFP4: return {64, 36};
    }
    throw std::runtime_error("gguf_min: tensor type " + std::to_string(t) + " is not known here");
}

size_t GTensor::row_bytes() const {
    const GTypeInfo ti = gtype_info(type);
    return (size_t) (ne[0] / ti.block) * ti.bytes;
}

namespace {
struct Rd {
    const uint8_t* p;
    const uint8_t* end;
    template <class T> T get() {
        if (p + sizeof(T) > end) throw std::runtime_error("gguf_min: header runs past the file");
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (p + n > end) throw std::runtime_error("gguf_min: string runs past the file");
        std::string s((const char*) p, (size_t) n);
        p += n;
        return s;
    }
};
// GGUF value types
enum { V_U8, V_I8, V_U16, V_I16, V_U32, V_I32, V_F32, V_BOOL, V_STR, V_ARR, V_U64, V_I64, V_F64 };
int scalar_bytes(uint32_t t) {
    switch (t) {
        case V_U8: case V_I8: case V_BOOL: return 1;
        case V_U16: case V_I16: return 2;
        case V_U32: case V_I32: case V_F32: return 4;
        case V_U64: case V_I64: case V_F64: return 8;
    }
    return 0;
}
bool is_float(uint32_t t) { return t == V_F32 || t == V_F64; }
uint64_t read_uint(Rd& r, uint32_t t) {
    switch (t) {
        case V_U8: return r.get<uint8_t>();
        case V_I8: return (uint64_t) (int64_t) r.get<int8_t>();
        case V_BOOL: return r.get<uint8_t>();
        case V_U16: return r.get<uint16_t>();
        case V_I16: return (uint64_t) (int64_t) r.get<int16_t>();
        case V_U32: return r.get<uint32_t>();
        case V_I32: return (uint64_t) (int64_t) r.get<int32_t>();
        case V_U64: return r.get<uint64_t>();
        case V_I64: return (uint64_t) r.get<int64_t>();
    }
    throw std::runtime_error("gguf_min: not an integer type");
}
double read_float(Rd& r, uint32_t t) { return t == V_F32 ? (double) r.get<float>() : r.get<double>(); }
}  // namespace

GgufMin::GgufMin(const std::string& first) {
    load(first, true);
    // the other shards: "...-00001-of-0000M.gguf" -> "-0000k-of-0000M"
    const size_t at = first.rfind("-00001-of-");
    if (at != std::string::npos) {
        const int n = std::stoi(first.substr(at + 10, 5));
        for (int k = 2; k <= n; ++k) {
            char num[8];
            std::snprintf(num, sizeof num, "%05d", k);
            std::string p = first;
            p.replace(at + 1, 5, num);
            load(p, false);
        }
    }
}

GgufMin::~GgufMin() {
    for (auto& m : maps_) {
        if (m.base) munmap(m.base, m.size);
        if (m.fd >= 0) close(m.fd);
    }
}

void GgufMin::load(const std::string& path, bool meta) {
    Map m;
    m.fd = open(path.c_str(), O_RDONLY);
    if (m.fd < 0) throw std::runtime_error("gguf_min: cannot open " + path);
    struct stat st{};
    fstat(m.fd, &st);
    m.size = (size_t) st.st_size;
    m.base = mmap(nullptr, m.size, PROT_READ, MAP_SHARED, m.fd, 0);
    if (m.base == MAP_FAILED) throw std::runtime_error("gguf_min: cannot map " + path);
    maps_.push_back(m);
    const uint8_t* base = (const uint8_t*) m.base;
    Rd r{base, base + m.size};
    if (r.get<uint32_t>() != 0x46554747u) throw std::runtime_error("gguf_min: not a GGUF file: " + path);
    const uint32_t ver = r.get<uint32_t>();
    if (ver < 2) throw std::runtime_error("gguf_min: GGUF version 1");
    const uint64_t n_t = r.get<uint64_t>(), n_kv = r.get<uint64_t>();
    uint64_t alignment = 32;
    for (uint64_t k = 0; k < n_kv; ++k) {
        const std::string key = r.str();
        const uint32_t t = r.get<uint32_t>();
        if (t == V_STR) {
            std::string v = r.str();
            if (meta) strs_[key] = v;
        } else if (t == V_ARR) {
            const uint32_t at = r.get<uint32_t>();
            const uint64_t n = r.get<uint64_t>();
            if (at == V_STR) {
                std::vector<std::string> v;
                if (meta) v.reserve(n);
                for (uint64_t i = 0; i < n; ++i) {
                    std::string s = r.str();
                    if (meta) v.push_back(std::move(s));
                }
                if (meta) sarr_[key] = std::move(v);
            } else if (is_float(at)) {
                std::vector<double> v(n);
                for (uint64_t i = 0; i < n; ++i) v[i] = read_float(r, at);
                if (meta) farr_[key] = std::move(v);
            } else if (scalar_bytes(at)) {
                std::vector<uint64_t> v(n);
                for (uint64_t i = 0; i < n; ++i) v[i] = read_uint(r, at);
                if (meta) uarr_[key] = std::move(v);
            } else {
                throw std::runtime_error("gguf_min: nested array in " + key);
            }
        } else if (is_float(t)) {
            const double v = read_float(r, t);
            if (meta) floats_[key] = v;
        } else {
            const uint64_t v = read_uint(r, t);
            if (meta) ints_[key] = (int64_t) v;
            if (key == "general.alignment") alignment = v;
        }
    }
    std::vector<GTensor> ts;
    std::vector<uint64_t> offs;
    for (uint64_t k = 0; k < n_t; ++k) {
        GTensor x;
        x.name = r.str();
        x.n_dims = (int) r.get<uint32_t>();
        for (int d = 0; d < x.n_dims; ++d) x.ne[d] = (int64_t) r.get<uint64_t>();
        x.type = r.get<uint32_t>();
        offs.push_back(r.get<uint64_t>());
        ts.push_back(std::move(x));
    }
    const size_t hdr = (size_t) (r.p - base);
    const size_t data0 = (hdr + alignment - 1) / alignment * alignment;
    for (size_t k = 0; k < ts.size(); ++k) {
        GTensor& x = ts[k];
        const GTypeInfo ti = gtype_info(x.type);
        const int64_t n = x.ne[0] * x.ne[1] * x.ne[2] * x.ne[3];
        x.nbytes = (size_t) (n / ti.block) * ti.bytes;
        x.data = base + data0 + offs[k];
        if (data0 + offs[k] + x.nbytes > m.size) throw std::runtime_error("gguf_min: tensor " + x.name + " runs past " + path);
        tensors_[x.name] = x;
    }
}

const GTensor* GgufMin::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}
const GTensor& GgufMin::t(const std::string& name) const {
    const GTensor* x = find(name);
    if (!x) throw std::runtime_error("gguf_min: no tensor " + name);
    return *x;
}
bool GgufMin::has(const std::string& key) const {
    return ints_.count(key) || floats_.count(key) || strs_.count(key) || uarr_.count(key) || farr_.count(key) || sarr_.count(key);
}
int64_t GgufMin::i(const std::string& key) const {
    auto it = ints_.find(key);
    if (it == ints_.end()) throw std::runtime_error("gguf_min: no integer " + key);
    return it->second;
}
double GgufMin::f(const std::string& key) const {
    auto it = floats_.find(key);
    if (it != floats_.end()) return it->second;
    return (double) i(key);
}
const std::string& GgufMin::s(const std::string& key) const {
    auto it = strs_.find(key);
    if (it == strs_.end()) throw std::runtime_error("gguf_min: no string " + key);
    return it->second;
}
std::vector<uint64_t> GgufMin::ua(const std::string& key) const {
    auto it = uarr_.find(key);
    if (it == uarr_.end()) throw std::runtime_error("gguf_min: no integer array " + key);
    return it->second;
}
std::vector<int64_t> GgufMin::ia(const std::string& key) const {
    std::vector<int64_t> v;
    for (uint64_t x : ua(key)) v.push_back((int64_t) x);
    return v;
}
std::vector<double> GgufMin::fa(const std::string& key) const {
    auto it = farr_.find(key);
    if (it == farr_.end()) throw std::runtime_error("gguf_min: no float array " + key);
    return it->second;
}
const std::vector<std::string>& GgufMin::sa(const std::string& key) const {
    auto it = sarr_.find(key);
    if (it == sarr_.end()) throw std::runtime_error("gguf_min: no string array " + key);
    return it->second;
}

}  // namespace ink2
