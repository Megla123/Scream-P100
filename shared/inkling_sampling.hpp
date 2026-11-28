// Host sampler from Strata v0.2.22 (src/inkling2/ink_serve.h), shared by both Inkling paths.
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors
#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <locale>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ink2 {

struct SampleOpts {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 64;
    float min_p = 0.0f;
    float rep = 1.0f;
    float freq = 0.0f;
    float pres = 0.0f;
    int last_n = 64;
    uint64_t seed = 0;
    int hold_budget = 0;       // hold=N:ID reasoning budget
    int32_t hold_token_id = -1;
    int pen_from = 0;

    bool greedy() const noexcept { return temperature <= 0.0f; }
    bool penalties() const noexcept { return rep != 1.0f || freq != 0.0f || pres != 0.0f; }
};

// Read the nine numeric settings shared by native startup and the Megla request protocol.
inline SampleOpts parse_sampling_fields(const std::string& fields, int max_top_k) {
    SampleOpts s;
    if (fields.empty()) return s;
    std::istringstream input(fields);
    input.imbue(std::locale::classic());
    std::string seed, extra;
    if (!(input >> s.temperature >> s.top_p >> s.top_k >> s.min_p >> s.rep >> s.freq >> s.pres >> s.last_n >> seed)
            || (input >> extra)) {
        throw std::runtime_error("invalid sampling fields");
    }
    const auto [end, error] = std::from_chars(seed.data(), seed.data() + seed.size(), s.seed);
    if (error != std::errc{} || end != seed.data() + seed.size()
            || !std::isfinite(s.temperature) || s.temperature < 0.0f
            || !(s.top_p > 0.0f && s.top_p <= 1.0f) || s.top_k < 1 || s.top_k > max_top_k
            || !(s.min_p >= 0.0f && s.min_p <= 1.0f)
            || !std::isfinite(s.rep) || s.rep <= 0.0f
            || !(s.freq >= -2.0f && s.freq <= 2.0f) || !(s.pres >= -2.0f && s.pres <= 2.0f)
            || s.last_n < 0) {
        throw std::runtime_error("sampling fields out of range");
    }
    return s;
}

// The K best token ids of lg[0..V-1], best first: larger logit first, the smaller id first on equal logits.  Exactly what
// std::partial_sort over an iota with that order leaves in idx[0..K-1], but one pass over the row (about 0.1 ms for V = 200058
// instead of a few ms), because every sampled token of every slot pays it on the host (#6779: 16..64 slots, temperature > 0).
inline void ink_top_k_ids(const float* lg, int V, int K, std::vector<int>& idx) {
    K = std::max(0, std::min(K, V));
    idx.assign((size_t)K, 0);
    if (K == 0) return;
    std::vector<float> val((size_t)K);
    int n = 0;
    for (int i = 0; i < V; ++i) {
        const float v = lg[i];
        if (n == K && !(v > val[K - 1])) continue;   // a later id with an equal logit sorts after the kept ones
        int j = (n < K) ? n++ : K - 1;
        while (j > 0 && v > val[j - 1]) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; --j; }
        val[j] = v;
        idx[j] = i;
    }
}

// The sampling step once the K best candidates are known: tv[i] is the logit of token ti[i], best first (larger logit first, the
// smaller id first on equal logits), K = min(top_k, vocab, 128).  Shared by sample_logits (candidates found on the host) and the
// candidate rows a backend returns (forward_batch_cand), so both give the same token for the same seed.
inline int32_t ink_sample_top(const float* tv, const int* ti, int K, const SampleOpts& s, std::mt19937_64& rng) {
    std::vector<double> p(K);
    double mx = tv[0], sum = 0.0;
    for (int i = 0; i < K; ++i) {
        p[i] = std::exp((tv[i] - mx) / static_cast<double>(s.temperature));
        sum += p[i];
    }
    for (double& x : p) x /= sum;
    int keep = K;
    if (s.min_p > 0.0f) {
        const double thr = static_cast<double>(s.min_p) * p[0];
        keep = 1;
        while (keep < K && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.0f) {
        double c = 0.0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= static_cast<double>(s.top_p)) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0.0;
    for (int i = 0; i < keep; ++i) z += p[i];
    std::uniform_real_distribution<double> u(0.0, z);
    const double r = u(rng);
    double c = 0.0;
    for (int i = 0; i < keep; ++i) {
        c += p[i];
        if (r <= c) return ti[i];
    }
    return ti[keep - 1];
}

// Sampling from a candidate row (ink2::CandRows, `cand` = the row's kInkCandidates entries).  Only for a slot without penalties:
// a penalty changes logits anywhere in the row before the best are picked, so such a slot needs its full row.
inline int32_t sample_candidates(const float* cv, const int32_t* ci, int V, const SampleOpts& s, std::mt19937_64& rng) {
    if (s.greedy()) return ci[0];
    const int K = std::min(std::max(1, s.top_k), std::min(V, 128));
    return ink_sample_top(cv, ci, K, s, rng);
}

// Replay exactly the RNG consumption of single-target sampling or prompt lookup at a serving handoff.
inline void skip_target_draws(std::mt19937_64& rng, size_t count) {
    std::uniform_real_distribution<double> draw;
    for (size_t i = 0; i < count; ++i) (void)draw(rng);
}

inline int32_t sample_logits(float* lg, int V, const SampleOpts& s, const std::vector<int32_t>& hist, std::mt19937_64& rng,
                             size_t history_end = std::numeric_limits<size_t>::max()) {
    if (s.penalties() && s.last_n > 0) {
        size_t start = static_cast<size_t>(std::max(0, s.pen_from));
        const size_t end = std::min(hist.size(), history_end);
        if (end > start) {
            const size_t from = end - std::min(end - start, static_cast<size_t>(s.last_n));
            std::unordered_map<int32_t, int> cnt;
            for (size_t i = from; i < end; ++i) ++cnt[hist[i]];
            for (const auto& kv : cnt) {
                if (kv.first < 0 || kv.first >= V) continue;
                float& x = lg[kv.first];
                if (s.rep != 1.0f) x = (x > 0.0f) ? (x / s.rep) : (x * s.rep);
                x -= static_cast<float>(kv.second) * s.freq + (s.pres != 0.0f ? s.pres : 0.0f);
            }
        }
    }
    if (s.greedy()) {
        int32_t best = 0;
        float mx = lg[0];
        for (int i = 1; i < V; ++i) {
            if (lg[i] > mx) { mx = lg[i]; best = i; }
        }
        return best;
    }
    const int K = std::min(std::max(1, s.top_k), std::min(V, 128));
    std::vector<int> idx;
    ink_top_k_ids(lg, V, K, idx);
    std::vector<float> tv((size_t)K);
    for (int i = 0; i < K; ++i) tv[(size_t)i] = lg[idx[(size_t)i]];
    return ink_sample_top(tv.data(), idx.data(), K, s, rng);
}

inline bool finite_logits(const float* lg, int V) noexcept {
    for (int i = 0; i < V; i += 67) if (!std::isfinite(lg[i])) return false;
    return true;
}

// Every row makes one target draw. A matching draft can be reused without changing the draw
// sequence. Stop at the first mismatch or terminal token. Keep the final sampled token pending.
template <typename Select, typename KeepDraft>
int sample_draft_prefix(const int* window, int rows, int& next, Select&& select, KeepDraft&& keep_draft) {
    int row = 0;
    next = select(row);
    while (row + 1 < rows && next == window[row + 1] && keep_draft(row + 1)) {
        next = select(++row);
    }
    return row;
}

}  // namespace ink2
