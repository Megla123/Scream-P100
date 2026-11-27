// src/stop_iso.hpp - a LEARNED STOP RULE for the MTP draft chain (FNX_DRAFT_STOP_ISO=<tables file>; off by default).
//
// Port of Keoian's LLAMA_SPEC_STOP_ISO (dual-P100 Qwen3.8-27B llama.cpp branch, CHANGES section 22, 2026-10-24).
// Draft d of a chain is kept while the estimated chance that the whole drafted prefix through d is accepted,
//
//     S(d) = prod_{i<d} iso_q(q_i) * iso_p1(p1_d),
//
// stays at or above theta; the first draft below it ends the chain (and is not verified).
//   q_i    the drafter's probability of the token it drew at step i (its argmax's probability when it drafts the argmax)
//   p1_d   the top probability of step d's distribution, known before step d's token is drawn
//   iso_q, iso_p1   isotonic (non-decreasing) maps fit offline on traced verify rounds: the target's acceptance of a
//          draft against q and against p1. The acceptance a tables file is fit on depends on the verify rule:
//          min(1, p/q) under speculative sampling, p(d) under exact-match sampling, 1{d = argmax} at temperature 0.
//
// EXACT. The decision about draft d reads step d's distribution (p1_d) and the tokens drawn before d, never the token
// drawn at d: the chain's length is a stopping time of the drafting process, which the speculative-sampling proof
// allows, so the output keeps the target's distribution (per-token and block verification alike). A rule that tested
// the drawn token's own q at its own step would bias the p/q rule; that is why q_d only enters from step d + 1
// (Chain::keep() before Chain::drew()). Under exact-match verification any rule is exact; the order matters there only
// for speed.
//
// The tables are per model, per draft layer and per sampling regime (the owner's 27B needed a second pair for thinking
// requests), not per machine; theta is per machine (what a verify row costs). Header only, no engine types
// (session.cpp reads FNX_DRAFT_STOP_ISO; tune/stop_iso_fit.py writes the file from an FNX_DRAFT_TRACE trace).
//
// Tables file (text, written by tune/stop_iso_fit.py; '#' starts a comment):
//     stop_iso 1
//     set <name> theta <theta> temp <lo> <hi> [top_p <lo> <hi>] [top_k <lo> <hi>] [presence <lo> <hi>]
//                                                  a table pair for requests whose temperature (and, where given, top-p,
//                                                  top-k with 0 = no cut, presence penalty) lie in those ranges
//     x  <k values, increasing>                     the knots of both maps (2 <= k <= 64)
//     q  <k values in [0, 1], non-decreasing>       iso_q at the knots
//     p1 <k values in [0, 1], non-decreasing>       iso_p1 at the knots
//     (more sets; the FIRST set whose ranges hold the request's sampling is used, none: the engine's own rule)
// Between knots the maps are linear, outside them constant (the owner's spec_iso_lookup).
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace fnx::stop_iso {

/// A piecewise-linear map on knots x (increasing) with values y; constant outside [x.front(), x.back()].
struct Map {
    std::vector<float> x, y;
    float operator()(float v) const {
        if (x.empty()) return 0.0f;
        if (!(v > x.front())) return y.front();   // (a NaN reads as the first knot)
        for (size_t j = 1; j < x.size(); ++j)
            if (v <= x[j]) {
                const float t = (v - x[j - 1]) / (x[j] - x[j - 1]);
                return y[j - 1] + t * (y[j] - y[j - 1]);
            }
        return y.back();
    }
};

/// One table pair, its threshold and the request temperatures it serves.
struct Set {
    std::string name;
    float theta = 0.0f;
    float t_lo = 0.0f, t_hi = 0.0f;
    float p_lo = -1e30f, p_hi = 1e30f, k_lo = -1e30f, k_hi = 1e30f, pr_lo = -1e30f, pr_hi = 1e30f;   // unset: any
    Map q, p1;
    bool serves(float temperature, float top_p = 1.0f, float top_k = 0.0f, float presence = 0.0f) const {
        return temperature >= t_lo && temperature <= t_hi && top_p >= p_lo && top_p <= p_hi && top_k >= k_lo && top_k <= k_hi &&
               presence >= pr_lo && presence <= pr_hi;
    }
};

struct Rule {
    std::vector<Set> sets;
    std::string path;
    bool on() const { return !sets.empty(); }
    /// The set for a request's sampling (the first whose ranges hold it); nullptr: the engine's own rule.
    const Set* pick(float temperature, float top_p = 1.0f, float top_k = 0.0f, float presence = 0.0f) const {
        for (const Set& s : sets)
            if (s.theta > 0.0f && s.serves(temperature, top_p, top_k, presence)) return &s;
        return nullptr;
    }
    /// Every set's theta replaced (a sweep: <switch>_THETA); theta <= 0 leaves them.
    void set_theta(float theta) {
        if (theta > 0.0f)
            for (Set& s : sets) s.theta = theta;
    }
    /// "chat theta 0.125 temp 0.001..1e+09, think theta 0.15 temp 0.95..1.05" for the engine's start line.
    std::string describe() const {
        std::string out;
        char b[160];
        for (const Set& s : sets) {
            std::snprintf(b, sizeof b, "%s%s theta %.3g temp %.3g..%.3g", out.empty() ? "" : ", ", s.name.c_str(),
                          (double) s.theta, (double) s.t_lo, (double) s.t_hi);
            out += b;
            const struct { const char* k; float lo, hi; } extra[] = {{"top_p", s.p_lo, s.p_hi}, {"top_k", s.k_lo, s.k_hi}, {"presence", s.pr_lo, s.pr_hi}};
            for (const auto& e : extra)
                if (e.lo > -1e29f || e.hi < 1e29f) {
                    std::snprintf(b, sizeof b, " %s %.3g..%.3g", e.k, (double) e.lo, (double) e.hi);
                    out += b;
                }
            std::snprintf(b, sizeof b, " (%zu knots)", s.q.x.size());
            out += b;
        }
        return out;
    }
};

/// One chain's state. keep() decides about the draft just computed from its distribution's top probability; drew()
/// then takes the drawn token's own probability into the survival for the drafts after it.
class Chain {
public:
    Chain() = default;
    explicit Chain(const Set* set) : set_(set) {}
    bool on() const { return set_ != nullptr; }
    /// Step d's distribution is known (top probability p1): verify the token drawn from it?  Reads only the drafts
    /// before d.  Off (no set): always true.
    bool keep(float p1) const { return set_ == nullptr || !(s_ * set_->p1(p1) < set_->theta); }
    /// The token drawn at step d, kept, had probability q under the drafter's distribution.
    void drew(float q) {
        if (set_ != nullptr) s_ *= set_->q(q);
    }
    float survival() const { return s_; }

private:
    const Set* set_ = nullptr;
    float s_ = 1.0f;   // float, as the owner's s_iso: the same ties

};

namespace detail {
inline bool read_values(std::istringstream& ls, std::vector<float>& v) {
    v.clear();
    double d;
    while (ls >> d) v.push_back((float) d);
    return !v.empty();
}
}  // namespace detail

/// Read a tables file; false with `err` set if it is missing or malformed (nothing half-read is kept).
inline bool load(const std::string& path, Rule& out, std::string& err) {
    out = Rule{};
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) { err = "stop_iso: cannot open " + path; return false; }
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    std::fclose(f);
    Rule r;
    r.path = path;
    std::istringstream in(text);
    std::string line;
    int lineno = 0;
    bool header = false;
    auto fail = [&](const std::string& why) {
        err = "stop_iso: " + path + ":" + std::to_string(lineno) + ": " + why;
        return false;
    };
    while (std::getline(in, line)) {
        ++lineno;
        if (const size_t h = line.find('#'); h != std::string::npos) line.resize(h);
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;
        if (!header) {
            int version = 0;
            if (key != "stop_iso" || !(ls >> version) || version != 1) return fail("not a 'stop_iso 1' tables file");
            header = true;
            continue;
        }
        if (key == "set") {
            Set s;
            std::string k1, k2;
            if (!(ls >> s.name >> k1 >> s.theta >> k2 >> s.t_lo >> s.t_hi) || k1 != "theta" || k2 != "temp")
                return fail("expected 'set <name> theta <theta> temp <lo> <hi>'");
            if (!std::isfinite(s.theta) || s.theta < 0.0f || s.theta > 1.0f || !(s.t_lo <= s.t_hi))
                return fail("theta must be in [0, 1] and lo <= hi");
            std::string k;
            while (ls >> k) {
                float lo, hi;
                if (!(ls >> lo >> hi) || !(lo <= hi)) return fail("'" + k + "' needs <lo> <hi> with lo <= hi");
                if (k == "top_p") s.p_lo = lo, s.p_hi = hi;
                else if (k == "top_k") s.k_lo = lo, s.k_hi = hi;
                else if (k == "presence") s.pr_lo = lo, s.pr_hi = hi;
                else return fail("unknown range '" + k + "' (top_p, top_k, presence)");
            }
            r.sets.push_back(std::move(s));
            continue;
        }
        if (r.sets.empty()) return fail("'" + key + "' before any 'set'");
        Set& s = r.sets.back();
        std::vector<float>* dst = key == "x" ? &s.q.x : key == "q" ? &s.q.y : key == "p1" ? &s.p1.y : nullptr;
        if (dst == nullptr) return fail("unknown key '" + key + "'");
        if (!dst->empty()) return fail("'" + key + "' given twice in set " + s.name);
        if (!detail::read_values(ls, *dst)) return fail("'" + key + "' has no values");
    }
    if (!header) { lineno = 0; return fail("empty file"); }
    if (r.sets.empty()) { lineno = 0; return fail("no 'set'"); }
    for (Set& s : r.sets) {
        s.p1.x = s.q.x;
        const size_t k = s.q.x.size();
        if (k < 2 || k > 64) return fail("set " + s.name + ": 2 to 64 knots");
        if (s.q.y.size() != k || s.p1.y.size() != k) return fail("set " + s.name + ": x, q and p1 must have the same length");
        for (size_t j = 0; j < k; ++j) {
            if (j > 0 && !(s.q.x[j] > s.q.x[j - 1])) return fail("set " + s.name + ": x must increase");
            for (const float v : {s.q.y[j], s.p1.y[j]})
                if (!(v >= 0.0f && v <= 1.0f)) return fail("set " + s.name + ": values must be in [0, 1]");
            if (j > 0 && (s.q.y[j] < s.q.y[j - 1] || s.p1.y[j] < s.p1.y[j - 1]))
                return fail("set " + s.name + ": q and p1 must not decrease (isotonic)");
        }
    }
    out = std::move(r);
    return true;
}

/// The engines' switch: `<prefix>` names the tables file, `<prefix>_THETA` overrides every set's theta (sweeps).
/// Unset or empty: off.  A file that does not load is reported and leaves the rule off (the engine's own rule runs).
inline Rule from_env(const char* prefix, const char* who) {
    Rule r;
    const char* p = std::getenv(prefix);
    if (p == nullptr || p[0] == '\0') return r;
    std::string err;
    if (!load(p, r, err)) {
        std::fprintf(stderr, "%s: %s=%s: %s; the learned stop rule stays off\n", who, prefix, p, err.c_str());
        return Rule{};
    }
    const std::string th = std::string(prefix) + "_THETA";
    if (const char* t = std::getenv(th.c_str()); t != nullptr && t[0] != '\0') r.set_theta((float) std::atof(t));
    std::fprintf(stderr, "%s: %s: the MTP chain stops on the learned survival rule (%s)\n", who, p, r.describe().c_str());
    return r;
}

}  // namespace fnx::stop_iso
