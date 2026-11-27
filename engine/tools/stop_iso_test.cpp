// tools/stop_iso_test.cpp - the learned MTP stop rule (src/stop_iso.hpp, FNX_DRAFT_STOP_ISO), against the rule it ports.
//   1. Map: bit for bit the owner's spec_iso_lookup (llama.cpp common/speculative.cpp) on the owner's 27B chat tables;
//   2. Chain: the owner's per-step decision (stop when s * iso_p1(p1_d) < theta, then s *= iso_q(q_d)) on random chains;
//   3. the tables file: a round trip, pick() by temperature, and the malformed files it refuses;
//   4. EXACT: speculative sampling (accept min(1, p/q), else the residual max(0, p - q), a bonus token after a full
//      window) with the rule keeps the target's distribution of the first three tokens (Monte Carlo, a Markov target and
//      drafter); a rule that tests the drawn token's own q at its own step does not (the test can see a bias).
#include "stop_iso.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace fnx;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-86s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// ---- the owner's code (Keoian/dual-p100-qwen-3-8-27b, common/speculative.cpp, c44158c77), verbatim but for names
const float own_x[24] = {0.000f, 0.020f, 0.050f, 0.100f, 0.150f, 0.200f, 0.250f, 0.300f, 0.350f, 0.400f, 0.450f, 0.500f,
                         0.550f, 0.600f, 0.650f, 0.700f, 0.750f, 0.800f, 0.850f, 0.900f, 0.950f, 0.980f, 0.995f, 1.000f};
const float own_q[24] = {0.000f, 0.115f, 0.147f, 0.231f, 0.250f, 0.267f, 0.299f, 0.299f, 0.299f, 0.321f, 0.350f, 0.389f,
                         0.402f, 0.419f, 0.419f, 0.419f, 0.437f, 0.503f, 0.530f, 0.570f, 0.603f, 0.734f, 0.821f, 0.994f};
const float own_p1[24] = {0.000f, 0.000f, 0.000f, 0.000f, 0.100f, 0.156f, 0.156f, 0.200f, 0.200f, 0.209f, 0.248f, 0.325f,
                          0.328f, 0.369f, 0.369f, 0.369f, 0.399f, 0.431f, 0.502f, 0.522f, 0.573f, 0.708f, 0.812f, 0.994f};
float own_lookup(const float* tab, const float x) {
    if (x <= own_x[0]) return tab[0];
    for (int j = 1; j < 24; ++j) {
        if (x <= own_x[j]) {
            const float t = (x - own_x[j - 1]) / (own_x[j] - own_x[j - 1]);
            return tab[j - 1] + t * (tab[j] - tab[j - 1]);
        }
    }
    return tab[23];
}
// the owner's chain: drafts kept of a chain with step distributions' top probabilities p1[] and drawn q[]
int own_chain(const std::vector<float>& p1, const std::vector<float>& q, float theta) {
    float s = 1.0f;
    for (size_t d = 0; d < p1.size(); ++d) {
        if (s * own_lookup(own_p1, p1[d]) < theta) return (int) d;
        s *= own_lookup(own_q, q[d]);
    }
    return (int) p1.size();
}

stop_iso::Set owner_set(float theta) {
    stop_iso::Set s;
    s.name = "chat27b";
    s.theta = theta;
    s.t_lo = 0.001f;
    s.t_hi = 1e9f;
    s.q.x.assign(own_x, own_x + 24);
    s.q.y.assign(own_q, own_q + 24);
    s.p1.x = s.q.x;
    s.p1.y.assign(own_p1, own_p1 + 24);
    return s;
}

bool write_file(const std::string& path, const std::string& text) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) return false;
    std::fputs(text.c_str(), f);
    std::fclose(f);
    return true;
}

// ---- 4: a Markov target P[prev][tok] and drafter Q[prev][tok] over V tokens; three tokens from token 0
constexpr int V = 3;
using Row = std::array<double, V>;
const std::array<Row, V> P = {{{0.50, 0.30, 0.20}, {0.10, 0.60, 0.30}, {0.40, 0.40, 0.20}}};
const std::array<Row, V> Q = {{{0.80, 0.15, 0.05}, {0.20, 0.35, 0.45}, {0.34, 0.33, 0.33}}};

int draw(const Row& r, double u) {
    double c = 0.0;
    for (int i = 0; i < V; ++i) {
        c += r[i];
        if (u < c) return i;
    }
    return V - 1;
}

// One sequence of 3 output tokens by speculative sampling with chains of up to K drafts; `cheat`: keep a draft by its
// own drawn q (s * iso_q(q_d) >= theta), the biased rule the port must not be.
int sample3(std::mt19937_64& rng, const stop_iso::Set& set, int K, bool cheat) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int out[3], n_out = 0, last = 0;
    while (n_out < 3) {
        // the chain
        stop_iso::Chain ch(&set);
        std::vector<int> drafts;
        int prev = last;
        for (int j = 0; j < K; ++j) {
            const Row& q = Q[prev];
            const double p1 = std::max(q[0], std::max(q[1], q[2]));
            if (!cheat && !ch.keep((float) p1)) break;
            const int d = draw(q, U(rng));
            if (cheat && ch.survival() * set.q((float) q[d]) < set.theta) break;
            ch.drew((float) q[d]);
            drafts.push_back(d);
            prev = d;
        }
        // the verify, row by row
        int ctx = last;
        bool ended = false;
        for (size_t j = 0; j < drafts.size() && n_out < 3; ++j) {
            const int d = drafts[j];
            const Row& p = P[ctx];
            const Row& q = Q[ctx];
            if (U(rng) * q[d] < p[d]) {
                out[n_out++] = d;
                ctx = d;
                continue;
            }
            Row r{};
            double sum = 0.0;
            for (int i = 0; i < V; ++i) sum += (r[i] = std::max(0.0, p[i] - q[i]));
            for (double& v : r) v /= sum;
            const int t = draw(r, U(rng));
            out[n_out++] = t;
            ctx = t;
            ended = true;
            break;
        }
        if (!ended && n_out < 3) {   // every draft stood: one more from the last row
            const int t = draw(P[ctx], U(rng));
            out[n_out++] = t;
            ctx = t;
        }
        last = ctx;
    }
    return out[0] * V * V + out[1] * V + out[2];
}

// the largest |empirical - exact| over the V^3 sequences
double max_gap(const stop_iso::Set& set, int K, bool cheat, int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<double> c(V * V * V, 0.0);
    for (int i = 0; i < n; ++i) c[(size_t) sample3(rng, set, K, cheat)] += 1.0;
    double gap = 0.0;
    for (int a = 0; a < V; ++a)
        for (int b = 0; b < V; ++b)
            for (int d = 0; d < V; ++d) {
                const double exact = P[0][a] * P[a][b] * P[b][d];
                gap = std::max(gap, std::fabs(c[(size_t) (a * V * V + b * V + d)] / n - exact));
            }
    return gap;
}
}  // namespace

int main() {
    std::printf("stop_iso_test\n");
    // 1. the map
    {
        const stop_iso::Set s = owner_set(0.125f);
        bool same = true;
        for (int i = -20; i <= 1020; ++i) {
            const float x = (float) i / 1000.0f;
            const float a = s.q(x), b = own_lookup(own_q, x), c = s.p1(x), d = own_lookup(own_p1, x);
            same = same && a == b && c == d;   // bit for bit: the same float operations
        }
        check(same, "Map = the owner's spec_iso_lookup, bit for bit, at 1,041 points in [-0.02, 1.02]");
        check(s.q(std::nanf("")) == own_q[0], "a NaN probability reads as the first knot");
    }
    // 2. the chain
    {
        std::mt19937_64 rng(7);
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        bool same = true;
        int stops[6] = {};
        for (int t = 0; t < 20000; ++t) {
            const float theta = (t % 4 == 0) ? 0.125f : (t % 4 == 1) ? 0.05f : (t % 4 == 2) ? 0.3f : 0.6f;
            const stop_iso::Set s = owner_set(theta);
            std::vector<float> p1(5), q(5);
            for (int d = 0; d < 5; ++d) {
                p1[d] = U(rng);
                q[d] = (t & 1) ? p1[d] : p1[d] * U(rng);   // argmax drafts (q = p1) and sampled ones (q <= p1)
            }
            stop_iso::Chain ch(&s);
            int kept = 0;
            for (int d = 0; d < 5; ++d) {
                if (!ch.keep(p1[d])) break;
                ch.drew(q[d]);
                ++kept;
            }
            same = same && kept == own_chain(p1, q, theta);
            ++stops[kept];
        }
        check(same, "Chain = the owner's per-step decision on 20,000 random chains (4 thetas)");
        check(stops[0] > 0 && stops[1] > 0 && stops[2] > 0 && stops[5] > 0, "  (the chains stop at every depth, so the test sees them)");
        stop_iso::Chain off;
        check(!off.on() && off.keep(0.0f), "no set: keep() always true");
    }
    // 3. the file
    {
        const std::string path = "stop_iso_test.tables";
        const std::string ok_text =
            "stop_iso 1   # a test file\n"
            "set think theta 0.15 temp 0.95 1.05\n"
            "x 0 0.5 1\nq 0.1 0.4 0.9\np1 0.2 0.5 0.95\n"
            "\n# chat: every other positive temperature\n"
            "set chat theta 0.125 temp 0.001 1e9\n"
            "x 0 0.25 0.75 1\nq 0 0.2 0.6 1\np1 0 0.1 0.5 1\n";
        stop_iso::Rule r;
        std::string err;
        const bool loaded = write_file(path, ok_text) && stop_iso::load(path, r, err);
        check(loaded && r.sets.size() == 2, "a two-set file loads");
        if (loaded) {
            const stop_iso::Set* a = r.pick(1.0f);
            const stop_iso::Set* b = r.pick(0.7f);
            check(a != nullptr && a->name == "think" && b != nullptr && b->name == "chat",
                  "pick(): the first set whose window holds the temperature (1.0 -> think, 0.7 -> chat)");
            check(r.pick(0.0f) == nullptr && r.pick(-1.0f) == nullptr, "pick(): greedy (0) and outside every window -> none");
            check(std::fabs(r.sets[1].q(0.5f) - 0.4f) < 1e-6f && r.sets[1].p1(2.0f) == 1.0f, "the loaded maps interpolate and clamp");
            const std::string keyed =
                "stop_iso 1\nset instruct theta 0.5 temp 0.6 0.8 top_p 0.7 0.9 presence 1 2\nx 0 1\nq 0 1\np1 0 1\n"
                "set chat theta 0.4 temp 0.001 1e9\nx 0 1\nq 0 1\np1 0 1\n";
            stop_iso::Rule rk;
            const bool kl = write_file(path, keyed) && stop_iso::load(path, rk, err);
            const stop_iso::Set* k1 = kl ? rk.pick(0.7f, 0.8f, 20.0f, 1.5f) : nullptr;
            const stop_iso::Set* k2 = kl ? rk.pick(0.7f, 0.95f, 20.0f, 1.5f) : nullptr;
            const stop_iso::Set* k3 = kl ? rk.pick(0.7f, 0.8f, 20.0f, 0.0f) : nullptr;
            check(k1 && k1->name == "instruct" && k2 && k2->name == "chat" && k3 && k3->name == "chat",
                  "top_p / presence ranges: 0.7-0.8-1.5 -> instruct; top-p 0.95 or presence 0 -> chat");
            r.set_theta(0.4f);
            check(r.sets[0].theta == 0.4f && r.sets[1].theta == 0.4f, "set_theta() overrides every set");
        }
        const char* bad[] = {
            "",                                                                    // empty
            "stop_iso 2\nset a theta 0.1 temp 0 1\nx 0 1\nq 0 1\np1 0 1\n",       // version
            "stop_iso 1\nx 0 1\n",                                                // before a set
            "stop_iso 1\nset a theta 0.1 temp 0 1\nx 0 1\nq 0 1\n",               // no p1
            "stop_iso 1\nset a theta 0.1 temp 0 1\nx 0 1\nq 0.5 0.4\np1 0 1\n",   // q decreases
            "stop_iso 1\nset a theta 0.1 temp 0 1\nx 0 0\nq 0 1\np1 0 1\n",       // x does not increase
            "stop_iso 1\nset a theta 0.1 temp 0 1\nx 0 1\nq 0 1.5\np1 0 1\n",     // above 1
            "stop_iso 1\nset a theta 0.1 temp 0 1\nx 0 0.5 1\nq 0 1\np1 0 1\n",   // lengths
            "stop_iso 1\nset a theta 2 temp 0 1\nx 0 1\nq 0 1\np1 0 1\n",         // theta
            "stop_iso 1\nset a theta 0.1 temp 0 1\nx 0 1\nq 0 1\np1 0 1\nz 1\n",  // unknown key
            "stop_iso 1\nset a theta 0.1 temp 0 1 top_q 0 1\nx 0 1\nq 0 1\np1 0 1\n",  // unknown range
            "stop_iso 1\nset a theta 0.1 temp 0 1 top_p 0.9 0.1\nx 0 1\nq 0 1\np1 0 1\n",  // lo > hi
        };
        int refused = 0;
        for (const char* t : bad) {
            stop_iso::Rule rb;
            std::string e2;
            if (write_file(path, t) && !stop_iso::load(path, rb, e2) && !rb.on() && !e2.empty()) ++refused;
        }
        check(refused == (int) (sizeof bad / sizeof bad[0]), "twelve malformed files are refused, each with a reason");
        stop_iso::Rule rm;
        check(!stop_iso::load("/nonexistent/stop_iso.tables", rm, err), "a missing file is refused");
        std::remove(path.c_str());
    }
    // 4. exactness
    {
        stop_iso::Set s;
        s.name = "mc";
        s.theta = 0.3f;
        s.t_lo = 0.0f;
        s.t_hi = 2.0f;
        s.q.x = {0.0f, 1.0f};
        s.q.y = {0.0f, 1.0f};   // iso_q(q) = q: the chain stops after a run of unlikely drafts
        s.p1.x = s.q.x;
        s.p1.y = {0.0f, 1.0f};
        const int n = 400000;
        const double g_rule = max_gap(s, 3, false, n, 11), g_cheat = max_gap(s, 3, true, n, 11);
        char b[160];
        std::snprintf(b, sizeof b, "the rule keeps the target's law of 3 tokens (max gap %.4f over 27, %d runs)", g_rule, n);
        check(g_rule < 0.004, b);
        std::snprintf(b, sizeof b, "a rule reading the drawn token's own q is biased (max gap %.4f)", g_cheat);
        check(g_cheat > 0.012, b);
    }
    std::printf("stop_iso_test: %s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
