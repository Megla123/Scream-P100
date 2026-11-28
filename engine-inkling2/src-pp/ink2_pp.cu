// ink2_pp.cu (2026-11-23, pp1): Inkling 2 Mini as a pipeline over the two P100s, several conversations at once (NACRT-pp.md).
// Card 0 holds layers 0-11 (+ the embedding's norm and the head), card 1 layers 12-23; each is a one-card Engine (NC = 1, the
// kernels of ink2_1c) over its own layers. A token's 48 passes go in 7 segments S0..S6 alternating between the cards; each card
// has one thread and a FIFO of segments, so while card 1 runs a segment of conversation A, card 0 runs one of B. A conversation
// is computed alone (no batching across conversations) with piece sizes that do not depend on the load, so its output is the
// same whether it runs alone or beside others.
//   ink2_pp --model M.gguf --serve SOCK [--slots 4] [--ctx 65536]          ink2_v1's socket protocol, a thread a connection
//   ink2_pp --model M.gguf --tokens zetoni.bin [--gen N] [--logits F]     one prompt (tests)
#define INK2_NC 1
#define INK2_PP 1
#include "ink2_v1.cu"

#include <condition_variable>
#include <deque>
#include <mutex>

namespace ink2pp {
using namespace ink2;

static Engine* E[2] = {};
static bool g_ds = false;   // pp4: the drafter (--mtp)
static int g_dscard = 0;    // pp6: the drafter's card (INK2_PP_RISAR=1: card 1, with the final mix; else card 0 with the head)
static int g_split = 12;   // the first layer of card 1
static double g_spin = 0.02;   // pp2: INK2_PP_SPIN_MS, how long a waiting thread spins before it sleeps

struct Conv;
struct Step {   // a piece in the pipe; a conversation has g_inflight of them (page-locked buffers)
    Conv* cv = nullptr;
    Engine::SegIn in;
    int last_seg = 6;
    bool ops_done[2] = {};
    bool done = true;
    float *x0 = nullptr, *tau = nullptr, *carry = nullptr, *logits = nullptr;
    int* pos = nullptr;
    int kind = 0;   // pp4: 0 a piece through the segments, 1 a draft on card 0
    cudaEvent_t ev[2] = {};   // pp8: the end of its last queued segment, on card 0's / card 1's stream
    int evc = -1;             // pp8: the card whose event the next segment waits for (-1: none)
    int anchor = 0, dk = 0, dpos = 0, dctx = -1;
    std::vector<int> drafts;
};
struct Conv {   // a slot: one conversation's caches on both cards and what they hold
    int id = 0;
    Engine::ConvDev dev[2];
    std::vector<int> processed;
    int n_past = 0, snap_n = -1;
    bool p_reset = true, p_restore = false;   // state ops for the next piece
    int p_roll = -1;
    bool busy = false;
    unsigned long long use = 0;
    std::vector<Step> steps;
    int next = 0;
    std::mutex m;
    std::condition_variable cvd;
};
static std::vector<Conv*> g_convs;
static std::mutex g_cm;   // the slots' busy flags
static std::condition_variable g_ccv;
static unsigned long long g_use = 0;

struct Queue {   // one card's segments, first come first served
    std::mutex m;
    std::condition_variable cv;
    std::deque<Step*> q;
    void push(Step* s) {
        {
            std::lock_guard<std::mutex> l(m);
            q.push_back(s);
        }
        cv.notify_one();
    }
    Step* pop() {
        for (const double t0 = now_s(); now_s() - t0 < g_spin;) {   // spin (the other card's segment ends soon), then sleep
            {
                std::lock_guard<std::mutex> l(m);
                if (!q.empty()) {
                    Step* s = q.front();
                    q.pop_front();
                    return s;
                }
            }
            std::this_thread::yield();
        }
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&]() { return !q.empty(); });
        Step* s = q.front();
        q.pop_front();
        return s;
    }
};
static Queue Q[2];
static int g_paket = 4;   // pp7: INK2_PP_PAKET, the most conversations in one batch (1 = no batches; pp9: 4)
static bool g_async = false;   // pp8: INK2_PP_ASYNC=1, hand-offs by events (pp9: off, slower)
// pp7: from a card's queue, later pieces of other conversations at s's segment that can join it (the writing path, the same
// segment, MAXT rows at most, g_paket conversations at most); they leave the queue
static void take_batch(Queue& q, Step* s, std::vector<Step*>& b) {
    b.assign(1, s);
    if (g_paket <= 1 || s->kind != 0 || s->in.read) return;
    int rows = s->in.T;
    std::lock_guard<std::mutex> l(q.m);
    for (auto it = q.q.begin(); it != q.q.end() && (int) b.size() < g_paket;) {
        Step* o = *it;
        bool ok = o->kind == 0 && !o->in.read && o->in.seg == s->in.seg && rows + o->in.T <= MAXT;
        for (Step* x : b) ok = ok && x->cv != o->cv;
        if (ok) {
            b.push_back(o);
            rows += o->in.T;
            it = q.q.erase(it);
        } else {
            ++it;
        }
    }
}
static std::atomic<long long> g_busy_us[2], g_nseg[2], g_nbat[2];   // pp7: g_nbat = batches run
static double g_t0 = 0;

static void finish(Step* s) {
    Conv& cv = *s->cv;
    {
        std::lock_guard<std::mutex> l(cv.m);
        s->done = true;
    }
    cv.cvd.notify_all();
}
static void worker(int c) {
    Engine& e = *E[c];
    CK(cudaSetDevice(e.cards[0].dev));
    for (;;) {
        Step* s = Q[c].pop();
        const double t = now_s();
        e.conv_use(s->cv->dev[c]);
        if (s->kind == 1) {   // pp4: a draft
            s->drafts = e.ds_draft_pp(s->cv->dev[c], s->anchor, s->dpos, s->dk, s->dctx);
            g_busy_us[c] += (long long) ((now_s() - t) * 1e6);
            finish(s);
            continue;
        }
        if (!s->ops_done[c]) {
            e.seg_ops(s->in);
            s->ops_done[c] = true;
        }
        std::vector<Step*> bat;   // pp7
        take_batch(Q[c], s, bat);
        for (Step* x : bat) {   // pp8: after the other card's segment of this piece; the last segment waits at its end
            if (x->evc >= 0 && x->evc != c) CK(cudaStreamWaitEvent(e.cards[0].st, x->ev[x->evc], 0));
            x->in.nosync = g_async && x->in.seg < x->last_seg;
        }
        if (bat.size() > 1) {
            std::vector<Engine::SegIn*> ins;
            std::vector<const Engine::ConvDev*> devs;
            for (Step* x : bat) {
                e.conv_use(x->cv->dev[c]);
                if (!x->ops_done[c]) {
                    e.seg_ops(x->in);
                    x->ops_done[c] = true;
                }
                ins.push_back(&x->in);
                devs.push_back(&x->cv->dev[c]);
            }
            e.seg_write_b(ins, devs);
            ++g_nbat[c];
        } else if (s->in.read) e.seg_read(s->in);
        else e.seg_write(s->in);
        g_busy_us[c] += (long long) ((now_s() - t) * 1e6);
        ++g_nseg[c];
        for (Step* x : bat) {
            if (x->in.nosync) {   // pp8
                CK(cudaEventRecord(x->ev[c], e.cards[0].st));
                x->evc = c;
            } else {
                x->evc = -1;
            }
        }
        for (Step* x : bat) {
            if (x->in.seg < x->last_seg) {
                ++x->in.seg;
                Q[Engine::pp_segs()[(size_t) x->in.seg].card].push(x);   // pp3
            } else {
                finish(x);
            }
        }
    }
}

static void wait_step(Conv& cv, Step& s) {
    for (const double t0 = now_s(); now_s() - t0 < g_spin;) {   // pp2: spin first (a token's segments take ~20 ms)
        {
            std::lock_guard<std::mutex> l(cv.m);
            if (s.done) return;
        }
        std::this_thread::yield();
    }
    std::unique_lock<std::mutex> l(cv.m);
    cv.cvd.wait(l, [&]() { return s.done; });
}
static void drain(Conv& cv) {
    for (Step& s : cv.steps) wait_step(cv, s);
}
static void embed_row(float* dst, int id) {   // Engine::embed into dst (no pictures in this build)
    if (id < 0 || id >= NV) throw std::runtime_error("token id out of range: " + std::to_string(id));
    const auto* tok = E[0]->tok;
    const uint8_t* row = tok->data + (size_t) id * tok->row_bytes();
    for (int b = 0; b < N / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * 34;
        uint16_t hb;
        std::memcpy(&hb, blk, 2);
        __half_raw hr;
        hr.x = hb;
        const float d = __half2float(__half(hr));
        for (int j = 0; j < 32; ++j) dst[b * 32 + j] = d * (float) (int8_t) blk[2 + j];
    }
}
// rows ids[0..T) at the conversation's n_past into the pipe: the reading path (read) or a window of the writing path; the head (S6)
// only for last (all rows: all_rows). A piece with state ops waits until nothing else of the conversation is in flight
static Step* submit(Conv& cv, const int* ids, int T, bool read, bool all_rows, bool last, bool keep_hist = false, bool snap = false,
                    bool bak = false) {
    if (cv.p_reset || cv.p_restore || cv.p_roll >= 0 || snap || bak) drain(cv);
    Step& s = cv.steps[(size_t) cv.next];
    cv.next = (cv.next + 1) % (int) cv.steps.size();
    wait_step(cv, s);
    if (cv.n_past + T > E[0]->ctx) throw std::runtime_error("context full");
    for (int t = 0; t < T; ++t) {
        embed_row(s.x0 + (size_t) t * N, ids[t]);
        const int P = cv.n_past + t;
        s.pos[t] = P;
        s.tau[t] = logf(fmaxf(((float) P + 1.0f) * (1.0f / E[0]->log_floor), 1.0f)) * E[0]->log_alpha + 1.0f;
    }
    Engine::SegIn& in = s.in;
    in = Engine::SegIn{};
    in.T = T;
    in.read = read;
    in.p0 = cv.n_past;
    in.nparts = (cv.n_past + T + KC - 1) / KC + 1;
    in.all_rows = all_rows;
    in.last = last;
    in.keep_hist = keep_hist;
    in.reset = cv.p_reset;
    in.restore = cv.p_restore;
    in.roll = cv.p_roll;
    in.snap = snap;
    in.bak = bak;
    cv.p_reset = cv.p_restore = false;
    cv.p_roll = -1;
    in.x0 = s.x0;
    in.pos = s.pos;
    in.tau = s.tau;
    in.carry = s.carry;
    in.logits = s.logits;
    s.last_seg = (int) Engine::pp_segs().size() - (last || (g_ds && g_dscard == 0) ? 1 : 2);   // pp6: with the drafter on card 1 it ingests at the last layers   // pp3: the head's segment is the last (pp4: the drafter ingests every piece there)
    s.kind = 0;
    s.ops_done[0] = s.ops_done[1] = false;
    s.evc = -1;   // pp8
    {
        std::lock_guard<std::mutex> l(cv.m);
        s.done = false;
    }
    cv.n_past += T;
    cv.processed.insert(cv.processed.end(), ids, ids + T);
    Q[0].push(&s);
    return &s;
}
// pp4: up to k drafts after the anchor (the pending token at n_past); ctx_row: the last verify's kept row (-1: none)
static std::vector<int> draft(Conv& cv, int anchor, int k, int ctx_row) {
    Step& s = cv.steps[(size_t) cv.next];
    cv.next = (cv.next + 1) % (int) cv.steps.size();
    wait_step(cv, s);
    s.kind = 1;
    s.anchor = anchor;
    s.dk = k;
    s.dpos = cv.n_past;
    s.dctx = ctx_row;
    s.drafts.clear();
    {
        std::lock_guard<std::mutex> l(cv.m);
        s.done = false;
    }
    Q[g_dscard].push(&s);   // pp6
    wait_step(cv, s);
    return s.drafts;
}
// Engine::feed: reading chunks (pf_min tokens or more) and windows of feed_t; only the last piece gets the head (when head)
static Step* feed(Conv& cv, const int* ids, int n, bool head) {
    const Engine& e = *E[0];
    Step* s = nullptr;
    for (int i = 0, T = 0; i < n; i += T) {
        const bool rd = n - i >= e.pf_min;
        T = std::min(rd ? e.pfc : e.feed_t, n - i);
        s = submit(cv, ids + i, T, rd, false, head && i + T >= n);
    }
    return s;
}
static bool restorable(const Conv& cv, size_t same, size_t n) {   // Engine::restore's test (and one token at least to read)
    return cv.snap_n > 0 && (int) same >= cv.snap_n && (int) cv.processed.size() >= cv.snap_n && cv.n_past - cv.snap_n <= RING - SWA - MAXT &&
           (size_t) cv.snap_n < n;
}
static unsigned long long fnv_logits(unsigned long long h, const float* row) {
    const unsigned* w = (const unsigned*) row;
    for (int i = 0; i < NVU; ++i) h = (h ^ w[i]) * 1099511628211ULL;
    return h;
}

struct Res {
    std::vector<int> out;
    int cached = 0, drafted = 0, accepted = 0, rounds = 0, slot = -1;
    double read_s = 0, write_s = 0, wait_s = 0, verify_s = 0, draft_s = 0;
    unsigned long long lh = 0, lh2 = 1469598103934665603ULL;
};
// generate() for one conversation: greedy, n-gram drafts (lookup), p8's snapshot before the last prompt token (snap_on)
static Res run(Conv& cv, const std::vector<int>& tok, int max, const std::vector<int>& stop, int eos, int lookup, int mtp, bool snap_on,
               bool lhall, const std::function<bool(const int*, int)>* on_out, const std::string& logits_out) {
    Res g;
    g.slot = cv.id;
    size_t sent = 0;
    auto flush = [&]() {
        if (!on_out || sent == g.out.size()) return true;
        const bool ok = (*on_out)(g.out.data() + sent, (int) (g.out.size() - sent));
        sent = g.out.size();
        return ok;
    };
    size_t same = 0;
    while (same < tok.size() && same < cv.processed.size() && tok[same] == cv.processed[same]) ++same;
    if (same != cv.processed.size() || same == tok.size()) {   // not a continuation: back to the snapshot, or start over
        if (snap_on && restorable(cv, same, tok.size())) {
            cv.p_restore = true;
            cv.p_reset = false;
            cv.n_past = cv.snap_n;
            cv.processed.resize((size_t) cv.snap_n);
            same = (size_t) cv.snap_n;
        } else {
            cv.p_reset = true;
            cv.p_restore = false;
            cv.n_past = 0;
            cv.processed.clear();
            cv.snap_n = -1;
            same = 0;
        }
        cv.p_roll = -1;
    }
    g.cached = (int) same;
    const double t0 = now_s();
    const int n = (int) tok.size();
    Step* last;
    if (snap_on && n - (int) same >= 2) {
        feed(cv, tok.data() + same, n - 1 - (int) same, false);
        last = submit(cv, tok.data() + n - 1, 1, false, false, true, false, true);
        cv.snap_n = n - 1;
    } else {
        last = feed(cv, tok.data() + same, n - (int) same, true);
    }
    drain(cv);
    g.read_s = now_s() - t0;
    g.lh = fnv_logits(1469598103934665603ULL, last->logits);
    if (!logits_out.empty()) {   // the --logits format of ink2_v1
        FILE* lf = std::fopen(logits_out.c_str(), "wb");
        if (lf) {
            const int32_t nn = n, nv = NVU, first = n - 1;
            std::fwrite(&nn, 4, 1, lf);
            std::fwrite(&nv, 4, 1, lf);
            std::fwrite(&first, 4, 1, lf);
            std::fwrite(last->logits, 4, NVU, lf);
            std::fclose(lf);
        }
    }
    const double t1 = now_s();
    int next = Engine::argmax(last->logits);
    if (lhall) g.lh2 = fnv_logits(g.lh2, last->logits);
    auto ends = [&](int t) { return t == eos || std::find(stop.begin(), stop.end(), t) != stop.end(); };
    int dcur = std::min(lookup, 4), cool = 0, ctx_row = -1;
    if (!g_ds) mtp = 0;
    static const int ng_min = getenv("INK2_NG_MIN") ? atoi(getenv("INK2_NG_MIN")) : 6;   // p7: an n-gram match that beats the drafter
    while ((int) g.out.size() < max) {
        g.out.push_back(next);
        if (!flush()) break;
        if (ends(next) || (int) g.out.size() >= max || cv.n_past >= E[0]->ctx) break;
        std::vector<int> d;
        if (lookup > 0) {
            int ml = 0;
            if (cool > 0) --cool;
            else {
                std::vector<int> hist = cv.processed;
                hist.push_back(next);
                d = ngram_draft(hist, std::min(dcur, std::min(MAXT - 1, max - (int) g.out.size())), &ml);
            }
            if (mtp > 0 && ml < ng_min) d.clear();
        }
        bool mtpd = false;
        if (d.empty() && mtp > 0) {   // pp4: the DSpark drafter (generate's p6 path)
            mtpd = true;
            const double td = now_s();
            d = draft(cv, next, std::min(mtp, max - (int) g.out.size()), ctx_row);
            g.draft_s += now_s() - td;
        }
        ctx_row = -1;
        if ((int) d.size() > E[0]->ctx - cv.n_past - 1) d.resize(std::max(0, E[0]->ctx - cv.n_past - 1));
        if (d.empty()) {
            Step* s = submit(cv, &next, 1, false, false, true);
            wait_step(cv, *s);
            next = Engine::argmax(s->logits);
            if (lhall) g.lh2 = fnv_logits(g.lh2, s->logits);
            continue;
        }
        std::vector<int> win{next};
        win.insert(win.end(), d.begin(), d.end());
        const int T = (int) win.size();
        const double tv = now_s();
        Step* s = submit(cv, win.data(), T, false, true, true, true, false, true);   // Engine::verify: stbak, keep_hist, all rows
        wait_step(cv, *s);
        int right = 0;
        while (right + 1 < T && Engine::argmax(s->logits + (size_t) right * NV) == win[right + 1]) ++right;
        const int nn = Engine::argmax(s->logits + (size_t) right * NV);
        if (lhall)
            for (int i = 0; i <= right; ++i) g.lh2 = fnv_logits(g.lh2, s->logits + (size_t) i * NV);
        const int keep = right + 1;
        ctx_row = right;   // pp4: the drafter's next context
        if (keep < T) {   // the conv states back to keep rows before the next piece (seg_ops)
            cv.p_roll = keep;
            cv.n_past -= T - keep;
            cv.processed.resize(cv.processed.size() - (size_t) (T - keep));
        }
        g.verify_s += now_s() - tv;
        ++g.rounds;
        g.drafted += (int) d.size();
        g.accepted += right;
        if (lookup > 0 && !mtpd) {   // the n-gram draft length follows the last n-gram window (p7: not the drafter's)
            if (right == (int) d.size()) dcur = std::min(lookup, std::max(2, dcur * 2));
            else dcur = std::max(2, right + 1);
            if (right == 0 && mtp == 0) cool = 4;
        }
        bool stop_now = false;
        for (int i = 0; i < right && (int) g.out.size() < max; ++i) {
            g.out.push_back(d[(size_t) i]);
            if (ends(d[(size_t) i])) {
                stop_now = true;
                break;
            }
        }
        if (!flush()) break;
        if (stop_now || (int) g.out.size() >= max) break;
        next = nn;
    }
    g.write_s = now_s() - t1;
    return g;
}

// a free slot for a prompt: the one whose caches it continues (or whose snapshot it repeats), else the least recently used
static Conv* take(const std::vector<int>& tok, bool snap_on, double& wait_s) {
    std::unique_lock<std::mutex> l(g_cm);
    const double t = now_s();
    for (;;) {
        Conv *best = nullptr, *lru = nullptr;
        size_t bu = 0;
        for (Conv* c : g_convs) {
            if (c->busy) continue;
            size_t same = 0;
            while (same < tok.size() && same < c->processed.size() && tok[same] == c->processed[same]) ++same;
            size_t u = 0;
            if (same == c->processed.size() && same < tok.size()) u = same;
            else if (snap_on && restorable(*c, same, tok.size())) u = (size_t) c->snap_n;
            if (u > bu) {
                bu = u;
                best = c;
            }
            if (!lru || c->use < lru->use) lru = c;
        }
        Conv* pick = best ? best : lru;
        if (pick) {
            pick->busy = true;
            pick->use = ++g_use;
            wait_s = now_s() - t;
            return pick;
        }
        g_ccv.wait(l);
    }
}
static void give(Conv* c) {
    {
        std::lock_guard<std::mutex> l(g_cm);
        c->busy = false;
    }
    g_ccv.notify_all();
}

static int g_lsock = -1;
static std::atomic<bool> g_quit{false};
static std::atomic<int> g_active{0};
static void handle(int c, int eos) {
    std::string req;
    char buf[1 << 16];
    for (;;) {
        const ssize_t n = read(c, buf, sizeof buf);
        if (n <= 0) break;
        req.append(buf, (size_t) n);
        if (req.find('\n') != std::string::npos) break;
    }
    if (json_int(req, "exit", 0)) {
        close(c);
        g_quit = true;
        shutdown(g_lsock, SHUT_RDWR);
        return;
    }
    std::string rep;
    if (json_int(req, "stats", 0)) {   // the cards' busy time since the start (or since "stats": 2, which zeroes it)
        char hd[400];
        const double up = now_s() - g_t0;
        std::snprintf(hd, sizeof hd, "{\"up_s\": %.3f, \"busy_s\": [%.3f, %.3f], \"segments\": [%lld, %lld], \"batches\": [%lld, %lld], \"active\": %d, \"slots\": %zu}\n", up,
                      g_busy_us[0] / 1e6, g_busy_us[1] / 1e6, (long long) g_nseg[0], (long long) g_nseg[1], (long long) g_nbat[0], (long long) g_nbat[1], g_active.load(), g_convs.size());
        if (json_int(req, "stats", 0) == 2) {
            g_t0 = now_s();
            g_busy_us[0] = g_busy_us[1] = 0;
            g_nseg[0] = g_nseg[1] = 0;
            g_nbat[0] = g_nbat[1] = 0;
        }
        rep = hd;
        (void) !write(c, rep.data(), rep.size());
        close(c);
        return;
    }
    Conv* cv = nullptr;
    try {
        std::vector<int> tok, stop;
        for (long long x : json_ints(req, "tokens")) tok.push_back((int) x);
        for (long long x : json_ints(req, "stop")) stop.push_back((int) x);
        if (tok.empty()) throw std::runtime_error("no tokens");
        if (!json_ints(req, "img").empty()) throw std::runtime_error("ink2_pp: no pictures in this build");
        const int ctx = E[0]->ctx;
        if ((int) tok.size() >= ctx) throw std::runtime_error("prompt longer than a slot's context");
        const int max = (int) std::min<long long>(json_int(req, "max", 256), std::max<long long>(1, (long long) ctx - (long long) tok.size()));
        const std::function<bool(const int*, int)> out_line = [&](const int* t, int n) {
            std::string l = "{\"t\": [";
            for (int i = 0; i < n; ++i) l += (i ? ", " : "") + std::to_string(t[i]);
            l += "]}\n";
            return write(c, l.data(), l.size()) == (ssize_t) l.size();
        };
        const bool streaming = json_int(req, "stream", 0) != 0;
        static const bool snap_on = getenv("INK2_SNAP") ? atoi(getenv("INK2_SNAP")) != 0 : true;
        if (json_int(req, "forget", 0)) {   // tests: no slot keeps a conversation
            std::lock_guard<std::mutex> l(g_cm);
            for (Conv* x : g_convs)
                if (!x->busy) {
                    x->processed.clear();
                    x->n_past = 0;
                    x->snap_n = -1;
                    x->p_reset = true;
                    x->p_restore = false;
                    x->p_roll = -1;
                }
        }
        double wait_s = 0;
        cv = take(tok, snap_on, wait_s);
        ++g_active;
        // "nosnap": 1 reads the whole prompt as ink2_v1's CLI does (no p8 snapshot): the tests compare its logits with that
        Res g = run(*cv, tok, max, stop, json_int(req, "noeos", 0) ? -1 : eos, (int) json_int(req, "lookup", 0), (int) json_int(req, "mtp", 0),
                    snap_on && !json_int(req, "nosnap", 0),
                    json_int(req, "lhall", 0) != 0, streaming ? &out_line : nullptr, json_str(req, "logits_out"));
        --g_active;
        g.wait_s = wait_s;
        give(cv);
        cv = nullptr;
        rep = "{\"tokens\": [";
        for (size_t i = 0; i < g.out.size(); ++i) rep += (i ? ", " : "") + std::to_string(g.out[i]);
        char hd[640];
        std::snprintf(hd, sizeof hd,
                      "], \"prompt_n\": %zu, \"cached\": %d, \"read_s\": %.4f, \"write_s\": %.4f, \"drafted\": %d, \"accepted\": %d, "
                      "\"rounds\": %d, \"draft_s\": %.4f, \"verify_s\": %.4f, \"back\": -1, \"ckpts\": 0, \"ck_s\": 0, \"park\": 0, "
                      "\"parks\": 0, \"park_mb\": 0, \"slot\": %d, \"wait_s\": %.4f, \"lh\": \"%016llx\", \"lh2\": \"%016llx\"}\n",
                      tok.size(), g.cached, g.read_s, g.write_s, g.drafted, g.accepted, g.rounds, g.draft_s, g.verify_s, g.slot, g.wait_s, g.lh, g.lh2);
        rep += hd;
    } catch (const std::exception& ex) {
        if (cv) {   // what the slot holds is unknown now: start it over
            drain(*cv);
            cv->processed.clear();
            cv->n_past = 0;
            cv->snap_n = -1;
            cv->p_reset = true;
            cv->p_restore = false;
            cv->p_roll = -1;
            --g_active;
            give(cv);
        }
        rep = std::string("{\"error\": \"") + ex.what() + "\"}\n";
    }
    (void) !write(c, rep.data(), rep.size());
    close(c);
}

static int serve(const std::string& path, int eos) {
    g_lsock = socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    std::snprintf(a.sun_path, sizeof a.sun_path, "%s", path.c_str());
    unlink(path.c_str());
    if (bind(g_lsock, (sockaddr*) &a, sizeof a) || listen(g_lsock, 64)) {
        std::perror("bind");
        return 1;
    }
    std::fprintf(stderr, "ink2_pp: posluša na %s (%zu rez)\n", path.c_str(), g_convs.size());
    while (!g_quit) {
        const int c = accept(g_lsock, nullptr, nullptr);
        if (c < 0) {
            if (g_quit) break;
            continue;
        }
        std::thread(handle, c, eos).detach();
    }
    unlink(path.c_str());
    return 0;
}

}  // namespace ink2pp

int main(int argc, char** argv) {
    using namespace ink2pp;
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    signal(SIGPIPE, SIG_IGN);
    setenv("INK2_CKPT", "0", 0);       // each conversation keeps its own slot: no checkpoints, no parked conversations
    setenv("INK2_PARK_MB", "0", 0);
    setenv("INK2_PF_CHUNK", "1024", 0);   // reading chunks of 1024 (one card computes the whole width)
    setenv("INK2_1C_STAGE_MB", "0", 0);   // ink2_1c's staging for RAM experts: none here (0 = its smallest, 5.6 MB)
    std::string model, tokens, sock, logits_out, mtp;   // pp4: --mtp the DSpark drafter, --draft-vocab its vocabulary
    int gen = 0, ctx = 65536, slots = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") model = nx();
        else if (a == "--tokens") tokens = nx();
        else if (a == "--gen") gen = std::stoi(nx());
        else if (a == "--ctx") ctx = std::stoi(nx());
        else if (a == "--slots") slots = std::max(1, std::stoi(nx()));
        else if (a == "--serve") sock = nx();
        else if (a == "--logits") logits_out = nx();
        else if (a == "--split") g_split = std::stoi(nx());
        else if (a == "--mtp") mtp = nx();
        else if (a == "--draft-vocab") g_draft_vocab = nx();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (model.empty() || (sock.empty() && tokens.empty())) {
        std::fprintf(stderr, "usage: ink2_pp --model M.gguf (--serve SOCK | --tokens zetoni.bin [--gen N] [--logits F]) [--slots S] [--ctx C]\n");
        return 1;
    }
    const double t0 = now_s();
    PPCfg c0, c1;
    c0.pp = c1.pp = true;
    c0.dev = 0;
    c0.lo = 0;
    c0.hi = g_split;
    c0.head = true;
    c1.dev = 1;
    c1.lo = g_split;
    c1.hi = NPHYS;
    c1.head = false;
    // pp3: which card holds which layer: INK2_PP_DELITEV "12" (pp1: card 0 = 0..split-1, card 1 the rest) or "il" (interleaved:
    // card 0 = 0-5 and 12-17, card 1 = 6-11 and 18-23); card 0 always holds layer 0 (the embedding) and the head
    const std::string dl = getenv("INK2_PP_DELITEV") ? getenv("INK2_PP_DELITEV") : "il";
    int owner[NPHYS];
    for (int p = 0; p < NPHYS; ++p) owner[p] = dl == "il" ? ((p >= 6 && p < 12) || p >= 18 ? 1 : 0) : (p >= g_split ? 1 : 0);
    if (owner[0] != 0) throw std::runtime_error("INK2_PP_DELITEV: card 0 holds layer 0");
    for (int p = 0; p < NPHYS; ++p) (owner[p] ? c1.mask : c0.mask) |= 1u << p;
    Engine::pp_build(owner, 0);
    {
        std::string t;
        for (const auto& g : Engine::pp_segs()) t += " " + std::to_string(g.card) + ":" + std::to_string(g.ps.size());
        std::fprintf(stderr, "ink2_pp: delitev %s, odseki (kartica:prehodov)%s\n", dl.c_str(), t.c_str());
    }
    if (getenv("INK2_PP_GLAVA") && atoi(getenv("INK2_PP_GLAVA")) != 0) {   // pp2: the head by vocabulary halves, the final mix on card 1 (pp3: off by default)
        c0.vlo = 0;
        c0.vhi = c1.vlo = NV / 2;
        c1.vhi = NV;
        c1.fin = true;
    } else {   // pp1: the whole head and the final mix on card 0
        c0.vlo = 0;
        c0.vhi = NV;
        c0.fin = true;
    }
    if (getenv("INK2_PP_ASYNC")) g_async = atoi(getenv("INK2_PP_ASYNC")) != 0;   // pp8
    if (getenv("INK2_PP_PAKET")) g_paket = std::max(1, atoi(getenv("INK2_PP_PAKET")));   // pp7
    if (getenv("INK2_PP_SPIN_MS")) g_spin = std::max(0.0, atof(getenv("INK2_PP_SPIN_MS")) / 1000.0);
    g_ds = !mtp.empty();
    c0.ds = c1.ds = g_ds;   // pp4: the drafter's features in the carry
    const bool risar1 = getenv("INK2_PP_RISAR") ? atoi(getenv("INK2_PP_RISAR")) == 1 : !g_draft_vocab.empty();   // pp9: card 1 by default
    if (g_ds && risar1) {   // pp6: the drafter and the final mix on card 1, the whole head on card 0
        g_dscard = 1;
        c0.fin = false;
        c1.fin = true;
        c0.vlo = 0;
        c0.vhi = NV;
        c1.vlo = c1.vhi = 0;
        if (Engine::pp_segs()[Engine::pp_segs().size() - 2].card != 1) throw std::runtime_error("INK2_PP_RISAR=1: the last layers must be on card 1");
        if (g_draft_vocab.empty()) throw std::runtime_error("INK2_PP_RISAR=1 needs --draft-vocab (card 1 has no head rows)");
    }
    if (g_ds && (g_dscard == 0) != c0.fin) throw std::runtime_error("--mtp: the drafter needs the final mix on its card (INK2_PP_GLAVA=0)");
    std::thread tl([&]() { E[1] = new Engine(model, ctx, g_dscard == 1 ? mtp : "", c1); });
    E[0] = new Engine(model, ctx, g_dscard == 0 ? mtp : "", c0);
    tl.join();
    const int inflight = getenv("INK2_PP_INFLIGHT") ? std::max(1, atoi(getenv("INK2_PP_INFLIGHT"))) : 2;
    const size_t rows = (size_t) std::max(MAXT, E[0]->pfc);
    for (int i = 0; i < slots; ++i) {
        Conv* cv = new Conv;
        cv->id = i;
        std::thread ta([&]() { E[1]->conv_alloc(cv->dev[1]); });
        E[0]->conv_alloc(cv->dev[0]);
        ta.join();
        cv->steps.resize((size_t) inflight);
        for (Step& s : cv->steps) {
            s.cv = cv;
            for (int dvi = 0; dvi < 2; ++dvi) {   // pp8: an event on each card
                CK(cudaSetDevice(E[dvi]->cards[0].dev));
                CK(cudaEventCreateWithFlags(&s.ev[dvi], cudaEventDisableTiming));
            }
            CK(cudaSetDevice(E[0]->cards[0].dev));
            CK(cudaHostAlloc((void**) &s.x0, rows * N * 4, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &s.pos, rows * 4, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &s.tau, rows * 4, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &s.carry, (size_t) (Engine::CV + Engine::CVF) * rows * N * 4, cudaHostAllocPortable));   // pp4: + the features
            CK(cudaHostAlloc((void**) &s.logits, (size_t) MAXT * NV * 4, cudaHostAllocPortable));
        }
        g_convs.push_back(cv);
    }
    for (int c = 0; c < 2; ++c) {
        size_t fr = 0, tot = 0;
        CK(cudaSetDevice(E[c]->cards[0].dev));
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "ink2_pp: kartica %d: VRAM pogona %.2f GiB (pogovor %.0f MiB), prosto %.2f GiB\n", c,
                     E[c]->cards[0].vram / 1073741824.0, E[c]->conv_bytes / 1048576.0, fr / 1073741824.0);
        if (getenv("INK2_VRAM_LOG"))
            for (auto& [k, v] : E[c]->vtag[0]) std::fprintf(stderr, "ink2_pp: kartica %d VRAM %-20s %8.1f MiB\n", c, k.c_str(), v / 1048576.0);
    }
    std::fprintf(stderr, "ink2_pp: naloženo v %.1f s, %d rez po %d, kos branja %d, v cevi do %d kosov pogovora\n", now_s() - t0, slots, ctx,
                 E[0]->pfc, inflight);
    g_t0 = now_s();
    std::thread(worker, 0).detach();
    std::thread(worker, 1).detach();
    const int eos = (int) E[0]->g.i("tokenizer.ggml.eos_token_id");
    if (!sock.empty()) {
        const int rc = serve(sock, eos);
        std::fflush(stderr);
        _exit(rc);
    }
    FILE* f = std::fopen(tokens.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot read %s\n", tokens.c_str()); return 1; }
    int32_t n = 0;
    if (std::fread(&n, 4, 1, f) != 1) return 1;
    std::vector<int> ids(n);
    if (std::fread(ids.data(), 4, n, f) != (size_t) n) return 1;
    std::fclose(f);
    Conv& cv = *g_convs[0];
    cv.busy = true;
    Res g = run(cv, ids, std::max(1, gen), {}, eos, 0, 0, false, false, nullptr, logits_out);   // as ink2_v1's CLI: no snapshot
    std::printf("poziv %d zetonov v %.3f s (%.1f tok/s)\n", n, g.read_s, n / std::max(1e-9, g.read_s));
    std::printf("prvi %d\n", g.out.empty() ? -1 : g.out[0]);
    if (gen > 0) {
        std::printf("pisanje %zu zetonov v %.3f s (%.2f tok/s):", g.out.size(), g.write_s, (g.out.size() - 1) / std::max(1e-9, g.write_s));
        for (int x : g.out) std::printf(" %d", x);
        std::printf("\n");
    }
    std::fflush(stdout);
    _exit(0);
}
