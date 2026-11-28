// ink2_tpb.cu (2026-11-24 night, tp1): Inkling 2 Mini in production's layout (both cards on every layer, ink2_v1's split by heads and
// expert halves) with conversation slots and batches: the writing pieces of up to INK2_PP_PAKET conversations (MAXT rows) go
// through one forward, every conversation with its own caches and states (the pipeline's ConvDev on each card), the kernels
// that read a conversation's state run group by group with the row count it would have alone (pp7); a reading chunk and a draft
// run for one conversation. One thread drives the engine (both cards through Engine::both). A conversation alone runs as
// production's ink2_v1 does; its output does not depend on the others (the same pieces, each row's sums its own).
//   ink2_tpb --model M.gguf --serve SOCK [--slots 4] [--ctx 40960] [--mtp D.gguf --draft-vocab V.bin]
#define INK2_PP 1
#define INK2_XSEQ 1024
#include "ink2_v1.cu"

#include <condition_variable>
#include <deque>
#include <mutex>

namespace ink2tp {
using namespace ink2;

static Engine* E = nullptr;
static bool g_ds = false;
static int g_paket = 4;        // INK2_PP_PAKET: conversations in one forward at most
static double g_spin = 0.08;   // INK2_PP_SPIN_MS (tp3: 80, a verify forward takes ~30 ms)
static double g_gather = 0.005;   // tp2: INK2_PP_ZBIRAJ_MS (tp3: 5), how long a forward waits for other writing conversations' pieces
static std::atomic<int> g_writers{0};   // tp2: conversations in their generation loop (their next piece comes soon)

struct Conv;
struct Job {   // a piece of a conversation (0 writing, 1 a reading chunk) or its draft (2)
    Conv* cv = nullptr;
    int kind = 0;
    int T = 0, p0 = 0, nparts = 0;
    bool all_rows = false, last = false, keep_hist = false;
    bool reset = false, restore = false, snap = false, bak = false;
    int roll = -1;
    std::vector<int> ids;
    float *x0 = nullptr, *tau = nullptr, *logits = nullptr;
    int* pos = nullptr;
    int anchor = 0, dk = 0, dpos = 0, dctx = -1;
    std::vector<int> drafts;
    bool done = true;
};
struct Conv {
    int id = 0;
    Engine::ConvDev dev[NC];
    std::vector<int> processed;
    int n_past = 0, snap_n = -1;
    bool p_reset = true, p_restore = false;
    int p_roll = -1;
    bool busy = false;
    unsigned long long use = 0;
    Job job;
    std::mutex m;
    std::condition_variable cvd;
};
static std::vector<Conv*> g_convs;
static std::mutex g_cm;
static std::condition_variable g_ccv;
static unsigned long long g_use = 0;

static std::mutex g_qm;
static std::condition_variable g_qcv;
static std::deque<Job*> g_q;
static std::atomic<long long> g_busy_us{0}, g_nfw{0}, g_nbat{0}, g_nrows{0};
static double g_t0 = 0;

static void finish(Job* j) {
    Conv& cv = *j->cv;
    {
        std::lock_guard<std::mutex> l(cv.m);
        j->done = true;
    }
    cv.cvd.notify_all();
}
static void wait_job(Conv& cv, Job& j) {
    for (const double t0 = now_s(); now_s() - t0 < g_spin;) {
        {
            std::lock_guard<std::mutex> l(cv.m);
            if (j.done) return;
        }
        std::this_thread::yield();
    }
    std::unique_lock<std::mutex> l(cv.m);
    cv.cvd.wait(l, [&]() { return j.done; });
}
static void push(Job* j) {
    {
        std::lock_guard<std::mutex> l(g_qm);
        g_q.push_back(j);
    }
    g_qcv.notify_one();
}

static Engine::SegIn ops_of(const Job& j) {
    Engine::SegIn in;
    in.reset = j.reset;
    in.restore = j.restore;
    in.roll = j.roll;
    in.snap = j.snap;
    in.bak = j.bak;
    return in;
}
// a batch on card c (both cards at once): the conversations' state ops, the 48 passes over all rows with each conversation's
// groups, the final mix and norm, the drafter's ingest group by group, the head over all rows, each piece's logit rows
static void fwd_card(Engine& e, int c, std::vector<Job*>& js) {
    Card& K = e.cards[c];
    CK(cudaSetDevice(K.dev));
    for (Job* j : js) {
        e.conv_use(j->cv->dev[c], c);
        e.seg_ops(ops_of(*j), c);
    }
    std::vector<Engine::BGroup>& bg = e.bgc[c];
    bg.clear();
    int T = 0;
    for (Job* j : js) {
        Engine::BGroup b;
        b.r0 = T;
        b.T = j->T;
        b.nparts = j->nparts;
        b.d = &j->cv->dev[c];
        b.hist = j->keep_hist;
        bg.push_back(b);
        T += j->T;
    }
    K.xidx = 0;
    K.keep_hist = false;
    k_tick<<<1, 1, 0, K.st>>>(K.dtok);
    for (size_t g = 0; g < js.size(); ++g) {
        const Job& j = *js[g];
        CK(cudaMemcpyAsync(K.x0 + (size_t) bg[g].r0 * N, j.x0, (size_t) j.T * N * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(K.pos + bg[g].r0, j.pos, j.T * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(K.tau + bg[g].r0, j.tau, j.T * 4, cudaMemcpyHostToDevice, K.st));
    }
    e.rms(K, K.x0, K.tok_norm, K.cur, T, e.eps);
    K.nb = 0;
    for (int p = 0; p < LS; ++p) e.layer(K, p, p, false, T, 0);
    e.copy(K, K.prel, K.cur, T);
    const int nprel = K.nb;
    for (int pass = 0; pass < NPASS; ++pass) {
        if (pass > 0) {
            e.copy(K, K.bankS, K.B[nprel + 1], T);
            e.copy(K, K.bankO, K.cur, T);
            e.copy(K, K.cur, K.prel, T);
            K.nb = nprel;
        }
        for (int ip = LS; ip < LE; ++ip) e.layer(K, ip, pass == 0 ? ip : NPHYS + (pass - 1) * NCORE + (ip - LS), pass > 0, T, 0);
    }
    for (int p = LE; p < NPHYS; ++p) e.layer(K, p, p, false, T, 0);
    bg.clear();
    e.mix(K, e.states(K, K.cur), K.out_rq, 1, K.F, T);
    e.rms(K, K.F, K.out_norm, K.hx, T, e.eps);
    if (e.ds_on) {   // forward_body's ingest, conversation by conversation
        int r0 = 0;
        for (Job* j : js) {
            e.conv_use(j->cv->dev[c], c);
            DSW& D = K.D;
            e.ds_feat(K, D.feat + (size_t) r0 * 4 * N, K.hx + (size_t) r0 * N, 3, j->T);
            e.ds_ingest(K, D.feat + (size_t) r0 * 4 * N, D.frnn, D.cbuf, D.kb, D.vb, K.pos + r0, j->T, false);
            CK(cudaMemcpyAsync(e.ds_frnnw_c[c], D.frnn, (size_t) j->T * N * 4, cudaMemcpyDeviceToDevice, K.st));
            r0 += j->T;
        }
    } else if (e.d2_on) {   // p74tp: DFlash 2's ingest, conversation by conversation
        int r0 = 0;
        for (Job* j : js) {
            e.conv_use(j->cv->dev[c], c);
            e.ds_feat(K, K.D.feat + (size_t) r0 * 4 * N, K.hx + (size_t) r0 * N, 3, j->T);
            e.d2_ingest(K, K.D.feat + (size_t) r0 * 4 * N, K.pos + r0, j->T, false);
            r0 += j->T;
        }
    }
    bool any = false;
    for (Job* j : js) any |= j->last;
    if (any) {
        if (T == 1) e.gemv8_t1(K, K.out, K.hx, e.segs({{NV_C, K.logits}}));
        else e.gemv8_t1T(K, K.out, K.hx, N, e.segs1(NV_C, K.logits), T);
        k_softcap<<<(T * NV_C + 255) / 256, 256, 0, K.st>>>(K.logits, T * NV_C, e.softcap);
        int r0 = 0;
        for (Job* j : js) {
            if (j->last) {
                const int a = j->all_rows ? 0 : j->T - 1;
                CK(cudaMemcpy2DAsync(j->logits + (size_t) c * NV_C, (size_t) NV * 4, K.logits + (size_t) (r0 + a) * NV_C, (size_t) NV_C * 4,
                                     (size_t) NV_C * 4, j->T - a, cudaMemcpyDeviceToHost, K.st));
            }
            r0 += j->T;
        }
    }
    CK(cudaStreamSynchronize(K.st));
}
// tp4: pf_card in 8 parts (the prelude; each repetition of the loop in two halves; the tail, the drafter's ingest and the head),
// the same kernels in the same order: between the parts other conversations' writing pieces and drafts run (the reading path
// keeps its rows in P.*, the writing path in K.*; the conversation's slot pointers and K.nb are set again at each part)
struct ReadState {
    int nb[NC] = {}, nprel[NC] = {};
};
static void pf_part(Engine& e, int c, Job& j, int part, ReadState& rs) {
    Card& K = e.cards[c];
    PFB& P = K.P;
    CK(cudaSetDevice(K.dev));
    e.conv_use(j.cv->dev[c], c);
    const int C = j.T, p0 = j.p0;
    const size_t CN = (size_t) C * N;
    if (part == 0) {
        e.seg_ops(ops_of(j), c);
        CK(cudaMemcpyAsync(P.x0, e.h_x0, CN * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(P.pos, e.h_pos, C * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(P.tau, e.h_tau, C * 4, cudaMemcpyHostToDevice, K.st));
        k_rms<<<C, 256, 0, K.st>>>(P.x0, N, K.tok_norm, P.cur, N, N, e.eps);
        e.raw_rows(K, P.x0, P.cur, C);
        K.nb = 0;
        for (int p = 0; p < LS; ++p) e.layer_pf(K, p, p, false, p0, C);
        CK(cudaMemcpyAsync(P.prel, P.cur, CN * 4, cudaMemcpyDeviceToDevice, K.st));
        rs.nprel[c] = K.nb;
    } else if (part <= 2 * NPASS) {
        const int pass = (part - 1) / 2, half = (part - 1) % 2;
        K.nb = rs.nb[c];
        if (half == 0 && pass > 0) {
            CK(cudaMemcpyAsync(P.bankS, P.B[rs.nprel[c] + 1], CN * 4, cudaMemcpyDeviceToDevice, K.st));
            CK(cudaMemcpyAsync(P.bankO, P.cur, CN * 4, cudaMemcpyDeviceToDevice, K.st));
            CK(cudaMemcpyAsync(P.cur, P.prel, CN * 4, cudaMemcpyDeviceToDevice, K.st));
            K.nb = rs.nprel[c];
        }
        const int a = half == 0 ? LS : (LS + LE) / 2, b = half == 0 ? (LS + LE) / 2 : LE;
        for (int ip = a; ip < b; ++ip) e.layer_pf(K, ip, pass == 0 ? ip : NPHYS + (pass - 1) * NCORE + (ip - LS), pass > 0, p0, C);
    } else {
        K.nb = rs.nb[c];
        for (int p = LE; p < NPHYS; ++p) e.layer_pf(K, p, p, false, p0, C);
        if (e.ds_on) {
            e.mix(K, e.states_pf(K, P.cur), K.out_rq, 1, P.F, C);
            k_rms<<<C, 256, 0, K.st>>>(P.F, N, K.out_norm, P.F2, N, N, e.eps);
            e.ds_feat(K, K.D.pfeat, P.F2, 3, C);
            e.ds_ingest(K, K.D.pfeat, K.D.pfrnn, K.D.pcbuf, K.D.pkb, K.D.pvb, P.pos, C, true);
        } else if (e.d2_on) {   // p74tp: the chunk's last D2RING rows into this conversation's rings
            e.mix(K, e.states_pf(K, P.cur), K.out_rq, 1, P.F, C);
            k_rms<<<C, 256, 0, K.st>>>(P.F, N, K.out_norm, P.F2, N, N, e.eps);
            e.ds_feat(K, K.D.pfeat, P.F2, 3, C);
            const int r0 = std::max(0, C - D2RING);
            e.d2_ingest(K, K.D.pfeat + (size_t) r0 * 4 * N, P.pos + r0, C - r0, true);
        }
        if (j.last) {
            States S{};
            for (int i = 0; i < K.nb; ++i) S.p[i] = P.B[i] + (size_t) (C - 1) * N;
            S.p[K.nb] = P.cur + (size_t) (C - 1) * N;
            S.n = K.nb + 1;
            e.mix(K, S, K.out_rq, 1, P.F, 1);
            e.rms(K, P.F, K.out_norm, P.hx1, 1, e.eps);
            e.gemv8_t1(K, K.out, P.hx1, e.segs({{NV_C, K.logits}}));
            k_softcap<<<(NV_C + 255) / 256, 256, 0, K.st>>>(K.logits, NV_C, e.softcap);
            CK(cudaMemcpyAsync(e.h_logits + (size_t) c * NV_C, K.logits, (size_t) NV_C * 4, cudaMemcpyDeviceToHost, K.st));
        }
    }
    rs.nb[c] = K.nb;
    CK(cudaStreamSynchronize(K.st));
}
static void run_waiting(Engine& e);   // tp4: the writing pieces and drafts queued meanwhile
static bool g_interleave = true;      // tp4: INK2_PP_PREPLETI (1): writes between a reading chunk's parts
static void run_read(Engine& e, Job& j) {   // a reading chunk (pf_card) of one conversation
    const int C = j.T;
    for (int t = 0; t < C; ++t) {
        e.h_raw[t] = e.embed(t, j.ids[t]);
        const int P = j.p0 + t;
        e.h_pos[t] = P;
        e.h_tau[t] = logf(fmaxf(((float) P + 1.0f) * (1.0f / e.log_floor), 1.0f)) * e.log_alpha + 1.0f;
    }
    ReadState rs;
    for (int part = 0; part <= 2 * NPASS + 1; ++part) {
        e.both([&](int c) { pf_part(e, c, j, part, rs); });
        if (g_interleave && part <= 2 * NPASS) run_waiting(e);
    }
    if (j.last) std::memcpy(j.logits, e.h_logits, (size_t) NV * 4);
}
static void run_draft(Engine& e, Job& j) {
    e.both([&](int c) {
        CK(cudaSetDevice(e.cards[c].dev));
        e.conv_use(j.cv->dev[c], c);
        Card& K = e.cards[c];
        if (j.dctx >= 0 && !e.d2_on)   // Engine::verify's D.ctxv = the kept row's f_rnn (DSpark's; p74tp)
            CK(cudaMemcpyAsync(K.D.ctxv, j.cv->dev[c].frnnw + (size_t) j.dctx * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, K.st));
    });   // tp2: no wait (ds_block follows on the same stream)
    e.n_past = j.dpos;
    j.drafts = e.d2_on ? e.d2_draft(j.anchor, j.dk) : e.ds_draft(j.anchor, j.dk);   // p74tp
}
static void run_waiting(Engine& e) {
    for (;;) {
        std::vector<Job*> bat;
        Job* dj = nullptr;
        {
            std::lock_guard<std::mutex> l(g_qm);
            for (auto it = g_q.begin(); it != g_q.end(); ++it)
                if ((*it)->kind == 2) {
                    dj = *it;
                    g_q.erase(it);
                    break;
                }
            if (!dj) {
                int rows = 0;
                for (auto it = g_q.begin(); it != g_q.end() && (int) bat.size() < g_paket;) {
                    Job* o = *it;
                    bool ok = o->kind == 0 && rows + o->T <= MAXT;
                    for (Job* x : bat) ok = ok && x->cv != o->cv;
                    if (ok) {
                        bat.push_back(o);
                        rows += o->T;
                        it = g_q.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
        }
        if (dj) {
            run_draft(e, *dj);
            finish(dj);
            continue;
        }
        if (bat.empty()) return;
        e.both([&](int c) { fwd_card(e, c, bat); });
        ++g_nfw;
        if (bat.size() > 1) ++g_nbat;
        for (Job* x : bat) g_nrows += x->T;
        for (Job* x : bat) finish(x);
        return;   // one batch between two parts
    }
}
static void sched() {
    Engine& e = *E;
    CK(cudaSetDevice(e.cards[0].dev));
    for (;;) {
        std::vector<Job*> bat;
        {
            std::unique_lock<std::mutex> l(g_qm);
            for (const double t0 = now_s(); g_q.empty() && now_s() - t0 < g_spin;) {
                l.unlock();
                std::this_thread::yield();
                l.lock();
            }
            g_qcv.wait(l, []() { return !g_q.empty(); });
            Job* j = g_q.front();
            g_q.pop_front();
            bat.push_back(j);
            if (j->kind == 0 && g_paket > 1) {   // other conversations' writing pieces into the same forward
                int rows = j->T;
                auto gather = [&]() {
                    for (auto it = g_q.begin(); it != g_q.end() && (int) bat.size() < g_paket;) {
                        Job* o = *it;
                        bool ok = o->kind == 0 && rows + o->T <= MAXT;
                        for (Job* x : bat) ok = ok && x->cv != o->cv;
                        if (ok) {
                            bat.push_back(o);
                            rows += o->T;
                            it = g_q.erase(it);
                        } else {
                            ++it;
                        }
                    }
                };
                gather();
                // tp2: other conversations that are writing send their next piece within ~1 ms (or after a draft): wait for them
                // a little, running the drafts that come meanwhile (their pieces follow)
                const double t0 = now_s();
                while ((int) bat.size() < std::min(g_paket, g_writers.load()) && rows < MAXT && now_s() - t0 < g_gather) {
                    Job* dj = nullptr;
                    for (auto it = g_q.begin(); it != g_q.end(); ++it)
                        if ((*it)->kind == 2) {
                            dj = *it;
                            g_q.erase(it);
                            break;
                        }
                    l.unlock();
                    if (dj) {
                        run_draft(e, *dj);
                        finish(dj);
                    } else {
                        std::this_thread::yield();
                    }
                    l.lock();
                    gather();
                }
            }
        }
        const double t = now_s();
        Job* j = bat[0];
        if (j->kind == 2) run_draft(e, *j);
        else if (j->kind == 1) run_read(e, *j);
        else {
            e.both([&](int c) { fwd_card(e, c, bat); });
            ++g_nfw;
            if (bat.size() > 1) ++g_nbat;
            for (Job* x : bat) g_nrows += x->T;
        }
        g_busy_us += (long long) ((now_s() - t) * 1e6);
        for (Job* x : bat) finish(x);
    }
}

static void embed_row(float* dst, int id) {
    if (id < 0 || id >= NV) throw std::runtime_error("token id out of range: " + std::to_string(id));
    const auto* tok = E->tok;
    const uint8_t* row = tok->data + (size_t) id * tok->row_bytes();
    for (int b = 0; b < N / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * 34;
        uint16_t hb;
        std::memcpy(&hb, blk, 2);
        __half_raw hr;
        hr.x = hb;
        const float d = __half2float(__half(hr));
        for (int k = 0; k < 32; ++k) dst[b * 32 + k] = d * (float) (int8_t) blk[2 + k];
    }
}
// a piece of the conversation (feed / a token / a draft window) through the engine; waits for it
static Job* run_piece(Conv& cv, const int* ids, int T, bool read, bool all_rows, bool last, bool keep_hist = false, bool snap = false,
                      bool bak = false) {
    Job& j = cv.job;
    wait_job(cv, j);
    if (cv.n_past + T > E->ctx) throw std::runtime_error("context full");
    j.kind = read ? 1 : 0;
    j.T = T;
    j.p0 = cv.n_past;
    j.nparts = (cv.n_past + T + KC - 1) / KC + 1;
    j.all_rows = all_rows;
    j.last = last;
    j.keep_hist = keep_hist;
    j.reset = cv.p_reset;
    j.restore = cv.p_restore;
    j.roll = cv.p_roll;
    j.snap = snap;
    j.bak = bak;
    cv.p_reset = cv.p_restore = false;
    cv.p_roll = -1;
    if (read) {
        j.ids.assign(ids, ids + T);
    } else {
        for (int t = 0; t < T; ++t) {
            embed_row(j.x0 + (size_t) t * N, ids[t]);
            const int P = cv.n_past + t;
            j.pos[t] = P;
            j.tau[t] = logf(fmaxf(((float) P + 1.0f) * (1.0f / E->log_floor), 1.0f)) * E->log_alpha + 1.0f;
        }
    }
    {
        std::lock_guard<std::mutex> l(cv.m);
        j.done = false;
    }
    cv.n_past += T;
    cv.processed.insert(cv.processed.end(), ids, ids + T);
    push(&j);
    wait_job(cv, j);
    return &j;
}
static std::vector<int> draft(Conv& cv, int anchor, int k, int ctx_row) {
    Job& j = cv.job;
    wait_job(cv, j);
    j.kind = 2;
    j.anchor = anchor;
    j.dk = k;
    j.dpos = cv.n_past;
    j.dctx = ctx_row;
    j.drafts.clear();
    {
        std::lock_guard<std::mutex> l(cv.m);
        j.done = false;
    }
    push(&j);
    wait_job(cv, j);
    return j.drafts;
}
static Job* feed(Conv& cv, const int* ids, int n, bool head) {   // Engine::feed: chunks of pf_min tokens or more, windows of feed_t
    const Engine& e = *E;
    Job* s = nullptr;
    for (int i = 0, T = 0; i < n; i += T) {
        const bool rd = n - i >= e.pf_min;
        T = std::min(rd ? e.pfc : e.feed_t, n - i);
        s = run_piece(cv, ids + i, T, rd, false, head && i + T >= n);
    }
    return s;
}
static bool restorable(const Conv& cv, size_t same, size_t n) {
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
// ink2_pp.cu's run(): the same generate() of one conversation, its pieces through the batching engine
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
    if (same != cv.processed.size() || same == tok.size()) {
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
    Job* last;
    if (snap_on && n - (int) same >= 2) {
        feed(cv, tok.data() + same, n - 1 - (int) same, false);
        last = run_piece(cv, tok.data() + n - 1, 1, false, false, true, false, true);
        cv.snap_n = n - 1;
    } else {
        last = feed(cv, tok.data() + same, n - (int) same, true);
    }
    g.read_s = now_s() - t0;
    g.lh = fnv_logits(1469598103934665603ULL, last->logits);
    if (!logits_out.empty()) {
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
    ++g_writers;   // tp2
    struct WriterGuard {
        ~WriterGuard() { --g_writers; }
    } writer_guard;
    static const int ng_min = getenv("INK2_NG_MIN") ? atoi(getenv("INK2_NG_MIN")) : 6;
    while ((int) g.out.size() < max) {
        g.out.push_back(next);
        if (!flush()) break;
        if (ends(next) || (int) g.out.size() >= max || cv.n_past >= E->ctx) break;
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
        if (d.empty() && mtp > 0) {
            mtpd = true;
            const double td = now_s();
            d = draft(cv, next, std::min(mtp, max - (int) g.out.size()), ctx_row);
            g.draft_s += now_s() - td;
        }
        ctx_row = -1;
        if ((int) d.size() > E->ctx - cv.n_past - 1) d.resize(std::max(0, E->ctx - cv.n_past - 1));
        if (d.empty()) {
            Job* s = run_piece(cv, &next, 1, false, false, true);
            next = Engine::argmax(s->logits);
            if (lhall) g.lh2 = fnv_logits(g.lh2, s->logits);
            continue;
        }
        std::vector<int> win{next};
        win.insert(win.end(), d.begin(), d.end());
        const int T = (int) win.size();
        const double tv = now_s();
        Job* s = run_piece(cv, win.data(), T, false, true, true, true, false, true);
        int right = 0;
        while (right + 1 < T && Engine::argmax(s->logits + (size_t) right * NV) == win[right + 1]) ++right;
        const int nn = Engine::argmax(s->logits + (size_t) right * NV);
        if (lhall)
            for (int i = 0; i <= right; ++i) g.lh2 = fnv_logits(g.lh2, s->logits + (size_t) i * NV);
        const int keep = right + 1;
        ctx_row = right;
        if (keep < T) {
            cv.p_roll = keep;
            cv.n_past -= T - keep;
            cv.processed.resize(cv.processed.size() - (size_t) (T - keep));
        }
        g.verify_s += now_s() - tv;
        ++g.rounds;
        g.drafted += (int) d.size();
        g.accepted += right;
        if (lookup > 0 && !mtpd) {
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
    if (json_int(req, "stats", 0)) {
        char hd[400];
        const double up = now_s() - g_t0;
        std::snprintf(hd, sizeof hd, "{\"up_s\": %.3f, \"busy_s\": [%.3f, %.3f], \"forwards\": %lld, \"batches\": %lld, \"rows\": %lld, \"active\": %d}\n", up,
                      g_busy_us / 1e6, g_busy_us / 1e6, (long long) g_nfw, (long long) g_nbat, (long long) g_nrows, g_active.load());
        if (json_int(req, "stats", 0) == 2) {
            g_t0 = now_s();
            g_busy_us = 0;
            g_nfw = g_nbat = g_nrows = 0;
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
        if (!json_ints(req, "img").empty()) throw std::runtime_error("ink2_tpb: no pictures in this build");
        const int ctx = E->ctx;
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
        if (json_int(req, "forget", 0)) {
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
        Res g = run(*cv, tok, max, stop, json_int(req, "noeos", 0) ? -1 : eos, (int) json_int(req, "lookup", 0), (int) json_int(req, "mtp", 0),
                    snap_on && !json_int(req, "nosnap", 0), json_int(req, "lhall", 0) != 0, streaming ? &out_line : nullptr, json_str(req, "logits_out"));
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
        if (cv) {
            wait_job(*cv, cv->job);
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
    std::fprintf(stderr, "ink2_tpb: posluša na %s (%zu rez)\n", path.c_str(), g_convs.size());
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

}  // namespace ink2tp

int main(int argc, char** argv) {
    using namespace ink2tp;
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    signal(SIGPIPE, SIG_IGN);
    setenv("INK2_CKPT", "0", 0);
    setenv("INK2_PARK_MB", "0", 0);
    std::string model, tokens, sock, logits_out, mtp;
    int gen = 0, ctx = 40960, slots = 4;
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
        else if (a == "--mtp") mtp = nx();
        else if (a == "--draft-vocab") g_draft_vocab = nx();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (model.empty() || (sock.empty() && tokens.empty())) {
        std::fprintf(stderr, "usage: ink2_tpb --model M.gguf (--serve SOCK | --tokens zetoni.bin [--gen N] [--logits F]) [--slots S] [--ctx C] [--mtp D --draft-vocab V]\n");
        return 1;
    }
    if (getenv("INK2_PP_PAKET")) g_paket = std::max(1, atoi(getenv("INK2_PP_PAKET")));
    if (getenv("INK2_PP_PREPLETI")) g_interleave = atoi(getenv("INK2_PP_PREPLETI")) != 0;   // tp4
    if (getenv("INK2_PP_ZBIRAJ_MS")) g_gather = std::max(0.0, atof(getenv("INK2_PP_ZBIRAJ_MS")) / 1000.0);   // tp2
    if (getenv("INK2_PP_SPIN_MS")) g_spin = std::max(0.0, atof(getenv("INK2_PP_SPIN_MS")) / 1000.0);
    const double t0 = now_s();
    PPCfg pc;
    pc.pp = true;   // the conversations' slots instead of the engine's own caches; every expert in VRAM; production's head (NC = 2)
    pc.lo = 0;
    pc.hi = NPHYS;
    pc.head = pc.fin = true;
    E = new Engine(model, ctx, mtp, pc);
    g_ds = !mtp.empty();
    for (int i = 0; i < slots; ++i) {
        Conv* cv = new Conv;
        cv->id = i;
        E->each_card([&](int c) { E->conv_alloc(cv->dev[c], c); });
        CK(cudaSetDevice(E->cards[0].dev));
        CK(cudaHostAlloc((void**) &cv->job.x0, (size_t) MAXT * N * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &cv->job.pos, (size_t) MAXT * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &cv->job.tau, (size_t) MAXT * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &cv->job.logits, (size_t) MAXT * NV * 4, cudaHostAllocPortable));
        cv->job.cv = cv;
        g_convs.push_back(cv);
    }
    for (int c = 0; c < NC; ++c) {
        size_t fr = 0, tot = 0;
        CK(cudaSetDevice(E->cards[c].dev));
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "ink2_tpb: kartica %d: VRAM pogona %.2f GiB (pogovor %.0f MiB), prosto %.2f GiB\n", c, E->cards[c].vram / 1073741824.0,
                     E->conv_bytes / 1048576.0, fr / 1073741824.0);
    }
    std::fprintf(stderr, "ink2_tpb: naloženo v %.1f s, %d rez po %d, paket do %d pogovorov, risar %s\n", now_s() - t0, slots, ctx, g_paket,
                 g_ds ? "da" : "ne");
    g_t0 = now_s();
    std::thread(sched).detach();
    const int eos = (int) E->g.i("tokenizer.ggml.eos_token_id");
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
    Res g = run(cv, ids, std::max(1, gen), {}, eos, 0, 0, false, false, nullptr, logits_out);
    std::printf("poziv %d zetonov v %.3f s (%.1f tok/s)\nprvi %d\n", n, g.read_s, n / std::max(1e-9, g.read_s), g.out.empty() ? -1 : g.out[0]);
    if (gen > 0) {
        std::printf("pisanje %zu zetonov v %.3f s (%.2f tok/s):", g.out.size(), g.write_s, (g.out.size() - 1) / std::max(1e-9, g.write_s));
        for (int x : g.out) std::printf(" %d", x);
        std::printf("\n");
    }
    std::fflush(stdout);
    _exit(0);
}
