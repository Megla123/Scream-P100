// ink2_v1.cu (2026-11-21): Inkling 2 Mini on both P100s at once, after Hy4's hy4_v1. Every layer pass runs on both cards: the
// attention's heads are divided (12 / 12 query heads, 4 / 4 KV heads; each card keeps only its heads' KV), every expert's FF width
// (576 / 576: gate and up rows, down columns), the output vocabulary (100512 / 100512). The depth mixing, the loop's messages and
// controllers, the norms, the router and the post-convolutions run on both cards on the same input (the same bits). Two sums
// across the cards a layer pass (the attention's wo and the MoE output) through page-locked mapped RAM (k_xput / k_xget).
// The whole model stays in VRAM (~8.6 GB a card). Prompts run in windows of MAXT tokens (the writing kernels); see VODNIK-inkling2.md.
//   ink2_v1 --model M.gguf --tokens zetoni.bin [--gen N] [--dump DIR] [--ctx C]
//   ink2_v1 --model M.gguf --serve SOCKET [--ctx C]
#include "ink2_kern.cuh"
#include "ink2_fuse.cuh"
#include "ink2_pf.cuh"
#include "ink2_dspark.cuh"
#include "ink2_dflash2.cuh"   // p74
#include "gguf_min.hpp"

#include <cublas_v2.h>

#include <poll.h>
#include <pthread.h>   // p41
#include <sched.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <climits>
#include <csignal>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace ink2 {

static FILE* g_rtrace = nullptr;   // p31: INK2_RTRACE
static std::string g_draft_vocab;   // p13: --draft-vocab FILE (int32 count + token ids), "" = the whole vocabulary
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

__global__ void k_f2h43(const float* x, __half* y, size_t n) {   // p43
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}
struct LW {   // one card's part of a physical layer
    bool full = false;
    int nrq = 0;
    const float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr, *attn_rq = nullptr, *ffn_rq = nullptr;
    Mat wq, wk, wv, wo, wgate, wrel, wsink, wkd, wvd;
    Mat cat;   // p4a: [wq | wgate | wrel | wsink | wk | (wkd) | wv | (wvd)] in one Q8S matrix; the Mats above are views into it
    const float* rel_bias_pad = nullptr;   // p3: [EXT + 1][NREL], the last row 0 (the reading path's R.B table)
    const float *q_clip, *rel_clip, *rel_bias, *k_conv, *v_conv, *a_pconv, *f_pconv, *ffn_norm, *gate_inp, *probs_b;
    bool fp4 = false;
    const uint8_t *gu = nullptr, *dn = nullptr;   // NVFP4 rows, or Q8S values
    const __half *gu_d = nullptr, *dn_d = nullptr;
    const float *gu_s = nullptr, *dn_s = nullptr;
    const uint8_t* const* gu_t = nullptr;   // p30: every expert's base (gate_up, down): in VRAM or in mapped page-locked RAM
    const uint8_t* const* dn_t = nullptr;
    int nres = NE;   // p30: experts in VRAM
    std::vector<char> res;                    // p31: which experts are in VRAM
    std::vector<const uint8_t*> h_gu, h_dn;   // p31: the bases on the host side (a RAM expert's mapped address)
    ECache ec;                                // p32: the writing path's cache of this layer's RAM experts
    const float* msg_q = nullptr;   // loop layers 6-17
    const float *c_down[4] = {}, *c_gain[4] = {}, *c_forget[4] = {};
};

// p6: the DSpark drafter on one card (whole, on both cards)
struct DSW {
    Mat wq[DSL], wk[DSL], wv[DSL], wo[DSL], kctx[DSL], vctx[DSL], gate[DSL], up[DSL], down[DSL];
    Mat dvout;   // p13: the output rows of this card's half of the draft vocabulary
    const float *attn_norm[DSL], *ffn_norm[DSL], *dflare[DSL], *qn[DSL], *kn[DSL], *kcn[DSL], *fuse, *mask, *dsnorm, *outnorm, *jb, *hb;
    const uint16_t *w1, *jw, *hw;
    __half *kc[DSL], *vc[DSL];
    float *feat, *frnn, *ctxv, *cbuf, *kb, *vb, *x, *xn, *q, *k, *v, *ao, *o, *g, *u, *h, *part, *hid, *state, *jin, *jout, *lat, *y, *amv, *ams;
    float *pfeat = nullptr, *pfrnn = nullptr, *pcbuf = nullptr, *pkb = nullptr, *pvb = nullptr;   // the reading path's rows
    int *ami, *prev, *bpos;
    int nparts = 0;
};

// p74: the DFlash 2 drafter on one card (whole, on both cards); its features are DSW's feat / pfeat (the same four as DSpark's)
struct D2W {
    Mat fc, selh, wq[D2L], wk[D2L], wv[D2L], wo[D2L], gate[D2L], up[D2L], down[D2L], aproj[D2L], mproj[D2L];
    const float *hnorm, *onorm, *anorm[D2L], *fnorm[D2L], *qn[D2L], *kn[D2L], *abase[D2L], *mbase[D2L];
    __half *kc[D2L], *vc[D2L];
    float *mrow, *cx, *xin, *kb, *vb, *x, *xn, *dyn, *c, *q, *k, *v, *ao, *o, *t1, *g, *u, *h, *part, *hid, *gout, *tv, *tv1;
    int *ti, *ti1, *bpos;
    int nparts = 0;
    float* part2 = nullptr;   // k_d2_gemv2's column chunks
    size_t partn = 0;
};

// p3: one card's buffers of the reading path (a chunk of up to C tokens)
struct PFB {
    int C = 0;
    size_t S_n = 0;   // p14: floats in S
    float* PV = nullptr;   // p14b: P.V parts by key pieces [3][nsbmax][QSF][HD]
    int pv_nsb = 0;
    bool shared = false;   // p18: Kf, Vf, RB and PV inside Xg, EY and GU (the int8 mode)
    cublasHandle_t cb;
    float *x0, *cur, *B[MAXB], *prel, *bankS, *bankO, *X3, *MS, *XC[4], *xm, *g64, *gp, *fpp, *NQ, *NK, *NV, *q, *k, *v, *kcv, *vcv, *r,
        *sink, *gate, *dk, *dv, *ao, *ab, *A, *F, *F2, *FN, *lg, *topw, *MO, *tau, *Xg, *GU, *HH, *EY, *Wf, *Wo, *Egu, *Edn, *Kf, *Vf, *S,
        *RB, *tmp, *hx1;
    float **pk = nullptr, **pq, **ps, **pv, **pao;   // p3f: pointer arrays of the window layers' batched attention
    int *pos, *topi, *cnt, *offs, *cursor, *slotrow, *rowtok, *rowexp;
    uint8_t* stg[2] = {};           // p31: staging buffers of the RAM experts
    size_t stg_bytes = 0;
    const uint8_t** d_stab = nullptr;   // p31: a group's expert table (gate_up, down) for each group, device and page-locked host
    const uint8_t** h_stab = nullptr;
    cudaEvent_t evS[2], evU[2];     // p31: a staging buffer filled / free again
    float* hbuf[2] = {};   // page-locked: this card's part of an exchange, by parity
    cudaEvent_t ev[2];
    static constexpr int XS_MAX = 4;
    cudaStream_t sd = nullptr, sh = nullptr;   // p19d: the exchange's copy streams (to RAM, back from RAM)
    cudaEvent_t evp[XS_MAX], evd[2][XS_MAX], evh;   // p19d: slice made, slice in RAM (by parity), the other card's slices back
    std::vector<int> h_cnt, h_offs;
    FTile* tiles = nullptr;   // p19b: k_fp4_gemm5's tile list
    std::vector<FTile> h_tiles;
    unsigned seq = 1;
};

// zip (INK2_PF_ZIP): a reading chunk's own buffers and its exchange in flight, for several chunks at once on a card
struct ZCtx {
    float *cur, *B[MAXB], *prel, *bankS, *bankO, *ms3 = nullptr, *ab, *MO, *tau, *pfeat = nullptr;   // ms3: chunks 1.. only
    int* pos = nullptr;
    int nb = 0;
    float* hbuf[2] = {};
    cudaEvent_t evp[PFB::XS_MAX], evd[2][PFB::XS_MAX], evh, evb;
    unsigned seq = 1;
    float* py = nullptr;   // the exchange in flight: its rows (ab or MO), their number, its parity, its number
    int pC = 0, ppar = 0;
    unsigned psq = 0;
    bool want = false;     // its slices are on their way to RAM, the other card's are not ordered back yet
};

struct Card {
    PFB P;
    static constexpr int ZMAX = 3;
    ZCtx Z[ZMAX];   // zip
    int zc = 0;     // zip: the chunk whose buffers P points at
    int zn = 1;     // zip: chunks at once in this read
    DSW D;   // p6
    D2W D2;   // p74
    float* amv_r = nullptr;   // p47: each written row's argmax on this card (value, index)
    float* mx2 = nullptr;     // p76: [MAXT][2] the rows' largest logits of both cards (min-p)
    int* ami_r = nullptr;
    int dev = 0, c = 0;
    cudaStream_t st;
    LW L[NPHYS];
    Mat out;
    const float *tok_norm, *out_norm, *out_rq;
    size_t vram = 0;
    unsigned seq = 1;
    // per slot: KV caches (f16) and the convolutions' last 3 inputs
    __half *kc[NSLOT], *vc[NSLOT];
    KV8 q8[NSLOT];   // p18: the full slots' int8 caches (INK2_KV8); window slots and the f16 mode leave them null
    size_t part_n = 0;   // p18: K.part's floats (the reading path's S shares it in the int8 mode)
    bool pf_rec = false;   // p63: alloc() records the reading path's allocations (alloc_pf) in pf_regions
    std::vector<std::pair<uint8_t*, size_t>> pf_regions;
    float *sk[NSLOT], *sv[NSLOT], *sa[NSLOT], *sf[NSLOT];
    // scratch
    float *x0, *cur, *B[MAXB], *prel, *bankS, *bankO, *X3, *MS, *XC[4], *xm, *g64, *gp, *fpp, *NQ, *NK, *NV, *q, *k, *v, *kcv, *vcv, *r,
        *sink, *gate, *dk, *dv, *ao, *part, *ab, *A, *F, *F2, *FN, *lg, *topw, *GU, *HH, *EY, *MO, *hx, *logits, *tau;
    int *pos, *topi;
    EJob* jobs;
    int* perm = nullptr;   // p11
    MJob* mjobs;   // p5b
    int* nmjobs;
    int nb = 0, nparts_max = 0;
    unsigned* dtok = nullptr;                   // p4c: forwards so far (device), the exchanges' sequence numbers
    float *stbuf = nullptr, *stbak = nullptr, *hist = nullptr;   // p5a: conv states, their snapshot, a window's inputs
    float* g64x = nullptr;   // p22: the three loop controllers' down outputs [3][MAXT][RANK]
    float* stsnap = nullptr;   // p8: the conv states before the last prompt token (the next request of a conversation goes back there)
    bool keep_hist = false;
    unsigned xidx = 0;                          // p4c: the next exchange of this forward
    std::map<int, cudaGraphExec_t> graphs;      // p4c: one T = 1 forward a parts bucket
};

struct Xch {
    float* buf[NC][2];
    volatile unsigned* flag[NC];
    unsigned long long* ll[NC][2] = {};   // p52: the 8-byte word buffers (INK2_XLL)
};

struct PrefillCancelled {};

struct Engine {
    GgufMin g;
    Card cards[NC];
    Xch X;
    const GTensor* tok = nullptr;
    int ctx;
    bool stare = getenv("INK2_STARE") != nullptr;   // p4a: v1's writing kernels
    bool kv8 = false;   // p18: the full layers' KV in int8 (INK2_KV8=1/0; by itself when ctx > 262144)
    int kv8_f16 = 32768, kv8_rot = 0, kv8_part = 3;   // p18: rows kept in f16 (INK2_KV8_F16), Hadamard (INK2_KV8_ROT), 1 K / 2 V / 3 both (INK2_KV8_PART)
    bool gemm2 = getenv("INK2_GEMM2") != nullptr;   // p3e
    bool win_seg = getenv("INK2_WIN_SEG") != nullptr;   // p6c off: windows' dense rows by k_gemv8_seg (other f32 order than T = 1)
    bool mt_off = getenv("INK2_MT") == nullptr;   // p5b: windows read each picked expert once (INK2_MT=1; measured 21. 11.: slower, off)
    bool attn_old = getenv("INK2_ATTN_OLD") != nullptr;   // p4f off: the attention by query head (k_attn_part)
    bool fuse = getenv("INK2_FUSE_OFF") == nullptr;   // p4e: fused writing kernels at T = 1
    // p22: merged launches (ink2_fuse.cuh), a mask: 1 mixes, 2 controllers, 4 norms + matrices, 8 routing. Measured 22. 11. against 0, the
    // same bits each: 1: +0.1 %; 2: single tokens +2.7 %, windows +1.2..1.5 %; 4: +7.1 % / +0.2..1.5 %; 8: +0.1 % / -1.8..-2.7 % (the last
    // block routes a window's rows one after another); 15: +9.3 % / -2.0..+0.2 %. Default 6
    int fuse2 = getenv("INK2_FUSE2") ? atoi(getenv("INK2_FUSE2")) : 6;
    // p49: the writing path's weight kernels in smaller blocks (the same sums): gate_up threads, down warps, dense Q8S T = 1 warps
    int r7tpb = getenv("INK2_R7TPB") ? atoi(getenv("INK2_R7TPB")) : 256;
    int r6wpb = getenv("INK2_R6WPB") ? atoi(getenv("INK2_R6WPB")) : 4;
    int g1wpb = getenv("INK2_G1WPB") ? atoi(getenv("INK2_G1WPB")) : 2;
    bool ilv = getenv("INK2_FP4_ILV") != nullptr;   // p4d off: NVFP4 experts interleaved as in the file (the older kernels need it)
    bool rows2 = getenv("INK2_ROWS2") != nullptr;   // p4b off: p4a's NVFP4 writing kernel
    bool gemm5 = getenv("INK2_GEMM5") ? atoi(getenv("INK2_GEMM5")) != 0 : true;
    int tile_s = getenv("INK2_TILE_S") ? atoi(getenv("INK2_TILE_S")) : 1;   // p24: expert tiles of 8 and 4 tokens for an expert's last few (0: 16-token tiles)
    int g6 = getenv("INK2_G6") ? atoi(getenv("INK2_G6")) : 0;   // p50: reading's 64-token expert tiles by k_fp4_g6 (32 / 16 = its step; 0 off)
    bool mix3 = getenv("INK2_MIX3") ? atoi(getenv("INK2_MIX3")) != 0 : true;   // p51: the depth mixes of few rows in two launches
    bool xll = getenv("INK2_XLL") ? atoi(getenv("INK2_XLL")) != 0 : true;   // p52: T = 1 exchanges as 8-byte words (value | sequence)
    int g1w = getenv("INK2_G1W") ? atoi(getenv("INK2_G1W")) : 4;   // p53: dense Q8S T = 1 by k_gemv8_t1w, rows a warp (1 / 2 / 4; 0 off)
    bool ckspare = getenv("INK2_CKSPARE") ? atoi(getenv("INK2_CKSPARE")) != 0 : true;   // p55: a spare checkpoint buffer made between requests
    const int att_w = getenv("INK2_ATT_W") ? atoi(getenv("INK2_ATT_W")) : 1;   // p61: a window's rows in one block of k_attn_decw (full layers)
    const int att_w_min = getenv("INK2_ATT_W_MIN") ? atoi(getenv("INK2_ATT_W_MIN")) : 64;   // p61: from so many parts (64 = 4k; kodachat -0.3 % below)
    int comb2 = getenv("INK2_COMB2") ? atoi(getenv("INK2_COMB2")) : 3;   // p57: 1 k_attn_comb2 (the parts' max and weights in parallel); p59: 3 k_attn_comb4 / comb5 from INK2_COMB4_MIN parts
    const int comb4_min = getenv("INK2_COMB4_MIN") ? atoi(getenv("INK2_COMB4_MIN")) : 48;   // p59
    int xsplit = getenv("INK2_XSPLIT") ? std::max(1, std::min(PFB::XS_MAX, atoi(getenv("INK2_XSPLIT")))) : 4;   // p19d: slices of an exchange (1: xch_pf; 22. 11.: 2 / 3 / 4 = 14k 876 / 896 / 914)   // p19b: one block a tile of 64 / 32 / 16 tokens (0: k_fp4_gemm)
    bool pf_deq = getenv("INK2_PF_DEQ") != nullptr;  // p3c: NVFP4 experts of the reading path dequantized to f32 + cuBLAS (p3b)
    bool catm = getenv("INK2_CATM") ? atoi(getenv("INK2_CATM")) != 0 : false;   // p27: the input projections in one call a group of rows (23. 11.: 128 tokens 322 -> 320 ms, routing ties move: off)
    bool q8g = getenv("INK2_Q8G") ? atoi(getenv("INK2_Q8G")) != 0 : true;   // p25: Q8 experts of the reading path by k_q8_gemm5 (0: k_q8s_f32 + cuBLAS)
    float eps, qk_clip, rel_clip_thr, log_alpha, log_floor, situ_b, situ_lb, wscale, softcap;
    // host side of a step
    float* h_x0 = nullptr;   // p4c: page-locked (a graph's memcpy nodes read them at each replay)
    int* h_pos = nullptr;
    float* h_tau = nullptr;
    bool graf = getenv("INK2_GRAF") != nullptr;   // p4c: the writing path's forward as a CUDA graph (measured 21. 11.: no gain, off)
    int pfc = RING >= 4096 ? 2048 : RING - SWA;   // p3: INK2_PF_CHUNK, tokens a reading chunk (p31: RING - SWA with a smaller ring)
    int pf_min = 31;                          // p3: INK2_PF_MIN, shortest piece that goes the reading path (INK2_PF_OFF: never); p23: 41, was MAXT + 1; p27: 31
    std::atomic<unsigned> pf_rec[NC];         // p3: the exchange each card has recorded
    // zip: INK2_PF_ZIP=n (2 or 3; 0 = off): n consecutive reading chunks at once, their steps taking turns on each card's
    // stream, so a chunk's exchange crosses PCIe while the other chunks compute. Every launch and cuBLAS call is the same,
    // chunk by chunk, so the output is the same to the bit. Off by itself when its buffers would push experts to RAM
    int zip = getenv("INK2_PF_ZIP") ? atoi(getenv("INK2_PF_ZIP")) : 2;
    std::atomic<unsigned> zrec[Card::ZMAX][NC];   // as pf_rec, for each of the chunks
    float* h_x0z[Card::ZMAX] = {};                // chunks 1..: rows, positions, tau and raw marks (chunk 0: h_x0, h_pos, h_tau, h_raw)
    int* h_posz[Card::ZMAX] = {};
    float* h_tauz[Card::ZMAX] = {};
    std::vector<char> h_rawz[Card::ZMAX];
    static constexpr int EB = 16, QS = 256;   // p3: experts dequantized at once; queries per attention block
    bool perm_on = getenv("INK2_PERM") ? atoi(getenv("INK2_PERM")) != 0 : false;   // p11: a window's expert jobs by expert (21. 11.: same speed, off)
    int feed_t = getenv("INK2_FEED_T") ? std::max(1, std::min(MAXT, atoi(getenv("INK2_FEED_T")))) : 8;   // p23: 8 rows (16 rows: 239 ms, 8 rows: 56 ms)   // windows of the writing path when reading (test: 1 = token by token)
    bool t1tt_on = getenv("INK2_T1TT") ? atoi(getenv("INK2_T1TT")) != 0 : true;   // p9: window GEMV with T at compile time
    bool ds_t1 = getenv("INK2_DS_T1") ? atoi(getenv("INK2_DS_T1")) != 0 : true;      // p9: the drafter's GEMV by gemv8_t1T
    bool pf_attb = getenv("INK2_PF_ATTB") ? atoi(getenv("INK2_PF_ATTB")) != 0 : true;   // p3f: window layers' attention in one batched call
    bool pf_rb = getenv("INK2_PF_RB") ? atoi(getenv("INK2_PF_RB")) != 0 : false;        // p3f: R.B by k_rb instead of cuBLAS (no gain, off)
    int pf_qs_full = getenv("INK2_PF_QSF") ? atoi(getenv("INK2_PF_QSF")) : 1024;
    bool qsf_fixed = getenv("INK2_PF_QSF") != nullptr;   // p19a: a fixed block (else qsf() sizes it by the chunk's keys)
    int qsf_max = getenv("INK2_PF_QSF_MAX") ? atoi(getenv("INK2_PF_QSF_MAX")) : 2048;
    size_t pv_cap = 0;   // p19c: the P.V pieces' room in floats
    bool share_on = getenv("INK2_SHARE") ? atoi(getenv("INK2_SHARE")) != 0 : true;   // p19c: the shared scratch in every mode (0: only int8)
    static constexpr int KFB = NC == 1 ? 8192 : 32768;   // p33: 8k with one card   // p14: full layers' keys converted to f32 at most this many at a time (longer: in blocks)
    static constexpr int PVK = 1024;    // p14b: the full layers' P.V in pieces of this many keys (one batched call, then k_pv_sum)
    bool pv_split = getenv("INK2_PV_SPLIT") ? atoi(getenv("INK2_PV_SPLIT")) != 0 : true;
    static constexpr int DS_RING = 4096, DS_RING1 = 16384, DS_WIN1 = 12288;   // p14: the drafter's caches above 64k (layer 1 sees 12k)
    bool ds_ring = false;   // 21. 11.: 14k 641 -> 695 (2048: 700, +0.7 GB)   // p3d: queries per block in the full layers
    float* h_logits = nullptr;   // page-locked [MAXT][NV]
    // p47: INK2_AMAX=1, the rows' argmax on the cards while writing; want_logits = the full rows are copied too
    bool dev_amax = getenv("INK2_AMAX") ? atoi(getenv("INK2_AMAX")) != 0 : true;
    bool want_logits = true;
    float* h_amv2 = nullptr;   // page-locked [NC][MAXT]
    int* h_ami2 = nullptr;
    std::FILE* fdump = nullptr;   // p43: the featdump request's file (card 0 writes every reading chunk's features)
    __half *fd_dev = nullptr, *fd_host = nullptr;
    bool force_pf = false;        // p43: every piece through the reading path
    bool share1c = NC == 1 && getenv("INK2_1C_SHARE") && atoi(getenv("INK2_1C_SHARE")) != 0;   // p63: cache slots in the reading path's buffers
    float s_invT = 0.0f;               // p36: 1 / temperature of this request (0: greedy)
    unsigned long long s_seed = 0;     // p36: its seed
    float s_lnminp = 0.0f;             // p76: ln(min_p) of this request (0: no min-p)
    bool row_raw = false;              // p76: h_logits' last row came from the reading path without noise (min-p)
    int n_past = 0;
    int snap_n = -1;   // p8: tokens before the snapshot of the conv states (stsnap), -1 = none
    std::vector<int> processed;   // the tokens whose state the caches hold
    // p21: checkpoints of the conversation, page-locked in RAM: the conv states and the last SWA rows of every window slot's ring
    // (with the drafter's rings: their windows). A request goes back to the latest one its prompt still shares, however far the
    // conversation went since: p8's one snapshot stood only at the prompt's end and died RING - SWA - MAXT tokens after it
    struct Ckpt {
        int pos = 0;
        char* buf[NC] = {};
    };
    std::vector<Ckpt> ckpts;     // by position; all of them describe a prefix of `processed`
    std::vector<Ckpt> ck_pool;   // buffers not in use
    bool ck_on = getenv("INK2_CKPT") ? atoi(getenv("INK2_CKPT")) != 0 : true;
    int ck_max = 0, ck_made = 0;   // INK2_CKPT_MB of RAM (16384) by both cards' buffers
    int ck_gap = getenv("INK2_CKPT_GAP") ? atoi(getenv("INK2_CKPT_GAP")) : 4096;        // hinted checkpoints this far apart (the last hint always)
    int ck_every = getenv("INK2_CKPT_EVERY") ? atoi(getenv("INK2_CKPT_EVERY")) : 16384;   // one at least every so many tokens of a long read
    static constexpr int DS_TAIL = SWA + 1;   // the drafter's layers 0 and 2 see P - 1024 .. P
    // p21b: conversations set aside in RAM (the cards hold one conversation's rows; a request of another one wiped it): the full
    // slots' rows [0, the last checkpoint) of both cards, the tokens, the first and the last three checkpoints
    struct Park {
        std::vector<int> tok;
        std::vector<Ckpt> cks;
        char* kv[NC] = {};
        size_t bytes = 0;
        unsigned long long use = 0;
    };
    std::vector<Park> parks;
    size_t park_bytes = 0, park_max = (size_t) (getenv("INK2_PARK_MB") ? std::max(0LL, atoll(getenv("INK2_PARK_MB"))) : (NC == 1 ? 0LL : 32768LL)) << 20;   // p30
    int park_min = getenv("INK2_PARK_MIN") ? atoi(getenv("INK2_PARK_MIN")) : 1024;   // fewer tokens than this are read again instead
    static constexpr int PARK_CK = 4;
    unsigned long long use_no = 0;
    struct KvArr {
        char* p;
        size_t rb;
        int r0, r1;   // the positions it holds: row q at index q - r0
    };
    // dumps (card 0): per slot l_out, attn_out, ffn_out rows of every window
    bool dumping = false;
    std::vector<std::vector<float>> d_lout, d_aout, d_fout;
    // card 1's worker
    std::thread w1;
    std::atomic<int> go{0}, done{0};
    std::function<void(int)> job;
    bool quit = false;

    std::map<std::string, size_t> vtag[NC];   // p18: VRAM by part (INK2_VRAM_LOG=1 prints it after loading)
    std::string vt[2] = {"ostalo", "ostalo"};
    void* alloc(Card& K, size_t bytes) {
        void* p = nullptr;
        CK(cudaSetDevice(K.dev));
        CK(cudaMalloc(&p, bytes));
        K.vram += bytes;
        vtag[K.c][vt[K.c]] += bytes;
        if (K.pf_rec) K.pf_regions.push_back({(uint8_t*) p, bytes});   // p63
        return p;
    }
    float* dev(Card& K, size_t floats) {
        float* p = (float*) alloc(K, floats * 4);
        CK(cudaMemset(p, 0, floats * 4));
        return p;
    }
    GgufMin* dsg = nullptr;   // p6: the drafter's file (--mtp)
    bool ds_on = false;
    bool d2_on = false;   // p74: the DFlash 2 drafter (--mtp with dflash.selector_rank)
    int d2_from = 0;      // p74: the reading path's first row the rings keep after this feed()
    int d2_mask = 0;
    float d2_cap = 30.0f, d2_eps = 1e-6f, d2_theta = 1e4f;
    const uint16_t *d2_A = nullptr, *d2_B = nullptr;   // the selector's codebooks (BF16 [vocabulary][D2R]) in the file's mapping
    float *h_d2v = nullptr, *h_d2g = nullptr;          // page-locked: each card's top-16 of rows 1.. (values, ids), card 0's gate
    int* h_d2i = nullptr;
    int d2_rows = getenv("INK2_D2_ROWS") ? std::max(2, std::min(DBLK, atoi(getenv("INK2_D2_ROWS")))) : DBLK;
    float d2_qmin = getenv("INK2_D2_QMIN") ? (float) atof(getenv("INK2_D2_QMIN")) : 0.5f;
    static constexpr int D2PIECE = 1024;   // the context's rows a pass (the reading path's chunk in pieces)
    const float* f32(Card& K, const std::string& name, size_t off = 0, size_t n = 0, GgufMin* gg = nullptr) {
        const GTensor& t = (gg ? *gg : g).t(name);
        if (t.type != G_F32) throw std::runtime_error(name + " is not F32");
        const size_t bytes = n ? n * 4 : t.nbytes;
        void* p = alloc(K, bytes);
        CK(cudaMemcpy(p, t.data + off * 4, bytes, cudaMemcpyHostToDevice));
        return (const float*) p;
    }
    // rows [r0, r1) and columns [c0, c1) of a Q8_0 matrix (rows = ne1 * ne2, cols = ne0) into Q8S
    Mat q8(Card& K, const std::string& name, int r0 = 0, int r1 = -1, int c0 = 0, int c1 = -1, GgufMin* gg = nullptr) {
        const GTensor& t = (gg ? *gg : g).t(name);
        if (t.type != G_Q8_0) throw std::runtime_error(name + " is not Q8_0");
        const int rows = (int) (t.ne[1] * t.ne[2] * t.ne[3]), cols = (int) t.ne[0];
        if (r1 < 0) r1 = rows;
        if (c1 < 0) c1 = cols;
        if (c0 % 32 || c1 % 32) throw std::runtime_error(name + ": column slice off the blocks");
        Mat m;
        m.type = G_Q8_0;
        m.rows = r1 - r0;
        m.cols = c1 - c0;
        const int nb = m.cols / 32;
        const size_t rb = t.row_bytes();
        std::vector<int8_t> qv((size_t) m.rows * m.cols);
        std::vector<uint16_t> dv((size_t) m.rows * nb);
        for (int r = 0; r < m.rows; ++r) {
            const uint8_t* src = t.data + (size_t) (r0 + r) * rb + (size_t) (c0 / 32) * 34;
            for (int b = 0; b < nb; ++b) {
                std::memcpy(&dv[(size_t) r * nb + b], src + b * 34, 2);
                std::memcpy(&qv[(size_t) r * m.cols + b * 32], src + b * 34 + 2, 32);
            }
        }
        void* pq = alloc(K, qv.size());
        void* pd = alloc(K, dv.size() * 2);
        CK(cudaMemcpy(pq, qv.data(), qv.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(pd, dv.data(), dv.size() * 2, cudaMemcpyHostToDevice));
        m.q = (const int8_t*) pq;
        m.d = pd;
        return m;
    }
    // p13: the given rows of a Q8_0 matrix as one Q8S matrix
    Mat q8_pick(Card& K, const std::string& name, const std::vector<int>& pick) {
        const GTensor& t = g.t(name);
        if (t.type != G_Q8_0) throw std::runtime_error(name + " is not Q8_0");
        Mat m;
        m.type = G_Q8_0;
        m.rows = (int) pick.size();
        m.cols = (int) t.ne[0];
        const int nb = m.cols / 32;
        const size_t rb = t.row_bytes();
        std::vector<int8_t> qv((size_t) m.rows * m.cols);
        std::vector<uint16_t> dv((size_t) m.rows * nb);
        for (int r = 0; r < m.rows; ++r) {
            const uint8_t* src = t.data + (size_t) pick[r] * rb;
            for (int b = 0; b < nb; ++b) {
                std::memcpy(&dv[(size_t) r * nb + b], src + b * 34, 2);
                std::memcpy(&qv[(size_t) r * m.cols + b * 32], src + b * 34 + 2, 32);
            }
        }
        void* pq = alloc(K, qv.size());
        void* pd = alloc(K, dv.size() * 2);
        CK(cudaMemcpy(pq, qv.data(), qv.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(pd, dv.data(), dv.size() * 2, cudaMemcpyHostToDevice));
        m.q = (const int8_t*) pq;
        m.d = pd;
        return m;
    }
    // p4a: several row ranges of Q8_0 matrices with the same columns as one Q8S matrix (rows one after another)
    Mat q8cat(Card& K, const std::vector<std::tuple<std::string, int, int>>& parts, std::vector<Mat>& views) {
        int rows = 0, cols = -1;
        for (auto& [name, r0, r1] : parts) {
            const GTensor& t = g.t(name);
            if (t.type != G_Q8_0) throw std::runtime_error(name + " is not Q8_0");
            if (cols >= 0 && cols != (int) t.ne[0]) throw std::runtime_error(name + ": columns differ");
            cols = (int) t.ne[0];
            rows += r1 - r0;
        }
        const int nb = cols / 32;
        std::vector<int8_t> qv((size_t) rows * cols);
        std::vector<uint16_t> dv((size_t) rows * nb);
        int o = 0;
        std::vector<int> starts;
        for (auto& [name, r0, r1] : parts) {
            const GTensor& t = g.t(name);
            const size_t rb = t.row_bytes();
            starts.push_back(o);
            for (int r = r0; r < r1; ++r, ++o) {
                const uint8_t* src = t.data + (size_t) r * rb;
                for (int b = 0; b < nb; ++b) {
                    std::memcpy(&dv[(size_t) o * nb + b], src + b * 34, 2);
                    std::memcpy(&qv[(size_t) o * cols + b * 32], src + b * 34 + 2, 32);
                }
            }
        }
        Mat m;
        m.type = G_Q8_0;
        m.rows = rows;
        m.cols = cols;
        void* pq = alloc(K, qv.size());
        void* pd = alloc(K, dv.size() * 2);
        CK(cudaMemcpy(pq, qv.data(), qv.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(pd, dv.data(), dv.size() * 2, cudaMemcpyHostToDevice));
        m.q = (const int8_t*) pq;
        m.d = pd;
        views.clear();
        for (size_t i = 0; i < parts.size(); ++i) {
            Mat v = m;
            v.rows = std::get<2>(parts[i]) - std::get<1>(parts[i]);
            v.q = m.q + (size_t) starts[i] * cols;
            v.d = (const __half*) m.d + (size_t) starts[i] * nb;
            views.push_back(v);
        }
        return m;
    }
    // this card's part of a layer's experts: gate_up rows [c*FF_C, +FF_C) and [FF + c*FF_C, +FF_C), down columns [c*FF_C, +FF_C).
    // p30: each expert contiguous (NVFP4: planar gate_up, then planar down; Q8: gate_up's int8 rows, their half scales, then down's
    // rows and scales); the experts with res[e] in one VRAM block, the others in one block of mapped page-locked RAM that the
    // kernels read over PCIe; L.gu_t / L.dn_t hold every expert's two bases
    size_t host_exp_bytes[NC] = {}, vram_exp_bytes[NC] = {};   // p40: a card each (the cards place their experts at once)
    bool q8fp4 = getenv("INK2_1C_Q8_FP4") && atoi(getenv("INK2_1C_Q8_FP4")) != 0;   // p35 (lossy)
    // p35: rows x cols floats (an expert matrix) to interleaved NVFP4 blocks (rows x cols / 64 x 36 bytes); returns its scale G
    static float to_nvfp4(const float* v, size_t rows, int cols, uint8_t* out) {
        static float ue[127];   // UE4M3 raw values of bytes 0..126 (ascending; 0x7F is NaN)
        static bool init = false;
        if (!init) {
            for (int b = 0; b < 127; ++b) {
                const int e = (b >> 3) & 0xF, m = b & 7;
                ue[b] = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
            }
            init = true;
        }
        static const float E2[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        auto code = [](float x) {   // nearest E2M1, sign in bit 3
            const float a = std::fabs(x);
            const int m = a < 0.25f ? 0 : a < 0.75f ? 1 : a < 1.25f ? 2 : a < 1.75f ? 3 : a < 2.5f ? 4 : a < 3.5f ? 5 : a < 5.0f ? 6 : 7;
            return (uint8_t) (m | (x < 0 && m ? 8 : 0));
        };
        float amax = 0.0f;
        for (size_t i = 0; i < rows * (size_t) cols; ++i) amax = std::max(amax, std::fabs(v[i]));
        const float G = amax > 0.0f ? amax / (6.0f * 448.0f) : 1.0f;
        const int nb = cols / 64;
        for (size_t r = 0; r < rows; ++r)
            for (int b = 0; b < nb; ++b) {
                uint8_t* blk = out + (r * nb + b) * 36;
                for (int sb = 0; sb < 4; ++sb) {
                    const float* x = v + r * cols + b * 64 + sb * 16;
                    float am = 0.0f;
                    for (int i = 0; i < 16; ++i) am = std::max(am, std::fabs(x[i]));
                    const float t = am / 6.0f / G;
                    const int hi = std::min(126, (int) (std::lower_bound(ue + 1, ue + 127, t) - ue));
                    int best = 0;
                    float berr = 3.4e38f;
                    for (int cand : {hi - 1, hi}) {
                        if (cand < 1 || am == 0.0f) continue;
                        const float sc = ue[cand] * G;
                        float err = 0.0f;
                        for (int i = 0; i < 16; ++i) {
                            const uint8_t c = code(x[i] / sc);
                            const float q = (c & 8 ? -1.0f : 1.0f) * E2[c & 7] * sc;
                            err += (q - x[i]) * (q - x[i]);
                        }
                        if (err < berr) {
                            berr = err;
                            best = cand;
                        }
                    }
                    blk[sb] = (uint8_t) best;
                    const float sc = best ? ue[best] * G : 1.0f;
                    uint8_t* qs = blk + 4 + sb * 8;
                    for (int i = 0; i < 8; ++i) {
                        const uint8_t lo = best ? code(x[i] / sc) : 0, hi4 = best ? code(x[i + 8] / sc) : 0;
                        qs[i] = (uint8_t) (lo | (hi4 << 4));
                    }
                }
            }
        return G;
    }
    size_t expert_bytes(int p) {   // gate_up + down of one expert of layer p, as experts() lays them out
        const GTensor& gu = g.t("blk." + std::to_string(p) + ".ffn_gate_up_exps.weight");
        if (gu.type == G_NVFP4 || q8fp4) return (size_t) 2 * FF_C * (N / 64) * 36 + (size_t) N * (FF_C / 64) * 36;
        return (size_t) 2 * FF_C * N + (size_t) 2 * FF_C * (N / 32) * 2 + (size_t) N * FF_C + (size_t) N * (FF_C / 32) * 2;
    }
    void experts(Card& K, LW& L, int p, const std::vector<char>& res) {
        const std::string b = "blk." + std::to_string(p) + ".";
        const GTensor& gu = g.t(b + "ffn_gate_up_exps.weight");
        const GTensor& dn = g.t(b + "ffn_down_exps.weight");
        const int c = K.c;
        std::vector<uint8_t> hg, hd;   // all experts, one after another
        size_t gb = 0, db = 0;         // one expert's gate_up / down bytes
        if (gu.type == G_NVFP4) {
            L.fp4 = true;
            const size_t rbg = gu.row_bytes(), rbd = dn.row_bytes(), cb = (size_t) (FF_C / 64) * 36;
            std::vector<uint8_t> ig((size_t) NE * 2 * FF_C * rbg), id((size_t) NE * N * cb);
            for (int e = 0; e < NE; ++e) {
                for (int h = 0; h < 2; ++h)
                    std::memcpy(&ig[((size_t) e * 2 * FF_C + h * FF_C) * rbg], gu.data + ((size_t) e * 2 * FF + h * FF + c * FF_C) * rbg,
                                (size_t) FF_C * rbg);
                for (int r = 0; r < N; ++r)
                    std::memcpy(&id[((size_t) e * N + r) * cb], dn.data + ((size_t) e * N + r) * rbd + (size_t) c * cb, cb);
            }
            hg.resize(ig.size());
            fp4_to_planar(ig.data(), hg.data(), (size_t) 2 * FF_C, N / 64, NE);
            hd.resize(id.size());
            fp4_to_planar(id.data(), hd.data(), (size_t) N, FF_C / 64, NE);
            gb = hg.size() / NE;
            db = hd.size() / NE;
            L.gu_s = f32(K, b + "ffn_gate_up_exps.scale");
            L.dn_s = f32(K, b + "ffn_down_exps.scale");
        } else if (gu.type == G_Q8_0 && q8fp4) {   // p35 (lossy): requantized to NVFP4, then as an NVFP4 layer
            L.fp4 = true;
            const size_t rbg = gu.row_bytes(), rbd = dn.row_bytes();
            const size_t cg = (size_t) 2 * FF_C * (N / 64) * 36, cd = (size_t) N * (FF_C / 64) * 36;
            std::vector<uint8_t> ig((size_t) NE * cg), id((size_t) NE * cd);
            std::vector<float> sg(NE), sd(NE);
            std::vector<std::thread> pool;
            std::atomic<int> next{0};
            auto deq = [](const uint8_t* src, int cols, float* dst) {   // a Q8_0 row piece of cols values
                for (int bb = 0; bb < cols / 32; ++bb) {
                    uint16_t hb;
                    std::memcpy(&hb, src + bb * 34, 2);
                    __half_raw hr;
                    hr.x = hb;
                    const float d = __half2float(__half(hr));
                    for (int j = 0; j < 32; ++j) dst[bb * 32 + j] = d * (float) (int8_t) src[bb * 34 + 2 + j];
                }
            };
            for (int t = 0; t < 16; ++t)
                pool.emplace_back([&]() {
                    std::vector<float> w((size_t) 2 * FF_C * N);
                    for (int e; (e = next.fetch_add(1)) < NE;) {
                        for (int h = 0; h < 2; ++h)
                            for (int r = 0; r < FF_C; ++r)
                                deq(gu.data + ((size_t) e * 2 * FF + h * FF + c * FF_C + r) * rbg, N, &w[((size_t) h * FF_C + r) * N]);
                        sg[e] = to_nvfp4(w.data(), (size_t) 2 * FF_C, N, &ig[(size_t) e * cg]);
                        for (int r = 0; r < N; ++r) deq(dn.data + ((size_t) e * N + r) * rbd + (size_t) (c * FF_C / 32) * 34, FF_C, &w[(size_t) r * FF_C]);
                        sd[e] = to_nvfp4(w.data(), (size_t) N, FF_C, &id[(size_t) e * cd]);
                    }
                });
            for (auto& t : pool) t.join();
            hg.resize(ig.size());
            fp4_to_planar(ig.data(), hg.data(), (size_t) 2 * FF_C, N / 64, NE);
            hd.resize(id.size());
            fp4_to_planar(id.data(), hd.data(), (size_t) N, FF_C / 64, NE);
            gb = cg;
            db = cd;
            float* a = (float*) alloc(K, NE * 4);
            float* bsc = (float*) alloc(K, NE * 4);
            CK(cudaMemcpy(a, sg.data(), NE * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(bsc, sd.data(), NE * 4, cudaMemcpyHostToDevice));
            L.gu_s = a;
            L.dn_s = bsc;
        } else if (gu.type == G_Q8_0) {
            L.fp4 = false;
            const size_t rbg = gu.row_bytes(), rbd = dn.row_bytes();
            const size_t gq = (size_t) 2 * FF_C * N, dq = (size_t) N * FF_C;
            gb = gq + (size_t) 2 * FF_C * (N / 32) * 2;
            db = dq + (size_t) N * (FF_C / 32) * 2;
            hg.resize((size_t) NE * gb);
            hd.resize((size_t) NE * db);
            for (int e = 0; e < NE; ++e) {
                uint8_t* qg = &hg[(size_t) e * gb];
                uint8_t* dg = qg + gq;
                for (int h = 0; h < 2; ++h)
                    for (int r = 0; r < FF_C; ++r) {
                        const uint8_t* src = gu.data + ((size_t) e * 2 * FF + h * FF + c * FF_C + r) * rbg;
                        const size_t row = (size_t) h * FF_C + r;
                        for (int bb = 0; bb < N / 32; ++bb) {
                            std::memcpy(dg + (row * (N / 32) + bb) * 2, src + bb * 34, 2);
                            std::memcpy(qg + row * N + bb * 32, src + bb * 34 + 2, 32);
                        }
                    }
                uint8_t* qd = &hd[(size_t) e * db];
                uint8_t* dd = qd + dq;
                for (int r = 0; r < N; ++r) {
                    const uint8_t* src = dn.data + ((size_t) e * N + r) * rbd + (size_t) (c * FF_C / 32) * 34;
                    for (int bb = 0; bb < FF_C / 32; ++bb) {
                        std::memcpy(dd + ((size_t) r * (FF_C / 32) + bb) * 2, src + bb * 34, 2);
                        std::memcpy(qd + (size_t) r * FF_C + bb * 32, src + bb * 34 + 2, 32);
                    }
                }
            }
        } else {
            throw std::runtime_error(b + "ffn_gate_up_exps: type " + std::to_string(gu.type));
        }
        int nres = 0;
        for (int e = 0; e < NE; ++e) nres += res[e] != 0;
        const size_t eb = gb + db;
        uint8_t* vblk = nres ? (uint8_t*) alloc(K, (size_t) nres * eb) : nullptr;
        uint8_t *hblk = nullptr, *hdev = nullptr;
        if (nres < NE) {
            CK(cudaSetDevice(K.dev));
            CK(cudaHostAlloc((void**) &hblk, (size_t) (NE - nres) * eb, cudaHostAllocMapped | cudaHostAllocPortable));
            CK(cudaHostGetDevicePointer((void**) &hdev, hblk, 0));
            host_exp_bytes[K.c] += (size_t) (NE - nres) * eb;
        }
        vram_exp_bytes[K.c] += (size_t) nres * eb;
        std::vector<const uint8_t*> tg(NE), td(NE);
        int iv = 0, ih = 0;
        for (int e = 0; e < NE; ++e) {
            const uint8_t* base;
            if (res[e]) {
                uint8_t* d = vblk + (size_t) iv++ * eb;
                CK(cudaMemcpy(d, &hg[(size_t) e * gb], gb, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d + gb, &hd[(size_t) e * db], db, cudaMemcpyHostToDevice));
                base = d;
            } else {
                std::memcpy(hblk + (size_t) ih * eb, &hg[(size_t) e * gb], gb);
                std::memcpy(hblk + (size_t) ih * eb + gb, &hd[(size_t) e * db], db);
                base = hdev + (size_t) ih++ * eb;
            }
            tg[e] = base;
            td[e] = base + gb;
        }
        void* t1 = alloc(K, NE * sizeof(void*));
        void* t2 = alloc(K, NE * sizeof(void*));
        CK(cudaMemcpy(t1, tg.data(), NE * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(t2, td.data(), NE * sizeof(void*), cudaMemcpyHostToDevice));
        L.gu_t = (const uint8_t* const*) t1;
        L.dn_t = (const uint8_t* const*) t2;
        L.nres = nres;
        L.res = res;   // p31
        L.h_gu = tg;
        L.h_dn = td;
        L.gu = L.dn = nullptr;
        L.gu_d = L.dn_d = nullptr;
    }
    // p30: which experts stay in VRAM, after every other allocation: what is free less INK2_1C_MARGIN_MB (400), at most
    // INK2_1C_EXP_MB, in INK2_1C_PLAN's order (lines "layer expert", the most used first; test/rstat_plan.py writes it), then
    // expert by expert over the layers in turn; an expert that does not fit is skipped and the next ones are tried
    void place_experts(Card& K) {
        CK(cudaSetDevice(K.dev));
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const long long margin = (getenv("INK2_1C_MARGIN_MB") ? atoll(getenv("INK2_1C_MARGIN_MB")) : NC == 1 ? 400LL : 150LL) << 20;   // p39: 150 with two cards
        long long budget = (long long) fr - margin - (long long) NPHYS * 2 * NE * 8;
        if (getenv("INK2_1C_EXP_MB")) budget = std::min(budget, atoll(getenv("INK2_1C_EXP_MB")) << 20);
        long long all = 0;   // p39: every expert fits: no cache, no scratch (nothing in RAM)
        for (int lp = 0; lp < NPHYS; ++lp) all += (long long) expert_bytes(lp) * NE;
        const bool fits = all <= budget;
        const long long cache = fits ? 0 : std::min(budget, (getenv("INK2_1C_CACHE_MB") ? atoll(getenv("INK2_1C_CACHE_MB")) : 1024LL) << 20);   // p32
        budget -= cache;
        const long long scratch = fits ? 0 : std::min(budget, (getenv("INK2_1C_SCRATCH_MB") ? atoll(getenv("INK2_1C_SCRATCH_MB")) : 256LL) << 20);   // p34
        budget -= scratch;
        std::vector<std::vector<double>> wt(NPHYS, std::vector<double>(NE, 1.0));   // p32: the plan's weights (1 without)
        std::vector<std::vector<char>> res(NPHYS, std::vector<char>(NE, 0)), seen(NPHYS, std::vector<char>(NE, 0));
        std::vector<std::pair<int, int>> order;
        if (getenv("INK2_1C_PLAN")) {
            FILE* f = std::fopen(getenv("INK2_1C_PLAN"), "r");
            if (!f) throw std::runtime_error(std::string("INK2_1C_PLAN: ") + getenv("INK2_1C_PLAN"));
            char line[256];
            while (std::fgets(line, sizeof line, f)) {
                int lp, le;
                double w = 1.0;
                const int k = std::sscanf(line, "%d %d %lf", &lp, &le, &w);
                if (k >= 2 && lp >= 0 && lp < NPHYS && le >= 0 && le < NE && !seen[lp][le]) {
                    seen[lp][le] = 1;
                    order.push_back({lp, le});
                    if (k == 3) wt[lp][le] = w;
                }
            }
            std::fclose(f);
        }
        for (int e = 0; e < NE; ++e)
            for (int lp = 0; lp < NPHYS; ++lp)
                if (!seen[lp][e]) order.push_back({lp, e});
        long long used = 0;
        for (auto [lp, e] : order) {
            const long long eb = (long long) expert_bytes(lp);
            if (used + eb <= budget) {
                res[lp][e] = 1;
                used += eb;
            }
        }
        for (int lp = 0; lp < NPHYS; ++lp) {
            vt[K.c] = lp >= NPHYS - 4 ? "strokovnjaki-q8" : (lp >= LS && lp < LE ? "strokovnjaki-zanka" : "strokovnjaki-fp4");
            experts(K, K.L[lp], lp, res[lp]);
        }
        // p32: the cache, split over the layers by the weight they leave in RAM, at most a slot a RAM expert
        double mwt = 0;
        std::vector<double> mw(NPHYS, 0.0);
        for (int lp = 0; lp < NPHYS; ++lp)
            for (int e = 0; e < NE; ++e)
                if (!res[lp][e]) mw[lp] += wt[lp][e];
        for (double x : mw) mwt += x;
        size_t cache_bytes = 0;
        std::string slots;
        vt[K.c] = "predpomnilnik";
        uint8_t* scr = scratch > 0 ? (uint8_t*) alloc(K, (size_t) scratch) : nullptr;   // p34
        std::vector<std::vector<uint8_t*>> shs(NPHYS);   // p63: each layer's slots in the reading path's buffers
        size_t sh_bytes = 0;
        if (share1c && mwt > 0) {
            std::vector<std::pair<size_t, uint8_t*>> rg;   // (bytes left, next free address), 256-byte aligned
            size_t sh_all = 0;
            for (auto [p, b] : K.pf_regions) {
                const size_t a = ((uintptr_t) p + 255) / 256 * 256 - (uintptr_t) p;
                if (b > a) {
                    rg.push_back({b - a, p + a});
                    sh_all += b - a;
                }
            }
            std::vector<int> want(NPHYS, 0), ord;
            for (int lp = 0; lp < NPHYS; ++lp) {
                const int s0 = std::min(NE - K.L[lp].nres, (int) ((double) cache * mw[lp] / mwt / (double) expert_bytes(lp)));
                want[lp] = std::max(0, std::min(NE - K.L[lp].nres - s0, (int) ((double) sh_all * mw[lp] / mwt / (double) expert_bytes(lp))));
                ord.push_back(lp);
            }
            std::sort(ord.begin(), ord.end(), [&](int a, int b) { return expert_bytes(a) > expert_bytes(b); });   // the big slots first
            for (bool more = true; more;) {   // a slot a layer in turn, so a short buffer list shares out evenly
                more = false;
                for (int lp : ord) {
                    if ((int) shs[lp].size() >= want[lp]) continue;
                    const size_t eb = expert_bytes(lp), need = (eb + 255) / 256 * 256;
                    std::sort(rg.begin(), rg.end(), [](auto& a, auto& b) { return a.first < b.first; });   // the tightest piece that fits
                    auto it = std::find_if(rg.begin(), rg.end(), [&](auto& r) { return r.first >= eb; });
                    if (it == rg.end()) {
                        want[lp] = (int) shs[lp].size();
                        continue;
                    }
                    shs[lp].push_back(it->second);
                    sh_bytes += eb;
                    const size_t take = std::min(need, it->first);
                    it->first -= take;
                    it->second += take;
                    more = true;
                }
            }
        }
        for (int lp = 0; lp < NPHYS; ++lp) {
            LW& L = K.L[lp];
            ECache& c = L.ec;
            c.eb = expert_bytes(lp);
            c.gb = (size_t) (L.h_dn[0] - L.h_gu[0]);
            c.S = mwt > 0 ? std::min(NE - L.nres, (int) ((double) cache * mw[lp] / mwt / (double) c.eb)) : 0;
            c.S = std::max(c.S, 0);
            c.S0 = c.S;   // p63
            c.S += (int) shs[lp].size();
            slots += " " + std::to_string(c.S0) + (shs[lp].empty() ? "" : "+" + std::to_string(shs[lp].size()));
            if (L.nres == NE) continue;   // p34: a layer with RAM experts gets the assign kernel even with no slot (the scratch)
            c.scr = scr;
            c.scr_bytes = (size_t) std::max(0LL, scratch);
            c.scr_list = (int*) alloc(K, (size_t) MAXT * TOPK * 4);
            c.nscr = (int*) alloc(K, 4);
            CK(cudaMemset(c.nscr, 0, 4));
            c.mem = c.S0 ? (uint8_t*) alloc(K, (size_t) c.S0 * c.eb) : nullptr;
            cache_bytes += (size_t) c.S0 * c.eb;
            {   // p63: the slots' addresses
                std::vector<uint8_t*> a(std::max(c.S, 1), nullptr);
                for (int s = 0; s < c.S0; ++s) a[s] = c.mem + (size_t) s * c.eb;
                for (int s = c.S0; s < c.S; ++s) a[s] = shs[lp][s - c.S0];
                void* d = alloc(K, a.size() * sizeof(void*));
                CK(cudaMemcpy(d, a.data(), a.size() * sizeof(void*), cudaMemcpyHostToDevice));
                c.sp = (uint8_t* const*) d;
            }
            c.slot_of = (int*) alloc(K, NE * 4);
            CK(cudaMemset(c.slot_of, 0xFF, NE * 4));
            c.exp_of = (int*) alloc(K, (size_t) std::max(c.S, 1) * 4);
            CK(cudaMemset(c.exp_of, 0xFF, (size_t) std::max(c.S, 1) * 4));
            c.used = (unsigned*) alloc(K, (size_t) std::max(c.S, 1) * 4);
            CK(cudaMemset(c.used, 0, (size_t) std::max(c.S, 1) * 4));
            std::vector<char> h(NE);
            for (int e = 0; e < NE; ++e) h[e] = !L.res[e];
            char* dh = (char*) alloc(K, NE);
            CK(cudaMemcpy(dh, h.data(), NE, cudaMemcpyHostToDevice));
            c.host = dh;
            void* hb = alloc(K, NE * sizeof(void*));
            CK(cudaMemcpy(hb, L.h_gu.data(), NE * sizeof(void*), cudaMemcpyHostToDevice));
            c.hb = (const uint8_t* const*) hb;
            c.fill = (int*) alloc(K, (size_t) 2 * MAXT * TOPK * 4);
            c.nfill = (int*) alloc(K, 4);
        }
        vt[K.c] = "ostalo";
        const std::string who = NC == 1 ? "ink2_1c" : "ink2_v1: kartica " + std::to_string(K.c);   // p40
        if (NC == 1 || cache_bytes)
            std::fprintf(stderr, "%s: predpomnilnik strokovnjakov iz RAM-a %.2f GiB (+ %.2f GiB v medpomnilnikih branja, INK2_1C_SHARE), reze po plasteh:%s\n",
                         who.c_str(), cache_bytes / 1073741824.0, sh_bytes / 1073741824.0, slots.c_str());
        std::string per;
        for (int lp = 0; lp < NPHYS; ++lp) per += " " + std::to_string(K.L[lp].nres);
        std::fprintf(stderr, "%s: strokovnjaki v VRAM %.2f GiB, v RAM-u %.2f GiB (prosto je bilo %.2f GiB, rezerva %lld MiB); po plasteh:%s\n",
                     who.c_str(), vram_exp_bytes[K.c] / 1073741824.0, host_exp_bytes[K.c] / 1073741824.0, fr / 1073741824.0, margin >> 20,
                     per.c_str());
    }
    template <class F> void each_card(F f) {   // p30: f(c) for every card at once (one card: here)
        if (NC == 1) {
            f(0);
            return;
        }
        std::thread t([&]() { f(NC - 1); });
        f(0);
        t.join();
    }

    void load_card(Card& K) {
        CK(cudaSetDevice(K.dev));
        CK(cudaStreamCreateWithFlags(&K.st, cudaStreamNonBlocking));
        const int c = K.c, h0 = c * NH_C, kv0 = c * NKV_C;
        vt[c] = "izhod";
        K.tok_norm = f32(K, "token_embd_norm.weight");
        K.out_norm = f32(K, "output_norm.weight");
        K.out_rq = f32(K, "output_res_query.weight");
        K.out = q8(K, "output.weight", c * NV_C, (c + 1) * NV_C);
        vt[c] = "plasti";
        const std::vector<int64_t> swa = g.ia("inkling2_loop.attention.sliding_window_pattern");
        for (int p = 0; p < NPHYS; ++p) {
            LW& L = K.L[p];
            const std::string b = "blk." + std::to_string(p) + ".";
            L.full = !swa[p];
            if (L.full) {
                L.q_norm = f32(K, b + "attn_q_inp_norm.weight");
                L.k_norm = f32(K, b + "attn_k_inp_norm.weight");
                L.v_norm = f32(K, b + "attn_v_inp_norm.weight");
            } else {
                L.attn_norm = f32(K, b + "attn_norm.weight");
            }
            if (p > 0) {
                L.attn_rq = f32(K, b + "attn_res_query.weight");
                L.nrq = (int) g.t(b + "attn_res_query.weight").ne[1];
            }
            L.ffn_rq = f32(K, b + "ffn_res_query.weight");
            {
                std::vector<std::tuple<std::string, int, int>> parts = {
                    {b + "attn_q.weight", h0 * HD, (h0 + NH_C) * HD}, {b + "attn_gate.weight", h0 * HD, (h0 + NH_C) * HD},
                    {b + "attn_rel.weight", h0 * NREL, (h0 + NH_C) * NREL}, {b + "attn_sink_proj.weight", h0, h0 + NH_C},
                    {b + "attn_k.weight", kv0 * HD, (kv0 + NKV_C) * HD}};
                if (L.full) parts.push_back({b + "attn_k_conv_proj.weight", kv0 * NCONV, (kv0 + NKV_C) * NCONV});
                parts.push_back({b + "attn_v.weight", kv0 * HD, (kv0 + NKV_C) * HD});
                if (L.full) parts.push_back({b + "attn_v_conv_proj.weight", kv0 * NCONV, (kv0 + NKV_C) * NCONV});
                std::vector<Mat> v;
                L.cat = q8cat(K, parts, v);
                L.wq = v[0];
                L.wgate = v[1];
                L.wrel = v[2];
                L.wsink = v[3];
                L.wk = v[4];
                if (L.full) {
                    L.wkd = v[5];
                    L.wv = v[6];
                    L.wvd = v[7];
                } else {
                    L.wv = v[5];
                }
            }
            L.wo = q8(K, b + "attn_output.weight", 0, N, h0 * HD, (h0 + NH_C) * HD);
            L.q_clip = f32(K, b + "attn_q_clip.weight");
            L.rel_clip = f32(K, b + "attn_rel_clip.weight");
            L.rel_bias = f32(K, b + "attn_rel_bias.weight");
            {
                float* bp = dev(K, (size_t) (EXT + 1) * NREL);
                CK(cudaMemcpy(bp, L.rel_bias, (size_t) EXT * NREL * 4, cudaMemcpyDeviceToDevice));
                L.rel_bias_pad = bp;
            }
            L.k_conv = f32(K, b + "attn_k_conv.weight", (size_t) kv0 * HD * NCONV, (size_t) KVC * NCONV);
            L.v_conv = f32(K, b + "attn_v_conv.weight", (size_t) kv0 * HD * NCONV, (size_t) KVC * NCONV);
            L.a_pconv = f32(K, b + "attn_post_conv.weight");
            L.f_pconv = f32(K, b + "ffn_post_conv.weight");
            L.ffn_norm = f32(K, b + "ffn_norm.weight");
            L.gate_inp = f32(K, b + "ffn_gate_inp.weight");
            L.probs_b = f32(K, b + "exp_probs_b.bias");
            // p30: the experts come last (place_experts: the VRAM left decides which stay on the card)
            if (p >= LS && p < LE) {
                L.msg_q = f32(K, b + "loop_msg_query.weight");
                const char* nm[4] = {"q", "k", "v", "ffn"};
                for (int j = 0; j < 4; ++j) {
                    const std::string lb = b + "loop_" + nm[j] + "_ctrl_";
                    L.c_down[j] = f32(K, lb + "down.weight");
                    L.c_gain[j] = f32(K, lb + "gain.weight");
                    L.c_forget[j] = f32(K, lb + "forget.weight");
                }
            }
        }
        // caches and states
        for (int s = 0; s < NSLOT; ++s) {
            const int p = s < NPHYS ? s : LS + (s - NPHYS) % NCORE;
            const size_t rows = K.L[p].full ? (size_t) ctx : (size_t) RING;
            vt[c] = K.L[p].full ? "kv-polno" : "kv-okno";
            if (kv8 && K.L[p].full) {   // p18: rows below f0 in f16, the rest in int8
                const size_t f = std::min(rows, (size_t) kv8_f16), nq = rows - f;
                const bool qk = (kv8_part & 1) && nq, qv = (kv8_part & 2) && nq;
                K.q8[s].f0 = (int) f;
                K.q8[s].rot = kv8_rot;
                K.kc[s] = (__half*) alloc(K, std::max<size_t>(1, qk ? f : rows) * KVC * 2);
                K.vc[s] = (__half*) alloc(K, std::max<size_t>(1, qv ? f : rows) * KVC * 2);
                if (qk) {
                    K.q8[s].k = (int8_t*) alloc(K, nq * KVC);
                    K.q8[s].ks = (__half*) alloc(K, nq * KQN * 2);
                }
                if (qv) {
                    K.q8[s].v = (int8_t*) alloc(K, nq * KVC);
                    K.q8[s].vs = (__half*) alloc(K, nq * KQN * 2);
                }
            } else {
                K.kc[s] = (__half*) alloc(K, rows * KVC * 2);
                K.vc[s] = (__half*) alloc(K, rows * KVC * 2);
            }
            vt[c] = "vmesni";
            if (s == 0) {   // p5a: all slots' convolution states in one buffer (a draft window's snapshot is one copy)
                K.stbuf = dev(K, (size_t) NSLOT * SLOTST);
                K.stbak = dev(K, (size_t) NSLOT * SLOTST);
                K.stsnap = dev(K, (size_t) NSLOT * SLOTST);
                K.hist = dev(K, (size_t) NSLOT * HISTS);
            }
            K.sk[s] = K.stbuf + (size_t) s * SLOTST;
            K.sv[s] = K.sk[s] + 3 * KVC;
            K.sa[s] = K.sk[s] + 6 * KVC;
            K.sf[s] = K.sa[s] + 3 * N;
        }
        const size_t TN = (size_t) MAXT * N;
        K.x0 = dev(K, TN);
        K.cur = dev(K, TN);
        for (int i = 0; i < MAXB; ++i) K.B[i] = dev(K, TN);
        K.prel = dev(K, TN);
        K.bankS = dev(K, TN);
        K.bankO = dev(K, TN);
        K.X3 = dev(K, 4 * TN);
        K.MS = dev(K, 4 * TN);
        for (int i = 0; i < 4; ++i) K.XC[i] = dev(K, TN);
        K.xm = dev(K, 2 * TN);
        K.g64 = dev(K, (size_t) MAXT * RANK);
        K.g64x = dev(K, (size_t) 3 * MAXT * RANK);
        K.gp = dev(K, TN);
        K.fpp = dev(K, TN);
        K.NQ = dev(K, TN);
        K.NK = dev(K, TN);
        K.NV = dev(K, TN);
        K.q = dev(K, (size_t) MAXT * QC);
        K.k = dev(K, (size_t) MAXT * KVC);
        K.v = dev(K, (size_t) MAXT * KVC);
        K.kcv = dev(K, (size_t) MAXT * KVC);
        K.vcv = dev(K, (size_t) MAXT * KVC);
        K.r = dev(K, (size_t) MAXT * NH_C * NREL);
        K.sink = dev(K, (size_t) MAXT * NH_C);
        K.gate = dev(K, (size_t) MAXT * QC);
        K.dk = dev(K, (size_t) MAXT * NKV_C * NCONV);
        K.dv = dev(K, (size_t) MAXT * NKV_C * NCONV);
        K.ao = dev(K, (size_t) MAXT * QC);
        K.nparts_max = (ctx + KC - 1) / KC + 1;
        K.part_n = (size_t) MAXT * NH_C * (kv8 ? 1 : 2) * K.nparts_max * (HD + 2);   // p4f: parts of 64 keys (p18: int8: half)
        if (NC == 1) K.part_n = std::max(K.part_n, (size_t) (pfc / QS) * NH_C * QS * (QS + SWA - 1));   // p33: the reading path's S fits
        K.part = dev(K, K.part_n);
        K.ab = dev(K, TN);
        K.A = dev(K, TN);
        K.F = dev(K, TN);
        K.F2 = dev(K, TN);
        K.FN = dev(K, TN);
        K.lg = dev(K, (size_t) MAXT * NE);
        K.topw = dev(K, (size_t) MAXT * TOPK);
        K.topi = (int*) alloc(K, (size_t) MAXT * TOPK * 4);
        K.jobs = (EJob*) alloc(K, sizeof(EJob) * MAXT * TOPK);
        K.perm = (int*) alloc(K, sizeof(int) * MAXT * TOPK);
        K.mjobs = (MJob*) alloc(K, sizeof(MJob) * MAXT * TOPK);
        K.nmjobs = (int*) alloc(K, 4);
        K.GU = dev(K, (size_t) MAXT * TOPK * 2 * FF_C);
        K.HH = dev(K, (size_t) MAXT * TOPK * FF_C);
        K.EY = dev(K, (size_t) MAXT * TOPK * N);
        K.MO = dev(K, TN);
        K.hx = dev(K, TN);
        K.logits = dev(K, (size_t) MAXT * NV_C);
        K.tau = dev(K, MAXT);
        K.pos = (int*) alloc(K, MAXT * 4);
        K.dtok = (unsigned*) alloc(K, 4);
        CK(cudaMemset(K.dtok, 0, 4));
    }

    Engine(const std::string& path, int ctx_, const std::string& mtp = "") : g(path), ctx(ctx_) {
        ds_ring = getenv("INK2_DS_RING") ? atoi(getenv("INK2_DS_RING")) != 0 : (NC == 1 || ctx > 65536);   // p14 (p33: always with one card)
        kv8 = getenv("INK2_KV8") ? atoi(getenv("INK2_KV8")) != 0 : ctx > 262144;   // p18
        if (getenv("INK2_KV8_F16")) kv8_f16 = std::max(0, atoi(getenv("INK2_KV8_F16")));
        if (getenv("INK2_KV8_ROT")) kv8_rot = atoi(getenv("INK2_KV8_ROT")) != 0;
        if (getenv("INK2_KV8_PART")) kv8_part = atoi(getenv("INK2_KV8_PART")) & 3;
        if (kv8 && (stare || attn_old)) throw std::runtime_error("INK2_KV8 needs the p4f attention (no INK2_STARE / INK2_ATTN_OLD)");
        if (!getenv("INK2_PF_QSF")) {   // p14: the full layers' score matrix 3 x QSF x (ctx + QSF) within the budget
            const double mb = getenv("INK2_PF_SMB") ? atof(getenv("INK2_PF_SMB")) : NC == 1 ? 100.0 : (ctx <= 98304 ? 1100.0 : ctx <= 196608 ? 450.0 : 220.0);   // p31
            while (pf_qs_full > 64 && 3.0 * pf_qs_full * (ctx + pf_qs_full) * 4 > mb * 1048576.0) pf_qs_full /= 2;
        }
        const std::string a = "inkling2_loop.";
        if (g.s("general.architecture") != "inkling2_loop" || g.i(a + "block_count") != NPHYS || g.i(a + "embedding_length") != N ||
            g.i(a + "expert_count") != NE || g.i(a + "expert_used_count") != TOPK || g.i(a + "expert_feed_forward_length") != FF ||
            g.i(a + "loop.start") != LS || g.i(a + "loop.end") != LE || g.i(a + "loop.count") != NPASS || g.i(a + "vocab_size") != NV ||
            g.i(a + "vocab_size_unpadded") != NVU || g.i(a + "attention.sliding_window") != SWA ||
            g.i(a + "attention.relative_extent") != EXT || g.i(a + "attention.relative_length") != NREL ||
            g.i(a + "attn_res.block_size") != BLK || g.i(a + "shortconv.l_cache") != NCONV || g.i(a + "loop.gate_rank") != RANK)
            throw std::runtime_error("geometry differs from v1's constants");
        eps = (float) g.f(a + "attention.layer_norm_rms_epsilon");
        qk_clip = (float) g.f(a + "attention.qk_clip");
        rel_clip_thr = (float) g.f(a + "attention.relative_clip");
        log_alpha = (float) g.f(a + "attention.log_scale_alpha");
        log_floor = (float) g.i(a + "attention.log_scale_floor");
        situ_b = (float) g.f(a + "activation.situ_beta");
        situ_lb = (float) g.f(a + "activation.situ_linear_beta");
        wscale = (float) g.f(a + "expert_weights_scale");
        softcap = (float) g.f(a + "final_logit_softcapping");
        tok = &g.t("token_embd.weight");
        if (tok->type != G_Q8_0) throw std::runtime_error("token_embd is not Q8_0");
        int ndev = 0;
        CK(cudaGetDeviceCount(&ndev));
        if (ndev < NC) throw std::runtime_error("needs two cards");
        for (int c = 0; c < NC; ++c) {
            cards[c].dev = c;
            cards[c].c = c;
        }
        const double t0 = now_s();
        each_card([&](int c) { load_card(cards[c]); });
        for (int c = 0; c < NC; ++c) {
            for (int p = 0; p < 2; ++p) CK(cudaHostAlloc((void**) &X.buf[c][p], (size_t) MAXT * N * 4, cudaHostAllocMapped | cudaHostAllocPortable));
            unsigned* f = nullptr;
            CK(cudaHostAlloc((void**) &f, 64, cudaHostAllocMapped | cudaHostAllocPortable));
            *f = 0;
            X.flag[c] = f;
            for (int p = 0; p < 2; ++p) {   // p52
                CK(cudaHostAlloc((void**) &X.ll[c][p], (size_t) MAXT * N * 8, cudaHostAllocMapped | cudaHostAllocPortable));
                std::memset(X.ll[c][p], 0, (size_t) MAXT * N * 8);
            }
        }
        CK(cudaHostAlloc((void**) &h_logits, (size_t) MAXT * NV * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &h_amv2, (size_t) NC * MAXT * 4, cudaHostAllocPortable));   // p47
        CK(cudaHostAlloc((void**) &h_ami2, (size_t) NC * MAXT * 4, cudaHostAllocPortable));
        each_card([&](int c) {
            CK(cudaSetDevice(cards[c].dev));
            cards[c].amv_r = (float*) alloc(cards[c], MAXT * 4);
            cards[c].mx2 = (float*) alloc(cards[c], MAXT * 2 * 4);   // p76
            cards[c].ami_r = (int*) alloc(cards[c], MAXT * 4);
        });
        if (getenv("INK2_PF_CHUNK")) pfc = std::max(MAXT, atoi(getenv("INK2_PF_CHUNK")));
        if (pfc + SWA - 1 > RING) throw std::runtime_error("INK2_PF_CHUNK: a chunk and its window (RING - SWA + 1) do not fit the ring");   // p31
        if (getenv("INK2_PF_MIN")) pf_min = std::max(1, atoi(getenv("INK2_PF_MIN")));
        if (getenv("INK2_PF_OFF")) pf_min = 1 << 30;
        if (!ilv && (pf_deq || stare || rows2 || getenv("INK2_ROWS3")))
            throw std::runtime_error("INK2_PF_DEQ / INK2_STARE / INK2_ROWS2 / INK2_ROWS3 read interleaved experts: add INK2_FP4_ILV=1");
        CK(cudaHostAlloc((void**) &h_x0, (size_t) std::max(MAXT, pfc) * N * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &h_pos, (size_t) std::max(MAXT, pfc) * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &h_tau, (size_t) std::max(MAXT, pfc) * 4, cudaHostAllocPortable));
        for (int c = 0; c < NC; ++c) pf_rec[c] = 0;
        for (int z = 0; z < Card::ZMAX; ++z)
            for (int c = 0; c < NC; ++c) zrec[z][c] = 0;
        if (zip && !(NC == 2 && xsplit > 1 && pf_min < (1 << 30) && zip > 1)) {
            std::fprintf(stderr, "ink2_v1: INK2_PF_ZIP off: it takes 2 or 3, two cards, the reading path and INK2_XSPLIT above 1\n");
            zip = 0;
        }
        zip = std::min(zip, (int) Card::ZMAX);
        if (pf_min < (1 << 30)) {
            each_card([&](int c) { alloc_pf(cards[c]); });
        }
        if (!mtp.empty()) {   // p6
            dsg = new GgufMin(mtp);
            if (dsg->has("dflash.selector_rank")) {   // p74: DFlash 2
                d2_init();
            } else {
            if (dsg->s("general.architecture") != "dflash" || dsg->i("dflash.block_size") != DBLK || dsg->i("dflash.block_count") != DSL)
                throw std::runtime_error("the drafter is not Inkling 2's DSpark");
            if (!g_draft_vocab.empty()) {   // p13: card c takes the c-th half of the (sorted) draft vocabulary
                FILE* f = std::fopen(g_draft_vocab.c_str(), "rb");
                int32_t n = 0;
                if (!f || std::fread(&n, 4, 1, f) != 1 || n <= 1) throw std::runtime_error("draft vocab: " + g_draft_vocab);
                std::vector<int> ids(n);
                if (std::fread(ids.data(), 4, n, f) != (size_t) n) throw std::runtime_error("draft vocab: " + g_draft_vocab);
                std::fclose(f);
                std::sort(ids.begin(), ids.end());
                for (int c = 0; c < NC; ++c) dvmap[c].assign(ids.begin() + (size_t) c * n / NC, ids.begin() + (size_t) (c + 1) * n / NC);
            }
            each_card([&](int c) { ds_load(cards[c]); });
            if (!g_draft_vocab.empty()) {
                auto pick = [&](int c) {   // own threads: both() needs card 1's worker, which starts after the constructor
                    CK(cudaSetDevice(cards[c].dev));
                    vt[c] = "risar-slovar";
                    cards[c].D.dvout = q8_pick(cards[c], "output.weight", dvmap[c]);
                };
                each_card(pick);
                std::fprintf(stderr, "ink2_v1: slovar risarja %zu + %zu zetonov (%s)\n", dvmap[0].size(), NC > 1 ? dvmap[NC - 1].size() : 0, g_draft_vocab.c_str());
            }
            CK(cudaHostAlloc((void**) &h_amv_pinned, NC * 4, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &h_ams_pinned, NC * 4, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &h_ami_pinned, NC * 4, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &h_prev_pinned, NC * 4, cudaHostAllocPortable));
            ds_on = true;
            }
        }
        if (zip) {   // zip: before the experts, which take the VRAM that is left, and only when every expert still fits beside it
            const long long zb = (long long) (zip - 1) * ((long long) pfc * N * 4 * (ds_on || d2_on ? 14 : 10) + (long long) pfc * 8);
            long long spare[NC];
            each_card([&](int c) { spare[c] = zip_spare(cards[c]); });
            long long sp = spare[0];
            for (int c = 1; c < NC; ++c) sp = std::min(sp, spare[c]);
            if (sp < zb) {
                std::fprintf(stderr, "ink2_v1: INK2_PF_ZIP off: its %lld MiB a card do not fit beside the experts (%lld MiB to spare)\n", zb >> 20, sp >> 20);
                zip = 0;
            } else {
                for (int z = 1; z < zip; ++z) {
                    CK(cudaHostAlloc((void**) &h_x0z[z], (size_t) std::max(MAXT, pfc) * N * 4, cudaHostAllocPortable));
                    CK(cudaHostAlloc((void**) &h_posz[z], (size_t) std::max(MAXT, pfc) * 4, cudaHostAllocPortable));
                    CK(cudaHostAlloc((void**) &h_tauz[z], (size_t) std::max(MAXT, pfc) * 4, cudaHostAllocPortable));
                    h_rawz[z].assign((size_t) 1 << 16, 0);
                }
                each_card([&](int c) { zip_alloc(cards[c]); });
                each_card([&](int c) { spare[c] = zip_spare(cards[c]); });   // what the driver really took (it rounds allocations up)
                long long left = spare[0];
                for (int c = 1; c < NC; ++c) left = std::min(left, spare[c]);
                if (left < 0) {
                    each_card([&](int c) { zip_free(cards[c]); });
                    std::fprintf(stderr, "ink2_v1: INK2_PF_ZIP off: its buffers took more than the %lld MiB to spare beside the experts\n", sp >> 20);
                    zip = 0;
                } else {
                    std::fprintf(stderr, "ink2_v1: INK2_PF_ZIP %d (reading chunks at once): %lld MiB a card of the %lld to spare beside the experts\n", zip, zb >> 20, sp >> 20);
                }
            }
        }
        each_card([&](int c) { place_experts(cards[c]); });   // p30
        if (ck_on) {   // p21
            const long long mb = getenv("INK2_CKPT_MB") ? atoll(getenv("INK2_CKPT_MB")) : (NC == 1 ? 2048 : 24576);   // p30: one card, less RAM
            ck_max = (int) std::max(0LL, mb * 1048576 / (long long) (NC * ck_bytes()));
            std::fprintf(stderr, "ink2_v1: kontrolne tocke pogovora v RAM-u: do %d po %.0f MiB (obe kartici); odlozeni pogovori do %zu MiB\n",
                         ck_max, NC * ck_bytes() / 1048576.0, park_max >> 20);
        }
        if (getenv("INK2_VRAM_LOG"))
            for (auto& [k, v] : vtag[0]) std::fprintf(stderr, "ink2_v1: VRAM %-20s %8.1f / %8.1f MiB\n", k.c_str(), v / 1048576.0, vtag[NC - 1][k] / 1048576.0);
        if (kv8)
            std::fprintf(stderr, "ink2_v1: KV polnih plasti int8 od vrstice %d (rot %d, del %d), S %s, Kf/Vf/RB/PV %s\n", std::min(kv8_f16, ctx),
                         kv8_rot, kv8_part, cards[0].P.S == cards[0].part ? "v K.part" : "svoj", cards[0].P.shared ? "v Xg/EY/GU" : "svoji");
        std::fprintf(stderr, "ink2_v1: naloženo v %.1f s, VRAM %.2f / %.2f GiB, ctx %d, kos branja %d\n", now_s() - t0,
                     cards[0].vram / 1073741824.0, cards[NC - 1].vram / 1073741824.0, ctx, pf_min < (1 << 30) ? pfc : 0);
        if (NC > 1) w1 = std::thread([this]() {   // p30: one card needs no worker
            CK(cudaSetDevice(cards[NC - 1].dev));
            int seen = 0;
            for (;;) {
                while (go.load(std::memory_order_acquire) == seen) std::this_thread::yield();
                seen = go.load();
                if (quit) return;
                job(1);
                done.fetch_add(1, std::memory_order_release);
            }
        });
    }
    ~Engine() {
        quit = true;
        go.fetch_add(1);
        if (w1.joinable()) w1.join();
    }
    void both(std::function<void(int)> f) {   // f(0) here and f(1) on card 1's thread, at the same time
        if (NC == 1) {   // p30
            CK(cudaSetDevice(cards[0].dev));
            f(0);
            return;
        }
        job = f;
        const int d = done.load();
        go.fetch_add(1, std::memory_order_release);
        CK(cudaSetDevice(cards[0].dev));
        f(0);
        while (done.load(std::memory_order_acquire) == d) std::this_thread::yield();
    }

    // ------------------------------------------------------------------------------------------------ the card's program
    void gemv8(Card& K, const Mat& m, const float* x, int ldx, float* y, int ldy, int T) {
        k_gemv8<<<(m.rows + GW_BR - 1) / GW_BR, 256, (size_t) T * 512 * 4, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, ldx, y, ldy, T);
    }
    // p4a: T = 1, the rows of a (view of a) Q8S matrix to the segments' outputs
    void gemv8_t1(Card& K, const Mat& m, const float* x, const Segs& sg) {
        const int blocks = (m.rows + G1_BR - 1) / G1_BR, nch = (m.cols + 511) / 512;
        const int S = blocks >= 112 ? 1 : std::min(nch, (112 + blocks - 1) / blocks);
        if (g1w == 1 || g1w == 2 || g1w == 4) {   // p53: a warp's rows with all S parts
            const int rpb = 8 * g1w;
            if (g1w == 1) k_gemv8_t1w<1><<<(m.rows + rpb - 1) / rpb, 256, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, sg, S);
            else if (g1w == 2) k_gemv8_t1w<2><<<(m.rows + rpb - 1) / rpb, 256, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, sg, S);
            else k_gemv8_t1w<4><<<(m.rows + rpb - 1) / rpb, 256, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, sg, S);
        } else
        if (g1wpb == 4) k_gemv8_t1<4><<<dim3((m.rows + 15) / 16, S), 128, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, sg, S);
        else if (g1wpb == 2) k_gemv8_t1<2><<<dim3((m.rows + 7) / 8, S), 64, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, sg, S);
        else k_gemv8_t1<8><<<dim3(blocks, S), 256, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, sg, S);
    }
    // p6c: T rows with k_gemv8_t1's sums (the same parts S as at T = 1)
    template <int TT>
    void t1tt(Card& K, const Mat& m, const float* x, int ldx, const SegsT& sg, dim3 g, int S) {
        k_gemv8_t1TT<TT><<<g, 256, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, ldx, sg, S);
    }
    void gemv8_t1T(Card& K, const Mat& m, const float* x, int ldx, const SegsT& sg, int T) {
        const int blocks = (m.rows + G1_BR - 1) / G1_BR, nch = (m.cols + 511) / 512;
        const int S = blocks >= 112 ? 1 : std::min(nch, (112 + blocks - 1) / blocks);
        const dim3 g(blocks, S);
        if (t1tt_on) switch (T) {   // p9
            case 2: return t1tt<2>(K, m, x, ldx, sg, g, S);
            case 3: return t1tt<3>(K, m, x, ldx, sg, g, S);
            case 4: return t1tt<4>(K, m, x, ldx, sg, g, S);
            case 5: return t1tt<5>(K, m, x, ldx, sg, g, S);
            case 6: return t1tt<6>(K, m, x, ldx, sg, g, S);
            case 7: return t1tt<7>(K, m, x, ldx, sg, g, S);
            case 8: return t1tt<8>(K, m, x, ldx, sg, g, S);
            case 9: return t1tt<9>(K, m, x, ldx, sg, g, S);
            case 10: return t1tt<10>(K, m, x, ldx, sg, g, S);
            case 11: return t1tt<11>(K, m, x, ldx, sg, g, S);
            case 12: return t1tt<12>(K, m, x, ldx, sg, g, S);
            case 13: return t1tt<13>(K, m, x, ldx, sg, g, S);
            case 14: return t1tt<14>(K, m, x, ldx, sg, g, S);
            case 15: return t1tt<15>(K, m, x, ldx, sg, g, S);
            case 16: return t1tt<16>(K, m, x, ldx, sg, g, S);
            default: break;
        }
        k_gemv8_t1T<<<g, 256, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, ldx, sg, S, T);
    }
    // p9: the drafter's matrices (DBLK = 8 rows, ingest windows) by the window GEMV instead of k_gemv8 (~85-186 GB/s)
    void dsgemv(Card& K, const Mat& m, const float* x, int ldx, float* y, int ldy, int T) {
        if (ds_t1 && T > 1 && T <= MAXT) gemv8_t1T(K, m, x, ldx, segs1(ldy, y), T);
        else gemv8(K, m, x, ldx, y, ldy, T);
    }
    static SegsT segs1(int ld, float* d) {
        SegsT sg{};
        sg.n = 1;
        sg.r0[0] = 0;
        sg.ld[0] = ld;
        sg.dst[0] = d;
        return sg;
    }
    static Segs segs(std::initializer_list<std::pair<int, float*>> l) {   // (rows, output) one after another
        Segs sg{};
        int r = 0;
        for (auto& [n, d] : l) {
            sg.r0[sg.n] = r;
            sg.dst[sg.n++] = d;
            r += n;
        }
        return sg;
    }
    static Mat rows_of(const Mat& m, int r0, int n) {
        Mat v = m;
        v.rows = n;
        v.q = m.q + (size_t) r0 * m.cols;
        v.d = (const __half*) m.d + (size_t) r0 * (m.cols / 32);
        return v;
    }
    void gemv32(Card& K, const float* W, int rows, int cols, const float* x, int ldx, float* y, int ldy, int T) {
        k_gemv32<<<(rows + GW_BR - 1) / GW_BR, 256, (size_t) T * 128 * 4, K.st>>>(W, rows, cols, x, ldx, y, ldy, T);
    }
    void rms(Card& K, const float* x, const float* w, float* y, int T, float e) { k_rms<<<T, 256, 0, K.st>>>(x, N, w, y, N, N, e); }
    void copy(Card& K, float* dst, const float* src, int T) { CK(cudaMemcpyAsync(dst, src, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, K.st)); }
    void xch(Card& K, float* y, int n) {   // p4c: sequence numbers from the card's forward counter (dtok)
        if (NC == 1) return;   // p30
        const unsigned i = K.xidx++;
        const int par = i & 1, c = K.c, oc = NC > 1 ? 1 - c : c;
        k_xput2<<<1, 1024, 0, K.st>>>(y, n, X.buf[c][par], X.flag[c], K.dtok, i);
        k_xget2<<<1, 1024, 0, K.st>>>(y, n, X.buf[oc][par], X.flag[oc], K.dtok, i, c);
    }
    void control(Card& K, const LW& L, int j, const float* x, const float* m, float* out, int T) {
        if (fuse && !stare && !dumping) {   // p4e: the rms of x and m inside the down projection (p6b: T rows)
            if (T == 1) k_ctrl_down1<<<RANK, 256, 0, K.st>>>(L.c_down[j], x, m, K.g64);
            else k_ctrl_downT<<<dim3(RANK, T), 256, 0, K.st>>>(L.c_down[j], x, m, K.g64);
            k_ctrl_gf<<<N / 8, 256, 0, K.st>>>(L.c_gain[j], L.c_forget[j], K.g64, x, m, N, out, T);
            return;
        }
        k_ctrl_in<<<T, 256, 0, K.st>>>(x, m, N, K.xm);
        if (!stare) {
            k_gemv32_rows<<<RANK, 256, 0, K.st>>>(L.c_down[j], 2 * N, K.xm, 2 * N, K.g64, RANK, T);
            k_ctrl_gf<<<N / 8, 256, 0, K.st>>>(L.c_gain[j], L.c_forget[j], K.g64, x, m, N, out, T);
            return;
        }
        gemv32(K, L.c_down[j], RANK, 2 * N, K.xm, 2 * N, K.g64, RANK, T);
        gemv32(K, L.c_gain[j], N, RANK, K.g64, RANK, K.gp, N, T);
        gemv32(K, L.c_forget[j], N, RANK, K.g64, RANK, K.fpp, N, T);
        k_ctrl_out<<<(T * N + 255) / 256, 256, 0, K.st>>>(x, m, N, K.gp, K.fpp, out, T);
    }
    void mix(Card& K, const States& S, const float* q, int nq, float* out, int T) {
        if (stare) k_mix<<<T, 256, 0, K.st>>>(S, q, nq, out);
        else if (mix3 && T <= MAXT) {   // p51
            k_mix3a<<<dim3(24, T), 32, 0, K.st>>>(S, q, nq);
            k_mix3b<<<dim3(N / 256, T), 256, 0, K.st>>>(S, nq, out, T);
        } else k_mix2<<<T, 768, 0, K.st>>>(S, q, nq, out);
    }
    States states(Card& K, const float* last) {
        States S{};
        for (int i = 0; i < K.nb; ++i) S.p[i] = K.B[i];
        S.p[K.nb] = last;
        S.n = K.nb + 1;
        return S;
    }
    void dump_rows(std::vector<float>& v, const float* d, int T) {
        const size_t o = v.size();
        v.resize(o + (size_t) T * N);
        CK(cudaMemcpy(v.data() + o, d, (size_t) T * N * 4, cudaMemcpyDeviceToHost));
    }

    void layer(Card& K, int p, int slot, bool loop, int T, int nparts) {
        const LW& L = K.L[p];
        const int c = K.c, T_N = T * N, oc = NC > 1 ? 1 - c : c;   // p30: oc = the other card (one card: itself, never read)
        const float* xin[3] = {K.cur, K.cur, K.cur};
        const int f2 = fuse && !stare && !dumping ? fuse2 : 0;   // p22: ink2_fuse.cuh's merged launches (a mask)
        if ((f2 & 1) && L.attn_rq && loop) {   // p22: the layer's depth mix and the loop messages' in one launch
            States S{};
            S.p[0] = K.bankS;
            S.p[1] = K.bankO;
            S.n = 2;
            k_mix2x<<<dim3(T, 2), 768, 0, K.st>>>(states(K, K.cur), L.attn_rq, L.nrq, K.X3, S, L.msg_q, 4, K.MS);
            for (int j = 0; j < 3; ++j) xin[j] = K.X3 + (size_t) (L.full ? j : 0) * T_N;
        } else {
            if (L.attn_rq) {
                mix(K, states(K, K.cur), L.attn_rq, L.nrq, K.X3, T);
                for (int j = 0; j < 3; ++j) xin[j] = K.X3 + (size_t) (L.full ? j : 0) * T_N;
            }
            if (loop) {
                States S{};
                S.p[0] = K.bankS;
                S.p[1] = K.bankO;
                S.n = 2;
                mix(K, S, L.msg_q, 4, K.MS, T);
            }
        }
        if (loop && (f2 & 2)) {   // p22: the three controllers' down projections in one launch, their gains and forgets in another
            Ctl3 A{};
            for (int j = 0; j < 3; ++j) {
                A.W[j] = L.c_down[j];
                A.Wg[j] = L.c_gain[j];
                A.Wf[j] = L.c_forget[j];
                A.x[j] = xin[j];
                A.m[j] = K.MS + (size_t) j * T_N;
                A.g[j] = K.g64x + (size_t) j * MAXT * RANK;
                A.y[j] = K.XC[j];
            }
            if (T == 1) k_ctrl_down1x<<<dim3(RANK, 3), 256, 0, K.st>>>(A);
            else k_ctrl_downTx<<<dim3(RANK, T, 3), 256, 0, K.st>>>(A);
            k_ctrl_gfx<<<dim3(N / 8, 3), 256, 0, K.st>>>(A, T);
            for (int j = 0; j < 3; ++j) xin[j] = K.XC[j];
        } else if (loop) {
            for (int j = 0; j < 3; ++j) {
                control(K, L, j, xin[j], K.MS + (size_t) j * T_N, K.XC[j], T);
                xin[j] = K.XC[j];
            }
        }
        const float *nq, *nk, *nv;
        if (!L.full && !loop) {
            rms(K, xin[0], L.attn_norm, K.NQ, T, eps);
            nq = nk = nv = K.NQ;
        } else {
            if (f2 & 4) {   // p22: the three norms in one launch
                k_rms3<<<dim3(T, 3), 256, 0, K.st>>>(xin[0], xin[1], xin[2], L.full ? L.q_norm : L.attn_norm, L.full ? L.k_norm : L.attn_norm,
                                                     L.full ? L.v_norm : L.attn_norm, K.NQ, K.NK, K.NV, eps);
            } else {
                rms(K, xin[0], L.full ? L.q_norm : L.attn_norm, K.NQ, T, eps);
                rms(K, xin[1], L.full ? L.k_norm : L.attn_norm, K.NK, T, eps);
                rms(K, xin[2], L.full ? L.v_norm : L.attn_norm, K.NV, T, eps);
            }
            nq = K.NQ;
            nk = K.NK;
            nv = K.NV;
        }
        const bool fzT = fuse && !stare && !dumping;   // p6b: the fused path for any T <= MAXT
        const bool g3 = (f2 & 4) && nq != nk && L.cat.cols == N && (T == 1 || (t1tt_on && !win_seg));   // p22: the three inputs' matrices in one launch
        if (g3) {
            const int nqr = 2 * QC + NH_C * NREL + NH_C, nkr = KVC + (L.full ? NKV_C * NCONV : 0);
            typedef std::vector<std::pair<int, float*>> Outs;
            const Outs oq = {{QC, K.q}, {QC, K.gate}, {NH_C * NREL, K.r}, {NH_C, K.sink}};
            Outs ok = {{KVC, K.k}}, ov = {{KVC, K.v}};
            if (L.full) {
                ok.push_back({NKV_C * NCONV, K.dk});
                ov.push_back({NKV_C * NCONV, K.dv});
            }
            struct PD {
                Mat m;
                const float* x;
                const Outs* out;
            };
            const PD pd[3] = {{rows_of(L.cat, 0, nqr), nq, &oq}, {rows_of(L.cat, nqr, nkr), nk, &ok}, {rows_of(L.cat, nqr + nkr, nkr), nv, &ov}};
            GParts1 A1{};
            GPartsT AT{};
            const int nch = (N + 511) / 512;
            int b0 = 0, p0 = 0, c0 = 0;
            for (int i = 0; i < 3; ++i) {
                GPart g{};
                g.q = pd[i].m.q;
                g.d = (const __half*) pd[i].m.d;
                g.x = pd[i].x;
                g.rows = pd[i].m.rows;
                g.blocks = (g.rows + G1_BR - 1) / G1_BR;
                g.S = g.blocks >= 112 ? 1 : std::min(nch, (112 + g.blocks - 1) / g.blocks);   // gemv8_t1's split
                g.b0 = b0;
                g.p0 = p0;
                g.c0 = c0;
                b0 += g.blocks * g.S;
                p0 += g.S * T * g.rows;
                c0 += g.blocks;
                A1.p[i] = AT.p[i] = g;
                int r = 0;
                for (const auto& [n, d] : *pd[i].out) {
                    Segs& s1 = A1.sg[i];
                    SegsT& sT = AT.sg[i];
                    s1.r0[s1.n] = sT.r0[sT.n] = r;
                    sT.ld[sT.n] = n;
                    s1.dst[s1.n++] = sT.dst[sT.n++] = d;
                    r += n;
                }
            }
            A1.n = AT.n = 3;
            if (p0 > GM_PART || c0 > 4096) throw std::runtime_error("k_gemv8_p: parts too big");
            switch (T) {
                case 1: k_gemv8_p1<<<b0, 256, 0, K.st>>>(A1); break;
                case 2: k_gemv8_pT<2><<<b0, 256, 0, K.st>>>(AT); break;
                case 3: k_gemv8_pT<3><<<b0, 256, 0, K.st>>>(AT); break;
                case 4: k_gemv8_pT<4><<<b0, 256, 0, K.st>>>(AT); break;
                case 5: k_gemv8_pT<5><<<b0, 256, 0, K.st>>>(AT); break;
                case 6: k_gemv8_pT<6><<<b0, 256, 0, K.st>>>(AT); break;
                case 7: k_gemv8_pT<7><<<b0, 256, 0, K.st>>>(AT); break;
                case 8: k_gemv8_pT<8><<<b0, 256, 0, K.st>>>(AT); break;
                case 9: k_gemv8_pT<9><<<b0, 256, 0, K.st>>>(AT); break;
                case 10: k_gemv8_pT<10><<<b0, 256, 0, K.st>>>(AT); break;
                case 11: k_gemv8_pT<11><<<b0, 256, 0, K.st>>>(AT); break;
                case 12: k_gemv8_pT<12><<<b0, 256, 0, K.st>>>(AT); break;
                case 13: k_gemv8_pT<13><<<b0, 256, 0, K.st>>>(AT); break;
                case 14: k_gemv8_pT<14><<<b0, 256, 0, K.st>>>(AT); break;
                case 15: k_gemv8_pT<15><<<b0, 256, 0, K.st>>>(AT); break;
                default: k_gemv8_pT<16><<<b0, 256, 0, K.st>>>(AT); break;
            }
        } else if (T > 1 && fzT) {   // p6b: the matrices of one input in one launch, T rows
            const int nqr = 2 * QC + NH_C * NREL + NH_C, nkr = KVC + (L.full ? NKV_C * NCONV : 0);
            auto st = [&](std::initializer_list<std::pair<int, float*>> l) {
                SegsT sg{};
                int r = 0;
                for (auto& [n, d] : l) {
                    sg.r0[sg.n] = r;
                    sg.ld[sg.n] = n;
                    sg.dst[sg.n++] = d;
                    r += n;
                }
                return sg;
            };
            auto run = [&](const Mat& m, const float* x, const SegsT& sg) {   // p6c: the bits of T = 1 (INK2_WIN_SEG: k_gemv8_seg, ~4 % faster)
                if (win_seg)
                    k_gemv8_seg<<<(m.rows + GW_BR - 1) / GW_BR, 256, (size_t) T * 512 * 4, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, N,
                                                                                                  sg, T);
                else
                    gemv8_t1T(K, m, x, N, sg, T);
            };
            const SegsT sq = st({{QC, K.q}, {QC, K.gate}, {NH_C * NREL, K.r}, {NH_C, K.sink}});
            const SegsT sk = L.full ? st({{KVC, K.k}, {NKV_C * NCONV, K.dk}}) : st({{KVC, K.k}});
            const SegsT sv = L.full ? st({{KVC, K.v}, {NKV_C * NCONV, K.dv}}) : st({{KVC, K.v}});
            if (nq == nk && nk == nv) {
                SegsT all = sq;
                for (int i = 0; i < sk.n; ++i) { all.r0[all.n] = nqr + sk.r0[i]; all.ld[all.n] = sk.ld[i]; all.dst[all.n++] = sk.dst[i]; }
                for (int i = 0; i < sv.n; ++i) { all.r0[all.n] = nqr + nkr + sv.r0[i]; all.ld[all.n] = sv.ld[i]; all.dst[all.n++] = sv.dst[i]; }
                run(L.cat, nq, all);
            } else {
                run(rows_of(L.cat, 0, nqr), nq, sq);
                run(rows_of(L.cat, nqr, nkr), nk, sk);
                run(rows_of(L.cat, nqr + nkr, nkr), nv, sv);
            }
        } else if (T == 1 && !stare) {   // p4a: the matrices of one input in one launch
            const int nqr = 2 * QC + NH_C * NREL + NH_C, nkr = KVC + (L.full ? NKV_C * NCONV : 0);
            const Segs sq = segs({{QC, K.q}, {QC, K.gate}, {NH_C * NREL, K.r}, {NH_C, K.sink}});
            const Segs sk = L.full ? segs({{KVC, K.k}, {NKV_C * NCONV, K.dk}}) : segs({{KVC, K.k}});
            const Segs sv = L.full ? segs({{KVC, K.v}, {NKV_C * NCONV, K.dv}}) : segs({{KVC, K.v}});
            if (nq == nk && nk == nv) {
                Segs all = sq;
                for (int i = 0; i < sk.n; ++i) {
                    all.r0[all.n] = nqr + sk.r0[i];
                    all.dst[all.n++] = sk.dst[i];
                }
                for (int i = 0; i < sv.n; ++i) {
                    all.r0[all.n] = nqr + nkr + sv.r0[i];
                    all.dst[all.n++] = sv.dst[i];
                }
                gemv8_t1(K, L.cat, nq, all);
            } else {
                gemv8_t1(K, rows_of(L.cat, 0, nqr), nq, sq);
                gemv8_t1(K, rows_of(L.cat, nqr, nkr), nk, sk);
                gemv8_t1(K, rows_of(L.cat, nqr + nkr, nkr), nv, sv);
            }
        } else {
            gemv8(K, L.wq, nq, N, K.q, QC, T);
            gemv8(K, L.wk, nk, N, K.k, KVC, T);
            gemv8(K, L.wv, nv, N, K.v, KVC, T);
            gemv8(K, L.wrel, nq, N, K.r, NH_C * NREL, T);
            gemv8(K, L.wsink, nq, N, K.sink, NH_C, T);
            gemv8(K, L.wgate, nq, N, K.gate, QC, T);
            if (L.full) {
                gemv8(K, L.wkd, nk, N, K.dk, NKV_C * NCONV, T);
                gemv8(K, L.wvd, nv, N, K.dv, NKV_C * NCONV, T);
            }
        }
        const float kconv = 1.0f / sqrtf((float) ((L.full ? 2 : 1) * NCONV + 1));
        const bool fz = fzT;   // p4e (p6b: any T)
        float* hk = K.keep_hist ? K.hist + (size_t) slot * HISTS : nullptr;   // p5a: the window's inputs for the rollback
        if (fz && T > 1) {
            k_post_qkvT<<<dim3(2 * NH_C + 2 * NKV_C, T), 128, 0, K.st>>>(
                K.q, K.r, K.k, K.v, L.full ? K.dk : nullptr, L.full ? K.dv : nullptr, K.sk[slot], K.sv[slot], L.k_conv, L.v_conv, L.q_clip,
                L.rel_clip, c * NH_C, qk_clip, rel_clip_thr, L.full ? K.tau : nullptr, eps, kconv, K.pos, !L.full, K.kc[slot], K.vc[slot], hk,
                hk ? hk + (size_t) MAXT * KVC : nullptr, K.q8[slot]);
            k_post_state<<<(2 * KVC + 255) / 256, 256, 0, K.st>>>(K.k, K.v, K.sk[slot], K.sv[slot], T);
        } else if (fz) {
            k_post_qkv<<<2 * NH_C + 2 * NKV_C, 128, 0, K.st>>>(K.q, K.r, K.k, K.v, L.full ? K.dk : nullptr, L.full ? K.dv : nullptr, K.sk[slot],
                                                               K.sv[slot], L.k_conv, L.v_conv, L.q_clip, L.rel_clip, c * NH_C, qk_clip,
                                                               rel_clip_thr, L.full ? K.tau : nullptr, eps, kconv, K.pos, !L.full,
                                                               K.kc[slot], K.vc[slot], K.q8[slot]);
        } else {
        if (K.keep_hist)
            CK(cudaMemcpyAsync(K.hist + (size_t) slot * HISTS + 0, K.k, (size_t) T * KVC * 4, cudaMemcpyDeviceToDevice, K.st));
        if (K.keep_hist)
            CK(cudaMemcpyAsync(K.hist + (size_t) slot * HISTS + (size_t) MAXT * KVC, K.v, (size_t) T * KVC * 4, cudaMemcpyDeviceToDevice, K.st));
        k_conv<<<(KVC + 127) / 128, 128, 0, K.st>>>(K.k, KVC, K.sk[slot], L.k_conv, L.full ? K.dk : nullptr, HD, kconv, K.kcv, T);
        k_conv<<<(KVC + 127) / 128, 128, 0, K.st>>>(K.v, KVC, K.sv[slot], L.v_conv, L.full ? K.dv : nullptr, HD, kconv, K.vcv, T);
        const float* tau = L.full ? K.tau : nullptr;
        k_headnorm<<<(T * NH_C + 7) / 8, 256, 0, K.st>>>(K.q, NH_C, HD, L.q_clip, c * NH_C, qk_clip, tau, 0, eps, T);
        k_headnorm<<<(T * NKV_C + 7) / 8, 256, 0, K.st>>>(K.kcv, NKV_C, HD, nullptr, 0, qk_clip, nullptr, 0, eps, T);
        k_headnorm<<<(T * NKV_C + 7) / 8, 256, 0, K.st>>>(K.vcv, NKV_C, HD, nullptr, 0, 0.0f, nullptr, 1, eps, T);
        k_headnorm<<<(T * NH_C + 7) / 8, 256, 0, K.st>>>(K.r, NH_C, NREL, L.rel_clip, c * NH_C, rel_clip_thr, tau, 0, eps, T);
        if (K.q8[slot].rot) k_rot<<<dim3(NH_C, T), 128, 0, K.st>>>(K.q, QC);   // p18
        if (K.q8[slot].rot)
            k_kv_store_rot<<<dim3(NKV_C, T, 2), 128, 0, K.st>>>(K.kcv, K.vcv, K.pos, K.kc[slot], K.vc[slot], K.q8[slot]);
        else
            k_kv_store<<<(T * KVC + 255) / 256, 256, 0, K.st>>>(K.kcv, K.vcv, K.pos, !L.full, K.kc[slot], K.vc[slot], T, K.q8[slot]);
        }
        if (!attn_old && !stare) {   // p4f: by KV head, 64 keys a block, the 3 query heads together (p5a: any T, grid z)
            const KV8 q8 = K.q8[slot];
            const bool q8k = kv8 && L.full, big = q8k && 2 * nparts > K.nparts_max;   // p18: K.part holds nparts_max parts a head: then 128 keys a part
            const int np = big ? nparts : L.full ? 2 * nparts : std::min(2 * nparts, (SWA + MAXT + KC2 - 1) / KC2 + 1);
            // p18: the parts wholly below f0 (f16 rows) by the f16 kernel, as in the 256k profile; the int8 one only from there
            const int kcb = big ? 128 : KC2, npf = !q8k ? np : (q8.k || q8.v) ? std::min(np, q8.f0 / kcb) : np;
            // p38: a window's rows side by side in grid x (they read the same keys at the same time: L2); INK2_ATT_ROWS=0 the old grid
            static const bool att_rows = !getenv("INK2_ATT_ROWS") || atoi(getenv("INK2_ATT_ROWS")) != 0;
            static const bool att_pv = !getenv("INK2_ATT_PV") || atoi(getenv("INK2_ATT_PV")) != 0;   // p70: int8 parts' V from shared memory
            const int nt = att_rows ? T : 1;
            const dim3 g16(npf * nt, NKV_C, T / nt), g8((np - npf) * nt, NKV_C, T / nt);
            if (att_w && L.full && T >= 2 && np >= att_w_min) {   // p61: 2 or 4 rows of the window a block, K and V read once
                // micro test (26. 11.): 2 rows a block for T = 2 and below 512 parts (32k), 4 from 512 parts for T >= 3
                const int tt = T >= 3 && np >= 512 ? 4 : 2;
                const dim3 w16(npf, NKV_C, (T + tt - 1) / tt), w8(np - npf, NKV_C, (T + tt - 1) / tt);
                if (npf > 0) {
                    if (tt == 4) {
                        if (big) k_attn_decw<128, false, 4><<<w16, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, K.part, np, q8, 0, T);
                        else k_attn_decw<64, false, 4><<<w16, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, K.part, np, q8, 0, T);
                    } else {
                        if (big) k_attn_decw<128, false, 2><<<w16, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, K.part, np, q8, 0, T);
                        else k_attn_decw<64, false, 2><<<w16, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, K.part, np, q8, 0, T);
                    }
                }
                if (np > npf) {
                    if (tt == 4) {
                        if (big) { if (att_pv) k_attn_decw<128, true, 4, true><<<w8, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, K.part, np, q8, npf, T); else k_attn_decw<128, true, 4><<<w8, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, K.part, np, q8, npf, T); }
                        else { if (att_pv) k_attn_decw<64, true, 4, true><<<w8, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, K.part, np, q8, npf, T); else k_attn_decw<64, true, 4><<<w8, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, K.part, np, q8, npf, T); }
                    } else {
                        if (big) k_attn_decw<128, true, 2><<<w8, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, K.part, np, q8, npf, T);   // p70: not staged here (micro test +8 %)
                        else { if (att_pv) k_attn_decw<64, true, 2, true><<<w8, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, K.part, np, q8, npf, T); else k_attn_decw<64, true, 2><<<w8, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, K.part, np, q8, npf, T); }
                    }
                }
            } else {
                if (npf > 0) {
                    if (big)
                        k_attn_dec<128, false><<<g16, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, !L.full, K.part, np, q8, 0, nt);
                    else
                        k_attn_dec<64, false><<<g16, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, !L.full, K.part, np, q8, 0, nt);
                }
                if (np > npf) {
                    if (big)
                        { if (att_pv) k_attn_dec<128, true, true><<<g8, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, !L.full, K.part, np, q8, npf, nt); else k_attn_dec<128, true><<<g8, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, !L.full, K.part, np, q8, npf, nt); }
                    else
                        { if (att_pv) k_attn_dec<64, true, true><<<g8, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, !L.full, K.part, np, q8, npf, nt); else k_attn_dec<64, true><<<g8, 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, !L.full, K.part, np, q8, npf, nt); }
                }
            }
            if (comb2 == 3 && np >= comb4_min && T <= 2) {   // p59: 4 blocks a head
                k_attn_comb4<4, 32><<<dim3(NH_C * 4, T), 32, 0, K.st>>>(K.part, np, K.ao);
                k_attn_comb4b<<<dim3(NH_C, T), 128, 0, K.st>>>(K.gate, K.ao, q8k && q8.rot);
            } else if (comb2 == 3 && np >= comb4_min) k_attn_comb5<<<dim3(NH_C, T), 128, 0, K.st>>>(K.part, np, K.gate, K.ao, q8k && q8.rot);   // p59
            else if (comb2) k_attn_comb2<<<dim3(NH_C, T), 128, 0, K.st>>>(K.part, np, K.gate, K.ao, q8k && q8.rot);   // p57
            else k_attn_comb<<<dim3(NH_C, T), 128, 0, K.st>>>(K.part, np, K.gate, K.ao, q8k && q8.rot);
        } else {
            const int np = L.full ? nparts : std::min(nparts, (SWA + KC - 1) / KC + 1);
            k_attn_part<<<dim3(np, NH_C, T), 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, !L.full, K.part, np);
            k_attn_comb<<<dim3(NH_C, T), 128, 0, K.st>>>(K.part, np, K.gate, K.ao, 0);
        }
        if (T == 1 && !stare) gemv8_t1(K, L.wo, K.ao, segs({{N, K.ab}}));
        else if (fzT) gemv8_t1T(K, L.wo, K.ao, QC, segs1(N, K.ab), T);   // p6c
        else gemv8(K, L.wo, K.ao, QC, K.ab, N, T);
        const int ip = slot < NPHYS ? slot : LS + (slot - NPHYS) % NCORE;
        if (fz) {   // p4e: the exchange, the post-convolution and the residual (or the block push) in one block
            const unsigned xi = K.xidx++;
            const int par = xi & 1;
            if (T > 1)
                k_xchg_convT<<<1, 1024, 0, K.st>>>(K.ab, nullptr, nullptr, X.buf[c][par], X.flag[c], X.buf[oc][par], X.flag[oc], K.dtok, xi,
                                                   c, K.sa[slot], L.a_pconv, 1.0f / sqrtf(NCONV + 1.0f), K.cur,
                                                   ip % BLK == 0 ? K.B[K.nb++] : nullptr, hk ? hk + (size_t) 2 * MAXT * KVC : nullptr, T, K.F2);
            else if (xll && NC > 1)   // p52
                k_xchg_conv_ll<<<1, 1024, 0, K.st>>>(K.ab, nullptr, nullptr, X.ll[c][par], X.ll[oc][par], K.dtok, xi, c, K.sa[slot], L.a_pconv,
                                                     1.0f / sqrtf(NCONV + 1.0f), K.cur, ip % BLK == 0 ? K.B[K.nb++] : nullptr);
            else
            k_xchg_conv<<<1, 1024, 0, K.st>>>(K.ab, nullptr, nullptr, X.buf[c][par], X.flag[c], X.buf[oc][par], X.flag[oc], K.dtok, xi, c,
                                              K.sa[slot], L.a_pconv, 1.0f / sqrtf(NCONV + 1.0f), K.cur,
                                              ip % BLK == 0 ? K.B[K.nb++] : nullptr);
        } else {
        xch(K, K.ab, T_N);
        if (K.keep_hist)
            CK(cudaMemcpyAsync(K.hist + (size_t) slot * HISTS + (size_t) 2 * MAXT * KVC, K.ab, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, K.st));
        k_conv<<<(N + 127) / 128, 128, 0, K.st>>>(K.ab, N, K.sa[slot], L.a_pconv, nullptr, 1, 1.0f / sqrtf(NCONV + 1.0f), K.A, T);
        if (ip % BLK == 0) {
            copy(K, K.B[K.nb++], K.cur, T);
            copy(K, K.cur, K.A, T);
        } else {
            k_add<<<(T_N + 255) / 256, 256, 0, K.st>>>(K.cur, K.cur, K.A, T_N);
        }
        }
        // MoE
        mix(K, states(K, K.cur), L.ffn_rq, 1, K.F, T);
        const float* f = K.F;
        if (loop) {
            control(K, L, 3, K.F, K.MS + (size_t) 3 * T_N, K.F2, T);
            f = K.F2;
        }
        rms(K, f, L.ffn_norm, K.FN, T, eps);
        const bool f2r = (f2 & 8) && fz;   // p22: the router's rows and the routing in one launch
        if (f2r) k_route2<<<NE, 256, 0, K.st>>>(L.gate_inp, K.FN, K.lg, T, L.probs_b, wscale, K.topi, K.topw, K.jobs);
        else if (stare) gemv32(K, L.gate_inp, NE, N, K.FN, N, K.lg, NE, T);
        else k_gemv32_rows<<<NE, 256, 0, K.st>>>(L.gate_inp, N, K.FN, N, K.lg, NE, T);
        if (f2r) {
        } else if (fz) {
            k_route_jobs<<<T, NE, 0, K.st>>>(K.lg, L.probs_b, wscale, K.topi, K.topw, K.jobs);
        } else {
            k_route<<<T, NE, 0, K.st>>>(K.lg, L.probs_b, wscale, K.topi, K.topw);
            k_jobs<<<1, MAXT * TOPK, 0, K.st>>>(K.topi, K.jobs, T);
        }
        const int nj = T * TOPK;
        if (L.ec.nfill) {   // p32: picked RAM experts into the layer's cache (the tables then point there; p34: or the scratch)
            k_cache_assign<<<1, 32, 0, K.st>>>(L.ec, K.topi, nj, K.dtok, (const uint8_t**) L.gu_t, (const uint8_t**) L.dn_t);
            k_cache_fill<<<dim3(32, nj), 256, 0, K.st>>>(L.ec);
        }
        if (g_rtrace) {   // p31: the writing path's routing, a line a slot: slot T e e e ...
            CK(cudaStreamSynchronize(K.st));
            int ti[MAXT * TOPK];
            CK(cudaMemcpy(ti, K.topi, (size_t) nj * 4, cudaMemcpyDeviceToHost));
            std::fprintf(g_rtrace, "%d %d", slot, T);
            for (int i = 0; i < nj; ++i) std::fprintf(g_rtrace, " %d", ti[i]);
            std::fprintf(g_rtrace, "\n");
        }
        const int* perm = nullptr;
        if (T > 1 && fz && perm_on) {   // p11: a window's jobs in the kernels' block order by expert
            k_job_perm<<<1, MAXT * TOPK, 0, K.st>>>(K.jobs, nj, K.perm);
            perm = K.perm;
        }
        if (L.fp4 && !ilv && T > 1 && !mt_off) {   // p5b: a window: each picked expert once for all its picks
            k_mt_jobs<<<1, 128, 0, K.st>>>(K.topi, nj, K.mjobs, K.nmjobs);
            k_fp4_mt<36, 4, 256><<<dim3(2 * FF_C / 64, nj), 256, 0, K.st>>>(L.gu, (size_t) 2 * FF_C * (N / 64) * 36, 2 * FF_C, K.FN, N, 0, L.gu_s,
                                                                         K.mjobs, K.nmjobs, K.GU, 2 * FF_C);
            k_situ<<<(nj * FF_C + 255) / 256, 256, 0, K.st>>>(K.GU, K.HH, nj, situ_b, situ_lb);
            k_fp4_mt<9, 1, 128><<<dim3(N / 128, nj), 128, 0, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, N, K.HH, FF_C, 1, L.dn_s, K.mjobs,
                                                                  K.nmjobs, K.EY, N);
        } else if (L.fp4 && !ilv) {   // p4d: planar experts, gate_up by rows of 4 lanes, down by 288-block tiles
            if (r7tpb == 64) k_fp4_rows7<36, 4, 64><<<dim3(2 * FF_C / 16, nj), 64, 0, K.st>>>(L.gu, (size_t) 2 * FF_C * (N / 64) * 36, 2 * FF_C, K.FN, N, 0,
                                                                            L.gu_s, K.jobs, K.GU, 2 * FF_C, perm, L.gu_t);
            else if (r7tpb == 128) k_fp4_rows7<36, 4, 128><<<dim3(2 * FF_C / 32, nj), 128, 0, K.st>>>(L.gu, (size_t) 2 * FF_C * (N / 64) * 36, 2 * FF_C, K.FN, N, 0,
                                                                            L.gu_s, K.jobs, K.GU, 2 * FF_C, perm, L.gu_t);
            else k_fp4_rows7<36, 4, 256><<<dim3(2 * FF_C / 64, nj), 256, 0, K.st>>>(L.gu, (size_t) 2 * FF_C * (N / 64) * 36, 2 * FF_C, K.FN, N, 0,
                                                                            L.gu_s, K.jobs, K.GU, 2 * FF_C, perm, L.gu_t);
            k_situ<<<(nj * FF_C + 255) / 256, 256, 0, K.st>>>(K.GU, K.HH, nj, situ_b, situ_lb);
            constexpr int DNB = FF_C / 64, DNR = 288 / DNB;   // p30: down's tile (9 x 32 rows a warp with two cards, 18 x 16 with one)
            if (r6wpb == 1) k_fp4_rows6<DNB, DNR, 1><<<dim3(N / DNR, nj), 32, 0, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, N, K.HH, FF_C, 1, L.dn_s, K.jobs,
                                                                          K.EY, N, perm, L.dn_t);
            else if (r6wpb == 2) k_fp4_rows6<DNB, DNR, 2><<<dim3(N / (2 * DNR), nj), 64, 0, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, N, K.HH, FF_C, 1, L.dn_s, K.jobs,
                                                                          K.EY, N, perm, L.dn_t);
            else k_fp4_rows6<DNB, DNR><<<dim3(N / (4 * DNR), nj), 128, 0, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, N, K.HH, FF_C, 1, L.dn_s, K.jobs,
                                                                          K.EY, N, perm, L.dn_t);
        } else if (L.fp4 && !stare && !rows2) {   // p4b: contiguous tiles of 288 blocks a warp (INK2_FP4_ILV)
            k_fp4_rows3<36, 8><<<dim3(2 * FF_C / 32, nj), 128, 0, K.st>>>(L.gu, (size_t) 2 * FF_C * (N / 64) * 36, 2 * FF_C, K.FN, N, 0, L.gu_s,
                                                                       K.jobs, K.GU, 2 * FF_C);
            k_situ<<<(nj * FF_C + 255) / 256, 256, 0, K.st>>>(K.GU, K.HH, nj, situ_b, situ_lb);
            k_fp4_rows3<9, 32><<<dim3(N / 128, nj), 128, 0, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, N, K.HH, FF_C, 1, L.dn_s, K.jobs, K.EY,
                                                                 N);
        } else if (L.fp4 && !stare) {
            k_fp4_rows2<<<dim3(2 * FF_C / 8, nj), 256, N * 4, K.st>>>(L.gu, (size_t) 2 * FF_C * (N / 64) * 36, 2 * FF_C, N, K.FN, N, 0, L.gu_s,
                                                                    K.jobs, K.GU, 2 * FF_C);
            k_situ<<<(nj * FF_C + 255) / 256, 256, 0, K.st>>>(K.GU, K.HH, nj, situ_b, situ_lb);
            k_fp4_rows2<<<dim3(N / 8, nj), 256, FF_C * 4, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, N, FF_C, K.HH, FF_C, 1, L.dn_s, K.jobs,
                                                               K.EY, N);
        } else if (L.fp4) {
            k_exp_rows<true><<<dim3(2 * FF_C / 8, nj), 256, N * 4, K.st>>>(L.gu, (size_t) 2 * FF_C * (N / 64) * 36, nullptr, 0, 2 * FF_C, N, K.FN, N,
                                                                         0, L.gu_s, K.jobs, K.GU, 2 * FF_C);
            k_situ<<<(nj * FF_C + 255) / 256, 256, 0, K.st>>>(K.GU, K.HH, nj, situ_b, situ_lb);
            k_exp_rows<true><<<dim3(N / 8, nj), 256, FF_C * 4, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, nullptr, 0, N, FF_C, K.HH, FF_C, 1,
                                                                    L.dn_s, K.jobs, K.EY, N);
        } else {
            k_exp_rows<false><<<dim3(2 * FF_C / 8, nj), 256, N * 4, K.st>>>(L.gu, (size_t) 2 * FF_C * N, L.gu_d, (size_t) 2 * FF_C * (N / 32),
                                                                          2 * FF_C, N, K.FN, N, 0, nullptr, K.jobs, K.GU, 2 * FF_C, perm, L.gu_t);
            k_situ<<<(nj * FF_C + 255) / 256, 256, 0, K.st>>>(K.GU, K.HH, nj, situ_b, situ_lb);
            k_exp_rows<false><<<dim3(N / 8, nj), 256, FF_C * 4, K.st>>>(L.dn, (size_t) N * FF_C, L.dn_d, (size_t) N * (FF_C / 32), N, FF_C,
                                                                     K.HH, FF_C, 1, nullptr, K.jobs, K.EY, N, perm, L.dn_t);
        }
        if (fz) {   // p4e: the experts' weighted sum, the exchange, the post-convolution and the residual in one block
            const unsigned xi = K.xidx++;
            const int par = xi & 1;
            if (T > 1) k_moe_sum<<<(T_N + 255) / 256, 256, 0, K.st>>>(K.EY, K.topw, K.MO, T);   // p6e: the experts' sum in parallel
            if (T > 1)
                k_xchg_convT<<<1, 1024, 0, K.st>>>(K.MO, nullptr, nullptr, X.buf[c][par], X.flag[c], X.buf[oc][par], X.flag[oc], K.dtok, xi,
                                                   c, K.sf[slot], L.f_pconv, 1.0f / sqrtf(NCONV + 1.0f), K.cur, nullptr,
                                                   hk ? hk + (size_t) 2 * MAXT * KVC + (size_t) MAXT * N : nullptr, T, K.F2);
            else if (xll && NC > 1)   // p52
                k_xchg_conv_ll<<<1, 1024, 0, K.st>>>(nullptr, K.EY, K.topw, X.ll[c][par], X.ll[oc][par], K.dtok, xi, c, K.sf[slot], L.f_pconv,
                                                     1.0f / sqrtf(NCONV + 1.0f), K.cur, nullptr);
            else
            k_xchg_conv<<<1, 1024, 0, K.st>>>(nullptr, K.EY, K.topw, X.buf[c][par], X.flag[c], X.buf[oc][par], X.flag[oc], K.dtok, xi, c,
                                              K.sf[slot], L.f_pconv, 1.0f / sqrtf(NCONV + 1.0f), K.cur, nullptr);
        } else {
            k_moe_sum<<<(T_N + 255) / 256, 256, 0, K.st>>>(K.EY, K.topw, K.MO, T);
            xch(K, K.MO, T_N);
            if (K.keep_hist)
                CK(cudaMemcpyAsync(K.hist + (size_t) slot * HISTS + (size_t) 2 * MAXT * KVC + (size_t) MAXT * N, K.MO, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, K.st));
            k_conv<<<(N + 127) / 128, 128, 0, K.st>>>(K.MO, N, K.sf[slot], L.f_pconv, nullptr, 1, 1.0f / sqrtf(NCONV + 1.0f), K.F2, T);
            k_add<<<(T_N + 255) / 256, 256, 0, K.st>>>(K.cur, K.cur, K.F2, T_N);
        }
        if ((ds_on || d2_on) && ds_feat_slot(slot) >= 0) ds_feat(K, K.D.feat, K.cur, ds_feat_slot(slot), T);   // p6 (p74)
        if (dumping && c == 0) {
            CK(cudaStreamSynchronize(K.st));
            dump_rows(d_lout[slot], K.cur, T);
            dump_rows(d_aout[slot], K.A, T);
            dump_rows(d_fout[slot], K.F2, T);
        }
    }


    // ------------------------------------------------------------------------------------------------ p6: the DSpark drafter
    const uint16_t* bf16(Card& K, const std::string& name) {
        const GTensor& t = dsg->t(name);
        if (t.type != G_BF16) throw std::runtime_error(name + " is not BF16");
        void* p = alloc(K, t.nbytes);
        CK(cudaMemcpy(p, t.data, t.nbytes, cudaMemcpyHostToDevice));
        return (const uint16_t*) p;
    }
    void ds_load(Card& K) {
        CK(cudaSetDevice(K.dev));
        vt[K.c] = "risar";
        DSW& D = K.D;
        GgufMin* G = dsg;
        for (int il = 0; il < DSL; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            D.wq[il] = q8(K, b + "attn_q.weight", 0, -1, 0, -1, G);
            D.wk[il] = q8(K, b + "attn_k.weight", 0, -1, 0, -1, G);
            D.wv[il] = q8(K, b + "attn_v.weight", 0, -1, 0, -1, G);
            D.wo[il] = q8(K, b + "attn_output.weight", 0, -1, 0, -1, G);
            D.kctx[il] = q8(K, b + "attn_k_ctx.weight", 0, -1, 0, -1, G);
            D.vctx[il] = q8(K, b + "attn_v_ctx.weight", 0, -1, 0, -1, G);
            D.gate[il] = q8(K, b + "ffn_gate.weight", 0, -1, 0, -1, G);
            D.up[il] = q8(K, b + "ffn_up.weight", 0, -1, 0, -1, G);
            D.down[il] = q8(K, b + "ffn_down.weight", 0, -1, 0, -1, G);
            D.attn_norm[il] = f32(K, b + "attn_norm.weight", 0, 0, G);
            D.ffn_norm[il] = f32(K, b + "ffn_norm.weight", 0, 0, G);
            D.dflare[il] = f32(K, b + "dflare_norm.weight", 0, 0, G);
            D.qn[il] = f32(K, b + "attn_q_norm.weight", 0, 0, G);
            D.kn[il] = f32(K, b + "attn_k_norm.weight", 0, 0, G);
            D.kcn[il] = f32(K, b + "attn_k_ctx_norm.weight", 0, 0, G);
            const size_t dr = ds_ring ? (size_t) (il == 1 ? DS_RING1 : DS_RING) : (size_t) ctx;   // p14
            D.kc[il] = (__half*) alloc(K, dr * NKVD * 2);
            D.vc[il] = (__half*) alloc(K, dr * NKVD * 2);
        }
        D.fuse = f32(K, "dspark_fuse.weight", 0, 0, G);
        D.mask = f32(K, "mask_embd.weight", 0, 0, G);
        D.dsnorm = f32(K, "dspark_norm.weight", 0, 0, G);
        D.outnorm = f32(K, "output_norm.weight", 0, 0, G);
        D.jb = f32(K, "joint_proj.bias", 0, 0, G);
        D.hb = f32(K, "dspark_hidden_out.bias", 0, 0, G);
        D.w1 = bf16(K, "markov_w1.weight");
        D.jw = bf16(K, "joint_proj.weight");
        D.hw = bf16(K, "dspark_hidden_out.weight");
        const size_t TN = (size_t) MAXT * N;
        D.feat = dev(K, 4 * TN);
        D.frnn = dev(K, TN);
        D.ctxv = dev(K, N);
        D.cbuf = dev(K, TN);
        D.kb = dev(K, (size_t) MAXT * NKVD);
        D.vb = dev(K, (size_t) MAXT * NKVD);
        D.x = dev(K, (size_t) DBLK * N);
        D.xn = dev(K, (size_t) DBLK * N);
        D.q = dev(K, (size_t) DBLK * NQ);
        D.k = dev(K, (size_t) DBLK * NKVD);
        D.v = dev(K, (size_t) DBLK * NKVD);
        D.ao = dev(K, (size_t) DBLK * NQ);
        D.o = dev(K, (size_t) DBLK * N);
        D.g = dev(K, (size_t) DBLK * DFF);
        D.u = dev(K, (size_t) DBLK * DFF);
        D.h = dev(K, (size_t) DBLK * DFF);
        D.nparts = ((ds_ring ? std::max(DS_WIN1, SWA + 1) : ctx) + DBLK + KC2 - 1) / KC2 + 1;   // p14
        D.part = dev(K, (size_t) DBLK * NH * D.nparts * (HD + 2));
        D.hid = dev(K, (size_t) DBLK * N);
        D.state = dev(K, DRANK);
        D.jin = dev(K, DJ);
        D.jout = dev(K, 3 * DRANK);
        D.lat = dev(K, DRANK);
        D.y = dev(K, N);
        D.amv = dev(K, 1);
        D.ams = dev(K, 1);
        D.ami = (int*) alloc(K, 4);
        D.prev = (int*) alloc(K, 4);
        D.bpos = (int*) alloc(K, DBLK * 4);
        if (pf_min < (1 << 30)) {
            const size_t CN = (size_t) pfc * N;
            D.pfeat = dev(K, 4 * CN);
            D.pfrnn = dev(K, CN);
            D.pcbuf = dev(K, CN);
            D.pkb = dev(K, (size_t) pfc * NKVD);
            D.pvb = dev(K, (size_t) pfc * NKVD);
        }
    }
    // the target's features of T rows (feat [T][4][N]) into the drafter's caches at pos[0..T); f_rnn of every row; ctxv = the last row's
    void ds_ingest(Card& K, float* feat, float* frnn, float* cbuf, float* kb, float* vb, const int* pos, int T, bool big) {
        DSW& D = K.D;
        for (int il = 0; il < DSL; ++il) {
            k_ds_mixfeat<<<T, 256, 0, K.st>>>(feat, D.fuse + il * 4, D.dflare[il], eps, cbuf);
            if (big) {
                PFB& P = K.P;
                deq8(K, D.kctx[il], P.Wf);
                sgemm(K, P.Wf, NKVD, N, cbuf, N, kb, NKVD, T);
                deq8(K, D.vctx[il], P.Wf);
                sgemm(K, P.Wf, NKVD, N, cbuf, N, vb, NKVD, T);
            } else {
                dsgemv(K, D.kctx[il], cbuf, N, kb, NKVD, T);
                dsgemv(K, D.vctx[il], cbuf, N, vb, NKVD, T);
            }
            const int rg = ds_ring ? (il == 1 ? DS_RING1 : DS_RING) : 0;   // p14
            k_ds_headrope<<<dim3(NKV, T), 128, 0, K.st>>>(kb, NKV, D.kcn[il], eps, pos, 0, 1e6f, D.kc[il], NKVD, rg);
            k_ds_vstore<<<(T * NKVD + 255) / 256, 256, 0, K.st>>>(vb, pos, 0, D.vc[il], T, rg);
        }
        k_ds_mixfeat<<<T, 256, 0, K.st>>>(feat, D.fuse + 3 * 4, D.dsnorm, eps, frnn);
        CK(cudaMemcpyAsync(D.ctxv, frnn + (size_t) (T - 1) * N, N * 4, cudaMemcpyDeviceToDevice, K.st));
    }
    void ds_feat(Card& K, float* feat, const float* rows, int j, int T) {   // feature j of T rows into feat [T][4][N]
        CK(cudaMemcpy2DAsync(feat + (size_t) j * N, (size_t) 4 * N * 4, rows, (size_t) N * 4, (size_t) N * 4, T, cudaMemcpyDeviceToDevice, K.st));
    }
    static int ds_feat_slot(int slot) { return slot == 4 ? 0 : slot == 28 ? 1 : slot == 46 ? 2 : -1; }
    // the drafter's block at positions P..P+7 for the anchor (the pending token): the 3 layers, then the final norm into hid (8 rows)
    void ds_block(int c, int anchor_row_ready, int P, int R = DBLK) {   // p10: R rows of the block (DBLK = as trained)
        Card& K = cards[c];
        DSW& D = K.D;
        CK(cudaSetDevice(K.dev));
        int hp[DBLK];
        for (int i = 0; i < DBLK; ++i) hp[i] = P + i;
        CK(cudaMemcpyAsync(D.bpos, hp, R * sizeof(int), cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(K.x0, h_x0, (size_t) N * 4, cudaMemcpyHostToDevice, K.st));   // the anchor's embedding row (embed(0, a))
        k_rms<<<1, 256, 0, K.st>>>(K.x0, N, K.tok_norm, D.x, N, N, eps);
        for (int i = 1; i < R; ++i) CK(cudaMemcpyAsync(D.x + (size_t) i * N, D.mask, N * 4, cudaMemcpyDeviceToDevice, K.st));
        for (int il = 0; il < DSL; ++il) {
            k_rms<<<R, 256, 0, K.st>>>(D.x, N, D.attn_norm[il], D.xn, N, N, eps);
            dsgemv(K, D.wq[il], D.xn, N, D.q, NQ, R);
            dsgemv(K, D.wk[il], D.xn, N, D.k, NKVD, R);
            dsgemv(K, D.wv[il], D.xn, N, D.v, NKVD, R);
            k_ds_headrope<<<dim3(NH, R), 128, 0, K.st>>>(D.q, NH, D.qn[il], eps, D.bpos, 0, 1e6f, nullptr, 0);
            const int rg = ds_ring ? (il == 1 ? DS_RING1 : DS_RING) : 0;   // p14
            k_ds_headrope<<<dim3(NKV, R), 128, 0, K.st>>>(D.k, NKV, D.kn[il], eps, D.bpos, 0, 1e6f, D.kc[il], NKVD, rg);
            k_ds_vstore<<<(R * NKVD + 255) / 256, 256, 0, K.st>>>(D.v, D.bpos, 0, D.vc[il], R, rg);
            const int lo = il == 1 ? (ds_ring ? std::max(0, P - DS_WIN1) : 0) : std::max(0, P - (SWA - 1) - 1), hi = P + R;   // window n_swa + 1 = 1025: keys from P - 1024
            const int np = (hi - lo + KC2 - 1) / KC2;
            k_ds_attn<<<dim3(np, NH, R), 128, 0, K.st>>>(D.q, D.kc[il], D.vc[il], lo, hi, D.part, np, rg);
            k_ds_comb<<<dim3(NH, R), 128, 0, K.st>>>(D.part, np, D.ao);
            dsgemv(K, D.wo[il], D.ao, NQ, D.o, N, R);
            k_add<<<(R * N + 255) / 256, 256, 0, K.st>>>(D.x, D.x, D.o, (size_t) R * N);
            k_rms<<<R, 256, 0, K.st>>>(D.x, N, D.ffn_norm[il], D.xn, N, N, eps);
            dsgemv(K, D.gate[il], D.xn, N, D.g, DFF, R);
            dsgemv(K, D.up[il], D.xn, N, D.u, DFF, R);
            k_ds_swiglu<<<(R * DFF + 255) / 256, 256, 0, K.st>>>(D.g, D.u, D.h, R * DFF);
            dsgemv(K, D.down[il], D.h, DFF, D.o, N, R);
            k_add<<<(R * N + 255) / 256, 256, 0, K.st>>>(D.x, D.x, D.o, (size_t) R * N);
        }
        k_rms<<<R, 256, 0, K.st>>>(D.x, N, D.outnorm, D.hid, N, N, eps);
        CK(cudaMemsetAsync(D.state, 0, DRANK * 4, K.st));
        CK(cudaStreamSynchronize(K.st));
        (void) anchor_row_ready;
    }
    // one Markov step at block position i: the card's half of output . y and its argmax
    float h_amv[NC];
    int h_ami[NC];
    void ds_step(int c, int i, int prev) {
        Card& K = cards[c];
        DSW& D = K.D;
        CK(cudaSetDevice(K.dev));
        CK(cudaMemcpyAsync(D.prev, &h_prev_pinned[c], 4, cudaMemcpyHostToDevice, K.st));
        (void) prev;
        k_ds_joint_in<<<(DJ + 255) / 256, 256, 0, K.st>>>(D.state, D.w1, D.prev, D.hid + (size_t) i * N, D.ctxv, D.jin);
        k_gemv_bf16<<<(3 * DRANK + 7) / 8, 256, 0, K.st>>>(D.jw, 3 * DRANK, DJ, D.jin, D.jb, D.jout);
        k_ds_gates<<<1, DRANK, 0, K.st>>>(D.jout, D.state, D.lat);
        k_gemv_bf16<<<N / 8, 256, 0, K.st>>>(D.hw, N, DRANK, D.lat, D.hb, D.y);
        k_add<<<(N + 255) / 256, 256, 0, K.st>>>(D.y, D.y, D.hid + (size_t) i * N, N);
        const bool dv = !dvmap[c].empty();   // p13
        if (dv) gemv8_t1(K, D.dvout, D.y, segs({{D.dvout.rows, K.logits}}));
        else gemv8_t1(K, K.out, D.y, segs({{NV_C, K.logits}}));
        const int valid = dv ? D.dvout.rows : std::min(NV_C, NVU - c * NV_C);
        k_argmax_n<<<1, 1024, 0, K.st>>>(K.logits, valid, D.amv, D.ami, ds_pmin > 0.0f ? D.ams : nullptr, softcap);
        CK(cudaMemcpyAsync(&h_amv_pinned[c], D.amv, 4, cudaMemcpyDeviceToHost, K.st));
        if (ds_pmin > 0.0f) CK(cudaMemcpyAsync(&h_ams_pinned[c], D.ams, 4, cudaMemcpyDeviceToHost, K.st));
        CK(cudaMemcpyAsync(&h_ami_pinned[c], D.ami, 4, cudaMemcpyDeviceToHost, K.st));
        CK(cudaStreamSynchronize(K.st));
    }
    float* h_amv_pinned = nullptr;
    float* h_ams_pinned = nullptr;   // p12
    std::vector<int> dvmap[NC];      // p13: the draft vocabulary's ids of each card's head rows (empty = the whole vocabulary)
    // p12: drafts stop at the first one whose probability is below this (FNX_DRAFT_MIN_P in Flash-Next); 0 = off
    float ds_pmin = getenv("INK2_DS_PMIN") ? (float) atof(getenv("INK2_DS_PMIN")) : 0.7f;   // 21. 11.: 0.7 best of 0 / 0.3 / 0.5 / 0.7
    int* h_ami_pinned = nullptr;
    int* h_prev_pinned = nullptr;
    // up to k drafts after the anchor (the pending token at position P = n_past)
    std::vector<int> ds_draft(int anchor, int k) {
        std::vector<int> d;
        if (!ds_on || k <= 0 || n_past + DBLK > ctx) return d;
        k = std::min(k, DBLK - 1);
        (void) embed(0, anchor);
        const int P = n_past;
        static const int rows_env = getenv("INK2_DS_ROWS") ? atoi(getenv("INK2_DS_ROWS")) : 1;   // p10: k + 1 rows (21. 11.: drafter 4.9 -> 3.2 ms a round, same acceptance); 0 = the block of DBLK
        const int R = rows_env > 0 ? std::min(DBLK, std::max(k + 1, rows_env)) : DBLK;
        both([&](int c) { ds_block(c, 0, P, R); });
        int prev = anchor;
        for (int i = 1; i <= k; ++i) {
            for (int c = 0; c < NC; ++c) h_prev_pinned[c] = prev;
            both([&](int c) { ds_step(c, i, prev); });
            const int t0 = dvmap[0].empty() ? h_ami_pinned[0] : dvmap[0][h_ami_pinned[0]];   // p13: card 0's ids are the smaller
            const int c1 = NC - 1;   // p30: one card is also the last card
            const int t1 = dvmap[c1].empty() ? h_ami_pinned[c1] + c1 * NV_C : dvmap[c1][h_ami_pinned[c1]];
            prev = h_amv_pinned[0] >= h_amv_pinned[c1] ? t0 : t1;   // the first maximum: card 0's tokens come first
            if (ds_pmin > 0.0f) {   // p12: p(top) = 1 / sum over both cards (each card's sum is relative to its own max)
                const float m0 = softcap * tanhf(h_amv_pinned[0] / softcap), m1 = softcap * tanhf(h_amv_pinned[c1] / softcap);
                const float m = std::max(m0, m1);
                const float tot = h_ams_pinned[0] * expf(m0 - m) + (NC > 1 ? h_ams_pinned[c1] * expf(m1 - m) : 0.0f);
                if (1.0f / tot < ds_pmin) break;
            }
            d.push_back(prev);
        }
        return d;
    }

    // ------------------------------------------------------------------------------------------------ p74: the DFlash 2 drafter
    void d2_init() {   // the file's geometry, both cards, the mask token's row, the host's buffers
        GgufMin* G = dsg;
        auto need = [](bool ok, const char* what) {
            if (!ok) throw std::runtime_error(std::string("DFlash 2: ") + what);
        };
        need(G->s("general.architecture") == "dflash" && G->i("dflash.block_count") == D2L && G->i("dflash.embedding_length") == N &&
                 G->i("dflash.feed_forward_length") == D2FF && G->i("dflash.attention.head_count") == D2H &&
                 G->i("dflash.attention.head_count_kv") == NKV && G->i("dflash.attention.key_length") == HD &&
                 G->i("dflash.attention.sliding_window") == D2WIN && G->i("dflash.block_size") == DBLK &&
                 G->i("dflash.selector_rank") == D2R && G->i("dflash.selector_top_k") == D2K && G->i("dflash.conv_kernel_size") == 2 &&
                 G->i("dflash.conv_group_size") == D2G,
             "the geometry differs from ink2_dflash2.cuh");
        need(G->s("dflash.selector_softcap_mode") == "after_correction", "selector_softcap_mode is not after_correction");
        need(G->ia("dflash.target_layers") == std::vector<int64_t>{5, 23, 41, 48}, "taps other than 5, 23, 41, 48");
        d2_mask = (int) G->i("dflash.mask_token_id");
        d2_cap = (float) G->f("dflash.final_logit_softcapping");
        d2_eps = (float) G->f("dflash.attention.layer_norm_rms_epsilon");
        d2_theta = (float) G->f("dflash.rope.freq_base");
        const GTensor &a = G->t("selector_predecessor.weight"), &b = G->t("selector_successor.weight");
        need(a.type == G_BF16 && b.type == G_BF16 && a.ne[0] == D2R && b.ne[0] == D2R && a.ne[1] >= NVU && b.ne[1] >= NVU,
             "the codebooks are not BF16 [vocabulary][256]");
        d2_A = (const uint16_t*) a.data;
        d2_B = (const uint16_t*) b.data;
        if (!g_draft_vocab.empty()) std::fprintf(stderr, "ink2_v1: --draft-vocab je samo za DSpark (DFlash 2 bere vse besedisce)\n");
        each_card([&](int c) { d2_load(cards[c]); });
        (void) embed(0, d2_mask);   // rms(tok_embd[mask]) * tok_norm, as the anchor's row (fix 3)
        each_card([&](int c) {
            Card& K = cards[c];
            CK(cudaSetDevice(K.dev));
            CK(cudaMemcpy(K.D2.mrow, h_x0, (size_t) N * 4, cudaMemcpyHostToDevice));
            k_rms<<<1, 256, 0, K.st>>>(K.D2.mrow, N, K.tok_norm, K.D2.mrow, N, N, eps);
            CK(cudaStreamSynchronize(K.st));
        });
        CK(cudaHostAlloc((void**) &h_d2v, (size_t) NC * DBLK * D2K * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &h_d2i, (size_t) NC * DBLK * D2K * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &h_d2g, (size_t) DBLK * D2R * 4, cudaHostAllocPortable));
        d2_on = true;
        std::fprintf(stderr, "ink2_v1: risar DFlash 2, %d vrstic bloka, prag q %.2f\n", d2_rows, d2_qmin);
    }
    void d2_load(Card& K) {
        CK(cudaSetDevice(K.dev));
        vt[K.c] = "risar";
        D2W& D = K.D2;
        GgufMin* G = dsg;
        if (pf_min < (1 << 30) && K.L[5].cat.rows < N) throw std::runtime_error("DFlash 2: P.Wf holds no 2304 x 2304 block of fc");
        D.fc = q8(K, "fc.weight", 0, -1, 0, -1, G);
        D.selh = q8(K, "selector_hidden.weight", 0, -1, 0, -1, G);
        D.hnorm = f32(K, "enc.output_norm.weight", 0, 0, G);
        D.onorm = f32(K, "output_norm.weight", 0, 0, G);
        for (int il = 0; il < D2L; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            D.wq[il] = q8(K, b + "attn_q.weight", 0, -1, 0, -1, G);
            D.wk[il] = q8(K, b + "attn_k.weight", 0, -1, 0, -1, G);
            D.wv[il] = q8(K, b + "attn_v.weight", 0, -1, 0, -1, G);
            D.wo[il] = q8(K, b + "attn_output.weight", 0, -1, 0, -1, G);
            D.gate[il] = q8(K, b + "ffn_gate.weight", 0, -1, 0, -1, G);
            D.up[il] = q8(K, b + "ffn_up.weight", 0, -1, 0, -1, G);
            D.down[il] = q8(K, b + "ffn_down.weight", 0, -1, 0, -1, G);
            D.aproj[il] = q8(K, b + "attn_conv_proj.weight", 0, -1, 0, -1, G);
            D.mproj[il] = q8(K, b + "ffn_conv_proj.weight", 0, -1, 0, -1, G);
            D.anorm[il] = f32(K, b + "attn_norm.weight", 0, 0, G);
            D.fnorm[il] = f32(K, b + "ffn_norm.weight", 0, 0, G);
            D.qn[il] = f32(K, b + "attn_q_norm.weight", 0, 0, G);
            D.kn[il] = f32(K, b + "attn_k_norm.weight", 0, 0, G);
            D.abase[il] = f32(K, b + "attn_conv_base", 0, 0, G);   // [2 sides][2 taps][N]
            D.mbase[il] = f32(K, b + "ffn_conv_base", 0, 0, G);
            D.kc[il] = (__half*) alloc(K, (size_t) D2RING * NKVD * 2);
            D.vc[il] = (__half*) alloc(K, (size_t) D2RING * NKVD * 2);
        }
        const size_t RM = (size_t) std::max(MAXT, std::min(D2PIECE, pfc));
        D.cx = dev(K, RM * N);
        D.xin = dev(K, RM * N);
        D.kb = dev(K, RM * NKVD);
        D.vb = dev(K, RM * NKVD);
        D.mrow = dev(K, N);
        D.x = dev(K, (size_t) DBLK * N);
        D.xn = dev(K, (size_t) DBLK * N);
        D.c = dev(K, (size_t) DBLK * N);
        D.o = dev(K, (size_t) DBLK * N);
        D.t1 = dev(K, (size_t) DBLK * N);
        D.hid = dev(K, (size_t) DBLK * N);
        D.dyn = dev(K, (size_t) DBLK * D2DYN);
        D.q = dev(K, (size_t) DBLK * D2Q);
        D.ao = dev(K, (size_t) DBLK * D2Q);
        D.k = dev(K, (size_t) DBLK * NKVD);
        D.v = dev(K, (size_t) DBLK * NKVD);
        D.g = dev(K, (size_t) DBLK * D2FF);
        D.u = dev(K, (size_t) DBLK * D2FF);
        D.h = dev(K, (size_t) DBLK * D2FF);
        D.nparts = (D2WIN + DBLK + D2KC - 1) / D2KC + 1;
        D.part = dev(K, (size_t) DBLK * D2H * D.nparts * (HD + 2));
        D.gout = dev(K, (size_t) DBLK * D2R);
        D.tv = dev(K, (size_t) DBLK * D2K);
        D.ti = (int*) alloc(K, (size_t) DBLK * D2K * 4);
        D.tv1 = dev(K, (size_t) DBLK * D2TS * D2K);
        D.ti1 = (int*) alloc(K, (size_t) DBLK * D2TS * D2K * 4);
        D.partn = (size_t) 1 << 20;   // floats: k_d2_gemv3's chunks (at most ~0.6 M: q, 18 chunks x 8 rows x 4096)
        D.part2 = dev(K, D.partn);
        D.bpos = (int*) alloc(K, DBLK * 4);
        K.D.feat = dev(K, (size_t) 4 * MAXT * N);   // the features: layer() and layer_pf() fill DSW's buffers
        if (pf_min < (1 << 30)) K.D.pfeat = dev(K, (size_t) 4 * pfc * N);
    }
    // the drafter's GEMV of T rows, 8 at a time, by the fastest kernel for the shape (test/d2_hitrost.cu, 27. 11.): k_d2_gemv for the
    // head and fc (many rows or 9216 columns), else k_d2_gemv2 (256 rows a block; few rows also split the columns into chunks,
    // ~224 blocks in all, added by k_d2_sum)
    template <int TT> void d2gemv_t(Card& K, const Mat& m, const float* x, int ldx, float* y, int ldy) {
        if (m.rows >= 32768 || m.cols >= 9216) {
            k_d2_gemv<TT, 4><<<(m.rows + 31) / 32, 256, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, ldx, y, ldy);
            return;
        }
        const int rb = (m.rows + 255) / 256;
        int nch = std::max(1, std::min((224 + rb - 1) / rb, (m.cols + D2PC - 1) / D2PC));
        int cch = ((m.cols + nch - 1) / nch + 31) / 32 * 32;
        nch = (m.cols + cch - 1) / cch;
        if (nch > 1 && (size_t) nch * TT * m.rows > K.D2.partn) {   // fewer chunks, as many as the buffer holds
            nch = std::max(1, (int) (K.D2.partn / ((size_t) TT * m.rows)));
            cch = ((m.cols + nch - 1) / nch + 31) / 32 * 32;
            nch = (m.cols + cch - 1) / cch;
        }
        k_d2_gemv2<TT><<<dim3(rb, nch), 256, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, ldx, cch, y, ldy, K.D2.part2);
        if (nch > 1) k_d2_sum<<<(m.rows * TT + 255) / 256, 256, 0, K.st>>>(K.D2.part2, nch, m.rows, TT, y, ldy);
    }
    void d2gemv(Card& K, const Mat& m, const float* x, int ldx, float* y, int ldy, int T) {
        for (int t0 = 0; t0 < T; t0 += 8) {
            const float* xt = x + (size_t) t0 * ldx;
            float* yt = y + (size_t) t0 * ldy;
            switch (std::min(8, T - t0)) {
                case 1: d2gemv_t<1>(K, m, xt, ldx, yt, ldy); break;
                case 2: d2gemv_t<2>(K, m, xt, ldx, yt, ldy); break;
                case 3: d2gemv_t<3>(K, m, xt, ldx, yt, ldy); break;
                case 4: d2gemv_t<4>(K, m, xt, ldx, yt, ldy); break;
                case 5: d2gemv_t<5>(K, m, xt, ldx, yt, ldy); break;
                case 6: d2gemv_t<6>(K, m, xt, ldx, yt, ldy); break;
                case 7: d2gemv_t<7>(K, m, xt, ldx, yt, ldy); break;
                default: d2gemv_t<8>(K, m, xt, ldx, yt, ldy); break;
            }
        }
    }
    void sgemm_b(Card& K, const float* W, int rows, int cols, const float* x, int ldx, float* y, int ldy, int C, float beta) {
        const float one = 1.0f;
        cublasSgemm(K.P.cb, CUBLAS_OP_T, CUBLAS_OP_N, rows, C, cols, &one, W, cols, x, ldx, &beta, y, ldy);
    }
    // the context of T rows (feat [T][4][N], device positions pos): x = rms(fc . feat) * hidden_norm, every layer's K / V into its
    // ring. big: the reading path (cuBLAS on f32 copies in P.Wf; fc by its four 2304-column blocks), else the window GEMV
    void d2_ingest(Card& K, const float* feat, const int* pos, int T, bool big) {
        D2W& D = K.D2;
        PFB& P = K.P;
        for (int r0 = 0; r0 < T; r0 += D2PIECE) {
            const int n = std::min(D2PIECE, T - r0);
            const float* f = feat + (size_t) r0 * 4 * N;
            const int* ps = pos + r0;
            if (big)
                for (int j = 0; j < 4; ++j) {
                    k_q8s_f32_cols<<<(unsigned) (((size_t) N * N + 255) / 256), 256, 0, K.st>>>(D.fc.q, (const __half*) D.fc.d, N, 4 * N, j * N, N, P.Wf);
                    sgemm_b(K, P.Wf, N, N, f + (size_t) j * N, 4 * N, D.cx, N, n, j ? 1.0f : 0.0f);
                }
            else
                d2gemv(K, D.fc, f, 4 * N, D.cx, N, n);
            k_rms<<<n, 256, 0, K.st>>>(D.cx, N, D.hnorm, D.xin, N, N, d2_eps);
            for (int il = 0; il < D2L; ++il) {
                if (big) {
                    deq8(K, D.wk[il], P.Wf);
                    sgemm(K, P.Wf, NKVD, N, D.xin, N, D.kb, NKVD, n);
                    deq8(K, D.wv[il], P.Wf);
                    sgemm(K, P.Wf, NKVD, N, D.xin, N, D.vb, NKVD, n);
                } else {
                    d2gemv(K, D.wk[il], D.xin, N, D.kb, NKVD, n);
                    d2gemv(K, D.wv[il], D.xin, N, D.vb, NKVD, n);
                }
                k_ds_headrope<<<dim3(NKV, n), 128, 0, K.st>>>(D.kb, NKV, D.kn[il], d2_eps, ps, 0, d2_theta, D.kc[il], NKVD, D2RING);
                k_ds_vstore<<<(n * NKVD + 255) / 256, 256, 0, K.st>>>(D.vb, ps, 0, D.vc[il], n, D2RING);
            }
        }
    }
    // card c: the block of R rows for the anchor (the pending token at P, its embedding row in h_x0[0]), the head's top-16 of
    // rows 1..R-1 on the card's half of the vocabulary, card 0 also the selector's gate; one sync
    void d2_block(int c, int P, int R) {
        Card& K = cards[c];
        D2W& D = K.D2;
        CK(cudaSetDevice(K.dev));
        int hp[DBLK];
        for (int i = 0; i < DBLK; ++i) hp[i] = P + i;
        CK(cudaMemcpyAsync(D.bpos, hp, R * sizeof(int), cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(K.x0, h_x0, (size_t) N * 4, cudaMemcpyHostToDevice, K.st));
        k_rms<<<1, 256, 0, K.st>>>(K.x0, N, K.tok_norm, D.x, N, N, eps);
        for (int i = 1; i < R; ++i) CK(cudaMemcpyAsync(D.x + (size_t) i * N, D.mrow, N * 4, cudaMemcpyDeviceToDevice, K.st));
        const int lo = P - (D2WIN - 1), hi = P + R, np = (hi - std::max(0, lo) + D2KC - 1) / D2KC;
        const dim3 cg((N + 255) / 256, R);
        const unsigned nr = (unsigned) ((R * N + 255) / 256);
        for (int il = 0; il < D2L; ++il) {
            k_rms<<<R, 256, 0, K.st>>>(D.x, N, D.anorm[il], D.xn, N, N, d2_eps);
            d2gemv(K, D.aproj[il], D.xn, N, D.dyn, D2DYN, R);
            k_d2_conv<<<cg, 256, 0, K.st>>>(D.xn, D.dyn, D.abase[il], 0, D.c);
            d2gemv(K, D.wq[il], D.c, N, D.q, D2Q, R);
            d2gemv(K, D.wk[il], D.c, N, D.k, NKVD, R);
            d2gemv(K, D.wv[il], D.c, N, D.v, NKVD, R);
            k_ds_headrope<<<dim3(D2H, R), 128, 0, K.st>>>(D.q, D2H, D.qn[il], d2_eps, D.bpos, 0, d2_theta, nullptr, 0);
            k_ds_headrope<<<dim3(NKV, R), 128, 0, K.st>>>(D.k, NKV, D.kn[il], d2_eps, D.bpos, 0, d2_theta, D.kc[il], NKVD, D2RING);
            k_ds_vstore<<<(R * NKVD + 255) / 256, 256, 0, K.st>>>(D.v, D.bpos, 0, D.vc[il], R, D2RING);
            k_d2_attn<<<dim3(np, NKV), 256, 0, K.st>>>(D.q, D.kc[il], D.vc[il], lo, hi, R, D.part, np);
            k_d2_comb<<<dim3(D2H, R), 128, 0, K.st>>>(D.part, np, D.ao);
            d2gemv(K, D.wo[il], D.ao, D2Q, D.o, N, R);
            k_d2_conv<<<cg, 256, 0, K.st>>>(D.o, D.dyn, D.abase[il] + 2 * N, 1, D.t1);
            k_add<<<nr, 256, 0, K.st>>>(D.x, D.x, D.t1, (size_t) R * N);
            k_rms<<<R, 256, 0, K.st>>>(D.x, N, D.fnorm[il], D.xn, N, N, d2_eps);
            d2gemv(K, D.mproj[il], D.xn, N, D.dyn, D2DYN, R);
            k_d2_conv<<<cg, 256, 0, K.st>>>(D.xn, D.dyn, D.mbase[il], 0, D.c);
            d2gemv(K, D.gate[il], D.c, N, D.g, D2FF, R);
            d2gemv(K, D.up[il], D.c, N, D.u, D2FF, R);
            k_ds_swiglu<<<(R * D2FF + 255) / 256, 256, 0, K.st>>>(D.g, D.u, D.h, R * D2FF);
            d2gemv(K, D.down[il], D.h, D2FF, D.o, N, R);
            k_d2_conv<<<cg, 256, 0, K.st>>>(D.o, D.dyn, D.mbase[il] + 2 * N, 1, D.t1);
            k_add<<<nr, 256, 0, K.st>>>(D.x, D.x, D.t1, (size_t) R * N);
        }
        k_rms<<<R - 1, 256, 0, K.st>>>(D.x + N, N, D.onorm, D.hid, N, N, d2_eps);
        d2gemv(K, K.out, D.hid, N, K.logits, NV_C, R - 1);   // raw logits (the cap comes after the selector's correction)
        const int valid = std::min(NV_C, NVU - c * NV_C), span = (valid + D2TS - 1) / D2TS;   // the top-16 in two passes
        k_d2_top16<<<dim3(R - 1, D2TS), 128, 0, K.st>>>(K.logits, nullptr, NV_C, valid, span, c * NV_C, D.tv1, D.ti1);
        k_d2_top16<<<dim3(R - 1, 1), 128, 0, K.st>>>(D.tv1, D.ti1, D2TS * D2K, D2TS * D2K, D2TS * D2K, 0, D.tv, D.ti);
        CK(cudaMemcpyAsync(h_d2v + (size_t) c * DBLK * D2K, D.tv, (size_t) (R - 1) * D2K * 4, cudaMemcpyDeviceToHost, K.st));
        CK(cudaMemcpyAsync(h_d2i + (size_t) c * DBLK * D2K, D.ti, (size_t) (R - 1) * D2K * 4, cudaMemcpyDeviceToHost, K.st));
        if (c == 0) {
            d2gemv(K, D.selh, D.hid, N, D.gout, D2R, R - 1);
            CK(cudaMemcpyAsync(h_d2g, D.gout, (size_t) (R - 1) * D2R * 4, cudaMemcpyDeviceToHost, K.st));
        }
        CK(cudaStreamSynchronize(K.st));
    }
    static float bf16f(uint16_t b) {
        const uint32_t u = (uint32_t) b << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
    // up to k drafts after the anchor (the pending token at position P = n_past): the block, then the greedy walk on the host. Draft
    // e: the cards' top-16 of row e + 1 merged (values down, card 0's smaller ids first on a tie), score[c] = cap tanh((u[c] +
    // <A[prev] * g_e, B[cand c]>) / cap) for the walked predecessor prev (the anchor, then the last draft) only; stops below d2_qmin
    std::vector<int> d2_draft(int anchor, int k) {
        std::vector<int> d;
        if (!d2_on || k <= 0 || n_past + DBLK > ctx) return d;
        k = std::min(k, DBLK - 1);
        (void) embed(0, anchor);
        const int P = n_past, R = std::max(k + 1, d2_rows);
        both([&](int c) { d2_block(c, P, R); });
        int prev = anchor;
        for (int e = 0; e < k; ++e) {
            float u[D2K];
            int id[D2K];
            const float *v0 = h_d2v + e * D2K, *v1 = h_d2v + (size_t) (NC - 1) * DBLK * D2K + e * D2K;
            const int *i0 = h_d2i + e * D2K, *i1 = h_d2i + (size_t) (NC - 1) * DBLK * D2K + e * D2K;
            for (int n = 0, a = 0, b = 0; n < D2K; ++n) {
                const bool first = NC == 1 || b >= D2K || (a < D2K && v0[a] >= v1[b]);
                u[n] = first ? v0[a] : v1[b];
                id[n] = first ? i0[a++] : i1[b++];
            }
            float ag[D2R];
            const uint16_t* ar = d2_A + (size_t) prev * D2R;
            const float* gv = h_d2g + (size_t) e * D2R;
            for (int r = 0; r < D2R; ++r) ag[r] = bf16f(ar[r]) * gv[r];
            float s[D2K];
            int best = -1;
            for (int c = 0; c < D2K; ++c) {
                s[c] = -INFINITY;
                if (id[c] < 0 || id[c] >= NVU) continue;
                const uint16_t* br = d2_B + (size_t) id[c] * D2R;
                float a = 0.0f;
                for (int r = 0; r < D2R; ++r) a += ag[r] * bf16f(br[r]);
                s[c] = d2_cap * tanhf((u[c] + a) / d2_cap);
                if (best < 0 || s[c] > s[best]) best = c;
            }
            if (best < 0) break;
            float tot = 0.0f;
            for (int c = 0; c < D2K; ++c) tot += s[c] == -INFINITY ? 0.0f : expf(s[c] - s[best]);
            if (1.0f / tot < d2_qmin) break;
            prev = id[best];
            d.push_back(prev);
        }
        return d;
    }

    // ------------------------------------------------------------------------------------------------ p3: the reading path
    // p41: a card's NUMA node (-1 unknown) from its PCI address; NodeBind keeps the calling thread on a node's CPUs while it
    // lives (page-locked pages land on the node of the thread that faults them in) and does nothing for -1 or a node it cannot read
    static int card_node(int dev) {
        char bus[32] = {};
        if (cudaDeviceGetPCIBusId(bus, sizeof bus, dev) != cudaSuccess) return -1;
        for (char* q = bus; *q; ++q) *q = (char) std::tolower((unsigned char) *q);
        FILE* f = std::fopen((std::string("/sys/bus/pci/devices/") + bus + "/numa_node").c_str(), "r");
        if (!f) return -1;
        int n = -1;
        if (std::fscanf(f, "%d", &n) != 1) n = -1;
        std::fclose(f);
        return n;
    }
    struct NodeBind {
        cpu_set_t old;
        bool on = false;
        explicit NodeBind(int node) {
            if (node < 0) return;
            FILE* f = std::fopen(("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist").c_str(), "r");
            if (!f) return;
            char line[512] = {};
            const bool got = std::fgets(line, sizeof line, f) != nullptr;
            std::fclose(f);
            if (!got) return;
            cpu_set_t set;
            CPU_ZERO(&set);
            int ncpu = 0;
            for (char* tok = std::strtok(line, ",\n"); tok; tok = std::strtok(nullptr, ",\n")) {
                int a = 0, b = 0;
                const int k = std::sscanf(tok, "%d-%d", &a, &b);
                if (k < 1) continue;
                if (k == 1) b = a;
                for (int c = a; c <= b && c < CPU_SETSIZE; ++c, ++ncpu) CPU_SET(c, &set);
            }
            if (!ncpu || pthread_getaffinity_np(pthread_self(), sizeof old, &old)) return;
            on = pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
        }
        ~NodeBind() {
            if (on) pthread_setaffinity_np(pthread_self(), sizeof old, &old);
        }
    };
    static int page_node(const void* ptr) {   // p41: the node a page is on (-1 unknown)
        void* pg = (void*) ptr;
        int st = -1;
        if (syscall(SYS_move_pages, 0, 1UL, &pg, nullptr, &st, 0) != 0) return -1;
        return st;
    }
    void alloc_pf(Card& K) {
        CK(cudaSetDevice(K.dev));
        vt[K.c] = "branje";
        K.pf_rec = share1c;   // p63
        PFB& P = K.P;
        const size_t C = pfc, CN = C * N;
        P.C = pfc;
        if (cublasCreate(&P.cb) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate");
        cublasSetStream(P.cb, K.st);
        P.x0 = dev(K, CN);
        P.cur = dev(K, CN);
        for (int i = 0; i < MAXB; ++i) P.B[i] = dev(K, CN);
        P.prel = dev(K, CN);
        P.bankS = dev(K, CN);
        P.bankO = dev(K, CN);
        P.X3 = dev(K, 4 * CN);
        P.MS = dev(K, 4 * CN);
        for (int i = 0; i < 4; ++i) P.XC[i] = dev(K, CN);
        P.xm = dev(K, 2 * CN);
        P.g64 = dev(K, C * RANK);
        P.gp = dev(K, CN);
        P.fpp = dev(K, CN);
        P.NQ = dev(K, CN);
        P.NK = dev(K, CN);
        P.NV = dev(K, CN);
        P.q = dev(K, C * QC);
        P.k = dev(K, C * KVC);
        P.v = dev(K, C * KVC);
        P.kcv = dev(K, C * KVC);
        P.vcv = dev(K, C * KVC);
        P.r = dev(K, C * NH_C * NREL);
        P.sink = dev(K, C * NH_C);
        P.gate = dev(K, C * QC);
        P.dk = dev(K, C * NKV_C * NCONV);
        P.dv = dev(K, C * NKV_C * NCONV);
        P.ao = dev(K, C * QC);
        P.ab = dev(K, CN);
        P.A = dev(K, CN);
        P.F = dev(K, CN);
        P.F2 = dev(K, CN);
        P.FN = dev(K, CN);
        P.lg = dev(K, C * NE);
        P.topw = dev(K, C * TOPK);
        P.MO = dev(K, CN);
        P.tau = dev(K, C);
        P.tmp = dev(K, CN);
        P.hx1 = dev(K, N);
        P.Xg = dev(K, CN * TOPK);
        P.GU = dev(K, C * TOPK * 2 * FF_C);
        P.HH = dev(K, C * TOPK * FF_C);
        P.EY = dev(K, NC == 1 ? std::max(CN * TOPK, C * NH_C * (EXT + 1)) : CN * TOPK);   // p33: R.B fits with one card
        P.Wf = dev(K, (size_t) K.L[5].cat.rows * N);   // a full layer's concatenated matrix is the largest
        P.Wo = dev(K, (size_t) N * QC);
        if (pf_deq || !q8g || catm) {   // p30: only the paths that use them (510 MB with one card)
            P.Egu = dev(K, (size_t) EB * 2 * FF_C * N);
            P.Edn = dev(K, (size_t) EB * N * FF_C);
        }
        const size_t kfr = (size_t) std::max(std::min(ctx, KFB), (int) C + SWA);   // p3f: a window chunk reads from p0 - (SWA - 1); p14: KFB
        // p18 (int8 mode): the attention's scratch (K / V rows in f32, R.B, the P.V pieces) lives in the MoE's (Xg, EY, GU): a
        // layer's attention and its experts run one after the other on the card's one stream, and every one of these is written
        // before it is read in its phase. ~274 MiB a card at 512k for the f16 rows below kv8_f16
        const bool share = (kv8 || share_on) && 2 * kfr * KVC <= CN * TOPK && (size_t) C * NH_C * (EXT + 1) <= CN * TOPK &&   // p19c: every mode
                           (size_t) 3 * ((ctx + PVK - 1) / PVK) * pf_qs_full * HD <= C * TOPK * 2 * FF_C;
        // p33: one card shares part by part
        const bool sh_kv = share || (NC == 1 && share_on && 2 * kfr * KVC <= CN * TOPK);
        const bool sh_rb = share || (NC == 1 && share_on);
        const bool sh_pv = share || (NC == 1 && share_on && (size_t) 3 * ((ctx + PVK - 1) / PVK) * pf_qs_full * HD <= C * TOPK * 2 * FF_C);
        P.Kf = sh_kv ? P.Xg : dev(K, kfr * KVC);
        P.Vf = sh_kv ? P.Xg + kfr * KVC : dev(K, kfr * KVC);
        const size_t s_full = (size_t) 3 * pf_qs_full * (ctx + pf_qs_full), s_win = (size_t) (C / QS) * NH_C * QS * (QS + SWA - 1),
                     s_tail = (size_t) 3 * QS * (QS + SWA);
        P.S_n = std::max(s_full, std::max(s_win, s_tail));
        if ((kv8 || share_on) && P.S_n <= K.part_n) {   // p18: K.part is the writing path's scratch, S the reading path's; one stream a card (p19c: every mode)
            P.S = K.part;
            P.S_n = K.part_n;
        } else {
            P.S = dev(K, P.S_n);
        }
        P.pv_nsb = (ctx + PVK - 1) / PVK;   // p14b
        P.PV = sh_pv ? P.GU : dev(K, (size_t) 3 * P.pv_nsb * pf_qs_full * HD);
        pv_cap = sh_pv ? (size_t) C * TOPK * 2 * FF_C : (size_t) 3 * P.pv_nsb * pf_qs_full * HD;   // p19c
        P.RB = sh_rb ? P.EY : dev(K, C * NH_C * (EXT + 1));
        P.shared = share;
        {   // p3f: query block b, head h = g * 3 + a: entry b * NH_C + h; S of a window block QS x (QS + SWA - 1)
            const int nb = (int) C / QS, nk = QS + SWA - 1, n = nb * NH_C;
            if ((size_t) n * QS * nk > P.S_n) throw std::runtime_error("p3f: S premajhen");
            std::vector<float*> a(n), b2(n), c(n), d(n), e(n);
            for (int bb = 0; bb < nb; ++bb)
                for (int h = 0; h < NH_C; ++h) {
                    const int i = bb * NH_C + h, g = h / (NH / NKV);
                    a[i] = P.Kf + (size_t) bb * QS * KVC + g * HD;
                    b2[i] = P.q + (size_t) bb * QS * QC + h * HD;
                    c[i] = P.S + (size_t) i * QS * nk;
                    d[i] = P.Vf + (size_t) bb * QS * KVC + g * HD;
                    e[i] = P.ao + (size_t) bb * QS * QC + h * HD;
                }
            float*** dst[5] = {&P.pk, &P.pq, &P.ps, &P.pv, &P.pao};
            std::vector<float*>* srcv[5] = {&a, &b2, &c, &d, &e};
            for (int k = 0; k < 5; ++k) {
                *dst[k] = (float**) alloc(K, (size_t) std::max(n, 1) * sizeof(float*));
                CK(cudaMemcpy(*dst[k], srcv[k]->data(), (size_t) n * sizeof(float*), cudaMemcpyHostToDevice));
            }
        }
        P.pos = (int*) alloc(K, C * 4);
        P.topi = (int*) alloc(K, C * TOPK * 4);
        P.cnt = (int*) alloc(K, NE * 4);
        P.offs = (int*) alloc(K, (NE + 1) * 4);
        P.h_tiles.resize((size_t) C * TOPK / 16 + 3 * NE);   // p19b: every tile holds >= 16 rows but an expert's last one (3 sizes)
        P.tiles = (FTile*) alloc(K, P.h_tiles.size() * sizeof(FTile));
        P.cursor = (int*) alloc(K, NE * 4);
        P.slotrow = (int*) alloc(K, C * TOPK * 4);
        P.rowtok = (int*) alloc(K, C * TOPK * 4);
        P.rowexp = (int*) alloc(K, C * TOPK * 4);
        {
            const int numa = getenv("INK2_NUMA") ? atoi(getenv("INK2_NUMA")) : 1;   // p41: 1 the writer's node, 2 the reader's, 0 off
            const int want = numa == 0 ? -1 : card_node(numa == 2 && NC == 2 ? cards[NC - 1 - K.c].dev : K.dev);
            NodeBind nb(want);
            for (int i = 0; i < 2; ++i) {
                CK(cudaHostAlloc((void**) &P.hbuf[i], CN * 4, cudaHostAllocPortable));
                CK(cudaEventCreateWithFlags(&P.ev[i], cudaEventDisableTiming));
                for (int j = 0; j < PFB::XS_MAX; ++j) CK(cudaEventCreateWithFlags(&P.evd[i][j], cudaEventDisableTiming));
            }
            if (NC == 2)
                std::fprintf(stderr, "ink2_v1: kartica %d (vozlisce %d): medpomnilnika izmenjave branja na vozliscu %d %d (INK2_NUMA %d)\n", K.c,
                             card_node(K.dev), page_node(P.hbuf[0]), page_node(P.hbuf[1]), numa);
        }
        for (int j = 0; j < PFB::XS_MAX; ++j) CK(cudaEventCreateWithFlags(&P.evp[j], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&P.evh, cudaEventDisableTiming));
        CK(cudaStreamCreateWithFlags(&P.sd, cudaStreamNonBlocking));   // p19d
        CK(cudaStreamCreateWithFlags(&P.sh, cudaStreamNonBlocking));
        if (NC == 1) {   // p31: the RAM experts' staging (both buffers at least one Q8 gate_up half)
            P.stg_bytes = (size_t) (getenv("INK2_1C_STAGE_MB") ? atoll(getenv("INK2_1C_STAGE_MB")) : 64LL) << 20;
            P.stg_bytes = std::max(P.stg_bytes, (size_t) 2 * FF_C * N + (size_t) 2 * FF_C * (N / 32) * 2);
            for (int i = 0; i < 2; ++i) {
                P.stg[i] = (uint8_t*) alloc(K, P.stg_bytes);
                CK(cudaEventCreateWithFlags(&P.evS[i], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&P.evU[i], cudaEventDisableTiming));
            }
            P.d_stab = (const uint8_t**) alloc(K, (size_t) 2 * NE * NE * sizeof(void*));
            CK(cudaHostAlloc((void**) &P.h_stab, (size_t) 2 * NE * NE * sizeof(void*), cudaHostAllocPortable));
        }
        P.h_cnt.resize(NE);
        P.h_offs.resize(NE + 1);
        K.pf_rec = false;   // p63
    }
    // y[C][rows] = x[C][cols] . W^T, W row-major f32 [rows][cols]
    void sgemm(Card& K, const float* W, int rows, int cols, const float* x, int ldx, float* y, int ldy, int C) {
        const float one = 1.0f, zero = 0.0f;
        cublasSgemm(K.P.cb, CUBLAS_OP_T, CUBLAS_OP_N, rows, C, cols, &one, W, cols, x, ldx, &zero, y, ldy);
    }
    void deq8(Card& K, const Mat& m, float* out) {
        const size_t n16 = (size_t) m.rows * m.cols / 16;
        k_q8s_f32<<<(unsigned) ((n16 + 255) / 256), 256, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, out);
    }
    void xch_pf(Card& K, float* y, size_t n) {   // through page-locked RAM by the copy engines; host-side order by pf_rec
        if (NC == 1) return;   // p30
        PFB& P = K.P;
        const int c = K.c;
        const unsigned sq = P.seq++;
        const int par = sq & 1;
        CK(cudaMemcpyAsync(P.hbuf[par], y, n * 4, cudaMemcpyDeviceToHost, K.st));
        CK(cudaEventRecord(P.ev[par], K.st));
        pf_rec[c].store(sq, std::memory_order_release);
        while (pf_rec[1 - c].load(std::memory_order_acquire) < sq) std::this_thread::yield();
        PFB& O = cards[1 - c].P;
        CK(cudaStreamWaitEvent(K.st, O.ev[par], 0));
        CK(cudaMemcpyAsync(P.tmp, O.hbuf[par], n * 4, cudaMemcpyHostToDevice, K.st));
        k_add_ord<<<(unsigned) ((n + 255) / 256), 256, 0, K.st>>>(y, P.tmp, n, c);
    }
    // p19d: an exchange whose C rows are made in xsplit slices by make(r0, r1) on K.st: each slice goes to RAM on the copy stream
    // sd as soon as it is made while the next one is made, the other card's slices come back on sh as soon as they are there. The
    // sum is xch_pf's (card 0's part first on both). Order: sd waits for each slice, sh for the other card's slices and for this
    // exchange's first slice (so tmp is free: K.st has added the last exchange by then); the other card overwrites the RAM buffer
    // of this parity only after it has added what it read from it (its K.st waits for its sh), as with xch_pf
    template <class F> void xch_pf2(Card& K, float* y, int C, F make) {
        if (NC == 1) {   // p30: nothing to exchange
            make(0, C);
            return;
        }
        PFB& P = K.P;
        const int c = K.c, ns = xsplit;
        const unsigned sq = P.seq++;
        const int par = sq & 1;
        for (int s = 0, r0 = 0; s < ns; ++s) {
            const int r1 = C * (s + 1) / ns;
            if (r1 > r0) make(r0, r1);
            CK(cudaEventRecord(P.evp[s], K.st));
            CK(cudaStreamWaitEvent(P.sd, P.evp[s], 0));
            if (r1 > r0)
                CK(cudaMemcpyAsync(P.hbuf[par] + (size_t) r0 * N, y + (size_t) r0 * N, (size_t) (r1 - r0) * N * 4, cudaMemcpyDeviceToHost, P.sd));
            CK(cudaEventRecord(P.evd[par][s], P.sd));
            r0 = r1;
        }
        pf_rec[c].store(sq, std::memory_order_release);
        while (pf_rec[1 - c].load(std::memory_order_acquire) < sq) std::this_thread::yield();
        PFB& O = cards[1 - c].P;
        CK(cudaStreamWaitEvent(P.sh, P.evp[0], 0));
        for (int s = 0, r0 = 0; s < ns; ++s) {
            const int r1 = C * (s + 1) / ns;
            CK(cudaStreamWaitEvent(P.sh, O.evd[par][s], 0));
            if (r1 > r0)
                CK(cudaMemcpyAsync(P.tmp + (size_t) r0 * N, O.hbuf[par] + (size_t) r0 * N, (size_t) (r1 - r0) * N * 4, cudaMemcpyHostToDevice, P.sh));
            r0 = r1;
        }
        CK(cudaEventRecord(P.evh, P.sh));
        CK(cudaStreamWaitEvent(K.st, P.evh, 0));
        CK(cudaStreamWaitEvent(K.st, P.evd[par][ns - 1], 0));   // our own slices are out before the sum overwrites y
        const size_t n = (size_t) C * N;
        k_add_ord<<<(unsigned) ((n + 255) / 256), 256, 0, K.st>>>(y, P.tmp, n, c);
    }
    // zip (INK2_PF_ZIP): chunk 0 keeps the reading path's own buffers, the later chunks get copies of the ones that live across
    // an exchange (the hidden state and its depth-mix states, the loop's banks and its fourth message, the exchanged rows,
    // positions, the drafter's features: 14 x a chunk's rows, 252 MiB at 2,048 tokens); everything else is scratch within a
    // step, and the other card's part of every chunk's exchange comes back into the one P.tmp (zx_recv)
    long long zip_spare(Card& K) {   // place_experts' rule: the bytes left on the card when every expert stays on it (< 0: they do not)
        CK(cudaSetDevice(K.dev));
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const long long margin = (getenv("INK2_1C_MARGIN_MB") ? atoll(getenv("INK2_1C_MARGIN_MB")) : NC == 1 ? 400LL : 150LL) << 20;
        long long budget = (long long) fr - margin - (long long) NPHYS * 2 * NE * 8;
        if (getenv("INK2_1C_EXP_MB")) budget = std::min(budget, atoll(getenv("INK2_1C_EXP_MB")) << 20);
        long long all = 0;
        for (int lp = 0; lp < NPHYS; ++lp) all += (long long) expert_bytes(lp) * NE;
        return budget - all;
    }
    void zip_alloc(Card& K) {
        PFB& P = K.P;
        const size_t C = pfc, CN = C * N;
        CK(cudaSetDevice(K.dev));
        vt[K.c] = "branje-zip";
        ZCtx& A = K.Z[0];
        A.cur = P.cur; A.prel = P.prel; A.bankS = P.bankS; A.bankO = P.bankO; A.ab = P.ab; A.MO = P.MO;
        A.tau = P.tau; A.pos = P.pos; A.pfeat = K.D.pfeat;
        for (int i = 0; i < MAXB; ++i) A.B[i] = P.B[i];
        for (int i = 0; i < 2; ++i) A.hbuf[i] = P.hbuf[i];
        for (int z = 1; z < zip; ++z) {
            ZCtx& B = K.Z[z];
            B.cur = dev(K, CN); B.prel = dev(K, CN); B.bankS = dev(K, CN); B.bankO = dev(K, CN); B.ms3 = dev(K, CN);
            B.ab = B.MO = dev(K, CN);   // one buffer: ab is used up (the convolution) before the experts' sum is made
            B.tau = dev(K, C); B.pos = (int*) alloc(K, C * 4);
            B.pfeat = K.D.pfeat ? dev(K, 4 * CN) : nullptr;
            for (int i = 0; i < MAXB; ++i) B.B[i] = dev(K, CN);
            const int numa = getenv("INK2_NUMA") ? atoi(getenv("INK2_NUMA")) : 1;   // p41's placement for these exchange buffers too
            NodeBind nb(numa == 0 ? -1 : card_node(numa == 2 && NC == 2 ? cards[NC - 1 - K.c].dev : K.dev));
            for (int i = 0; i < 2; ++i) CK(cudaHostAlloc((void**) &B.hbuf[i], CN * 4, cudaHostAllocPortable));
        }
        for (int z = 0; z < zip; ++z) {
            ZCtx& Z = K.Z[z];
            for (int j = 0; j < PFB::XS_MAX; ++j) {
                CK(cudaEventCreateWithFlags(&Z.evp[j], cudaEventDisableTiming));
                for (int i = 0; i < 2; ++i) CK(cudaEventCreateWithFlags(&Z.evd[i][j], cudaEventDisableTiming));
            }
            CK(cudaEventCreateWithFlags(&Z.evh, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&Z.evb, cudaEventDisableTiming));
        }
        vt[K.c] = "ostalo";
    }
    void zip_free(Card& K) {   // zip_alloc undone (the experts would not all stay on the card beside it)
        CK(cudaSetDevice(K.dev));
        const size_t C = pfc, CN = C * N;
        for (int z = 1; z < zip; ++z) {
            ZCtx& B = K.Z[z];
            for (float* q : {B.cur, B.prel, B.bankS, B.bankO, B.ms3, B.ab, B.tau, B.pfeat})
                if (q) CK(cudaFree(q));
            CK(cudaFree(B.pos));
            for (int i = 0; i < MAXB; ++i) CK(cudaFree(B.B[i]));
            for (int i = 0; i < 2; ++i) CK(cudaFreeHost(B.hbuf[i]));
            const size_t b = ((B.pfeat ? 14 : 10) * CN + 2 * C) * 4;
            K.vram -= b;
            vtag[K.c]["branje-zip"] -= b;
        }
    }
    void zip_load(Card& K, int id) {   // P points at chunk id's buffers
        PFB& P = K.P;
        const ZCtx& Z = K.Z[id];
        P.cur = Z.cur; P.prel = Z.prel; P.bankS = Z.bankS; P.bankO = Z.bankO; P.ab = Z.ab; P.MO = Z.MO;
        P.tau = Z.tau; P.pos = Z.pos; K.D.pfeat = Z.pfeat;
        for (int i = 0; i < MAXB; ++i) P.B[i] = Z.B[i];
        K.nb = Z.nb;
        K.zc = id;
    }
    void zip_save(Card& K) { K.Z[K.zc].nb = K.nb; }
    // xch_pf2 in three parts for the chunks of a zip. zx_start: the slices of the chunk P points at made and sent to RAM.
    // zx_recv(id): the other card's slices of chunk id's exchange ordered back into P.tmp, which every chunk uses: it is
    // called right after a sum (or at a chunk's first step), for the chunk whose sum comes next on the stream, so tmp is
    // free (evb, recorded here on K.st, is behind the sum before) and the copy runs under the step that follows. zx_add, at
    // the chunk's next step: wait for them and add
    template <class F> void zx_start(Card& K, float* y, int C, F make) {
        PFB& P = K.P;
        ZCtx& Z = K.Z[K.zc];
        const int c = K.c, ns = xsplit;
        const unsigned sq = Z.seq++;
        const int par = sq & 1;
        for (int s = 0, r0 = 0; s < ns; ++s) {
            const int r1 = C * (s + 1) / ns;
            if (r1 > r0) make(r0, r1);
            CK(cudaEventRecord(Z.evp[s], K.st));
            CK(cudaStreamWaitEvent(P.sd, Z.evp[s], 0));
            if (r1 > r0)
                CK(cudaMemcpyAsync(Z.hbuf[par] + (size_t) r0 * N, y + (size_t) r0 * N, (size_t) (r1 - r0) * N * 4, cudaMemcpyDeviceToHost, P.sd));
            CK(cudaEventRecord(Z.evd[par][s], P.sd));
            r0 = r1;
        }
        zrec[K.zc][c].store(sq, std::memory_order_release);
        Z.py = y;
        Z.pC = C;
        Z.ppar = par;
        Z.psq = sq;
        Z.want = true;
    }
    void zx_recv(Card& K, int id) {
        PFB& P = K.P;
        ZCtx& Z = K.Z[id];
        if (!Z.want) return;
        Z.want = false;
        const int c = K.c, ns = xsplit, par = Z.ppar, C = Z.pC;
        while (zrec[id][1 - c].load(std::memory_order_acquire) < Z.psq) std::this_thread::yield();
        ZCtx& O = cards[1 - c].Z[id];
        CK(cudaEventRecord(Z.evb, K.st));
        CK(cudaStreamWaitEvent(P.sh, Z.evb, 0));
        for (int s = 0, r0 = 0; s < ns; ++s) {
            const int r1 = C * (s + 1) / ns;
            CK(cudaStreamWaitEvent(P.sh, O.evd[par][s], 0));
            if (r1 > r0)
                CK(cudaMemcpyAsync(P.tmp + (size_t) r0 * N, O.hbuf[par] + (size_t) r0 * N, (size_t) (r1 - r0) * N * 4, cudaMemcpyHostToDevice, P.sh));
            r0 = r1;
        }
        CK(cudaEventRecord(Z.evh, P.sh));
    }
    void zx_add(Card& K) {
        PFB& P = K.P;
        ZCtx& Z = K.Z[K.zc];
        zx_recv(K, K.zc);   // still to order only when no other chunk's step came in between
        CK(cudaStreamWaitEvent(K.st, Z.evh, 0));
        CK(cudaStreamWaitEvent(K.st, Z.evd[Z.ppar][xsplit - 1], 0));   // our own slices are out before the sum overwrites y
        const size_t n = (size_t) Z.pC * N;
        k_add_ord<<<(unsigned) ((n + 255) / 256), 256, 0, K.st>>>(Z.py, P.tmp, n, K.c);
        Z.py = nullptr;
        zx_recv(K, (K.zc + 1) % K.zn);   // the chunk whose sum comes next: its part comes back under this step
    }
    void moe_comb(Card& K, const LW& L, int r0, int r1) {   // p19d: rows r0 .. r1 - 1 of k_moe_comb
        PFB& P = K.P;
        const size_t o = (size_t) r0 * TOPK;
        k_moe_comb<<<(unsigned) (((size_t) (r1 - r0) * N + 255) / 256), 256, 0, K.st>>>(P.EY, P.slotrow + o, P.topi + o, P.topw + o,
                                                                                         L.fp4 ? L.dn_s : nullptr, P.MO + (size_t) r0 * N, r1 - r0);
    }
    void conv_pf(Card& K, const float* x, int Cc, float* st, const float* w, const float* dyn, int hs, float k, float* y, int T) {
        const size_t n = (size_t) T * Cc;
        k_conv_par<<<(unsigned) ((n + 255) / 256), 256, 0, K.st>>>(x, Cc, st, w, dyn, hs, k, y, T);
        k_conv_state<<<(Cc + 255) / 256, 256, 0, K.st>>>(x, Cc, st, T);
    }
    void control_pf(Card& K, const LW& L, int j, const float* x, const float* m, float* out, int C) {
        PFB& P = K.P;
        k_ctrl_in<<<C, 256, 0, K.st>>>(x, m, N, P.xm);
        sgemm(K, L.c_down[j], RANK, 2 * N, P.xm, 2 * N, P.g64, RANK, C);
        sgemm(K, L.c_gain[j], N, RANK, P.g64, RANK, P.gp, N, C);
        sgemm(K, L.c_forget[j], N, RANK, P.g64, RANK, P.fpp, N, C);
        k_ctrl_out<<<(C * N + 255) / 256, 256, 0, K.st>>>(x, m, N, P.gp, P.fpp, out, C);
    }
    States states_pf(Card& K, const float* last) {
        States S{};
        for (int i = 0; i < K.nb; ++i) S.p[i] = K.P.B[i];
        S.p[K.nb] = last;
        S.n = K.nb + 1;
        return S;
    }
    // p19a: the full layers' query block for a chunk whose keys end at nk: as big as S and the P.V pieces allow for nk keys
    // (pf_qs_full is the block that fits the whole context; at 262144 that is 64, and a 37.6k prompt ran its P.V in calls of a
    // few CUDA blocks), at most qsf_max (2048 = a chunk). INK2_PF_QSF fixes the block as before. p19c: S at most 3 x 2^25 floats
    // (402 MB): the 256k and the 512k modes both hold that in K.part, so they choose the same blocks and stay bit for bit the same
    // below the int8 rows
    int qsf(const PFB& P, int nk) const {
        if (qsf_fixed) return pf_qs_full;
        const size_t s_cap = std::min(P.S_n, (size_t) 3 << 25);
        const size_t nsb = (size_t) (nk + PVK - 1) / PVK;
        int q = pf_qs_full;
        while (2 * q <= qsf_max && (size_t) 3 * (2 * q) * nk <= s_cap && 3 * nsb * (2 * q) * HD <= pv_cap) q *= 2;
        return q;
    }
    // attention of a chunk on one slot: positions p0 .. p0 + C - 1 are in the cache already
    void attn_pf(Card& K, const LW& L, int slot, int p0, int C) {
        PFB& P = K.P;
        const bool win = !L.full, rot = !win && K.q8[slot].rot;   // p18
        if (rot) k_rot<<<dim3(NH_C, C), 128, 0, K.st>>>(P.q, QC);
        const float one = 1.0f, zero = 0.0f, scale = 1.0f / sqrtf((float) HD);
        // R.B for d = 0..EXT: RB[(i * NH_C + h)][d]
        if (pf_rb)
            k_rb<<<(C * NH_C + 7) / 8, 256, 0, K.st>>>(P.r, L.rel_bias_pad, C * NH_C, P.RB);
        else
            cublasSgemm(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, EXT + 1, C * NH_C, NREL, &one, L.rel_bias_pad, NREL, P.r, NREL, &zero, P.RB, EXT + 1);
        const bool bat = win && pf_attb;   // p3f: keys from p0 - (SWA - 1) even below 0 (zero rows, masked), so every block has the same shape
        const int klo = bat ? p0 - (SWA - 1) : win ? std::max(0, p0 - (SWA - 1)) : 0, khi = p0 + C;
        const int nkall = khi - klo;
        const int QF = win ? QS : qsf(P, khi), nsbm = (khi + PVK - 1) / PVK;   // p19a: the block and the P.V pieces' stride
        if (!win && nkall > KFB) {   // p14: long context: per query block and KV head, Q.K^T by key blocks into S, softmax, P.V by blocks
            for (int i0 = 0; i0 < C; i0 += QF) {
                const int qs = std::min(QF, C - i0), nk = p0 + i0 + qs;
                for (int g = 0; g < NKV_C; ++g) {
                    const int h0 = g * (NH / NKV);
                    for (int b0 = 0; b0 < nk; b0 += KFB) {
                        const int bn = std::min(KFB, nk - b0);
                        k_kv_f32g<<<(unsigned) (((size_t) bn * HD + 255) / 256), 256, 0, K.st>>>(K.kc[slot], K.vc[slot], b0, bn, g, P.Kf, P.Vf,
                                                                                                   K.q8[slot]);
                        cublasSgemmStridedBatched(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, bn, qs, HD, &one, P.Kf, HD, 0, P.q + (size_t) i0 * QC + h0 * HD, QC,
                                                  HD, &zero, P.S + b0, nk, (long long) nk * qs, NH / NKV);
                    }
                    k_pf_softmax<<<dim3(qs, NH / NKV), 256, 0, K.st>>>(P.S, nk, 0, P.pos, i0, h0, P.RB, P.sink, 0, scale);
                    for (int b0 = 0; b0 < nk; b0 += KFB) {   // p14b: P.V by key pieces (KFB is a multiple of PVK)
                        const int bn = std::min(KFB, nk - b0);
                        k_kv_f32g<<<(unsigned) (((size_t) bn * HD + 255) / 256), 256, 0, K.st>>>(K.kc[slot], K.vc[slot], b0, bn, g, nullptr, P.Vf,
                                                                                                   K.q8[slot]);
                        pv_parts(K, P.Vf, HD, b0, bn, nk, qs, nsbm);
                    }
                    const int nsb = (nk + PVK - 1) / PVK;
                    k_pv_sum<<<(unsigned) ((3 * (size_t) qs * HD + 255) / 256), 256, 0, K.st>>>(P.PV, nsb, nsbm, qs, P.ao, i0, h0);
                }
            }
            if (rot) k_rot<<<dim3(NH_C, C), 128, 0, K.st>>>(P.ao, QC);
            k_sigmul<<<(unsigned) (((size_t) C * QC + 255) / 256), 256, 0, K.st>>>(P.ao, P.gate, (size_t) C * QC);
            return;
        }
        k_kv_f32<<<(unsigned) (((size_t) nkall * KVC + 255) / 256), 256, 0, K.st>>>(K.kc[slot], K.vc[slot], klo, nkall, win, P.Kf, P.Vf,
                                                                                    K.q8[slot]);
        int ib = 0;
        if (bat && C / QS > 0) {
            const int nb = C / QS, nk = QS + SWA - 1;
            cublasSgemmBatched(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, nk, QS, HD, &one, (const float**) P.pk, KVC, (const float**) P.pq, QC, &zero,
                               P.ps, nk, nb * NH_C);
            k_pf_softmax_b<<<dim3(QS, NH_C, nb), 256, 0, K.st>>>(P.S, nk, p0, QS, P.pos, P.RB, P.sink, scale);
            cublasSgemmBatched(P.cb, CUBLAS_OP_N, CUBLAS_OP_N, HD, QS, nk, &one, (const float**) P.pv, KVC, (const float**) P.ps, nk, &zero,
                               P.pao, QC, nb * NH_C);
            ib = nb * QS;
        }
        const int QSL = QF;   // p3d: bigger query blocks for the full layers (bigger cuBLAS calls); p19a: by the chunk's keys
        for (int i0 = ib; i0 < C; i0 += QSL) {
            const int qs = std::min(QSL, C - i0);
            const int lo = bat ? p0 + i0 - (SWA - 1) : win ? std::max(0, p0 + i0 - (SWA - 1)) : 0, hi = p0 + i0 + qs, nk = hi - lo;
            const float* kf = P.Kf + (size_t) (lo - klo) * KVC;
            const float* vf = P.Vf + (size_t) (lo - klo) * KVC;
            for (int g = 0; g < NKV_C; ++g) {
                const int h0 = g * (NH / NKV);
                cublasSgemmStridedBatched(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, nk, qs, HD, &one, kf + g * HD, KVC, 0,
                                          P.q + (size_t) i0 * QC + h0 * HD, QC, HD, &zero, P.S, nk, (long long) nk * qs, NH / NKV);
                k_pf_softmax<<<dim3(qs, NH / NKV), 256, 0, K.st>>>(P.S, nk, lo, P.pos, i0, h0, P.RB, P.sink, win, scale);
                if (!win && pv_split && nk > 4 * PVK) {   // p14b: many keys, few output tiles: P.V by key pieces
                    pv_parts(K, vf + g * HD, KVC, 0, nk, nk, qs, nsbm);
                    k_pv_sum<<<(unsigned) ((3 * (size_t) qs * HD + 255) / 256), 256, 0, K.st>>>(P.PV, (nk + PVK - 1) / PVK, nsbm, qs, P.ao, i0, h0);
                } else
                cublasSgemmStridedBatched(P.cb, CUBLAS_OP_N, CUBLAS_OP_N, HD, qs, nk, &one, vf + g * HD, KVC, 0, P.S, nk, (long long) nk * qs,
                                          &zero, P.ao + (size_t) i0 * QC + h0 * HD, QC, HD, NH / NKV);
            }
        }
        if (rot) k_rot<<<dim3(NH_C, C), 128, 0, K.st>>>(P.ao, QC);
        k_sigmul<<<(unsigned) (((size_t) C * QC + 255) / 256), 256, 0, K.st>>>(P.ao, P.gate, (size_t) C * QC);
    }
    // p14b: the P.V of KV head group h0 for keys [b0, b0 + bn) (V rows at vb, row stride ldv) into the parts of pieces b0 / PVK ..
    void pv_parts(Card& K, const float* vb, int ldv, int b0, int bn, int nk, int qs, int nsbm) {   // p19a: nsbm = the parts' stride
        PFB& P = K.P;
        const float one = 1.0f, zero = 0.0f;
        const int sb0 = b0 / PVK, nfull = bn / PVK, tail = bn % PVK;
        for (int a = 0; a < 3; ++a) {
            const float* Sa = P.S + (size_t) a * nk * qs + b0;
            float* Pa = P.PV + ((size_t) a * nsbm + sb0) * qs * HD;
            if (nfull)
                cublasSgemmStridedBatched(P.cb, CUBLAS_OP_N, CUBLAS_OP_N, HD, qs, PVK, &one, vb, ldv, (long long) PVK * ldv, Sa, nk, PVK, &zero,
                                          Pa, HD, (long long) qs * HD, nfull);
            if (tail)
                cublasSgemm(P.cb, CUBLAS_OP_N, CUBLAS_OP_N, HD, qs, tail, &one, vb + (size_t) nfull * PVK * ldv, ldv, Sa + (size_t) nfull * PVK, nk,
                            &zero, Pa + (size_t) nfull * qs * HD, HD);
        }
    }
    // p50: k_fp4_gemm5 over the tiles [0, nt) of P.tiles; with INK2_G6 the first n64 (the 64-token tiles) by k_fp4_g6 (the same sums)
    template <typename WT, typename ETT>
    void fp4_gemm_pf(Card& K, int nt, WT W, size_t es, int rows, int cols, const float* x, int ldx, float* y, int ldy, ETT ET) {
        PFB& P = K.P;
        int n64 = 0;
        if (g6 == 32 || g6 == 16)
            while (n64 < nt && P.h_tiles[n64].nk > 32) ++n64;
        if (n64 > 0) {
            if (g6 == 16) k_fp4_g6<16, 2><<<dim3(rows / F4M, n64), 128, 0, K.st>>>(W, es, rows, cols, P.tiles, x, ldx, y, ldy, ET);
            else k_fp4_g6<32, 1><<<dim3(rows / F4M, n64), 128, 0, K.st>>>(W, es, rows, cols, P.tiles, x, ldx, y, ldy, ET);
        }
        if (nt > n64)
            k_fp4_gemm5<<<dim3(rows / F4M, nt - n64), 256, 0, K.st>>>(W, es, rows, cols, P.tiles + n64, x, ldx, y, ldy, tile_s, ET);
    }
    void experts_pf(Card& K, const LW& L, int C, bool comb = true) {   // p19d: comb false = without the last k_moe_comb (xch_pf2 makes it)
        PFB& P = K.P;
        const int n = C * TOPK;
        const float one = 1.0f, zero = 0.0f;
        CK(cudaMemsetAsync(P.cnt, 0, NE * 4, K.st));
        k_exp_count<<<(n + 255) / 256, 256, 0, K.st>>>(P.topi, n, P.cnt);
        CK(cudaMemcpyAsync(P.h_cnt.data(), P.cnt, NE * 4, cudaMemcpyDeviceToHost, K.st));
        CK(cudaStreamSynchronize(K.st));
        P.h_offs[0] = 0;
        for (int e = 0; e < NE; ++e) P.h_offs[e + 1] = P.h_offs[e] + P.h_cnt[e];
        CK(cudaMemcpyAsync(P.offs, P.h_offs.data(), (NE + 1) * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemsetAsync(P.cursor, 0, NE * 4, K.st));
        k_exp_fill<<<(n + 255) / 256, 256, 0, K.st>>>(P.topi, n, P.offs, P.cursor, P.slotrow, P.rowtok, P.rowexp);
        k_gather_rows<<<(unsigned) (((size_t) n * (N / 4) + 255) / 256), 256, 0, K.st>>>(P.FN, P.rowtok, n, P.Xg);
        const size_t gu_rows = 2 * FF_C, gu_e = gu_rows * N, dn_e = (size_t) N * FF_C;
        if (L.nres < NE && P.stg[0] && !pf_deq && gemm5 && !ilv && !gemm2 && (L.fp4 || q8g)) {   // p31
            experts_pf_staged(K, L, C);
            if (comb) moe_comb(K, L, 0, C);
            return;
        }
        if (L.fp4 && !pf_deq && gemm5 && !ilv && !gemm2) {   // p19b: the tile list from the counts the host has already
            const int nt = fp4_tiles(P.h_offs.data(), P.h_tiles.data());
            CK(cudaMemcpyAsync(P.tiles, P.h_tiles.data(), (size_t) nt * sizeof(FTile), cudaMemcpyHostToDevice, K.st));
            fp4_gemm_pf(K, nt, L.gu, gu_rows * (N / 64) * 36, (int) gu_rows, N, P.Xg, N, P.GU, (int) gu_rows, L.gu_t);   // p50
            k_situ_s<<<(n * FF_C + 255) / 256, 256, 0, K.st>>>(P.GU, P.HH, n, P.rowexp, L.gu_s, situ_b, situ_lb);
            fp4_gemm_pf(K, nt, L.dn, (size_t) N * (FF_C / 64) * 36, N, FF_C, P.HH, FF_C, P.EY, N, L.dn_t);   // p50
            if (comb) moe_comb(K, L, 0, C);
            return;
        }
        if (!L.fp4 && q8g) {   // p25: the Q8 experts by the same tiles, straight from int8 rows and half scales
            const int nt = fp4_tiles(P.h_offs.data(), P.h_tiles.data());
            CK(cudaMemcpyAsync(P.tiles, P.h_tiles.data(), (size_t) nt * sizeof(FTile), cudaMemcpyHostToDevice, K.st));
            k_q8_gemm5<<<dim3(gu_rows / F4M, nt), 256, 0, K.st>>>((const int8_t*) L.gu, L.gu_d, gu_e, gu_rows * (N / 32), N, P.tiles, P.Xg, N,
                                                                   P.GU, (int) gu_rows, tile_s, L.gu_t);
            k_situ_s<<<(n * FF_C + 255) / 256, 256, 0, K.st>>>(P.GU, P.HH, n, P.rowexp, nullptr, situ_b, situ_lb);
            k_q8_gemm5<<<dim3(N / F4M, nt), 256, 0, K.st>>>((const int8_t*) L.dn, L.dn_d, dn_e, (size_t) N * (FF_C / 32), FF_C, P.tiles, P.HH,
                                                             FF_C, P.EY, N, tile_s, L.dn_t);
            if (comb) moe_comb(K, L, 0, C);
            return;
        }
        if (L.fp4 && !pf_deq) {   // p3c: the NVFP4 tiles straight from the weights (p4d: planar unless INK2_FP4_ILV)
            auto gemm = ilv ? k_fp4_gemm<false> : (gemm2 ? k_fp4_gemm2<true> : k_fp4_gemm<true>);   // p3e: INK2_GEMM2 = 128 x 128 tiles
            gemm<<<dim3(gu_rows / F4M, NE), 256, 0, K.st>>>(L.gu, gu_rows * (N / 64) * 36, (int) gu_rows, N, P.offs, P.Xg, N, P.GU, (int) gu_rows);
            k_situ_s<<<(n * FF_C + 255) / 256, 256, 0, K.st>>>(P.GU, P.HH, n, P.rowexp, L.gu_s, situ_b, situ_lb);
            gemm<<<dim3(N / F4M, NE), 256, 0, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, N, FF_C, P.offs, P.HH, FF_C, P.EY, N);
            if (comb) moe_comb(K, L, 0, C);
            return;
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (int e0 = 0; e0 < NE; e0 += EB) {
                if (P.h_offs[e0 + EB] == P.h_offs[e0]) continue;
                float* E = pass == 0 ? P.Egu : P.Edn;
                const size_t rows = (pass == 0 ? gu_rows : (size_t) N) * EB;
                const int cols = pass == 0 ? N : FF_C;
                if (L.fp4) {
                    const uint8_t* W = pass == 0 ? L.gu + (size_t) e0 * gu_rows * (N / 64) * 36 : L.dn + (size_t) e0 * N * (FF_C / 64) * 36;
                    const size_t n4 = rows * (cols / 4);
                    k_fp4_f32b<<<(unsigned) ((n4 + 255) / 256), 256, 0, K.st>>>(W, rows, cols, E);
                } else {
                    const int8_t* q = (const int8_t*) (pass == 0 ? L.gu + (size_t) e0 * gu_e : L.dn + (size_t) e0 * dn_e);
                    const __half* d = pass == 0 ? L.gu_d + (size_t) e0 * gu_rows * (N / 32) : L.dn_d + (size_t) e0 * N * (FF_C / 32);
                    const size_t n16 = rows * cols / 16;
                    k_q8s_f32<<<(unsigned) ((n16 + 255) / 256), 256, 0, K.st>>>(q, d, rows, cols, E);
                }
                for (int e = e0; e < e0 + EB; ++e) {
                    const int ne = P.h_cnt[e], o = P.h_offs[e];
                    if (!ne) continue;
                    if (pass == 0)
                        cublasSgemm(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) gu_rows, ne, N, &one, E + (size_t) (e - e0) * gu_e, N,
                                    P.Xg + (size_t) o * N, N, &zero, P.GU + (size_t) o * gu_rows, (int) gu_rows);
                    else
                        cublasSgemm(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, N, ne, FF_C, &one, E + (size_t) (e - e0) * dn_e, FF_C,
                                    P.HH + (size_t) o * FF_C, FF_C, &zero, P.EY + (size_t) o * N, N);
                }
            }
            if (pass == 0)
                k_situ_s<<<(n * FF_C + 255) / 256, 256, 0, K.st>>>(P.GU, P.HH, n, P.rowexp, L.fp4 ? L.gu_s : nullptr, situ_b, situ_lb);
        }
        if (comb) moe_comb(K, L, 0, C);
    }
    // p31: experts_pf for a layer with experts in RAM (the counts, offsets, gathered rows and rowexp are made). VRAM experts' tiles
    // first, then the RAM experts with rows in groups of G through the staging buffers: gate_up halves, SITU over all rows, then the
    // down halves; group g in buffer g & 1, copied on sd while the group before it is multiplied (evS: filled, evU: free again)
    void experts_pf_staged(Card& K, const LW& L, int C) {
        PFB& P = K.P;
        const int n = C * TOPK;
        const size_t gu_rows = 2 * FF_C, gu_e = gu_rows * N, dn_e = (size_t) N * FF_C;
        const size_t gb = L.fp4 ? gu_rows * (N / 64) * 36 : gu_e + gu_rows * (N / 32) * 2;
        const size_t db = L.fp4 ? (size_t) N * (FF_C / 64) * 36 : dn_e + (size_t) N * (FF_C / 32) * 2;
        std::vector<char> inv(NE, 0);
        std::vector<int> ram;
        for (int e = 0; e < NE; ++e)
            if (P.h_cnt[e]) {
                if (L.res[e]) inv[e] = 1;
                else ram.push_back(e);
            }
        const int G = std::max(1, (int) (P.stg_bytes / gb)), ng = ((int) ram.size() + G - 1) / G;
        int nt = 0;
        auto tiles_of = [&](auto pick) {   // fp4_tiles() for the experts pick(e) chooses: 64-token tiles, then 32s, then 16s
            const int s = nt;
            for (int pass = 0; pass < 3; ++pass)
                for (int e = 0; e < NE; ++e) {
                    if (!pick(e)) continue;
                    const int cnt = P.h_offs[e + 1] - P.h_offs[e];
                    for (int q = 0; q < cnt;) {
                        const int left = cnt - q, tn = left > 32 ? 64 : left > 16 ? 32 : 16;
                        if (tn == (pass == 0 ? 64 : pass == 1 ? 32 : 16)) P.h_tiles[nt++] = FTile{e, P.h_offs[e] + q, std::min(tn, left)};
                        q += tn;
                    }
                }
            return s;
        };
        std::vector<int> ts(ng + 2);
        ts[0] = tiles_of([&](int e) { return inv[e] != 0; });
        for (int g = 0; g < ng; ++g) {
            std::vector<char> m(NE, 0);
            for (int i = g * G; i < std::min((g + 1) * G, (int) ram.size()); ++i) m[ram[i]] = 1;
            ts[g + 1] = tiles_of([&](int e) { return m[e] != 0; });
        }
        ts[ng + 1] = nt;
        for (int g = 0; g < ng; ++g)   // group g's tables: gate_up at [2g], down at [2g + 1]
            for (int i = g * G; i < std::min((g + 1) * G, (int) ram.size()); ++i) {
                const int e = ram[i], j = i - g * G;
                P.h_stab[(size_t) (2 * g) * NE + e] = P.stg[g & 1] + (size_t) j * gb;
                P.h_stab[(size_t) (2 * g + 1) * NE + e] = P.stg[g & 1] + (size_t) j * db;
            }
        CK(cudaMemcpyAsync(P.tiles, P.h_tiles.data(), (size_t) nt * sizeof(FTile), cudaMemcpyHostToDevice, K.st));
        if (ng) CK(cudaMemcpyAsync(P.d_stab, P.h_stab, (size_t) 2 * ng * NE * sizeof(void*), cudaMemcpyHostToDevice, K.st));
        auto stage = [&](int g, bool down) {
            const int b = g & 1;
            CK(cudaStreamWaitEvent(P.sd, P.evU[b], 0));
            for (int i = g * G; i < std::min((g + 1) * G, (int) ram.size()); ++i) {
                const int e = ram[i], j = i - g * G;
                CK(cudaMemcpyAsync(P.stg[b] + (size_t) j * (down ? db : gb), down ? L.h_dn[e] : L.h_gu[e], down ? db : gb, cudaMemcpyDefault, P.sd));
            }
            CK(cudaEventRecord(P.evS[b], P.sd));
        };
        auto gemm = [&](int t0, int t1, bool down, const uint8_t* const* ET) {
            if (t1 <= t0) return;
            const dim3 grid(down ? N / F4M : (unsigned) (gu_rows / F4M), t1 - t0);
            if (L.fp4) {
                if (down) k_fp4_gemm5<<<grid, 256, 0, K.st>>>(nullptr, 0, N, FF_C, P.tiles + t0, P.HH, FF_C, P.EY, N, tile_s, ET);
                else k_fp4_gemm5<<<grid, 256, 0, K.st>>>(nullptr, 0, (int) gu_rows, N, P.tiles + t0, P.Xg, N, P.GU, (int) gu_rows, tile_s, ET);
            } else {
                if (down) k_q8_gemm5<<<grid, 256, 0, K.st>>>(nullptr, nullptr, dn_e, 0, FF_C, P.tiles + t0, P.HH, FF_C, P.EY, N, tile_s, ET);
                else k_q8_gemm5<<<grid, 256, 0, K.st>>>(nullptr, nullptr, gu_e, 0, N, P.tiles + t0, P.Xg, N, P.GU, (int) gu_rows, tile_s, ET);
            }
        };
        for (int ph = 0; ph < 2; ++ph) {
            const bool down = ph == 1;
            for (int g = 0; g < std::min(ng, 2); ++g) stage(g, down);
            gemm(ts[0], ts[1], down, down ? L.dn_t : L.gu_t);
            for (int g = 0; g < ng; ++g) {
                const int b = g & 1;
                CK(cudaStreamWaitEvent(K.st, P.evS[b], 0));
                gemm(ts[g + 1], ts[g + 2], down, (const uint8_t* const*) P.d_stab + (size_t) (2 * g + ph) * NE);
                CK(cudaEventRecord(P.evU[b], K.st));
                if (g + 2 < ng) stage(g + 2, down);
            }
            if (!down) k_situ_s<<<(n * FF_C + 255) / 256, 256, 0, K.st>>>(P.GU, P.HH, n, P.rowexp, L.fp4 ? L.gu_s : nullptr, situ_b, situ_lb);
        }
    }
    // zip (INK2_PF_ZIP): part -1 = the whole layer; 0 = up to the attention's exchange on its way, 1 = from its sum to the
    // experts' exchange on its way, 2 = from that sum to the layer's end
    void layer_pf(Card& K, int p, int slot, bool loop, int p0, int C, int part = -1) {
        const LW& L = K.L[p];
        PFB& P = K.P;
        const size_t CN = (size_t) C * N;
        auto rmsC = [&](const float* x, const float* w, float* y) { k_rms<<<C, 256, 0, K.st>>>(x, N, w, y, N, N, eps); };
        if (part <= 0) {
        const float* xin[3] = {P.cur, P.cur, P.cur};
        if (L.attn_rq) {
            mix(K, states_pf(K, P.cur), L.attn_rq, L.nrq, P.X3, C);
            for (int j = 0; j < 3; ++j) xin[j] = P.X3 + (L.full ? j : 0) * CN;
        }
        if (loop) {
            States S{};
            S.p[0] = P.bankS;
            S.p[1] = P.bankO;
            S.n = 2;
            // zip: a later chunk's messages go to the experts' scratch (free here; chunk 0's are in P.MS as ever) and the
            // fourth, which is read after the attention's exchange, from there to the chunk's own ms3
            float* MS = part >= 0 && K.zc > 0 ? P.Xg : P.MS;
            mix(K, S, L.msg_q, 4, MS, C);
            for (int j = 0; j < 3; ++j) {
                control_pf(K, L, j, xin[j], MS + j * CN, P.XC[j], C);
                xin[j] = P.XC[j];
            }
            if (MS != P.MS) CK(cudaMemcpyAsync(K.Z[K.zc].ms3, MS + 3 * CN, CN * 4, cudaMemcpyDeviceToDevice, K.st));
        }
        const float *nq, *nk, *nv;
        if (!L.full && !loop) {
            rmsC(xin[0], L.attn_norm, P.NQ);
            nq = nk = nv = P.NQ;
        } else {
            rmsC(xin[0], L.full ? L.q_norm : L.attn_norm, P.NQ);
            rmsC(xin[1], L.full ? L.k_norm : L.attn_norm, P.NK);
            rmsC(xin[2], L.full ? L.v_norm : L.attn_norm, P.NV);
            nq = P.NQ;
            nk = P.NK;
            nv = P.NV;
        }
        deq8(K, L.cat, P.Wf);
        const int nqr = 2 * QC + NH_C * NREL + NH_C, nkr = KVC + (L.full ? NKV_C * NCONV : 0);
        if (catm) {   // p27: a call a group of rows with the same input, into P.Egu, then each part to its buffer
            CatSegs s{};
            int r0 = 0, nr = 0;
            auto part = [&](float* dst, int w) { s.dst[s.n] = dst; s.ld[s.n] = w; s.off[s.n] = nr; nr += w; ++s.n; };
            auto flush = [&](const float* x) {
                sgemm(K, P.Wf + (size_t) r0 * N, nr, N, x, N, P.Egu, nr, C);
                k_cat_scatter<<<(unsigned) (((size_t) C * nr + 255) / 256), 256, 0, K.st>>>(P.Egu, nr, C, s);
                r0 += nr;
                nr = 0;
                s = CatSegs{};
            };
            part(P.q, QC);
            part(P.gate, QC);
            part(P.r, NH_C * NREL);
            part(P.sink, NH_C);
            const bool one = nq == nk && nk == nv;
            if (!one) flush(nq);
            part(P.k, KVC);
            if (L.full) part(P.dk, NKV_C * NCONV);
            if (!one) flush(nk);
            part(P.v, KVC);
            if (L.full) part(P.dv, NKV_C * NCONV);
            flush(nv);
        } else {
        sgemm(K, P.Wf, QC, N, nq, N, P.q, QC, C);
        sgemm(K, P.Wf + (size_t) QC * N, QC, N, nq, N, P.gate, QC, C);
        sgemm(K, P.Wf + (size_t) 2 * QC * N, NH_C * NREL, N, nq, N, P.r, NH_C * NREL, C);
        sgemm(K, P.Wf + (size_t) (2 * QC + NH_C * NREL) * N, NH_C, N, nq, N, P.sink, NH_C, C);
        sgemm(K, P.Wf + (size_t) nqr * N, KVC, N, nk, N, P.k, KVC, C);
        sgemm(K, P.Wf + (size_t) (nqr + nkr) * N, KVC, N, nv, N, P.v, KVC, C);
        }
        if (L.full && !catm) {
            sgemm(K, P.Wf + (size_t) (nqr + KVC) * N, NKV_C * NCONV, N, nk, N, P.dk, NKV_C * NCONV, C);
            sgemm(K, P.Wf + (size_t) (nqr + nkr + KVC) * N, NKV_C * NCONV, N, nv, N, P.dv, NKV_C * NCONV, C);
        }
        const float kconv = 1.0f / sqrtf((float) ((L.full ? 2 : 1) * NCONV + 1));
        conv_pf(K, P.k, KVC, K.sk[slot], L.k_conv, L.full ? P.dk : nullptr, HD, kconv, P.kcv, C);
        conv_pf(K, P.v, KVC, K.sv[slot], L.v_conv, L.full ? P.dv : nullptr, HD, kconv, P.vcv, C);
        const float* tau = L.full ? P.tau : nullptr;
        k_headnorm<<<(C * NH_C + 7) / 8, 256, 0, K.st>>>(P.q, NH_C, HD, L.q_clip, K.c * NH_C, qk_clip, tau, 0, eps, C);
        k_headnorm<<<(C * NKV_C + 7) / 8, 256, 0, K.st>>>(P.kcv, NKV_C, HD, nullptr, 0, qk_clip, nullptr, 0, eps, C);
        k_headnorm<<<(C * NKV_C + 7) / 8, 256, 0, K.st>>>(P.vcv, NKV_C, HD, nullptr, 0, 0.0f, nullptr, 1, eps, C);
        k_headnorm<<<(C * NH_C + 7) / 8, 256, 0, K.st>>>(P.r, NH_C, NREL, L.rel_clip, K.c * NH_C, rel_clip_thr, tau, 0, eps, C);
        if (K.q8[slot].rot)
            k_kv_store_rot<<<dim3(NKV_C, C, 2), 128, 0, K.st>>>(P.kcv, P.vcv, P.pos, K.kc[slot], K.vc[slot], K.q8[slot]);
        else
            k_kv_store<<<(C * KVC + 255) / 256, 256, 0, K.st>>>(P.kcv, P.vcv, P.pos, !L.full, K.kc[slot], K.vc[slot], C, K.q8[slot]);
        attn_pf(K, L, slot, p0, C);
        deq8(K, L.wo, P.Wo);
        if (part == 0) {   // zip: xch_pf2's slices, the sum at the chunk's next step
            zx_start(K, P.ab, C, [&](int r0, int r1) { sgemm(K, P.Wo, N, QC, P.ao + (size_t) r0 * QC, QC, P.ab + (size_t) r0 * N, N, r1 - r0); });
            return;
        }
        if (xsplit > 1) {   // p19d
            xch_pf2(K, P.ab, C, [&](int r0, int r1) { sgemm(K, P.Wo, N, QC, P.ao + (size_t) r0 * QC, QC, P.ab + (size_t) r0 * N, N, r1 - r0); });
        } else {
            sgemm(K, P.Wo, N, QC, P.ao, QC, P.ab, N, C);
            xch_pf(K, P.ab, CN);
        }
        }   // part <= 0
        if (part == 1) zx_add(K);
        if (part < 0 || part == 1) {
        conv_pf(K, P.ab, N, K.sa[slot], L.a_pconv, nullptr, 1, 1.0f / sqrtf(NCONV + 1.0f), P.A, C);
        const int ip = slot < NPHYS ? slot : LS + (slot - NPHYS) % NCORE;
        if (ip % BLK == 0) {
            CK(cudaMemcpyAsync(P.B[K.nb++], P.cur, CN * 4, cudaMemcpyDeviceToDevice, K.st));
            CK(cudaMemcpyAsync(P.cur, P.A, CN * 4, cudaMemcpyDeviceToDevice, K.st));
        } else {
            k_add<<<(unsigned) ((CN + 255) / 256), 256, 0, K.st>>>(P.cur, P.cur, P.A, CN);
        }
        mix(K, states_pf(K, P.cur), L.ffn_rq, 1, P.F, C);
        const float* f = P.F;
        if (loop) {
            control_pf(K, L, 3, P.F, part == 1 && K.zc > 0 ? K.Z[K.zc].ms3 : P.MS + 3 * CN, P.F2, C);
            f = P.F2;
        }
        rmsC(f, L.ffn_norm, P.FN);
        sgemm(K, L.gate_inp, NE, N, P.FN, N, P.lg, NE, C);
        k_route<<<C, NE, 0, K.st>>>(P.lg, L.probs_b, wscale, P.topi, P.topw);
        if (part == 1) {   // zip
            experts_pf(K, L, C, false);
            zx_start(K, P.MO, C, [&](int r0, int r1) { moe_comb(K, L, r0, r1); });
            return;
        }
        experts_pf(K, L, C, xsplit <= 1);
        if (xsplit > 1)   // p19d
            xch_pf2(K, P.MO, C, [&](int r0, int r1) { moe_comb(K, L, r0, r1); });
        else
            xch_pf(K, P.MO, CN);
        }   // part < 0 || part == 1
        if (part == 2) zx_add(K);
        conv_pf(K, P.MO, N, K.sf[slot], L.f_pconv, nullptr, 1, 1.0f / sqrtf(NCONV + 1.0f), P.F2, C);
        k_add<<<(unsigned) ((CN + 255) / 256), 256, 0, K.st>>>(P.cur, P.cur, P.F2, CN);
        if ((ds_on || d2_on) && ds_feat_slot(slot) >= 0) ds_feat(K, K.D.pfeat, P.cur, ds_feat_slot(slot), C);   // p6 (p74)
    }
    // a chunk of C tokens at positions p0.. through both cards; the last row's logits into h_logits when last
    // zip (INK2_PF_ZIP): pf_card's work for consecutive chunks at once, in steps that end where an exchange is on its way: the
    // first chunk's step, the next chunk's same step, ... then everyone's next step. A later chunk's layer k comes after the
    // earlier chunks' layer k (their cache rows and convolution states), and each chunk's own steps keep pf_card's order, so
    // every value is the one pf_card gives chunk after chunk
    struct PfRun { int C, p0; bool last; int nprel; const float* hx0; const int* hpos; const float* htau; const char* hraw; };
    static constexpr int NLAY = LS + NPASS * NCORE + (NPHYS - LE);   // a chunk's layers in order (= NSLOT)
    static void pf_layer_of(int k, int& p, int& slot, bool& loop) {
        if (k < LS) {
            p = slot = k;
            loop = false;
        } else if (k < LS + NPASS * NCORE) {
            const int pass = (k - LS) / NCORE, ip = LS + (k - LS) % NCORE;
            p = ip;
            slot = pass == 0 ? ip : NPHYS + (pass - 1) * NCORE + (ip - LS);
            loop = pass > 0;
        } else {
            p = slot = LE + (k - LS - NPASS * NCORE);
            loop = false;
        }
    }
    void pf_step(int c, PfRun& R, int i) {   // step i of 0 .. 2 NLAY
        Card& K = cards[c];
        PFB& P = K.P;
        const int C = R.C;
        const size_t CN = (size_t) C * N;
        int p, slot;
        bool loop;
        if (i % 2 == 1) {
            pf_layer_of(i / 2, p, slot, loop);
            layer_pf(K, p, slot, loop, R.p0, C, 1);
            return;
        }
        if (i == 0) {   // pf_card's start
            zx_recv(K, (K.zc + 1) % K.zn);
            CK(cudaMemcpyAsync(P.x0, R.hx0, (size_t) C * N * 4, cudaMemcpyHostToDevice, K.st));
            CK(cudaMemcpyAsync(P.pos, R.hpos, C * 4, cudaMemcpyHostToDevice, K.st));
            CK(cudaMemcpyAsync(P.tau, R.htau, C * 4, cudaMemcpyHostToDevice, K.st));
            k_rms<<<C, 256, 0, K.st>>>(P.x0, N, K.tok_norm, P.cur, N, N, eps);
            raw_rows(K, P.x0, P.cur, C, R.hraw);
            K.nb = 0;
        } else {
            pf_layer_of(i / 2 - 1, p, slot, loop);
            layer_pf(K, p, slot, loop, R.p0, C, 2);
        }
        const int k = i / 2;
        if (k == NLAY) {
            pf_tail(c, C, R.p0, R.last);
            return;
        }
        if (k == LS) {   // pf_card's copies between the layers
            CK(cudaMemcpyAsync(P.prel, P.cur, CN * 4, cudaMemcpyDeviceToDevice, K.st));
            R.nprel = K.nb;
        } else if (k > LS && k < LS + NPASS * NCORE && (k - LS) % NCORE == 0) {
            CK(cudaMemcpyAsync(P.bankS, P.B[R.nprel + 1], CN * 4, cudaMemcpyDeviceToDevice, K.st));
            CK(cudaMemcpyAsync(P.bankO, P.cur, CN * 4, cudaMemcpyDeviceToDevice, K.st));
            CK(cudaMemcpyAsync(P.cur, P.prel, CN * 4, cudaMemcpyDeviceToDevice, K.st));
            K.nb = R.nprel;
        }
        pf_layer_of(k, p, slot, loop);
        layer_pf(K, p, slot, loop, R.p0, C, 0);
    }
    void pf_multi(int c, int n, const int* Cs, int p0, bool last) {   // n chunks of Cs[j] tokens from position p0 on
        Card& K = cards[c];
        CK(cudaSetDevice(K.dev));
        PfRun R[Card::ZMAX];
        K.zn = n;
        for (int j = 0, p = p0; j < n; p += Cs[j], ++j)
            R[j] = PfRun{Cs[j], p, last && j == n - 1, 0, j ? h_x0z[j] : h_x0, j ? h_posz[j] : h_pos, j ? h_tauz[j] : h_tau,
                         j ? h_rawz[j].data() : nullptr};
        for (int i = 0; i <= 2 * NLAY; ++i)
            for (int j = 0; j < n; ++j) {
                zip_load(K, j);
                pf_step(c, R[j], i);
                zip_save(K);
            }
        zip_load(K, 0);
        CK(cudaStreamSynchronize(K.st));
    }
    void pf_card(int c, int C, int p0, bool last) {
        Card& K = cards[c];
        PFB& P = K.P;
        CK(cudaSetDevice(K.dev));
        CK(cudaMemcpyAsync(P.x0, h_x0, (size_t) C * N * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(P.pos, h_pos, C * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(P.tau, h_tau, C * 4, cudaMemcpyHostToDevice, K.st));
        k_rms<<<C, 256, 0, K.st>>>(P.x0, N, K.tok_norm, P.cur, N, N, eps);
        raw_rows(K, P.x0, P.cur, C);   // p17
        K.nb = 0;
        const size_t CN = (size_t) C * N;
        for (int p = 0; p < LS; ++p) layer_pf(K, p, p, false, p0, C);
        CK(cudaMemcpyAsync(P.prel, P.cur, CN * 4, cudaMemcpyDeviceToDevice, K.st));
        const int nprel = K.nb;
        for (int pass = 0; pass < NPASS; ++pass) {
            if (pass > 0) {
                CK(cudaMemcpyAsync(P.bankS, P.B[nprel + 1], CN * 4, cudaMemcpyDeviceToDevice, K.st));
                CK(cudaMemcpyAsync(P.bankO, P.cur, CN * 4, cudaMemcpyDeviceToDevice, K.st));
                CK(cudaMemcpyAsync(P.cur, P.prel, CN * 4, cudaMemcpyDeviceToDevice, K.st));
                K.nb = nprel;
            }
            for (int ip = LS; ip < LE; ++ip) layer_pf(K, ip, pass == 0 ? ip : NPHYS + (pass - 1) * NCORE + (ip - LS), pass > 0, p0, C);
        }
        for (int p = LE; p < NPHYS; ++p) layer_pf(K, p, p, false, p0, C);
        pf_tail(c, C, p0, last);
        CK(cudaStreamSynchronize(K.st));
    }
    void pf_tail(int c, int C, int p0, bool last) {   // zip: pf_card's end as a function of its own (a chunk's last step too)
        Card& K = cards[c];
        PFB& P = K.P;
        if (ds_on || d2_on) {   // p6: every row's final hidden, then the drafter's caches
            mix(K, states_pf(K, P.cur), K.out_rq, 1, P.F, C);
            k_rms<<<C, 256, 0, K.st>>>(P.F, N, K.out_norm, P.F2, N, N, eps);
            ds_feat(K, K.D.pfeat, P.F2, 3, C);
            if (ds_on) ds_ingest(K, K.D.pfeat, K.D.pfrnn, K.D.pcbuf, K.D.pkb, K.D.pvb, P.pos, C, true);
            else {   // p74: only the rows the rings keep after this feed() (its last D2WIN + DBLK), and at most D2RING of the chunk
                const int r0 = std::min(C, std::max(0, std::max(d2_from - p0, C - D2RING)));
                if (r0 < C) d2_ingest(K, K.D.pfeat + (size_t) r0 * 4 * N, P.pos + r0, C - r0, true);
            }
            if (fdump && c == 0) {   // p43: [C][4][N] as f16 to the dump file
                const size_t nv = (size_t) C * 4 * N;
                if (!fd_dev) {
                    CK(cudaMalloc((void**) &fd_dev, (size_t) pfc * 4 * N * 2));
                    CK(cudaHostAlloc((void**) &fd_host, (size_t) pfc * 4 * N * 2, cudaHostAllocPortable));
                }
                k_f2h43<<<(unsigned) ((nv + 255) / 256), 256, 0, K.st>>>(K.D.pfeat, fd_dev, nv);
                CK(cudaMemcpyAsync(fd_host, fd_dev, nv * 2, cudaMemcpyDeviceToHost, K.st));
                CK(cudaStreamSynchronize(K.st));
                if (std::fwrite(fd_host, 2, nv, fdump) != nv) throw std::runtime_error("featdump: short write");
            }
        }
        if (last) {   // the last row only: the depth mix, the norm and the head
            States S{};
            for (int i = 0; i < K.nb; ++i) S.p[i] = P.B[i] + (size_t) (C - 1) * N;
            S.p[K.nb] = P.cur + (size_t) (C - 1) * N;
            S.n = K.nb + 1;
            mix(K, S, K.out_rq, 1, P.F, 1);
            rms(K, P.F, K.out_norm, P.hx1, 1, eps);
            gemv8_t1(K, K.out, P.hx1, segs({{NV_C, K.logits}}));
            k_softcap_g<<<(NV_C + 255) / 256, 256, 0, K.st>>>(K.logits, NV_C, 1, softcap, s_lnminp < 0.0f ? 0.0f : s_invT, s_seed, p0 + C,
                                                           c * NV_C);   // p36 (p76: with min-p the host adds the noise)
            row_raw = s_invT > 0.0f && s_lnminp < 0.0f;   // p76
            CK(cudaMemcpyAsync(h_logits + (size_t) c * NV_C, K.logits, (size_t) NV_C * 4, cudaMemcpyDeviceToHost, K.st));
        }
    }

    // T tokens at positions n_past.. through both cards; logits of the last row (or of all rows) into h_logits
    void forward_card(int c, int T, int nparts, bool all_rows) {
        Card& K = cards[c];
        CK(cudaSetDevice(K.dev));
        if (T == 1 && !all_rows && !dumping && graf) {   // p4c: replay (or first capture) this parts bucket's graph
            auto it = K.graphs.find(nparts);
            if (it == K.graphs.end()) {
                CK(cudaStreamBeginCapture(K.st, cudaStreamCaptureModeThreadLocal));
                forward_body(c, T, nparts, all_rows);
                cudaGraph_t gr;
                CK(cudaStreamEndCapture(K.st, &gr));
                cudaGraphExec_t ex;
                CK(cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
                CK(cudaGraphDestroy(gr));
                it = K.graphs.emplace(nparts, ex).first;
            }
            CK(cudaGraphLaunch(it->second, K.st));
        } else {
            forward_body(c, T, nparts, all_rows);
        }
        if (dev_amax && want_logits)   // p47
            CK(cudaMemcpy2DAsync(h_logits + (size_t) c * NV_C, (size_t) NV * 4, K.logits, (size_t) NV_C * 4, (size_t) NV_C * 4,
                                 all_rows ? T : 1, cudaMemcpyDeviceToHost, K.st));
        CK(cudaStreamSynchronize(K.st));
    }
    void forward_body(int c, int T, int nparts, bool all_rows) {
        Card& K = cards[c];
        K.xidx = 0;
        k_tick<<<1, 1, 0, K.st>>>(K.dtok);
        CK(cudaMemcpyAsync(K.x0, h_x0, (size_t) T * N * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(K.pos, h_pos, T * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(K.tau, h_tau, T * 4, cudaMemcpyHostToDevice, K.st));
        rms(K, K.x0, K.tok_norm, K.cur, T, eps);
        raw_rows(K, K.x0, K.cur, T);   // p17
        K.nb = 0;
        for (int p = 0; p < LS; ++p) layer(K, p, p, false, T, nparts);
        copy(K, K.prel, K.cur, T);
        const int nprel = K.nb;
        for (int pass = 0; pass < NPASS; ++pass) {
            if (pass > 0) {
                copy(K, K.bankS, K.B[nprel + 1], T);
                copy(K, K.bankO, K.cur, T);
                copy(K, K.cur, K.prel, T);
                K.nb = nprel;
            }
            for (int ip = LS; ip < LE; ++ip) {
                const int slot = pass == 0 ? ip : NPHYS + (pass - 1) * NCORE + (ip - LS);
                layer(K, ip, slot, pass > 0, T, nparts);
            }
        }
        for (int p = LE; p < NPHYS; ++p) layer(K, p, p, false, T, nparts);
        mix(K, states(K, K.cur), K.out_rq, 1, K.F, T);
        rms(K, K.F, K.out_norm, K.hx, T, eps);
        if (ds_on) {   // p6: the final hidden is feature 3; the drafter's caches get every row
            ds_feat(K, K.D.feat, K.hx, 3, T);
            ds_ingest(K, K.D.feat, K.D.frnn, K.D.cbuf, K.D.kb, K.D.vb, K.pos, T, false);
        } else if (d2_on) {   // p74
            ds_feat(K, K.D.feat, K.hx, 3, T);
            d2_ingest(K, K.D.feat, K.pos, T, false);
        }
        const int r0 = all_rows ? 0 : T - 1, nr = T - r0;
        if (nr == 1 && !stare) gemv8_t1(K, K.out, K.hx + (size_t) r0 * N, segs({{NV_C, K.logits}}));
        else if (fuse && !stare && !dumping) gemv8_t1T(K, K.out, K.hx + (size_t) r0 * N, N, segs1(NV_C, K.logits), nr);   // p6c
        else gemv8(K, K.out, K.hx + (size_t) r0 * N, N, K.logits, NV_C, nr);
        row_raw = false;   // p76
        if (s_invT > 0.0f && s_lnminp < 0.0f) {   // p76: min-p over both cards' halves of each row
            k_softcap_g<<<(nr * NV_C + 255) / 256, 256, 0, K.st>>>(K.logits, NV_C, nr, softcap, 0.0f, 0, 0, 0);
            k_rowmax2<<<nr, 1024, 0, K.st>>>(K.logits, NV_C, std::min(NV_C, NVU - c * NV_C), K.mx2, c);
            xch(K, K.mx2, 2 * nr);
            k_gumbel_minp<<<(nr * NV_C + 255) / 256, 256, 0, K.st>>>(K.logits, NV_C, nr, s_invT, s_seed, n_past + r0 + 1, c * NV_C, K.mx2,
                                                                       NC, s_lnminp);
        } else
        k_softcap_g<<<(nr * NV_C + 255) / 256, 256, 0, K.st>>>(K.logits, NV_C, nr, softcap, s_invT, s_seed, n_past + r0 + 1, c * NV_C);   // p36
        if (dev_amax) {   // p47: the rows' argmax here; the full rows (want_logits) are copied by forward_card, outside a graph
            const int valid = std::min(NV_C, NVU - c * NV_C);
            for (int r = 0; r < nr; ++r)
                k_argmax_n<<<1, 1024, 0, K.st>>>(K.logits + (size_t) r * NV_C, valid, K.amv_r + r, K.ami_r + r);
            CK(cudaMemcpyAsync(h_amv2 + (size_t) c * MAXT, K.amv_r, nr * 4, cudaMemcpyDeviceToHost, K.st));
            CK(cudaMemcpyAsync(h_ami2 + (size_t) c * MAXT, K.ami_r, nr * 4, cudaMemcpyDeviceToHost, K.st));
        } else {
        CK(cudaMemcpy2DAsync(h_logits + (size_t) c * NV_C, (size_t) NV * 4, K.logits, (size_t) NV_C * 4, (size_t) NV_C * 4, nr,
                             cudaMemcpyDeviceToHost, K.st));
        }
    }
    // p17: pictures. A request's image rows (ink2_vis: mtmd's projector output, n_embd f32 a row) and where they go: token ids
    // from IMG_BASE name rows (img_map: first id, rows, offset into img_rows); the server makes the ids from the picture's
    // hash, so the conversation cache never mixes two pictures. Image rows skip token_embd_norm, as in llama.cpp
    // ("do not normalize raw embeddings input"): h_raw marks them, raw_rows() puts them back after the norm.
    static constexpr int IMG_BASE = 1 << 28;
    std::vector<float> img_rows;
    std::vector<std::array<int, 3>> img_map;
    std::vector<char> h_raw = std::vector<char>((size_t) 1 << 16, 0);   // rows of a window or a reading chunk
    void raw_rows(Card& K, const float* x0, float* cur, int T, const char* raw = nullptr) {   // zip: raw = a later chunk's marks
        const char* h_raw = raw ? raw : this->h_raw.data();
        for (int t = 0; t < T;) {
            if (!h_raw[t]) { ++t; continue; }
            int e = t;
            while (e < T && h_raw[e]) ++e;
            CK(cudaMemcpyAsync(cur + (size_t) t * N, x0 + (size_t) t * N, (size_t) (e - t) * N * 4, cudaMemcpyDeviceToDevice, K.st));
            t = e;
        }
    }
    bool embed(int t, int id) {
        if (id >= IMG_BASE) {
            for (const auto& m : img_map)
                if (id >= m[0] && id < m[0] + m[1]) {
                    std::memcpy(h_x0 + (size_t) t * N, img_rows.data() + ((size_t) m[2] + (id - m[0])) * N, (size_t) N * 4);
                    return true;
                }
            throw std::runtime_error("image row without data (id " + std::to_string(id) + ")");
        }
        if (id < 0 || id >= NV) throw std::runtime_error("token id out of range: " + std::to_string(id));
        const size_t rb = tok->row_bytes();
        const uint8_t* row = tok->data + (size_t) id * rb;
        for (int b = 0; b < N / 32; ++b) {
            const uint8_t* blk = row + (size_t) b * 34;
            uint16_t hb;
            std::memcpy(&hb, blk, 2);
            __half_raw hr;
            hr.x = hb;
            const float d = __half2float(__half(hr));
            for (int j = 0; j < 32; ++j) h_x0[(size_t) t * N + b * 32 + j] = d * (float) (int8_t) blk[2 + j];
        }
        return false;
    }
    // feeds tokens (windows of MAXT); returns the logits rows of the last window (all its rows when all_rows)
    int feed(const int* ids, int n, bool all_rows, const std::function<bool()>* cancelled = nullptr) {
        int T = 0;
        d2_from = n_past + n - (D2WIN + DBLK);   // p74
        bool dropped = false;   // p63
        for (int i = 0; i < n; i += T) {
            if (cancelled && (*cancelled)()) throw PrefillCancelled{};
            if (!all_rows && (n - i >= pf_min || force_pf)) {   // p3: the reading path, chunks of up to pfc (p43: always while dumping)
                if (share1c && !dropped) {   // p63: the cache's slots in the reading path's buffers are emptied first
                    dropped = true;
                    Card& K = cards[0];
                    CK(cudaSetDevice(K.dev));
                    CK(cudaStreamSynchronize(K.st));   // the staging copies (their own stream) overwrite slots only after the last write read them
                    for (int lp = 0; lp < NPHYS; ++lp)
                        if (K.L[lp].ec.S > K.L[lp].ec.S0)
                            k_cache_drop<<<1, 64, 0, K.st>>>(K.L[lp].ec, (const uint8_t**) K.L[lp].gu_t, (const uint8_t**) K.L[lp].dn_t);
                }
                T = std::min(pfc, n - i);
                if (n_past + T > ctx) throw std::runtime_error("context full");
                for (int t = 0; t < T; ++t) {
                    h_raw[t] = embed(t, ids[i + t]);
                    const int P = n_past + t;
                    h_pos[t] = P;
                    h_tau[t] = logf(fmaxf(((float) P + 1.0f) * (1.0f / log_floor), 1.0f)) * log_alpha + 1.0f;
                }
                const bool last = i + T >= n;
                const int p0 = n_past;
                int Cs[Card::ZMAX] = {T}, nz = 1, tot = T;   // zip: the next chunks, as this loop would cut them
                while (nz < zip && Cs[nz - 1] == pfc && n - i - tot >= pf_min) {
                    const int Tz = std::min(pfc, n - i - tot);
                    if (n_past + tot + Tz > ctx) break;   // the loop throws "context full" when it comes to this chunk, as ever
                    bool ok = true;
                    std::swap(h_x0, h_x0z[nz]);   // embed() writes h_x0
                    try {
                        for (int t = 0; t < Tz; ++t) {
                            h_rawz[nz][t] = embed(t, ids[i + tot + t]);
                            const int P = n_past + tot + t;
                            h_posz[nz][t] = P;
                            h_tauz[nz][t] = logf(fmaxf(((float) P + 1.0f) * (1.0f / log_floor), 1.0f)) * log_alpha + 1.0f;
                        }
                    } catch (const std::exception&) {
                        ok = false;   // a token embed() refuses: the loop meets it again when it comes to this chunk
                    }
                    std::swap(h_x0, h_x0z[nz]);
                    if (!ok) break;
                    Cs[nz++] = Tz;
                    tot += Tz;
                }
                if (nz > 1) {
                    const bool lastz = i + tot >= n;
                    both([&](int c) { pf_multi(c, nz, Cs, p0, lastz); });
                    T = tot;
                    n_past += T;
                    processed.insert(processed.end(), ids + i, ids + i + T);
                    continue;
                }
                both([&](int c) { pf_card(c, T, p0, last); });
                n_past += T;
                processed.insert(processed.end(), ids + i, ids + i + T);
                continue;
            }
            T = std::min(all_rows ? MAXT : feed_t, n - i);   // p23b: a draft window (verify, --logits-all) is ONE window of up to MAXT rows; feed_t is for reading short pieces only
            if (n_past + T > ctx) throw std::runtime_error("context full");
            for (int t = 0; t < T; ++t) {
                h_raw[t] = embed(t, ids[i + t]);
                const int P = n_past + t;
                h_pos[t] = P;
                h_tau[t] = logf(fmaxf(((float) P + 1.0f) * (1.0f / log_floor), 1.0f)) * log_alpha + 1.0f;
            }
            int nparts = (n_past + T + KC - 1) / KC + 1;
            if (T == 1 && graf) {   // p4c: graphs by buckets of parts (empty parts return at once)
                int b = 4;
                while (b < nparts) b <<= 1;
                nparts = std::min(b, cards[0].nparts_max);
            }
            const bool last = i + T >= n;
            both([&](int c) { forward_card(c, T, nparts, all_rows && last); });
            n_past += T;
            processed.insert(processed.end(), ids + i, ids + i + T);
        }
        return all_rows ? T : 1;
    }
    // p8: the states before the last prompt token, so a request that repeats the prompt (a conversation's next turn re-renders
    // the reply differently, e.g. without its reasoning) reads only from there. The caches need nothing: rows of later positions
    // are written again, earlier rows stay while the window ring was not overrun (RING - SWA - MAXT positions since)
    void snapshot() {
        both([&](int c) {
            Card& K = cards[c];
            CK(cudaSetDevice(K.dev));
            CK(cudaMemcpyAsync(K.stsnap, K.stbuf, (size_t) NSLOT * SLOTST * 4, cudaMemcpyDeviceToDevice, K.st));
            CK(cudaStreamSynchronize(K.st));
        });
        snap_n = n_past;
    }
    bool restore(size_t same) {
        if (snap_n <= 0 || (int) same < snap_n || (int) processed.size() < snap_n || n_past - snap_n > RING - SWA - MAXT) return false;
        both([&](int c) {
            Card& K = cards[c];
            CK(cudaSetDevice(K.dev));
            CK(cudaMemcpyAsync(K.stbuf, K.stsnap, (size_t) NSLOT * SLOTST * 4, cudaMemcpyDeviceToDevice, K.st));
            CK(cudaStreamSynchronize(K.st));
        });
        n_past = snap_n;
        processed.resize(snap_n);
        return true;
    }
    // p21: the checkpoints
    bool win_slot(int s) const { return !cards[0].L[s < NPHYS ? s : LS + (s - NPHYS) % NCORE].full; }
    size_t ck_bytes() const {
        size_t b = (size_t) NSLOT * SLOTST * 4;
        for (int s = 0; s < NSLOT; ++s)
            if (win_slot(s)) b += (size_t) 2 * SWA * KVC * 2;
        if (ds_on && ds_ring) b += (size_t) (2 * DS_TAIL + DS_WIN1) * NKVD * 2 * 2;
        if (d2_on) b += (size_t) D2L * 2 * D2WIN * NKVD * 2;   // p74
        return b;
    }
    // card c's state at position pos to RAM (save) or back: the conv states, then of every cache kept by position modulo a ring
    // the rows of positions [pos - W, pos) (the ones below 0 are skipped; they are never read unmasked)
    void ck_copy(int c, char* h, int pos, bool save) {
        Card& K = cards[c];
        CK(cudaSetDevice(K.dev));
        auto mv = [&](void* d, char* hp, size_t n) {
            CK(save ? cudaMemcpyAsync(hp, d, n, cudaMemcpyDeviceToHost, K.st) : cudaMemcpyAsync(d, hp, n, cudaMemcpyHostToDevice, K.st));
        };
        auto rows = [&](void* dev, int W, int ring, size_t rb) {
            for (int q = std::max(0, pos - W); q < pos;) {
                const int r = q % ring, n = std::min(pos - q, ring - r);
                mv((char*) dev + (size_t) r * rb, h + (size_t) (q - (pos - W)) * rb, (size_t) n * rb);
                q += n;
            }
            h += (size_t) W * rb;
        };
        mv(K.stbuf, h, (size_t) NSLOT * SLOTST * 4);
        h += (size_t) NSLOT * SLOTST * 4;
        for (int s = 0; s < NSLOT; ++s)
            if (win_slot(s)) {
                rows(K.kc[s], SWA, RING, (size_t) KVC * 2);
                rows(K.vc[s], SWA, RING, (size_t) KVC * 2);
            }
        if (ds_on && ds_ring)
            for (int il = 0; il < DSL; ++il) {
                const int W = il == 1 ? DS_WIN1 : DS_TAIL, rg = il == 1 ? DS_RING1 : DS_RING;
                rows(K.D.kc[il], W, rg, (size_t) NKVD * 2);
                rows(K.D.vc[il], W, rg, (size_t) NKVD * 2);
            }
        if (d2_on)   // p74: the rings' last D2WIN rows
            for (int il = 0; il < D2L; ++il) {
                rows(K.D2.kc[il], D2WIN, D2RING, (size_t) NKVD * 2);
                rows(K.D2.vc[il], D2WIN, D2RING, (size_t) NKVD * 2);
            }
        CK(cudaStreamSynchronize(K.st));
    }
    // p55: one spare checkpoint buffer in ck_pool, made while the engine waits (at start and after a reply), not before a first token
    void ck_spare() {
        if (!ckspare || !ck_on || ck_max <= 0 || !ck_pool.empty() || ck_made >= ck_max) return;
        Ckpt cp;
        for (int c = 0; c < NC; ++c) CK(cudaHostAlloc((void**) &cp.buf[c], ck_bytes(), cudaHostAllocPortable));
        ++ck_made;
        ck_pool.push_back(cp);
    }
    // a checkpoint of the state now (n_past tokens). With no free buffer the closest pair gives up its later one (never the newest)
    void ck_take() {
        if (!ck_on || ck_max <= 0 || n_past <= 0) return;
        const int pos = n_past;
        auto by_pos = [](const Ckpt& a, int q) { return a.pos < q; };
        auto it = std::lower_bound(ckpts.begin(), ckpts.end(), pos, by_pos);
        Ckpt cp;   // (not `ck`: that is CK()'s function)
        if (it != ckpts.end() && it->pos == pos) {
            cp = *it;
            ckpts.erase(it);
        } else if (!ck_pool.empty()) {
            cp = ck_pool.back();
            ck_pool.pop_back();
        } else if (ck_made < ck_max) {
            for (int c = 0; c < NC; ++c) CK(cudaHostAlloc((void**) &cp.buf[c], ck_bytes(), cudaHostAllocPortable));
            ++ck_made;
        } else if (!parks.empty()) {   // p21b: the least recently used parked conversation gives one up (its last one last)
            const size_t i = park_lru();
            if (parks[i].cks.size() > 1) {
                cp = parks[i].cks.front();
                parks[i].cks.erase(parks[i].cks.begin());
            } else {
                park_drop(i);
                if (ck_pool.empty()) return;
                cp = ck_pool.back();
                ck_pool.pop_back();
            }
        } else if (!ckpts.empty()) {
            size_t v = 0;
            int best = INT_MAX;
            for (size_t i = 0; i + 1 < ckpts.size(); ++i) {
                const int gap = ckpts[i].pos - (i ? ckpts[i - 1].pos : 0);
                if (gap < best) {
                    best = gap;
                    v = i;
                }
            }
            cp = ckpts[v];
            ckpts.erase(ckpts.begin() + (long) v);
        } else {
            return;
        }
        cp.pos = pos;
        both([&](int c) { ck_copy(c, cp.buf[c], pos, true); });
        ckpts.insert(std::lower_bound(ckpts.begin(), ckpts.end(), pos, by_pos), cp);
    }
    // the latest checkpoint at or before limit becomes the state; later ones are dropped. -1: there is none (the caller starts over)
    int ck_back(int limit) {
        while (!ckpts.empty() && ckpts.back().pos > limit) {
            ck_pool.push_back(ckpts.back());
            ckpts.pop_back();
        }
        if (ckpts.empty() || (int) processed.size() < ckpts.back().pos) return -1;
        const Ckpt cp = ckpts.back();
        both([&](int c) { ck_copy(c, cp.buf[c], cp.pos, false); });
        n_past = cp.pos;
        processed.resize((size_t) cp.pos);
        return cp.pos;
    }
    // p21b: parked conversations
    static int ck_at(const std::vector<Ckpt>& v, int limit) {   // the latest checkpoint's position at or before limit; 0: none
        int u = 0;
        for (const Ckpt& k : v)
            if (k.pos <= limit) u = std::max(u, k.pos);
        return u;
    }
    // the arrays with card c's rows that are kept by position: the full slots' caches (f16; in the int8 mode rows from f0 on in
    // int8 with their scales) and the drafter's caches when they are no rings
    std::vector<KvArr> kv_arrays(int c) {
        Card& K = cards[c];
        std::vector<KvArr> a;
        for (int s = 0; s < NSLOT; ++s) {
            if (win_slot(s)) continue;
            const KV8& q = K.q8[s];
            const int fk = q.k ? q.f0 : ctx, fv = q.v ? q.f0 : ctx;
            a.push_back({(char*) K.kc[s], (size_t) KVC * 2, 0, fk});
            if (q.k) {
                a.push_back({(char*) q.k, (size_t) KVC, fk, ctx});
                a.push_back({(char*) q.ks, (size_t) KQN * 2, fk, ctx});
            }
            a.push_back({(char*) K.vc[s], (size_t) KVC * 2, 0, fv});
            if (q.v) {
                a.push_back({(char*) q.v, (size_t) KVC, fv, ctx});
                a.push_back({(char*) q.vs, (size_t) KQN * 2, fv, ctx});
            }
        }
        if (ds_on && !ds_ring)
            for (int il = 0; il < DSL; ++il) {
                a.push_back({(char*) K.D.kc[il], (size_t) NKVD * 2, 0, ctx});
                a.push_back({(char*) K.D.vc[il], (size_t) NKVD * 2, 0, ctx});
            }
        return a;
    }
    static size_t kv_rows(const KvArr& x, int pos) { return (size_t) std::max(0, std::min(pos, x.r1) - x.r0); }
    size_t kv_bytes(int c, int pos) {
        size_t b = 0;
        for (const KvArr& x : kv_arrays(c)) b += x.rb * kv_rows(x, pos);
        return b;
    }
    // card c's rows of positions [0, upto) to RAM (save) or back; the RAM buffer is laid out for positions [0, lay)
    void kv_copy(int c, char* h, int lay, int upto, bool save) {
        Card& K = cards[c];
        CK(cudaSetDevice(K.dev));
        CK(cudaStreamSynchronize(K.st));
        for (const KvArr& x : kv_arrays(c)) {
            const size_t n = x.rb * kv_rows(x, std::min(upto, lay));
            if (n) CK(save ? cudaMemcpy(h, x.p, n, cudaMemcpyDeviceToHost) : cudaMemcpy(x.p, h, n, cudaMemcpyHostToDevice));
            h += x.rb * kv_rows(x, lay);
        }
    }
    size_t park_lru() const {
        size_t v = 0;
        for (size_t i = 1; i < parks.size(); ++i)
            if (parks[i].use < parks[v].use) v = i;
        return v;
    }
    void park_drop(size_t i) {
        for (int c = 0; c < NC; ++c) std::free(parks[i].kv[c]);
        for (const Ckpt& k : parks[i].cks) ck_pool.push_back(k);
        park_bytes -= parks[i].bytes;
        parks.erase(parks.begin() + (long) i);
    }
    // the conversation on the cards into RAM: its rows up to its last checkpoint, and the checkpoints after `above` (the ones up
    // to there stay: the request goes back to one of them). False: too short, or more than the room there is
    bool park_live(int above) {
        if (park_max == 0 || ckpts.empty()) return false;
        const int pos = ckpts.back().pos;
        if (pos - above < park_min || (int) processed.size() < pos) return false;
        size_t need[NC], bytes = 0;
        for (int c = 0; c < NC; ++c) bytes += need[c] = kv_bytes(c, pos);
        if (bytes > park_max) return false;
        while (park_bytes + bytes > park_max && !parks.empty()) park_drop(park_lru());
        Park pk;
        for (int c = 0; c < NC; ++c)
            if (!(pk.kv[c] = (char*) std::malloc(need[c]))) {
                for (int d = 0; d < c; ++d) std::free(pk.kv[d]);
                return false;
            }
        both([&](int c) { kv_copy(c, pk.kv[c], pos, pos, true); });
        pk.tok.assign(processed.begin(), processed.begin() + pos);
        std::vector<Ckpt> mine;
        while (!ckpts.empty() && ckpts.back().pos > above) {
            mine.insert(mine.begin(), ckpts.back());
            ckpts.pop_back();
        }
        for (size_t i = 0; i < mine.size(); ++i)   // its first and its last PARK_CK - 1
            (i == 0 || i + (PARK_CK - 1) >= mine.size() ? pk.cks : ck_pool).push_back(mine[i]);
        pk.bytes = bytes;
        pk.use = ++use_no;
        park_bytes += bytes;
        parks.push_back(pk);
        return true;
    }
    // parked conversation i onto the cards at its checkpoint u (the caller parked or gave up the one that was there). whole: the
    // prompt goes on from its end, so it moves back; else it stays parked and the cards get a copy of its rows up to u
    void park_load(size_t i, int u, bool whole) {
        for (const Ckpt& k : ckpts) ck_pool.push_back(k);
        ckpts.clear();
        const int lay = (int) parks[i].tok.size();
        const Park pk = parks[i];
        both([&](int c) { kv_copy(c, pk.kv[c], lay, u, false); });
        if (whole) {
            processed = pk.tok;
            ckpts = pk.cks;
            for (int c = 0; c < NC; ++c) std::free(pk.kv[c]);
            park_bytes -= pk.bytes;
            parks.erase(parks.begin() + (long) i);
            if (ck_back(u) != u) throw std::runtime_error("park_load: no checkpoint");
        } else {
            const Ckpt* src = nullptr;
            for (const Ckpt& k : pk.cks)
                if (k.pos == u) src = &k;
            if (!src) throw std::runtime_error("park_load: no checkpoint");
            processed.assign(pk.tok.begin(), pk.tok.begin() + u);
            both([&](int c) { ck_copy(c, src->buf[c], u, false); });
            n_past = u;
            parks[i].use = ++use_no;
            ck_take();   // the cards' own checkpoint there
        }
    }
    void reset() {
        n_past = 0;
        snap_n = -1;
        processed.clear();
        for (const Ckpt& k : ckpts) ck_pool.push_back(k);   // p21
        ckpts.clear();
        both([&](int c) {
            Card& K = cards[c];
            CK(cudaSetDevice(K.dev));
            CK(cudaMemsetAsync(K.stbuf, 0, (size_t) NSLOT * SLOTST * 4, K.st));
            CK(cudaStreamSynchronize(K.st));
        });
    }
    // p5a: tokens win[0..T) through the writing path with every row's logits; win[0] is the pending token, win[1..] drafts. Returns
    // how many drafts were right (row i's argmax == win[i + 1]); the caches then hold win[0..keep] (keep = right + 1) and the
    // convolutions' states are set back to that point. out_next = the argmax after the last kept token
    int verify(const int* win, int T, int& out_next) {
        both([&](int c) {
            Card& K = cards[c];
            CK(cudaSetDevice(K.dev));
            CK(cudaMemcpyAsync(K.stbak, K.stbuf, (size_t) NSLOT * SLOTST * 4, cudaMemcpyDeviceToDevice, K.st));
            K.keep_hist = true;
        });
        feed(win, T, true);
        int right = 0;
        while (right + 1 < T && amax_row(right) == win[right + 1]) ++right;
        out_next = amax_row(right);
        const int keep = right + 1;
        if (ds_on && !dev_amax)   // p6: the drafter's context is the last kept row's f_rnn (the rejected rows' caches are rewritten later)
            both([&](int c) {
                Card& K = cards[c];
                CK(cudaSetDevice(K.dev));
                CK(cudaMemcpyAsync(K.D.ctxv, K.D.frnn + (size_t) right * N, N * 4, cudaMemcpyDeviceToDevice, K.st));
                CK(cudaStreamSynchronize(K.st));
            });
        both([&](int c) {
            Card& K = cards[c];
            CK(cudaSetDevice(K.dev));
            if (ds_on && dev_amax)   // p47: the same copy, in the closing call (one sync instead of two)
                CK(cudaMemcpyAsync(K.D.ctxv, K.D.frnn + (size_t) right * N, N * 4, cudaMemcpyDeviceToDevice, K.st));
            K.keep_hist = false;
            if (keep < T) {
                const size_t n = (size_t) NSLOT * SLOTST;
                k_conv_roll<<<(unsigned) ((n + 255) / 256), 256, 0, K.st>>>(K.stbuf, K.stbak, K.hist, keep);
            }
            CK(cudaStreamSynchronize(K.st));
        });
        n_past -= T - keep;
        processed.resize(processed.size() - (T - keep));
        return right;
    }
    static int argmax(const float* l) {
        int b = 0;
        for (int i = 1; i < NVU; ++i)
            if (l[i] > l[b]) b = i;
        return b;
    }
    // p47: row r's token: from the cards' argmax (card 0's ids are the smaller: it wins a tie) or from the copied logits
    // p76: the host's sample of a full row of softcapped logits without noise (the reading path's last row with min-p), the same
    // noise as k_gumbel_minp: the token predicting position pos
    static unsigned long long mix64h(unsigned long long z) {
        z += 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    int sample_row(const float* l, int pos) const {
        float m = -INFINITY;
        for (int j = 0; j < NVU; ++j) m = std::max(m, l[j]);
        const float thr = m + s_lnminp / s_invT;
        int best = 0;
        float bv = -INFINITY;
        for (int j = 0; j < NVU; ++j) {
            if (l[j] < thr) continue;
            const unsigned long long h = mix64h(s_seed ^ mix64h((unsigned long long) pos * 0x100000000ull + (unsigned long long) j));
            const float u = ((float) (h >> 40) + 0.5f) * (1.0f / 16777216.0f);
            const float v = l[j] * s_invT - logf(-logf(u));
            if (v > bv) { bv = v; best = j; }
        }
        return best;
    }
    int amax_row(int r) const {
        if (!dev_amax || want_logits) return argmax(h_logits + (size_t) r * NV);
        const float v0 = h_amv2[r], v1 = NC > 1 ? h_amv2[MAXT + r] : -INFINITY;
        return (NC == 1 || v0 >= v1) ? h_ami2[r] : h_ami2[MAXT + r] + NV_C;
    }
};

static void write_dump(const std::string& path, const std::vector<float>& v, int cols) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    const int32_t nd = 2;
    const int64_t ne[4] = {cols, (int64_t) (v.size() / cols), 1, 1};
    std::fwrite(&nd, 4, 1, f);
    std::fwrite(ne, 8, 4, f);
    std::fwrite(v.data(), 4, v.size(), f);
    std::fclose(f);
}

// --serve SOCKET: one JSON line a connection, {"tokens": [...], "max": N, "stop": [...], "noeos": 0} -> {"tokens": [...], "prompt_n", "cached",
// "read_s", "write_s"}. A prompt that continues exactly the tokens already processed reads only the rest; anything else starts over
static std::vector<long long> json_ints(const std::string& j, const std::string& key) {
    std::vector<long long> v;
    const size_t k = j.find("\"" + key + "\"");
    if (k == std::string::npos) return v;
    size_t p = j.find('[', k), e = j.find(']', p);
    if (p == std::string::npos || e == std::string::npos) return v;
    const char* c = j.c_str() + p + 1;
    const char* end = j.c_str() + e;
    while (c < end) {
        char* nx;
        const long long x = std::strtoll(c, &nx, 10);
        if (nx == c) { ++c; continue; }
        v.push_back(x);
        c = nx;
    }
    return v;
}
static std::string json_str(const std::string& j, const std::string& key) {   // p17: a plain string value (no escapes)
    const size_t k = j.find("\"" + key + "\"");
    if (k == std::string::npos) return "";
    const size_t a = j.find('"', j.find(':', k) + 1);
    const size_t b = a == std::string::npos ? a : j.find('"', a + 1);
    return a == std::string::npos || b == std::string::npos ? "" : j.substr(a + 1, b - a - 1);
}
static long long json_int(const std::string& j, const std::string& key, long long def) {
    const size_t k = j.find("\"" + key + "\"");
    if (k == std::string::npos) return def;
    const size_t p = j.find(':', k);
    return std::strtoll(j.c_str() + p + 1, nullptr, 10);
}

struct Gen {
    std::vector<int> out;
    int cached = 0, drafted = 0, accepted = 0, rounds = 0;
    double read_s = 0, write_s = 0, draft_s = 0, verify_s = 0;   // p6: time in the drafter and in the target's windows
    int park = 0;    // p21b: the request started from a parked conversation
    int back = -1;   // p21: the checkpoint the request went back to (-1: none), the time taking new ones, a hash of the first logits
    double ck_s = 0;
    unsigned long long lh = 0, lh2 = 1469598103934665603ULL;   // p22b: lh2 over every step's logits (g_lhall)
};
// p5a: draft tokens from the history: the latest earlier place where the last 3 tokens (then 2) occurred, and what followed it
// p7: the longest suffix of the history (3..NG_MAX tokens) that occurred before, at its most recent place, and the tokens that
// followed it there (p5a took the latest 3-token match: in code that is often a common trigram elsewhere). *len = match length
static constexpr int NG_MAX = 16;
static std::vector<int> ngram_draft(const std::vector<int>& h, int D, int* len = nullptr) {
    std::vector<int> d;
    if (len) *len = 0;
    const int n = (int) h.size();
    if (D <= 0 || n < 4) return d;
    int best = 0, at = -1;
    for (int p = n - 2; p >= 0 && best < NG_MAX; --p) {
        if (h[p] != h[n - 1]) continue;
        int l = 1;
        while (l < NG_MAX && p - l >= 0 && h[p - l] == h[n - 1 - l]) ++l;
        if (l > best) {
            best = l;
            at = p;
        }
    }
    if (best < 3) return d;
    if (len) *len = best;
    for (int i = at + 1; i < n && (int) d.size() < D && h[i] < NVU; ++i) d.push_back(h[i]);   // p17: not image rows
    return d;
}
// p15: on_out(new tokens) after each step when given (the server streams them); false stops the generation (the client left)
static bool g_lhall = false;   // p22b (tests): hash the logits of every step
static unsigned long long fnv_row(unsigned long long h, const float* row) {
    const unsigned* w = (const unsigned*) row;
    for (int i = 0; i < NVU; ++i) h = (h ^ w[i]) * 1099511628211ULL;
    return h;
}
static Gen generate(Engine& e, const std::vector<int>& tok, int max, const std::vector<int>& stop, int eos, int lookup = 0, int mtp = 0,
                    const std::function<bool(const int*, int)>* on_out = nullptr, const std::vector<int>* hints = nullptr,
                    const std::function<bool()>* cancelled = nullptr) {
    Gen g;
    e.want_logits = true;   // p47
    size_t sent = 0;
    auto flush = [&]() {
        if (!on_out || sent == g.out.size()) return true;
        const bool ok = (*on_out)(g.out.data() + sent, (int) (g.out.size() - sent));
        sent = g.out.size();
        return ok;
    };
    static const bool snap = getenv("INK2_SNAP") ? atoi(getenv("INK2_SNAP")) != 0 : true;   // p8
    size_t same = 0;
    while (same < tok.size() && same < e.processed.size() && tok[same] == e.processed[same]) ++same;
    const double t0 = now_s();
    const int n = (int) tok.size();
    if (same != e.processed.size() || same == tok.size()) {   // not a continuation (or nothing new to read)
        if (e.ck_on) {   // p21: back to the latest checkpoint the prompt still shares (one token is always read), or start over
            const int lim = (int) std::min(same, tok.size() - 1), u_live = Engine::ck_at(e.ckpts, lim);
            const int last = e.ckpts.empty() ? 0 : e.ckpts.back().pos;
            int best = -1, u_best = u_live;
            size_t same_best = 0;
            for (size_t i = 0; i < e.parks.size(); ++i) {   // p21b: a parked conversation that shares more
                const std::vector<int>& pt = e.parks[i].tok;
                size_t sp = 0;
                while (sp < tok.size() && sp < pt.size() && tok[sp] == pt[sp]) ++sp;
                const int u = Engine::ck_at(e.parks[i].cks, (int) std::min(sp, tok.size() - 1));
                if (u > u_best) {
                    u_best = u;
                    best = (int) i;
                    same_best = sp;
                }
            }
            if (best >= 0) {
                const std::vector<int> want = e.parks[(size_t) best].tok;   // park_live may drop others, never the newest... find it again
                e.parks[(size_t) best].use = ++e.use_no;
                e.park_live(0);
                size_t at = e.parks.size();
                for (size_t i = 0; i < e.parks.size(); ++i)
                    if (e.parks[i].tok == want) at = i;
                if (at < e.parks.size()) {
                    e.park_load(at, u_best, same_best + 64 >= want.size());
                    g.back = u_best;
                    g.park = 1;
                } else {
                    e.reset();
                }
            } else {
                if (u_live < last / 2) e.park_live(u_live);   // another conversation (less than half is shared): this one is set aside
                g.back = e.ck_back(lim);
                if (g.back < 0) e.reset();
            }
            same = (size_t) std::max(g.back, 0);
        } else {
            if (!(snap && e.restore(same))) {   // p8: back to the last prompt's snapshot, or start over
                e.reset();
                same = 0;
            }
            same = std::min(same, e.processed.size());
        }
    }
    g.cached = (int) same;
    if (e.ck_on) {
        // p21: checkpoints on the way: at the hints (where the prompt's messages start: the last one always, others when ck_gap
        // tokens past the stop before), every ck_every tokens of a long stretch, and before the last token (p8's place)
        std::vector<int> cand, stops;
        if (hints)
            for (int h : *hints)
                if (h > (int) same && h < n - 1) cand.push_back(h);
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
        int at = (int) same;
        for (size_t i = 0; i <= cand.size(); ++i) {
            const int h = i < cand.size() ? cand[i] : n - 1;
            while (h - at > e.ck_every) stops.push_back(at += e.ck_every);
            if (h > at && (i + 1 >= cand.size() || h - at >= e.ck_gap)) stops.push_back(at = h);   // the last hint and n - 1 always
        }
        at = (int) same;
        for (int st : stops) {
            e.feed(tok.data() + at, st - at, false, cancelled);
            at = st;
            const double tc = now_s();
            e.ck_take();
            g.ck_s += now_s() - tc;
        }
        e.feed(tok.data() + at, n - at, false, cancelled);
    } else if (snap && n - (int) same >= 2) {   // p8: all but the last token, the snapshot, then the last token
        e.feed(tok.data() + same, n - 1 - (int) same, false, cancelled);
        e.snapshot();
        e.feed(tok.data() + n - 1, 1, false, cancelled);
    } else {
        e.feed(tok.data() + same, n - (int) same, false, cancelled);
    }
    if (cancelled && (*cancelled)()) throw PrefillCancelled{};
    g.read_s = now_s() - t0;
    {   // p21: FNV-1a of the logits after the prompt (tests compare a read after going back with a read from the start)
        unsigned long long h = 1469598103934665603ULL;
        const unsigned* w = (const unsigned*) e.h_logits;
        for (int i = 0; i < NVU; ++i) h = (h ^ w[i]) * 1099511628211ULL;
        g.lh = h;
    }
    const double t1 = now_s();
    int next = e.row_raw ? e.sample_row(e.h_logits, e.n_past) : Engine::argmax(e.h_logits);   // p76: the reading path's raw row
    e.want_logits = !e.dev_amax || g_lhall;   // p47
    auto ends = [&](int t) { return t == eos || std::find(stop.begin(), stop.end(), t) != stop.end(); };
    int dcur = std::min(lookup, 4), cool = 0;   // p5a: the draft length follows the last window; after a miss no drafts for a while
    static const int ng_min = getenv("INK2_NG_MIN") ? atoi(getenv("INK2_NG_MIN")) : 6;   // p7: n-gram match that beats the drafter
    while ((int) g.out.size() < max) {
        g.out.push_back(next);
        if (!flush()) break;
        if (ends(next) || (int) g.out.size() >= max || e.n_past >= e.ctx) break;
        std::vector<int> hist = e.processed;
        hist.push_back(next);
        std::vector<int> d;
        if (lookup > 0) {   // n-grams first (p6d: with mtp too, where the history has a match; p7: with mtp only a match of ng_min)
            int ml = 0;
            if (cool > 0) --cool;
            else d = ngram_draft(hist, std::min(dcur, std::min(MAXT - 1, max - (int) g.out.size())), &ml);
            if (mtp > 0 && ml < ng_min) d.clear();
        }
        bool mtpd = false;
        const int p43_at = e.n_past;   // p43: the anchor's position
        if (d.empty() && mtp > 0) {   // p6: the DSpark drafter
            mtpd = true;
            const double td = now_s();
            d = e.d2_on ? e.d2_draft(next, std::min(mtp, max - (int) g.out.size())) : e.ds_draft(next, std::min(mtp, max - (int) g.out.size()));   // p74
            g.draft_s += now_s() - td;
        }
        if ((int) d.size() > e.ctx - e.n_past - 1) d.resize(std::max(0, e.ctx - e.n_past - 1));
        if (d.empty()) {
            e.feed(&next, 1, false);
            next = e.amax_row(0);   // p47 (= Engine::argmax(e.h_logits) without INK2_AMAX)
            if (g_lhall) g.lh2 = fnv_row(g.lh2, e.h_logits);
            continue;
        }
        std::vector<int> win{next};
        win.insert(win.end(), d.begin(), d.end());
        int nn = 0;
        const double tv = now_s();
        const int right = e.verify(win.data(), (int) win.size(), nn);
        if (g_lhall)
            for (int i = 0; i <= right; ++i) g.lh2 = fnv_row(g.lh2, e.h_logits + (size_t) i * NV);
        g.verify_s += now_s() - tv;
        ++g.rounds;
        if (mtpd) {   // p43: INK2_DS_LOG
            static std::FILE* dsl = getenv("INK2_DS_LOG") ? std::fopen(getenv("INK2_DS_LOG"), "a") : nullptr;
            if (dsl) {
                std::fprintf(dsl, "%d %d :", p43_at, win[0]);
                for (int t : d) std::fprintf(dsl, " %d", t);
                std::fprintf(dsl, " : %d\n", right);
                std::fflush(dsl);
            }
        }
        g.drafted += (int) d.size();
        g.accepted += right;
        if (lookup > 0 && !mtpd) {   // the n-gram draft length follows the last n-gram window (p7: not the drafter's)
            if (right == (int) d.size()) dcur = std::min(lookup, std::max(2, dcur * 2));
            else dcur = std::max(2, right + 1);
            if (right == 0 && mtp == 0) cool = 4;
        }
        bool stop_now = false;
        for (int i = 0; i < right && (int) g.out.size() < max; ++i) {
            g.out.push_back(d[i]);
            if (ends(d[i])) { stop_now = true; break; }
        }
        if (!flush()) break;
        if (stop_now || (int) g.out.size() >= max) break;
        next = nn;
    }
    g.write_s = now_s() - t1;
    e.want_logits = true;   // p47
    return g;
}

static bool peer_gone(int fd) {
    pollfd peer{fd, 0, 0};
    return poll(&peer, 1, 0) > 0 && (peer.revents & (POLLHUP | POLLERR | POLLNVAL));
}

static int serve(Engine& e, const std::string& path, int eos) {
    const int s = socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    std::snprintf(a.sun_path, sizeof a.sun_path, "%s", path.c_str());
    unlink(path.c_str());
    if (bind(s, (sockaddr*) &a, sizeof a) || listen(s, 4)) {
        std::perror("bind");
        return 1;
    }
    std::fprintf(stderr, "ink2_v1: posluša na %s\n", path.c_str());
    e.ck_spare();   // p55
    for (;;) {
        const int c = accept(s, nullptr, nullptr);
        if (c < 0) continue;
        const std::function<bool()> cancelled = [c] { return peer_gone(c); };
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
            break;
        }
        // Requests canceled in the socket backlog can still contain a whole prompt.
        if (cancelled()) {
            close(c);
            continue;
        }
        std::string rep;
        try {
            std::vector<int> tok, stop;
            for (long long x : json_ints(req, "tokens")) tok.push_back((int) x);
            for (long long x : json_ints(req, "stop")) stop.push_back((int) x);
            {   // p17: the pictures' rows of this request
                const std::vector<long long> im = json_ints(req, "img");
                e.img_map.clear();
                e.img_rows.clear();
                size_t rows = 0;
                for (size_t i = 0; i + 2 < im.size(); i += 3) {
                    e.img_map.push_back({(int) im[i], (int) im[i + 1], (int) im[i + 2]});
                    rows = std::max(rows, (size_t) (im[i + 1] + im[i + 2]));
                }
                if (rows) {
                    const std::string path = json_str(req, "embd");
                    FILE* f = std::fopen(path.c_str(), "rb");
                    if (!f) throw std::runtime_error("cannot read the image rows " + path);
                    e.img_rows.resize(rows * N);
                    const size_t got = std::fread(e.img_rows.data(), 4, rows * N, f);
                    std::fclose(f);
                    if (got != rows * N) throw std::runtime_error("short image rows file " + path);
                }
            }
            if (tok.empty()) throw std::runtime_error("no tokens");
            if (!json_str(req, "featdump").empty()) {   // p43: every row's drafter features from position 0, nothing kept after
                if (!e.ds_on && !e.d2_on) throw std::runtime_error("featdump: no drafter loaded");
                if ((long long) tok.size() > e.ctx) throw std::runtime_error("featdump: longer than the context");
                const std::string path = json_str(req, "featdump");
                e.reset();
                while (!e.parks.empty()) e.park_drop(e.parks.size() - 1);
                e.fdump = std::fopen(path.c_str(), "wb");
                if (!e.fdump) throw std::runtime_error("featdump: cannot write " + path);
                e.force_pf = true;
                try {
                    e.feed(tok.data(), (int) tok.size(), false);
                } catch (...) {
                    e.force_pf = false;
                    std::fclose(e.fdump);
                    e.fdump = nullptr;
                    throw;
                }
                e.force_pf = false;
                std::fclose(e.fdump);
                e.fdump = nullptr;
                e.reset();
                rep = "{\"featdump\": " + std::to_string(tok.size()) + "}\n";
                (void) !write(c, rep.data(), rep.size());
                close(c);
                continue;
            }
            const int max = (int) std::min<long long>(json_int(req, "max", 256), std::max<long long>(1, (long long) e.ctx - (long long) tok.size()));
            // p15: "stream": 1 sends {"t": [ids]} lines as the tokens come; a failed write (the client left) stops the generation
            const std::function<bool(const int*, int)> out_line = [&](const int* t, int n) {
                std::string l = "{\"t\": [";
                for (int i = 0; i < n; ++i) l += (i ? ", " : "") + std::to_string(t[i]);
                l += "]}\n";
                return write(c, l.data(), l.size()) == (ssize_t) l.size();
            };
            const bool streaming = json_int(req, "stream", 0) != 0;
            if (json_int(req, "forget", 0)) {   // p21b (tests): no conversation on the cards or in RAM
                e.reset();
                while (!e.parks.empty()) e.park_drop(e.parks.size() - 1);
            }
            g_lhall = json_int(req, "lhall", 0) != 0;
            {   // p36: sampling (temp_milli 0 = greedy)
                const long long tm = json_int(req, "temp_milli", 0);
                e.s_invT = tm > 0 ? 1000.0f / (float) tm : 0.0f;
                e.s_seed = (unsigned long long) json_int(req, "seed", 0);
                const long long mp = json_int(req, "minp_milli", 0);   // p76
                e.s_lnminp = tm > 0 && mp > 0 && mp < 1000 ? logf((float) mp / 1000.0f) : 0.0f;
            }
            std::vector<int> hints;   // p21: where the prompt's messages start
            for (long long x : json_ints(req, "ckpt")) hints.push_back((int) x);
            Gen g = generate(e, tok, max, stop, json_int(req, "noeos", 0) ? -1 : eos, (int) json_int(req, "lookup", 0),
                             (int) json_int(req, "mtp", 0), streaming ? &out_line : nullptr, &hints, &cancelled);   // noeos: a fixed length
            rep = "{\"tokens\": [";
            for (size_t i = 0; i < g.out.size(); ++i) rep += (i ? ", " : "") + std::to_string(g.out[i]);
            char hd[640];
            std::snprintf(hd, sizeof hd,
                          "], \"prompt_n\": %zu, \"cached\": %d, \"read_s\": %.4f, \"write_s\": %.4f, \"drafted\": %d, \"accepted\": %d, "
                          "\"rounds\": %d, \"draft_s\": %.4f, \"verify_s\": %.4f, \"back\": %d, \"ckpts\": %zu, \"ck_s\": %.4f, "
                          "\"park\": %d, \"parks\": %zu, \"park_mb\": %zu, \"lh\": \"%016llx\", \"lh2\": \"%016llx\"}\n",
                          tok.size(), g.cached, g.read_s, g.write_s, g.drafted, g.accepted, g.rounds, g.draft_s, g.verify_s, g.back,
                          e.ckpts.size(), g.ck_s, g.park, e.parks.size(), e.park_bytes >> 20, g.lh, g.lh2);
            rep += hd;
        } catch (const PrefillCancelled&) {
            // feed() only checks between windows, with the processed prefix still valid.
            close(c);
            continue;
        } catch (const std::exception& ex) {
            e.reset();
            rep = std::string("{\"error\": \"") + ex.what() + "\"}\n";
        }
        (void) !write(c, rep.data(), rep.size());
        close(c);
        e.ck_spare();   // p55: the client has its reply
    }
    close(s);
    unlink(path.c_str());
    return 0;
}

}  // namespace ink2

int main(int argc, char** argv) {
    using namespace ink2;
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    if (getenv("INK2_RTRACE")) g_rtrace = std::fopen(getenv("INK2_RTRACE"), "a");   // p31
    signal(SIGPIPE, SIG_IGN);   // p15: a client that left makes write() fail instead of ending the server   // lazy loading can sync a context on a kernel's first launch (a cross-card wait deadlocks)
    std::string model, tokens, dump, sock, logits_out, mtp;   // --mtp F: the DSpark drafter (dspark-Inkling-2-Mini-Q8_0.gguf)   // --logits F: the last row's logits after the prompt (logiti.bin format)
    int gen = 0, ctx = 32768;
    bool logits_all = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") model = nx();
        else if (a == "--tokens") tokens = nx();
        else if (a == "--gen") gen = std::stoi(nx());
        else if (a == "--dump") dump = nx();
        else if (a == "--ctx") ctx = std::stoi(nx());
        else if (a == "--serve") sock = nx();
        else if (a == "--logits") logits_out = nx();
        else if (a == "--logits-all") { logits_out = nx(); logits_all = true; }   // every row of the last window (writing path)
        else if (a == "--mtp") mtp = nx();
        else if (a == "--draft-vocab") g_draft_vocab = nx();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (model.empty()) { std::fprintf(stderr, "usage: ink2_v1 --model M.gguf (--tokens zetoni.bin [--gen N] [--dump DIR] | --serve SOCK) [--ctx C]\n"); return 1; }
    Engine e(model, ctx, mtp);
    const int eos = (int) e.g.i("tokenizer.ggml.eos_token_id");
    if (!sock.empty()) return serve(e, sock, eos);
    FILE* f = std::fopen(tokens.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot read %s\n", tokens.c_str()); return 1; }
    int32_t n = 0;
    if (std::fread(&n, 4, 1, f) != 1) return 1;
    std::vector<int> ids(n);
    if (std::fread(ids.data(), 4, n, f) != (size_t) n) return 1;
    std::fclose(f);
    if (!dump.empty()) {
        e.dumping = true;
        e.d_lout.resize(NSLOT);
        e.d_aout.resize(NSLOT);
        e.d_fout.resize(NSLOT);
    }
    const double t0 = now_s();
    const int nr = e.feed(ids.data(), n, !dump.empty() || logits_all);
    const double t1 = now_s();
    std::printf("poziv %d zetonov v %.3f s (%.1f tok/s)\n", n, t1 - t0, n / (t1 - t0));
    if (!dump.empty()) {
        for (int s = 0; s < NSLOT; ++s) {
            write_dump(dump + "/l_out-" + std::to_string(s) + ".bin", e.d_lout[s], N);
            write_dump(dump + "/attn_out-" + std::to_string(s) + ".bin", e.d_aout[s], N);
            write_dump(dump + "/ffn_out-" + std::to_string(s) + ".bin", e.d_fout[s], N);
        }
        FILE* lf = std::fopen((dump + "/logiti.bin").c_str(), "wb");
        const int32_t first = n - nr, nv = NVU;
        std::fwrite(&n, 4, 1, lf);
        std::fwrite(&nv, 4, 1, lf);
        std::fwrite(&first, 4, 1, lf);
        for (int r = 0; r < nr; ++r) std::fwrite(e.h_logits + (size_t) r * NV, 4, NVU, lf);
        std::fclose(lf);
        e.dumping = false;
    }
    if (!logits_out.empty()) {
        FILE* lf = std::fopen(logits_out.c_str(), "wb");
        const int32_t first = logits_all ? n - nr : n - 1, nv = NVU;
        std::fwrite(&n, 4, 1, lf);
        std::fwrite(&nv, 4, 1, lf);
        std::fwrite(&first, 4, 1, lf);
        for (int r = logits_all ? 0 : nr - 1; r < nr; ++r) std::fwrite(e.h_logits + (size_t) r * NV, 4, NVU, lf);
        std::fclose(lf);
    }
    int next = Engine::argmax(e.h_logits + (size_t) (nr - 1) * NV);
    std::printf("prvi %d\n", next);
    if (gen > 0) {
        std::vector<int> out{next};
        const double t2 = now_s();
        for (int i = 1; i < gen && next != eos; ++i) {
            e.feed(&next, 1, false);
            next = Engine::argmax(e.h_logits);
            out.push_back(next);
        }
        const double t3 = now_s();
        std::printf("pisanje %zu zetonov v %.3f s (%.2f tok/s):", out.size(), t3 - t2, (out.size() - 1) / std::max(1e-9, t3 - t2));
        for (int x : out) std::printf(" %d", x);
        std::printf("\n");
    }
    return 0;
}
