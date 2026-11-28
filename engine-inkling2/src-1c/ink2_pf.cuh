// ink2_pf.cuh (2026-11-21, p3): the reading path of Inkling 2 Mini, prompt chunks of up to INK2_PF_CHUNK tokens. The same math as
// the writing kernels in f32: dense matrices dequantized to f32 and multiplied with cuBLAS SGEMM, experts grouped by expert
// (dequantized in groups, SGEMM each), attention as Q.K^T and P.V with cuBLAS around a softmax kernel that adds the relative
// bias from a table R.B (d = 0..EXT, the last column 0) and the sink.
#pragma once
#include "ink2_kern.cuh"

#include <algorithm>

namespace ink2 {

// Q8S rows -> f32 rows: a thread per 16 values
__global__ void k_q8s_f32(const int8_t* __restrict__ q, const __half* __restrict__ d, size_t rows, int cols, float* out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x, n16 = rows * cols / 16;
    if (i >= n16) return;
    const size_t c0 = i * 16, r = c0 / cols;
    const int c = (int) (c0 % cols);
    const float sc = __half2float(d[r * (cols / 32) + c / 32]);
    const int4 w4 = __ldg((const int4*) (q + c0));
    const int8_t* w = (const int8_t*) &w4;
    float4* o = (float4*) (out + c0);
#pragma unroll
    for (int k = 0; k < 4; ++k) o[k] = make_float4(sc * w[4 * k], sc * w[4 * k + 1], sc * w[4 * k + 2], sc * w[4 * k + 3]);
}
// NVFP4 rows (cols / 64 blocks of 36 B a row) -> f32 rows: a thread per sub-block of 16 values
__global__ void k_fp4_f32(const uint8_t* __restrict__ W, size_t rows, int cols, float* out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x, nsub = rows * (cols / 16);
    if (i >= nsub) return;
    const size_t r = i / (cols / 16);
    const int s = (int) (i % (cols / 16));
    const uint8_t* blk = W + r * (size_t) (cols / 64) * 36 + (size_t) (s >> 2) * 36;
    const float d = ue4m3(blk[s & 3]);
    const uint8_t* qs = blk + 4 + (s & 3) * 8;
    float* o = out + r * cols + (size_t) s * 16;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint8_t b = qs[j];
        o[j] = c_fp4[b & 15] * d;
        o[j + 8] = c_fp4[b >> 4] * d;
    }
}
// p3b: NVFP4 rows -> f32 rows with one float4 store a thread (coalesced): thread u of a sub-block of 16 values writes values
// 4u..4u+3 = the low (u < 2) or high (u >= 2) nibbles of code bytes 4(u & 1)..4(u & 1)+3
__global__ void k_fp4_f32b(const uint8_t* __restrict__ W, size_t rows, int cols, float* out) {
    __shared__ float c_fp4s[16];   // lanes reading different entries of __constant__ memory serialize
    if (threadIdx.x < 16) c_fp4s[threadIdx.x] = c_fp4[threadIdx.x];
    __syncthreads();
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x, n4 = rows * (cols / 4);
    if (i >= n4) return;
    const size_t r = i / (cols / 4);
    const int c4 = (int) (i % (cols / 4)), s = c4 >> 2, u = c4 & 3;
    const uint8_t* blk = W + r * (size_t) (cols / 64) * 36 + (size_t) (s >> 2) * 36;
    const float d = ue4m3(__ldg(blk + (s & 3)));
    const uint32_t q = __ldg((const uint32_t*) (blk + 4 + (s & 3) * 8 + (u & 1) * 4));
    const int sh = u >= 2 ? 4 : 0;
    float4 o;
    o.x = c_fp4s[(q >> sh) & 15] * d;
    o.y = c_fp4s[(q >> (8 + sh)) & 15] * d;
    o.z = c_fp4s[(q >> (16 + sh)) & 15] * d;
    o.w = c_fp4s[(q >> (24 + sh)) & 15] * d;
    ((float4*) out)[i] = o;
}
// p3b: the short convolution of T rows in parallel over rows (an output needs only the 3 inputs before it); the state is read for
// the first 3 rows and the last 3 inputs are written back by k_conv_state after all rows are done
__global__ void k_conv_par(const float* x, int C, const float* st, const float* w, const float* dyn, int hs, float k, float* y, int T) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * C) return;
    const int t = (int) (i / C), c = (int) (i % C);
    auto in = [&](int tt) { return tt >= 0 ? x[(size_t) tt * C + c] : st[(size_t) (3 + tt) * C + c]; };
    const float s0 = in(t - 3), s1 = in(t - 2), s2 = in(t - 1), xt = x[i];
    float o = 0.0f;
    o += s0 * w[c * 4 + 0];
    o += s1 * w[c * 4 + 1];
    o += s2 * w[c * 4 + 2];
    o += xt * w[c * 4 + 3];
    if (dyn) {
        const float* dt = dyn + ((size_t) t * (C / hs) + c / hs) * NCONV;
        o = o + dt[0] * xt;
        o = o + dt[1] * s2;
        o = o + dt[2] * s1;
        o = o + dt[3] * s0;
    }
    y[i] = (xt + o) * k;
}
__global__ void k_conv_state(const float* x, int C, float* st, int T) {   // the state = the last 3 inputs (rows before 0 from the state)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
    for (int k = 0; k < 3; ++k) {
        const int tt = T - 3 + k;
        v[k] = tt >= 0 ? x[(size_t) tt * C + c] : st[(size_t) (3 + tt) * C + c];
    }
    for (int k = 0; k < 3; ++k) st[(size_t) k * C + c] = v[k];
}
// the expert lists of a chunk: counts per expert, then each (t, k) its row in the expert-sorted order (cursor by atomics: the order
// within an expert does not change any row's result)
__global__ void k_exp_count(const int* topi, int n, int* cnt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&cnt[topi[i]], 1);
}
__global__ void k_exp_fill(const int* topi, int n, const int* offs, int* cur, int* slotrow, int* rowtok, int* rowexp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = topi[i], r = offs[e] + atomicAdd(&cur[e], 1);
    slotrow[i] = r;
    rowtok[r] = i / TOPK;
    rowexp[r] = e;
}
__global__ void k_gather_rows(const float* x, const int* rowtok, int nrows, float* out) {   // out[r] = x[rowtok[r]], N floats a row
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) nrows * (N / 4)) return;
    const int r = (int) (i / (N / 4)), c = (int) (i % (N / 4));
    ((float4*) out)[(size_t) r * (N / 4) + c] = ((const float4*) x)[(size_t) rowtok[r] * (N / 4) + c];
}
// SITU on expert-sorted rows, with the row's expert's gate_up scale
__global__ void k_situ_s(const float* gu, float* h, int nrows, const int* rowexp, const float* gs, float beta, float lb) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows * FF_C) return;
    const int j = i / FF_C, c = i % FF_C;
    const float s = gs ? gs[rowexp[j]] : 1.0f;
    const float g = gu[(size_t) j * 2 * FF_C + c] * s, u = gu[(size_t) j * 2 * FF_C + FF_C + c] * s;
    const float act = (beta * tanhf(g / beta)) * sigm(g);
    const float up = lb > 0.0f ? lb * tanhf(u / lb) : u;
    h[i] = act * up;
}
// moe[t] = sum_k w[t][k] * (EY[row(t, k)] * down scale of its expert), k in order
__global__ void k_moe_comb(const float* ey, const int* slotrow, const int* topi, const float* topw, const float* ds, float* moe, int T) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * N) return;
    const int t = (int) (i / N), c = (int) (i % N);
    float v = 0;
    for (int k = 0; k < TOPK; ++k) {
        const int s = t * TOPK + k;
        const float y = ey[(size_t) slotrow[s] * N + c];
        v += (ds ? y * ds[topi[s]] : y) * topw[s];
    }
    moe[i] = v;
}
// K and V rows of positions [j0, j0 + n) of a slot's cache (ring for windows) as f32 rows of KVC
__global__ void k_kv_f32(const __half* kc, const __half* vc, int j0, int n, int ring, float* kf, float* vf, KV8 q8) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * KVC) return;
    const int j = (int) (i / KVC), c = (int) (i % KVC);
    if (j0 + j < 0) {   // p3f: the rows before position 0 of a window block (masked; zero V adds nothing)
        kf[i] = 0.0f;
        vf[i] = 0.0f;
        return;
    }
    const size_t row = ring ? (size_t) ((j0 + j) % RING) : (size_t) (j0 + j);
    kf[i] = kv_get(q8, true, kc, row, c);   // p18: int8 rows of a full slot by q8
    vf[i] = kv_get(q8, false, vc, row, c);
}
// p14: rows j0..j0+n of KV head g as f32 [n][HD] (V only when kf is null): the long context's key blocks
__global__ void k_kv_f32g(const __half* kc, const __half* vc, int j0, int n, int g, float* kf, float* vf, KV8 q8) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * HD) return;
    const size_t j = i / HD, c = i % HD, src = (size_t) (j0 + j) * KVC + g * HD + c;
    (void) src;
    if (kf) kf[i] = kv_get(q8, true, kc, j0 + j, (int) (g * HD + c));   // p18
    vf[i] = kv_get(q8, false, vc, j0 + j, (int) (g * HD + c));
}
// p14b: P.V split by keys: part[a][sb][i][HD] of key piece sb summed in order sb = 0..nsb-1 into ao (query i0 + i, head h0 + a)
__global__ void k_pv_sum(const float* part, int nsb, int nsbmax, int qs, float* ao, int i0, int h0) {
    const size_t e = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (size_t) 3 * qs * HD) return;
    const int a = (int) (e / ((size_t) qs * HD)), r = (int) (e % ((size_t) qs * HD)), i = r / HD, d = r % HD;
    const float* p = part + ((size_t) a * nsbmax * qs + i) * HD + d;
    float v = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) v += p[(size_t) sb * qs * HD];
    ao[(size_t) (i0 + i) * QC + (h0 + a) * HD + d] = v;
}
// scores of query i (row of nk keys, positions j0..) of head h: s = S * scale + RB[i][h][min(d, EXT)] - sink[i][h], masked to the
// window / causal range; softmax with the zero sink logit (one more 1 in the sum when the max is below 0). In place. A block per row
__global__ void __launch_bounds__(256) k_pf_softmax(float* S, int nk, int j0, const int* pos, int i0, int h0, const float* RB, const float* sink,
                                                    int window, float scale) {
    __shared__ float red[32];
    const int i = blockIdx.x, hh = blockIdx.y, h = h0 + hh, QS = gridDim.x;
    float* row = S + ((size_t) hh * QS + i) * nk;
    const int P = pos[i0 + i];
    const int lo = window ? P - (SWA - 1) : 0;
    const float* rb = RB + ((size_t) (i0 + i) * NH_C + h) * (EXT + 1);
    const float t = sink[(i0 + i) * NH_C + h];
    // p3g: the score is computed again from the raw row in each pass instead of being written back twice (same values)
    auto score = [&](int j) {
        const int pj = j0 + j;
        return pj <= P && pj >= lo && pj >= 0 ? row[j] * scale + (rb[min(P - pj, EXT)] - t) : -INFINITY;
    };
    float m = 0.0f;   // the sink logit
    for (int j = threadIdx.x; j < nk; j += blockDim.x) m = fmaxf(m, score(j));
    m = block_max(m, red);
    float l = 0;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) l += expf(score(j) - m);
    l = block_sum(l, red) + expf(0.0f - m);
    const float inv = 1.0f / l;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) row[j] = expf(score(j) - m) * inv;
}
// p3f: k_pf_softmax for the window layers' query blocks of one batched call: block (i, h, b) is query b * QS + i of local head h,
// row idx = b * NH_C + h of S (nk = QS + SWA - 1 keys from position p0 + b * QS - (SWA - 1), negative ones masked). Same sums
__global__ void __launch_bounds__(256) k_pf_softmax_b(float* S, int nk, int p0, int QS, const int* pos, const float* RB, const float* sink,
                                                      float scale) {
    __shared__ float red[32];
    const int i = blockIdx.x, h = blockIdx.y, b = blockIdx.z, ii = b * QS + i;
    float* row = S + ((size_t) (b * NH_C + h) * QS + i) * nk;
    const int P = pos[ii], j0 = p0 + b * QS - (SWA - 1);
    const int lo = max(P - (SWA - 1), 0);
    const float* rb = RB + ((size_t) ii * NH_C + h) * (EXT + 1);
    const float t = sink[ii * NH_C + h];
    auto score = [&](int j) {   // p3g as in k_pf_softmax
        const int pj = j0 + j;
        return pj <= P && pj >= lo ? row[j] * scale + (rb[min(P - pj, EXT)] - t) : -INFINITY;
    };
    float m = 0.0f;   // the sink logit
    for (int j = threadIdx.x; j < nk; j += blockDim.x) m = fmaxf(m, score(j));
    m = block_max(m, red);
    float l = 0;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) l += expf(score(j) - m);
    l = block_sum(l, red) + expf(0.0f - m);
    const float inv = 1.0f / l;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) row[j] = expf(score(j) - m) * inv;
}
// p3f: RB[n][d] = r[n] . B[d] for d = 0..EXT (cuBLAS ran this with an inner size of 16 at ~0.35 TFLOPS): 8 rows n a block, the
// 16 values of B[d] from the cache, sums in order k = 0..15
__global__ void __launch_bounds__(256) k_rb(const float* __restrict__ r, const float* __restrict__ Bpad, int nrows, float* RB) {
    __shared__ float rs[8][NREL];
    const int n0 = blockIdx.x * 8;
    if (threadIdx.x < 8 * NREL) {
        const int a = threadIdx.x / NREL, k = threadIdx.x % NREL;
        rs[a][k] = n0 + a < nrows ? r[(size_t) (n0 + a) * NREL + k] : 0.0f;
    }
    __syncthreads();
    for (int d = threadIdx.x; d <= EXT; d += blockDim.x) {
        float bb[NREL];
#pragma unroll
        for (int k4 = 0; k4 < NREL / 4; ++k4) {
            const float4 v = __ldg((const float4*) (Bpad + (size_t) d * NREL) + k4);
            bb[k4 * 4 + 0] = v.x;
            bb[k4 * 4 + 1] = v.y;
            bb[k4 * 4 + 2] = v.z;
            bb[k4 * 4 + 3] = v.w;
        }
#pragma unroll
        for (int a = 0; a < 8; ++a) {
            if (n0 + a >= nrows) break;
            float acc = 0.0f;
#pragma unroll
            for (int k = 0; k < NREL; ++k) acc += rs[a][k] * bb[k];
            RB[(size_t) (n0 + a) * (EXT + 1) + d] = acc;
        }
    }
}
__global__ void k_sigmul(float* a, const float* g, size_t n) {   // a *= sigmoid(g)
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= sigm(g[i]);
}
__global__ void k_add_ord(float* y, const float* other, size_t n, int card) {   // the cross-card sum, card 0's part first on both
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = card == 0 ? y[i] + other[i] : other[i] + y[i];
}

// p27: the attention's input projections of a piece by groups of rows with the same input into T [C][ldT], then each part
// (dst[g], its width ld[g], from column off[g] of T) to its own buffer
struct CatSegs { float* dst[8]; int ld[8], off[8], n; };
__global__ void k_cat_scatter(const float* __restrict__ T, int ldT, int C, CatSegs s) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) C * ldT) return;
    const int t = (int) (i / ldT), j = (int) (i % ldT);
    int g = 0;
    while (g + 1 < s.n && j >= s.off[g + 1]) ++g;
    s.dst[g][(size_t) t * s.ld[g] + j - s.off[g]] = T[i];
}

// p3c: NVFP4 experts of a chunk without an f32 copy (after Hy4's k_iq3_gemm3): a block = 128 rows of one expert by up to 64 of its
// tokens at a time (rows of x in the expert-sorted order: expert e's from offs[e] to offs[e + 1]), steps of 32 input values (two
// NVFP4 sub-blocks: thread (row dr, half dh) decodes one into shared memory), each thread an 8-row by 4-token tile, the next step's
// bytes and inputs loaded during this step's products. y[row of x][r] = W[e][r] . x (the expert scales are applied after)
constexpr int F4M = 128, F4N = 64, F4K = 32;
template <bool PLANAR>
__global__ void __launch_bounds__(256) k_fp4_gemm(const uint8_t* __restrict__ W, size_t estride, int rows, int cols, const int* offs,
                                                  const float* x, int ldx, float* y, int ldy) {
    __shared__ __align__(16) float ws[F4K][F4M];
    __shared__ __align__(16) float xs[F4K][F4N];
    __shared__ float tab[16];
    const int e = blockIdx.y, o0 = offs[e], ne = offs[e + 1] - o0;
    if (ne == 0) return;
    if (threadIdx.x < 16) tab[threadIdx.x] = c_fp4[threadIdx.x];
    const int r0 = blockIdx.x * F4M, ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
    const int dr = threadIdx.x & (F4M - 1), dh = threadIdx.x >> 7;
    const int tok = threadIdx.x >> 2, seg = (threadIdx.x & 3) * 8;
    const int nblk = cols / 64, rb = nblk * 36, nsteps = cols / F4K;
    const uint8_t* rowp = W + (size_t) e * estride + (size_t) (r0 + dr) * rb;   // interleaved rows
    const uint8_t* pcodes = W + (size_t) e * estride + (size_t) (r0 + dr) * nblk * 32;   // p4d: planar (codes, then scales)
    const uint8_t* pscales = W + (size_t) e * estride + (size_t) rows * nblk * 32 + (size_t) (r0 + dr) * nblk * 4;
    for (int t0 = 0; t0 < ne; t0 += F4N) {
        const int nk = min(F4N, ne - t0);
        const float* xr = tok < nk ? x + (size_t) (o0 + t0 + tok) * ldx + seg : nullptr;
        uint8_t rd;
        uint32_t q0, q1;
        float4 v0, v1;
        auto load = [&](int ks) {   // step ks: block ks / 2, sub-block (ks & 1) * 2 + dh
            const int sb = (ks & 1) * 2 + dh;
            if (PLANAR) {
                const uint8_t* c = pcodes + (size_t) (ks >> 1) * 32 + sb * 8;
                rd = __ldg(pscales + (size_t) (ks >> 1) * 4 + sb);
                q0 = __ldg((const uint32_t*) c);
                q1 = __ldg((const uint32_t*) (c + 4));
            } else {
                const uint8_t* blk = rowp + (size_t) (ks >> 1) * 36;
                rd = __ldg(blk + sb);
                q0 = __ldg((const uint32_t*) (blk + 4 + sb * 8));
                q1 = __ldg((const uint32_t*) (blk + 8 + sb * 8));
            }
            v0 = v1 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (xr) {
                v0 = __ldg((const float4*) (xr + ks * F4K));
                v1 = __ldg((const float4*) (xr + ks * F4K + 4));
            }
        };
        load(0);
        float acc[8][4];
#pragma unroll
        for (int a = 0; a < 8; ++a)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[a][k] = 0.0f;
        for (int ks = 0; ks < nsteps; ++ks) {
            __syncthreads();
            {
                const float d = ue4m3(rd);
                const int k = dh * 16;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t b0 = (q0 >> (8 * j)) & 0xFFu, b1 = (q1 >> (8 * j)) & 0xFFu;
                    ws[k + j][dr] = tab[b0 & 15] * d;
                    ws[k + 4 + j][dr] = tab[b1 & 15] * d;
                    ws[k + 8 + j][dr] = tab[b0 >> 4] * d;
                    ws[k + 12 + j][dr] = tab[b1 >> 4] * d;
                }
                xs[seg + 0][tok] = v0.x; xs[seg + 1][tok] = v0.y; xs[seg + 2][tok] = v0.z; xs[seg + 3][tok] = v0.w;
                xs[seg + 4][tok] = v1.x; xs[seg + 5][tok] = v1.y; xs[seg + 6][tok] = v1.z; xs[seg + 7][tok] = v1.w;
            }
            __syncthreads();
            if (ks + 1 < nsteps) load(ks + 1);
#pragma unroll 8
            for (int k = 0; k < F4K; ++k) {
                const float4 w0 = *(const float4*) &ws[k][ty * 8], w1 = *(const float4*) &ws[k][ty * 8 + 4];
                const float4 xv = *(const float4*) &xs[k][tx * 4];
                const float w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w}, xx[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                for (int a = 0; a < 8; ++a)
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) acc[a][kk] += w[a] * xx[kk];
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int tk = tx * 4 + i;
            if (tk >= nk) continue;
            float* yr = y + (size_t) (o0 + t0 + tk) * ldy + r0 + ty * 8;
#pragma unroll
            for (int a = 0; a < 8; ++a) yr[a] = acc[a][i];
        }
        __syncthreads();
    }
}

// p19b: NVFP4 expert tiles of 64, 32 or 16 tokens, one block a tile from a list (k_fp4_gemm5): an expert's last tile takes the
// smallest size that holds its remaining tokens (64-token tiles were 80 % filled on a chunk's routing), each thread 8 rows by
// TN / 16 tokens. Every sum is the same chain of FMA in k order as in k_fp4_gemm, so y is the same to the bit. Planar weights
template <int TN>
__device__ __forceinline__ void fp4_tile(const uint8_t* __restrict__ pcodes, const uint8_t* __restrict__ pscales, int nsteps, const float* tab,
                                         float (*ws)[F4M], float* xsm, const float* __restrict__ x, int ldx, int row0, int nk, float* y,
                                         int ldy, int r0) {
    constexpr int TT = TN / 16;   // tokens a thread
    float (*xs)[TN] = (float (*)[TN]) xsm;
    const int ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
    const int dr = threadIdx.x & (F4M - 1), dh = threadIdx.x >> 7;
    const int tok = threadIdx.x >> 2, seg = (threadIdx.x & 3) * 8;
    const float* xr = tok < nk ? x + (size_t) (row0 + tok) * ldx + seg : nullptr;
    uint8_t rd;
    uint32_t q0, q1;
    float4 v0, v1;
    auto load = [&](int ks) {
        const int sb = (ks & 1) * 2 + dh;
        const uint8_t* c = pcodes + (size_t) (ks >> 1) * 32 + sb * 8;
        rd = __ldg(pscales + (size_t) (ks >> 1) * 4 + sb);
        q0 = __ldg((const uint32_t*) c);
        q1 = __ldg((const uint32_t*) (c + 4));
        v0 = v1 = make_float4(0.f, 0.f, 0.f, 0.f);
        if (xr) {
            v0 = __ldg((const float4*) (xr + ks * F4K));
            v1 = __ldg((const float4*) (xr + ks * F4K + 4));
        }
    };
    load(0);
    float acc[8][TT];
#pragma unroll
    for (int a = 0; a < 8; ++a)
#pragma unroll
        for (int k = 0; k < TT; ++k) acc[a][k] = 0.0f;
    for (int ks = 0; ks < nsteps; ++ks) {
        __syncthreads();
        {
            const float d = ue4m3(rd);
            const int k = dh * 16;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t b0 = (q0 >> (8 * j)) & 0xFFu, b1 = (q1 >> (8 * j)) & 0xFFu;
                ws[k + j][dr] = tab[b0 & 15] * d;
                ws[k + 4 + j][dr] = tab[b1 & 15] * d;
                ws[k + 8 + j][dr] = tab[b0 >> 4] * d;
                ws[k + 12 + j][dr] = tab[b1 >> 4] * d;
            }
            if (tok < TN) {
                xs[seg + 0][tok] = v0.x; xs[seg + 1][tok] = v0.y; xs[seg + 2][tok] = v0.z; xs[seg + 3][tok] = v0.w;
                xs[seg + 4][tok] = v1.x; xs[seg + 5][tok] = v1.y; xs[seg + 6][tok] = v1.z; xs[seg + 7][tok] = v1.w;
            }
        }
        __syncthreads();
        if (ks + 1 < nsteps) load(ks + 1);
#pragma unroll 8
        for (int k = 0; k < F4K; ++k) {
            const float4 w0 = *(const float4*) &ws[k][ty * 8], w1 = *(const float4*) &ws[k][ty * 8 + 4];
            const float w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            float xx[TT];
            if constexpr (TT == 4) {
                const float4 xv = *(const float4*) &xs[k][tx * 4];
                xx[0] = xv.x; xx[1] = xv.y; xx[2] = xv.z; xx[3] = xv.w;
            } else if constexpr (TT == 2) {
                const float2 xv = *(const float2*) &xs[k][tx * 2];
                xx[0] = xv.x; xx[1] = xv.y;
            } else {
                xx[0] = xs[k][tx];
            }
#pragma unroll
            for (int a = 0; a < 8; ++a)
#pragma unroll
                for (int kk = 0; kk < TT; ++kk) acc[a][kk] += w[a] * xx[kk];
        }
    }
#pragma unroll
    for (int i = 0; i < TT; ++i) {
        const int tk = tx * TT + i;
        if (tk >= nk) continue;
        float* yr = y + (size_t) (row0 + tk) * ldy + r0 + ty * 8;
#pragma unroll
        for (int a = 0; a < 8; ++a) yr[a] = acc[a][i];
    }
    __syncthreads();
}
// p24: tiles of TN = 8 or 4 tokens for an expert's last few tokens (short pieces of a prompt: ~8 or fewer tokens an expert): RT
// rows x 1 token a thread (RT 4 or 2; 128 / RT row groups x TN tokens = 256 threads). The same sums as fp4_tile, column by column
template <int TN, int RT>
__device__ __forceinline__ void fp4_tile_s(const uint8_t* __restrict__ pcodes, const uint8_t* __restrict__ pscales, int nsteps, const float* tab,
                                           float (*ws)[F4M], float* xsm, const float* __restrict__ x, int ldx, int row0, int nk, float* y,
                                           int ldy, int r0) {
    static_assert(F4M / RT * TN == 256, "a thread a row group and token");
    float (*xs)[TN] = (float (*)[TN]) xsm;
    const int ty = threadIdx.x / TN, tx = threadIdx.x % TN;
    const int dr = threadIdx.x & (F4M - 1), dh = threadIdx.x >> 7;
    const int tok = threadIdx.x >> 2, seg = (threadIdx.x & 3) * 8;
    const float* xr = tok < nk ? x + (size_t) (row0 + tok) * ldx + seg : nullptr;
    uint8_t rd;
    uint32_t q0, q1;
    float4 v0, v1;
    auto load = [&](int ks) {
        const int sb = (ks & 1) * 2 + dh;
        const uint8_t* c = pcodes + (size_t) (ks >> 1) * 32 + sb * 8;
        rd = __ldg(pscales + (size_t) (ks >> 1) * 4 + sb);
        q0 = __ldg((const uint32_t*) c);
        q1 = __ldg((const uint32_t*) (c + 4));
        v0 = v1 = make_float4(0.f, 0.f, 0.f, 0.f);
        if (xr) {
            v0 = __ldg((const float4*) (xr + ks * F4K));
            v1 = __ldg((const float4*) (xr + ks * F4K + 4));
        }
    };
    load(0);
    float acc[RT];
#pragma unroll
    for (int a = 0; a < RT; ++a) acc[a] = 0.0f;
    for (int ks = 0; ks < nsteps; ++ks) {
        __syncthreads();
        {
            const float d = ue4m3(rd);
            const int k = dh * 16;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t b0 = (q0 >> (8 * j)) & 0xFFu, b1 = (q1 >> (8 * j)) & 0xFFu;
                ws[k + j][dr] = tab[b0 & 15] * d;
                ws[k + 4 + j][dr] = tab[b1 & 15] * d;
                ws[k + 8 + j][dr] = tab[b0 >> 4] * d;
                ws[k + 12 + j][dr] = tab[b1 >> 4] * d;
            }
            if (tok < TN) {
                xs[seg + 0][tok] = v0.x; xs[seg + 1][tok] = v0.y; xs[seg + 2][tok] = v0.z; xs[seg + 3][tok] = v0.w;
                xs[seg + 4][tok] = v1.x; xs[seg + 5][tok] = v1.y; xs[seg + 6][tok] = v1.z; xs[seg + 7][tok] = v1.w;
            }
        }
        __syncthreads();
        if (ks + 1 < nsteps) load(ks + 1);
#pragma unroll 8
        for (int k = 0; k < F4K; ++k) {
            float w[RT];
            if constexpr (RT == 4) {
                const float4 w0 = *(const float4*) &ws[k][ty * 4];
                w[0] = w0.x; w[1] = w0.y; w[2] = w0.z; w[3] = w0.w;
            } else {
                const float2 w0 = *(const float2*) &ws[k][ty * 2];
                w[0] = w0.x; w[1] = w0.y;
            }
            const float xx = xs[k][tx];
#pragma unroll
            for (int a = 0; a < RT; ++a) acc[a] += w[a] * xx;
        }
    }
    if (tx < nk) {
        float* yr = y + (size_t) (row0 + tx) * ldy + r0 + ty * RT;
#pragma unroll
        for (int a = 0; a < RT; ++a) yr[a] = acc[a];
    }
    __syncthreads();
}
// p19b: one block a tile: the tiles (expert, first row of x, rows) from the host's routing counts, the 64-token ones first, so a
// big expert's tiles run side by side instead of one after the other in one block (2048 tokens: up to ~1,500 rows on one expert)
struct FTile { int e, row0, nk; };
__global__ void __launch_bounds__(256) k_fp4_gemm5(const uint8_t* __restrict__ W, size_t estride, int rows, int cols, const FTile* tiles,
                                                   const float* x, int ldx, float* y, int ldy, int small = 1,   // p24: small = tiles of 8 and 4 tokens
                                                   const uint8_t* const* ET = nullptr) {   // p30: expert bases
    __shared__ __align__(16) float ws[F4K][F4M];
    __shared__ __align__(16) float xs[F4K * F4N];
    __shared__ float tab[16];
    const FTile t = tiles[blockIdx.y];
    if (threadIdx.x < 16) tab[threadIdx.x] = c_fp4[threadIdx.x];
    const int r0 = blockIdx.x * F4M, dr = threadIdx.x & (F4M - 1);
    const int nblk = cols / 64, nsteps = cols / F4K;
    const uint8_t* EB = ET ? ET[t.e] : W + (size_t) t.e * estride;   // p30
    const uint8_t* pcodes = EB + (size_t) (r0 + dr) * nblk * 32;
    const uint8_t* pscales = EB + (size_t) rows * nblk * 32 + (size_t) (r0 + dr) * nblk * 4;
    if (t.nk > 32)
        fp4_tile<64>(pcodes, pscales, nsteps, tab, ws, xs, x, ldx, t.row0, t.nk, y, ldy, r0);
    else if (t.nk > 16)
        fp4_tile<32>(pcodes, pscales, nsteps, tab, ws, xs, x, ldx, t.row0, t.nk, y, ldy, r0);
    else if (t.nk > 8 || !small)
        fp4_tile<16>(pcodes, pscales, nsteps, tab, ws, xs, x, ldx, t.row0, t.nk, y, ldy, r0);
    else if (t.nk > 4)   // p24
        fp4_tile_s<8, 4>(pcodes, pscales, nsteps, tab, ws, xs, x, ldx, t.row0, t.nk, y, ldy, r0);
    else
        fp4_tile_s<4, 2>(pcodes, pscales, nsteps, tab, ws, xs, x, ldx, t.row0, t.nk, y, ldy, r0);
}
// the tile list of k_fp4_gemm5 from the experts' row offsets (offs[e] .. offs[e + 1]): 64-token tiles, then the 32s, then the 16s
inline int fp4_tiles(const int* offs, FTile* out) {
    int n = 0;
    for (int pass = 0; pass < 3; ++pass)
        for (int e = 0; e < NE; ++e)
            for (int t0 = 0; t0 < offs[e + 1] - offs[e];) {
                const int left = offs[e + 1] - offs[e] - t0, tn = left > 32 ? 64 : left > 16 ? 32 : 16;
                if (tn == (pass == 0 ? 64 : pass == 1 ? 32 : 16)) out[n++] = FTile{e, offs[e] + t0, std::min(tn, left)};
                t0 += tn;
            }
    return n;
}

// p25: Q8 experts (int8 rows, a half scale for 32 values: the last 4 layers) by k_fp4_gemm5's tiles, straight from the bytes
// instead of k_q8s_f32 copies and a cublasSgemm an expert. Step of 32 values = one Q8 block a row: thread (row dr, half dh)
// decodes 16 of them (scale x int8 in f32, as k_q8s_f32) into shared memory; inputs, products and stores as fp4_tile
template <int TN>
__device__ __forceinline__ void q8_tile(const int8_t* __restrict__ qrow, const unsigned short* __restrict__ drow, int nsteps,
                                        float (*ws)[F4M], float* xsm, const float* __restrict__ x, int ldx, int row0, int nk, float* y,
                                        int ldy, int r0) {
    constexpr int TT = TN / 16;   // tokens a thread
    float (*xs)[TN] = (float (*)[TN]) xsm;
    const int ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
    const int dr = threadIdx.x & (F4M - 1), dh = threadIdx.x >> 7;
    const int tok = threadIdx.x >> 2, seg = (threadIdx.x & 3) * 8;
    const float* xr = tok < nk ? x + (size_t) (row0 + tok) * ldx + seg : nullptr;
    int4 q;
    unsigned short dq;
    float4 v0, v1;
    auto load = [&](int ks) {
        q = __ldg((const int4*) (qrow + (size_t) ks * F4K + dh * 16));
        dq = __ldg(drow + ks);
        v0 = v1 = make_float4(0.f, 0.f, 0.f, 0.f);
        if (xr) {
            v0 = __ldg((const float4*) (xr + ks * F4K));
            v1 = __ldg((const float4*) (xr + ks * F4K + 4));
        }
    };
    load(0);
    float acc[8][TT];
#pragma unroll
    for (int a = 0; a < 8; ++a)
#pragma unroll
        for (int k = 0; k < TT; ++k) acc[a][k] = 0.0f;
    for (int ks = 0; ks < nsteps; ++ks) {
        __syncthreads();
        {
            const float sc = __half2float(__ushort_as_half(dq));
            const int8_t* w = (const int8_t*) &q;
            const int k = dh * 16;
#pragma unroll
            for (int j = 0; j < 16; ++j) ws[k + j][dr] = sc * w[j];
            if (tok < TN) {
                xs[seg + 0][tok] = v0.x; xs[seg + 1][tok] = v0.y; xs[seg + 2][tok] = v0.z; xs[seg + 3][tok] = v0.w;
                xs[seg + 4][tok] = v1.x; xs[seg + 5][tok] = v1.y; xs[seg + 6][tok] = v1.z; xs[seg + 7][tok] = v1.w;
            }
        }
        __syncthreads();
        if (ks + 1 < nsteps) load(ks + 1);
#pragma unroll 8
        for (int k = 0; k < F4K; ++k) {
            const float4 w0 = *(const float4*) &ws[k][ty * 8], w1 = *(const float4*) &ws[k][ty * 8 + 4];
            const float w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            float xx[TT];
            if constexpr (TT == 4) {
                const float4 xv = *(const float4*) &xs[k][tx * 4];
                xx[0] = xv.x; xx[1] = xv.y; xx[2] = xv.z; xx[3] = xv.w;
            } else if constexpr (TT == 2) {
                const float2 xv = *(const float2*) &xs[k][tx * 2];
                xx[0] = xv.x; xx[1] = xv.y;
            } else {
                xx[0] = xs[k][tx];
            }
#pragma unroll
            for (int a = 0; a < 8; ++a)
#pragma unroll
                for (int kk = 0; kk < TT; ++kk) acc[a][kk] += w[a] * xx[kk];
        }
    }
#pragma unroll
    for (int i = 0; i < TT; ++i) {
        const int tk = tx * TT + i;
        if (tk >= nk) continue;
        float* yr = y + (size_t) (row0 + tk) * ldy + r0 + ty * 8;
#pragma unroll
        for (int a = 0; a < 8; ++a) yr[a] = acc[a][i];
    }
    __syncthreads();
}
// p26: p24's small tiles for the Q8 experts: q8_tile's decode, fp4_tile_s's products (RT rows x 1 token a thread)
template <int TN, int RT>
__device__ __forceinline__ void q8_tile_s(const int8_t* __restrict__ qrow, const unsigned short* __restrict__ drow, int nsteps,
                                          float (*ws)[F4M], float* xsm, const float* __restrict__ x, int ldx, int row0, int nk, float* y,
                                          int ldy, int r0) {
    static_assert(F4M / RT * TN == 256, "a thread a row group and token");
    float (*xs)[TN] = (float (*)[TN]) xsm;
    const int ty = threadIdx.x / TN, tx = threadIdx.x % TN;
    const int dr = threadIdx.x & (F4M - 1), dh = threadIdx.x >> 7;
    const int tok = threadIdx.x >> 2, seg = (threadIdx.x & 3) * 8;
    const float* xr = tok < nk ? x + (size_t) (row0 + tok) * ldx + seg : nullptr;
    int4 q;
    unsigned short dq;
    float4 v0, v1;
    auto load = [&](int ks) {
        q = __ldg((const int4*) (qrow + (size_t) ks * F4K + dh * 16));
        dq = __ldg(drow + ks);
        v0 = v1 = make_float4(0.f, 0.f, 0.f, 0.f);
        if (xr) {
            v0 = __ldg((const float4*) (xr + ks * F4K));
            v1 = __ldg((const float4*) (xr + ks * F4K + 4));
        }
    };
    load(0);
    float acc[RT];
#pragma unroll
    for (int a = 0; a < RT; ++a) acc[a] = 0.0f;
    for (int ks = 0; ks < nsteps; ++ks) {
        __syncthreads();
        {
            const float sc = __half2float(__ushort_as_half(dq));
            const int8_t* w = (const int8_t*) &q;
            const int k = dh * 16;
#pragma unroll
            for (int j = 0; j < 16; ++j) ws[k + j][dr] = sc * w[j];
            if (tok < TN) {
                xs[seg + 0][tok] = v0.x; xs[seg + 1][tok] = v0.y; xs[seg + 2][tok] = v0.z; xs[seg + 3][tok] = v0.w;
                xs[seg + 4][tok] = v1.x; xs[seg + 5][tok] = v1.y; xs[seg + 6][tok] = v1.z; xs[seg + 7][tok] = v1.w;
            }
        }
        __syncthreads();
        if (ks + 1 < nsteps) load(ks + 1);
#pragma unroll 8
        for (int k = 0; k < F4K; ++k) {
            float w[RT];
            if constexpr (RT == 4) {
                const float4 w0 = *(const float4*) &ws[k][ty * 4];
                w[0] = w0.x; w[1] = w0.y; w[2] = w0.z; w[3] = w0.w;
            } else {
                const float2 w0 = *(const float2*) &ws[k][ty * 2];
                w[0] = w0.x; w[1] = w0.y;
            }
            const float xx = xs[k][tx];
#pragma unroll
            for (int a = 0; a < RT; ++a) acc[a] += w[a] * xx;
        }
    }
    if (tx < nk) {
        float* yr = y + (size_t) (row0 + tx) * ldy + r0 + ty * RT;
#pragma unroll
        for (int a = 0; a < RT; ++a) yr[a] = acc[a];
    }
    __syncthreads();
}
// Q: int8 [NE][rows][cols], D: half [NE][rows][cols / 32] (qstride / dstride an expert); rows a multiple of F4M, cols of F4K
__global__ void __launch_bounds__(256) k_q8_gemm5(const int8_t* __restrict__ Q, const __half* __restrict__ D, size_t qstride, size_t dstride,
                                                  int cols, const FTile* tiles, const float* x, int ldx, float* y, int ldy, int small = 1,
                                                  const uint8_t* const* ET = nullptr) {   // p30: expert bases (int8 rows, then the scales at + qstride)
    __shared__ __align__(16) float ws[F4K][F4M];
    __shared__ __align__(16) float xs[F4K * F4N];
    const FTile t = tiles[blockIdx.y];
    const int r0 = blockIdx.x * F4M, dr = threadIdx.x & (F4M - 1), nsteps = cols / F4K;
    const int8_t* qrow = (ET ? (const int8_t*) ET[t.e] : Q + (size_t) t.e * qstride) + (size_t) (r0 + dr) * cols;
    const unsigned short* drow = (ET ? (const unsigned short*) (ET[t.e] + qstride) : (const unsigned short*) D + (size_t) t.e * dstride)
                                 + (size_t) (r0 + dr) * (cols / 32);
    if (t.nk > 32)
        q8_tile<64>(qrow, drow, nsteps, ws, xs, x, ldx, t.row0, t.nk, y, ldy, r0);
    else if (t.nk > 16)
        q8_tile<32>(qrow, drow, nsteps, ws, xs, x, ldx, t.row0, t.nk, y, ldy, r0);
    else if (t.nk > 8 || !small)
        q8_tile<16>(qrow, drow, nsteps, ws, xs, x, ldx, t.row0, t.nk, y, ldy, r0);
    else if (t.nk > 4)   // p26
        q8_tile_s<8, 4>(qrow, drow, nsteps, ws, xs, x, ldx, t.row0, t.nk, y, ldy, r0);
    else
        q8_tile_s<4, 2>(qrow, drow, nsteps, ws, xs, x, ldx, t.row0, t.nk, y, ldy, r0);
}


constexpr int F4N2 = 128;   // p3e: 128 tokens a tile, 8 rows x 8 tokens a thread
template <bool PLANAR>
__global__ void __launch_bounds__(256) k_fp4_gemm2(const uint8_t* __restrict__ W, size_t estride, int rows, int cols, const int* offs,
                                                  const float* x, int ldx, float* y, int ldy) {
    __shared__ __align__(16) float ws[F4K][F4M];
    __shared__ __align__(16) float xs[F4K][F4N2];
    __shared__ float tab[16];
    const int e = blockIdx.y, o0 = offs[e], ne = offs[e + 1] - o0;
    if (ne == 0) return;
    if (threadIdx.x < 16) tab[threadIdx.x] = c_fp4[threadIdx.x];
    const int r0 = blockIdx.x * F4M, ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
    const int dr = threadIdx.x & (F4M - 1), dh = threadIdx.x >> 7;
    const int tok = threadIdx.x >> 1, seg = (threadIdx.x & 1) * 16;   // 128 tokens, 16 inputs each
    const int nblk = cols / 64, rb = nblk * 36, nsteps = cols / F4K;
    const uint8_t* rowp = W + (size_t) e * estride + (size_t) (r0 + dr) * rb;   // interleaved rows
    const uint8_t* pcodes = W + (size_t) e * estride + (size_t) (r0 + dr) * nblk * 32;   // p4d: planar (codes, then scales)
    const uint8_t* pscales = W + (size_t) e * estride + (size_t) rows * nblk * 32 + (size_t) (r0 + dr) * nblk * 4;
    for (int t0 = 0; t0 < ne; t0 += F4N2) {
        const int nk = min(F4N2, ne - t0);
        const float* xr = tok < nk ? x + (size_t) (o0 + t0 + tok) * ldx + seg : nullptr;
        uint8_t rd;
        uint32_t q0, q1;
        float4 v0, v1, v2, v3;
        auto load = [&](int ks) {   // step ks: block ks / 2, sub-block (ks & 1) * 2 + dh
            const int sb = (ks & 1) * 2 + dh;
            if (PLANAR) {
                const uint8_t* c = pcodes + (size_t) (ks >> 1) * 32 + sb * 8;
                rd = __ldg(pscales + (size_t) (ks >> 1) * 4 + sb);
                q0 = __ldg((const uint32_t*) c);
                q1 = __ldg((const uint32_t*) (c + 4));
            } else {
                const uint8_t* blk = rowp + (size_t) (ks >> 1) * 36;
                rd = __ldg(blk + sb);
                q0 = __ldg((const uint32_t*) (blk + 4 + sb * 8));
                q1 = __ldg((const uint32_t*) (blk + 8 + sb * 8));
            }
            v0 = v1 = v2 = v3 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (xr) {
                v0 = __ldg((const float4*) (xr + ks * F4K));
                v1 = __ldg((const float4*) (xr + ks * F4K + 4));
                v2 = __ldg((const float4*) (xr + ks * F4K + 8));
                v3 = __ldg((const float4*) (xr + ks * F4K + 12));
            }
        };
        load(0);
        float acc[8][8];
#pragma unroll
        for (int a = 0; a < 8; ++a)
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[a][k] = 0.0f;
        for (int ks = 0; ks < nsteps; ++ks) {
            __syncthreads();
            {
                const float d = ue4m3(rd);
                const int k = dh * 16;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t b0 = (q0 >> (8 * j)) & 0xFFu, b1 = (q1 >> (8 * j)) & 0xFFu;
                    ws[k + j][dr] = tab[b0 & 15] * d;
                    ws[k + 4 + j][dr] = tab[b1 & 15] * d;
                    ws[k + 8 + j][dr] = tab[b0 >> 4] * d;
                    ws[k + 12 + j][dr] = tab[b1 >> 4] * d;
                }
                xs[seg + 0][tok] = v0.x; xs[seg + 1][tok] = v0.y; xs[seg + 2][tok] = v0.z; xs[seg + 3][tok] = v0.w;
                xs[seg + 4][tok] = v1.x; xs[seg + 5][tok] = v1.y; xs[seg + 6][tok] = v1.z; xs[seg + 7][tok] = v1.w;
                xs[seg + 8][tok] = v2.x; xs[seg + 9][tok] = v2.y; xs[seg + 10][tok] = v2.z; xs[seg + 11][tok] = v2.w;
                xs[seg + 12][tok] = v3.x; xs[seg + 13][tok] = v3.y; xs[seg + 14][tok] = v3.z; xs[seg + 15][tok] = v3.w;
            }
            __syncthreads();
            if (ks + 1 < nsteps) load(ks + 1);
#pragma unroll 8
            for (int k = 0; k < F4K; ++k) {
                const float4 w0 = *(const float4*) &ws[k][ty * 8], w1 = *(const float4*) &ws[k][ty * 8 + 4];
                const float4 xa = *(const float4*) &xs[k][tx * 8], xb = *(const float4*) &xs[k][tx * 8 + 4];
                const float w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w}, xx[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int a = 0; a < 8; ++a)
#pragma unroll
                    for (int kk = 0; kk < 8; ++kk) acc[a][kk] += w[a] * xx[kk];
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int tk = tx * 8 + i;
            if (tk >= nk) continue;
            float* yr = y + (size_t) (o0 + t0 + tk) * ldy + r0 + ty * 8;
#pragma unroll
            for (int a = 0; a < 8; ++a) yr[a] = acc[a][i];
        }
        __syncthreads();
    }
}

}  // namespace ink2
