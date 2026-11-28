// ink2_dflash2.cuh (2026-11-27, p74): the DFlash 2 drafter (incoai/Inkling-2-Mini-DFlash2, Inco AI + z-lab, Apache-2.0) beside
// DSpark; risar/dflash2_torch.py is its reference (SGLang's DFlash2DraftModel with the checkpoint's Inkling patch, llama.cpp
// b12777's dflash.cpp + dflash2.cpp with the four fixes of zzzprojectpage/inkling2-p100-dflash2). It runs whole on both cards
// like DSpark (the same inputs give the same bits); only the target's head is split by vocabulary, each card a top-16 of its half.
//  - context: per committed position p the same four features as DSpark's (D.feat): x = rms(fc . [F0|F1|F2|F3]) * hidden_norm; layer
//    il keeps K = rope(rms_head(k_proj . x) * k_norm), V = v_proj . x in a ring of D2RING rows (sliding window 2048)
//  - block: [rms(tok_embd[anchor]) * tok_norm, the mask token's row x 7] at P..P+7 through 5 layers, each sublayer between a 2-tap
//    grouped dynamic convolution of its input (side 0) and of its output (side 1), both from one projection of the normed input;
//    32 query heads on 8 KV heads, NeoX RoPE base 1e4, row t sees the keys from P + t - 2047 and the whole block
//  - head: the target's output on rows 1..7, a top-16 per card; the host merges them and walks the selector's lattice (Engine::d2_draft)
#pragma once
#include "ink2_dspark.cuh"   // DBLK, k_ds_headrope, k_ds_vstore, k_ds_swiglu (and ink2_kern.cuh)

namespace ink2 {

constexpr int D2L = 5, D2FF = 6912, D2H = 32, D2Q = D2H * HD, D2G = 16, D2NG = N / D2G, D2DYN = 4 * D2NG, D2K = 16, D2R = 256;
constexpr int D2WIN = 2048, D2RING = 2560, D2KC = 32;   // the window; ring rows (>= window + block, a reading chunk's tail fits); keys a part
static_assert(D2H / NKV == 4, "k_d2_attn: 4 query heads a KV head");

// the convolution of R rows: out[t][e] = (b[0][e] + d[t][s][0][e/16]) y[t][e] + (t >= 1: (b[1][e] + d[t][s][1][e/16]) y[t-1][e]);
// d = the rows' projections [R][2 sides][2 taps][144], b = the side's base kernel [2 taps][N]
__global__ void k_d2_conv(const float* __restrict__ y, const float* __restrict__ d, const float* __restrict__ b, int side, float* __restrict__ out) {
    const int t = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= N) return;
    const float* dt = d + (size_t) t * D2DYN + side * 2 * D2NG;
    float o = (b[e] + dt[e / D2G]) * y[(size_t) t * N + e];
    if (t > 0) o += (b[N + e] + dt[D2NG + e / D2G]) * y[(size_t) (t - 1) * N + e];
    out[(size_t) t * N + e] = o;
}

// the block's attention, a part of D2KC keys of one KV head for all R rows x 4 query heads at once (the keys and values are read once):
// keys j in [j0, j0 + D2KC) of the ring (row j % D2RING), row t sees lo + t <= j < hi (lo = P - 2047, hi = P + R); out (m, l, acc[HD])
// per (t, head, part). Threads: 8 warps; scores lane = key, warp w = queries 4w..4w+3 (query = t * 4 + head in the group)
__global__ void __launch_bounds__(256) k_d2_attn(const float* __restrict__ q, const __half* __restrict__ kc, const __half* __restrict__ vc,
                                                 int lo, int hi, int R, float* __restrict__ part, int nparts) {
    __shared__ __align__(16) float qs[DBLK * 4][HD];   // 16 KB
    __shared__ __align__(16) __half ks[D2KC][HD + 2];   // padded: lane j reads row j, banks apart
    __shared__ __align__(16) __half vs[D2KC][HD];
    __shared__ float ps[DBLK * 4][D2KC];
    const int p = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, w = tid >> 5, l = tid & 31;
    const int j0 = max(0, lo) + p * D2KC, j1 = min(hi, j0 + D2KC), nq = R * 4;
    for (int i = tid; i < nq * HD; i += 256) {   // q row t, head g * 4 + h: q[t][g * 4 + h][d]
        const int qi = i / HD, dd = i % HD;
        qs[qi][dd] = q[((size_t) (qi >> 2) * D2H + g * 4 + (qi & 3)) * HD + dd];
    }
    for (int i = tid; i < D2KC * HD / 2; i += 256) {
        const int jj = i / (HD / 2), dd = (i % (HD / 2)) * 2, j = j0 + jj;
        __half2 kv = __float2half2_rn(0.0f), vv = kv;
        if (j < j1) {
            const size_t r = (size_t) (j % D2RING) * NKVD + g * HD + dd;
            kv = *(const __half2*) (kc + r);
            vv = *(const __half2*) (vc + r);
        }
        *(__half2*) &ks[jj][dd] = kv;
        *(__half2*) &vs[jj][dd] = vv;
    }
    __syncthreads();
    const float sc = 1.0f / sqrtf((float) HD);
    for (int qq = 0; qq < 4; ++qq) {
        const int qi = w * 4 + qq;
        if (qi >= nq) break;
        const int j = j0 + l, t = qi >> 2;
        float s = -INFINITY;
        if (j < j1 && j >= lo + t) {
            float a = 0.0f;
#pragma unroll 8
            for (int dd = 0; dd < HD; dd += 2) {
                const float2 kf = __half22float2(*(const __half2*) &ks[l][dd]);
                a += qs[qi][dd] * kf.x + qs[qi][dd + 1] * kf.y;
            }
            s = a * sc;
        }
        const float m = warp_max(s);
        const float e = m == -INFINITY ? 0.0f : expf(s - m);
        const float sum = warp_sum(e);
        ps[qi][l] = e;
        float* o = part + (((size_t) t * D2H + g * 4 + (qi & 3)) * nparts + p) * (HD + 2);
        if (l == 0) { o[0] = m; o[1] = sum; }
    }
    __syncwarp();
    for (int qq = 0; qq < 4; ++qq) {
        const int qi = w * 4 + qq;
        if (qi >= nq) break;
        const int t = qi >> 2;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int jj = 0; jj < D2KC; ++jj) {
            const float pj = ps[qi][jj];
            const uint2 vw = *(const uint2*) &vs[jj][l * 4];   // 8 bytes a lane: no bank conflict
            const float2 v01 = __half22float2(*(const __half2*) &vw.x), v23 = __half22float2(*(const __half2*) &vw.y);
            a0 += pj * v01.x; a1 += pj * v01.y; a2 += pj * v23.x; a3 += pj * v23.y;
        }
        float* o = part + (((size_t) t * D2H + g * 4 + (qi & 3)) * nparts + p) * (HD + 2) + 2 + l * 4;
        o[0] = a0; o[1] = a1; o[2] = a2; o[3] = a3;
    }
}
__global__ void __launch_bounds__(128) k_d2_comb(const float* part, int nparts, float* out) {   // out [R][D2H][HD]
    const int t = blockIdx.y, h = blockIdx.x;
    const float* p = part + ((size_t) t * D2H + h) * nparts * (HD + 2);
    float M = -INFINITY;
    for (int i = 0; i < nparts; ++i) M = fmaxf(M, p[(size_t) i * (HD + 2)]);
    float L = 0.0f, acc = 0.0f;
    for (int i = 0; i < nparts; ++i) {
        const float* pi = p + (size_t) i * (HD + 2);
        if (pi[1] == 0.0f) continue;
        const float f = expf(pi[0] - M);
        L += pi[1] * f;
        acc += pi[2 + threadIdx.x] * f;
    }
    out[((size_t) t * D2H + h) * HD + threadIdx.x] = acc / L;
}

// a top-16 (values descending, ties: the smaller id first) of the entries [p * span, min(n, (p + 1) * span)) of row r (r = blockIdx.x,
// p = blockIdx.y; row r's entries at v + r * ldv): each thread its own top-16, then 16 rounds of a block argmax. The ids are the
// entry's index + id0, or ids_in[] at the same place. Out: [rows][gridDim.y][16]. Two passes make a row's top-16 (D2TS parts, then
// the D2TS x 16 kept): ~30 us for 7 rows of 100k instead of 1.2 ms with one block a row (27. 11.)
constexpr int D2TS = 32;
__device__ __forceinline__ bool d2_better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }
__global__ void __launch_bounds__(128) k_d2_top16(const float* __restrict__ v, const int* __restrict__ ids_in, int ldv, int n, int span, int id0,
                                                  float* __restrict__ ov, int* __restrict__ oi) {
    __shared__ float sv[128 * D2K];
    __shared__ int si[128 * D2K];
    __shared__ float rv[4];
    __shared__ int ri[4], rp[4];
    const int tid = threadIdx.x, row = blockIdx.x, p = blockIdx.y;
    const float* x = v + (size_t) row * ldv;
    const int* xi = ids_in ? ids_in + (size_t) row * ldv : nullptr;
    const int e0 = p * span, e1 = min(n, e0 + span);
    float tv[D2K];
    int ti[D2K];
#pragma unroll
    for (int k = 0; k < D2K; ++k) { tv[k] = -INFINITY; ti[k] = 0x7fffffff; }
    for (int i = e0 + tid; i < e1; i += 128) {
        const float a = x[i];
        const int id = xi ? xi[i] : i + id0;
        if (id >= 0 && d2_better(a, id, tv[D2K - 1], ti[D2K - 1])) {   // an unrolled swap chain: the list stays in registers
            float ca = a;
            int ci = id;
#pragma unroll
            for (int k = 0; k < D2K; ++k)
                if (d2_better(ca, ci, tv[k], ti[k])) {
                    const float sa = tv[k];
                    const int sj = ti[k];
                    tv[k] = ca;
                    ti[k] = ci;
                    ca = sa;
                    ci = sj;
                }
        }
    }
#pragma unroll
    for (int k = 0; k < D2K; ++k) { sv[tid * D2K + k] = tv[k]; si[tid * D2K + k] = ti[k]; }
    __syncthreads();
    int head = 0;   // this thread's next unused entry (its list is sorted)
    float* o = ov + ((size_t) row * gridDim.y + p) * D2K;
    int* oid = oi + ((size_t) row * gridDim.y + p) * D2K;
    for (int r = 0; r < D2K; ++r) {
        float bv = head < D2K ? sv[tid * D2K + head] : -INFINITY;
        int bi = head < D2K ? si[tid * D2K + head] : 0x7fffffff, bp = tid;
        for (int s = 16; s; s >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, bv, s);
            const int i2 = __shfl_xor_sync(0xffffffffu, bi, s), p2 = __shfl_xor_sync(0xffffffffu, bp, s);
            if (d2_better(v2, i2, bv, bi)) { bv = v2; bi = i2; bp = p2; }
        }
        if ((tid & 31) == 0) { rv[tid >> 5] = bv; ri[tid >> 5] = bi; rp[tid >> 5] = bp; }
        __syncthreads();
        if (tid == 0) {
            for (int k = 1; k < 4; ++k)
                if (d2_better(rv[k], ri[k], rv[0], ri[0])) { rv[0] = rv[k]; ri[0] = ri[k]; rp[0] = rp[k]; }
            o[r] = rv[0];
            oid[r] = ri[0] == 0x7fffffff ? -1 : ri[0];
        }
        __syncthreads();
        if (tid == rp[0]) ++head;
        __syncthreads();
    }
}

// the drafter's own Q8S GEMV for T <= 8 rows of x (not the target's kernel: no bits to keep, only the drafter's on both cards):
// y[t][r] = sum_c q[r][c] d[r][c / 32] x[t][c]. A block's 512-column chunk of x is staged once in shared memory (lane l's 16 columns
// at l * 17: no bank conflicts), RW rows of q a warp, 8 warps a block, every column in the block (fixed order: the same bits on
// both cards). k_gemv8_t1TT read x from L2 once a warp: 8 rows cost ~107 us a matrix of the block (profile 27. 11.)
template <int TT, int RW>
__global__ void __launch_bounds__(256) k_d2_gemv(const int8_t* __restrict__ q, const __half* __restrict__ d, int rows, int cols,
                                                 const float* __restrict__ x, int ldx, float* __restrict__ y, int ldy) {
    __shared__ float xs[TT][32 * 17];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r0 = (blockIdx.x * 8 + warp) * RW;
    float acc[RW][TT];
#pragma unroll
    for (int a = 0; a < RW; ++a)
#pragma unroll
        for (int t = 0; t < TT; ++t) acc[a][t] = 0.0f;
    for (int c0 = 0; c0 < cols; c0 += 512) {
        const int nc = min(512, cols - c0);
        __syncthreads();
        for (int i = threadIdx.x; i < TT * 512; i += 256) {
            const int t = i >> 9, c = i & 511;
            xs[t][(c >> 4) * 17 + (c & 15)] = c < nc ? x[(size_t) t * ldx + c0 + c] : 0.0f;
        }
        __syncthreads();
        const int c = c0 + lane * 16;
        if (c < cols) {
            float wf[RW][16], sc[RW];
#pragma unroll
            for (int a = 0; a < RW; ++a) {
                const int r = r0 + a;
                const int4 w4 = r < rows ? __ldg((const int4*) (q + (size_t) r * cols + c)) : make_int4(0, 0, 0, 0);
                sc[a] = r < rows ? __half2float(d[(size_t) r * (cols / 32) + c / 32]) : 0.0f;
                const int8_t* w = (const int8_t*) &w4;
#pragma unroll
                for (int j = 0; j < 16; ++j) wf[a][j] = (float) w[j];
            }
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                float xv[16];
#pragma unroll
                for (int j = 0; j < 16; ++j) xv[j] = xs[t][lane * 17 + j];
#pragma unroll
                for (int a = 0; a < RW; ++a) {
                    float s = 0.0f;
#pragma unroll
                    for (int j = 0; j < 16; ++j) s += wf[a][j] * xv[j];
                    acc[a][t] += sc[a] * s;
                }
            }
        }
    }
#pragma unroll
    for (int a = 0; a < RW; ++a) {
#pragma unroll
        for (int t = 0; t < TT; ++t) acc[a][t] = warp_sum(acc[a][t]);
        if (lane == 0 && r0 + a < rows)
#pragma unroll
            for (int t = 0; t < TT; ++t) y[(size_t) t * ldy + r0 + a] = acc[a][t];
    }
}

// the drafter's GEMV, take 2 (27. 11.: k_d2_gemv went ~180 GB/s, a barrier and x's staging before every 512 columns of only 4 rows
// a warp): a lane a row of q, 32 bytes (one Q8 block, a whole sector) a step; x's 512-column piece in shared memory as [column][8]
// (all lanes read the same column: a broadcast). A block is 256 rows x the columns [ch * cch, (ch + 1) * cch); with one column
// chunk (gridDim.y == 1) it writes y, else part[ch][t][rows], which k_d2_sum adds in chunk order (fixed: the same bits on both cards)
constexpr int D2PC = 512;   // columns of x staged at once
template <int TT>
__global__ void __launch_bounds__(256) k_d2_gemv2(const int8_t* __restrict__ q, const __half* __restrict__ d, int rows, int cols,
                                                  const float* __restrict__ x, int ldx, int cch, float* __restrict__ y, int ldy,
                                                  float* __restrict__ part) {
    __shared__ float4 xs[D2PC * 2];   // 16 KB
    const int r = blockIdx.x * 256 + threadIdx.x, ch = blockIdx.y, cb = ch * cch, ce = min(cols, cb + cch);
    const bool live = r < rows;
    const int8_t* qr = q + (size_t) (live ? r : 0) * cols;
    const __half* dr = d + (size_t) (live ? r : 0) * (cols / 32);
    float acc[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.0f;
    for (int p0 = cb; p0 < ce; p0 += D2PC) {
        const int nc = min(D2PC, ce - p0);
        __syncthreads();
        for (int i = threadIdx.x; i < nc * 8; i += 256) {
            const int c = i >> 3, t = i & 7;
            ((float*) xs)[i] = t < TT ? x[(size_t) t * ldx + p0 + c] : 0.0f;
        }
        __syncthreads();
        if (!live) continue;
#pragma unroll 2
        for (int b = 0; b < nc / 32; ++b) {
            const int c = p0 + b * 32;
            const int4 w0 = __ldg((const int4*) (qr + c)), w1 = __ldg((const int4*) (qr + c + 16));
            const float sc = __half2float(dr[c / 32]);
            const int8_t* wa = (const int8_t*) &w0;
            const int8_t* wb = (const int8_t*) &w1;
            float s[TT];
#pragma unroll
            for (int t = 0; t < TT; ++t) s[t] = 0.0f;
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float w = (float) (j < 16 ? wa[j] : wb[j - 16]);
                const float4 xa = xs[(b * 32 + j) * 2], xb = xs[(b * 32 + j) * 2 + 1];
                const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int t = 0; t < TT; ++t) s[t] += w * xv[t];
            }
#pragma unroll
            for (int t = 0; t < TT; ++t) acc[t] += sc * s[t];
        }
    }
    if (!live) return;
    if (gridDim.y == 1) {
#pragma unroll
        for (int t = 0; t < TT; ++t) y[(size_t) t * ldy + r] = acc[t];
    } else {
#pragma unroll
        for (int t = 0; t < TT; ++t) part[((size_t) ch * TT + t) * rows + r] = acc[t];
    }
}
// take 3 (27. 11.: k_d2_gemv2 ~143 GB/s, bound by its two LDS.128 of x a column and row): a lane takes 4 rows (r, r + 128, r + 256,
// r + 384 of the block's 512), so each x read from shared memory serves 4 rows; the weight is scaled once (sc * w) and then used for
// the TT rows of x. 128 threads a block; the column chunks (cch, a multiple of 32) as in take 2, part [ch][t][rows] for k_d2_sum
template <int TT>
__global__ void __launch_bounds__(128) k_d2_gemv3(const int8_t* __restrict__ q, const __half* __restrict__ d, int rows, int cols,
                                                  const float* __restrict__ x, int ldx, int cch, float* __restrict__ y, int ldy,
                                                  float* __restrict__ part) {
    __shared__ float4 xs[D2PC * 2];   // 16 KB
    const int ch = blockIdx.y, cb = ch * cch, ce = min(cols, cb + cch);
    int rr[4];
    bool lv[4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        rr[a] = blockIdx.x * 512 + a * 128 + threadIdx.x;
        lv[a] = rr[a] < rows;
    }
    float acc[4][TT];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int t = 0; t < TT; ++t) acc[a][t] = 0.0f;
    for (int p0 = cb; p0 < ce; p0 += D2PC) {
        const int nc = min(D2PC, ce - p0);
        __syncthreads();
        for (int i = threadIdx.x; i < nc * 8; i += 128) {
            const int c = i >> 3, t = i & 7;
            ((float*) xs)[i] = t < TT ? x[(size_t) t * ldx + p0 + c] : 0.0f;
        }
        __syncthreads();
        for (int b = 0; b < nc / 32; ++b) {
            const int c = p0 + b * 32;
            int4 w[4][2];
            float sc[4];
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const int8_t* qr = q + (size_t) (lv[a] ? rr[a] : 0) * cols + c;
                w[a][0] = lv[a] ? __ldg((const int4*) qr) : make_int4(0, 0, 0, 0);
                w[a][1] = lv[a] ? __ldg((const int4*) (qr + 16)) : make_int4(0, 0, 0, 0);
                sc[a] = lv[a] ? __half2float(d[(size_t) rr[a] * (cols / 32) + c / 32]) : 0.0f;
            }
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float4 xa = xs[(b * 32 + j) * 2], xb = xs[(b * 32 + j) * 2 + 1];
                const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    const float wf = sc[a] * (float) ((const int8_t*) &w[a][j >> 4])[j & 15];
#pragma unroll
                    for (int t = 0; t < TT; ++t) acc[a][t] += wf * xv[t];
                }
            }
        }
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        if (!lv[a]) continue;
        if (gridDim.y == 1) {
#pragma unroll
            for (int t = 0; t < TT; ++t) y[(size_t) t * ldy + rr[a]] = acc[a][t];
        } else {
#pragma unroll
            for (int t = 0; t < TT; ++t) part[((size_t) ch * TT + t) * rows + rr[a]] = acc[a][t];
        }
    }
}
__global__ void k_d2_sum(const float* __restrict__ part, int nch, int rows, int T, float* __restrict__ y, int ldy) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * T) return;
    const int t = i / rows, r = i % rows;
    float s = 0.0f;
    for (int ch = 0; ch < nch; ++ch) s += part[((size_t) ch * T + t) * rows + r];
    y[(size_t) t * ldy + r] = s;
}

// columns [c0, c0 + nc) of a Q8S matrix as f32 [rows][nc] (fc's four 2304-column blocks for the reading path's sgemm)
__global__ void k_q8s_f32_cols(const int8_t* __restrict__ q, const __half* __restrict__ d, int rows, int cols, int c0, int nc, float* out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) rows * nc) return;
    const int r = (int) (i / nc), c = c0 + (int) (i % nc);
    out[i] = (float) q[(size_t) r * cols + c] * __half2float(d[(size_t) r * (cols / 32) + c / 32]);
}

}  // namespace ink2
