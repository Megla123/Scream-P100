// ink2_kern.cuh (2026-11-21): Inkling 2 Mini on both P100s, the kernels of v1 (pisanje, okna do MAXT zetonov). The math is
// llama.cpp's src/models/inkling2.cpp at 690b87d, see VODNIK-inkling2.md "Model". Q8S GEMV (k_gemv8) and F32 GEMV (k_gemv32)
// come from Hy4 (hy4_kern.cuh); the rest is new. Activations are f32, the KV cache f16 as in llama.cpp.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ink2 {

constexpr int N = 2304, NH = 24, NKV = 8, HD = 128, NQ = NH * HD, NKVD = NKV * HD, NREL = 16, EXT = 1024;
constexpr int NE = 128, TOPK = 8, FF = 1152, NV = 201024, NVU = 200058;
constexpr int NPHYS = 24, LS = 6, LE = 18, NPASS = 3, NCORE = LE - LS, NSLOT = NPHYS + (NPASS - 1) * NCORE, BLK = 6;
constexpr int RANK = 64, NCONV = 4, SWA = 1024, MAXT = 16, MAXB = 4;
#ifndef INK2_RING
#define INK2_RING 4096
#endif
constexpr int RING = INK2_RING;   // window slots: rows by position modulo RING (an MAXT window or a prompt chunk overwrites no row it still reads)
#ifndef INK2_NC
#define INK2_NC 2
#endif
constexpr int NC = INK2_NC, NH_C = NH / NC, NKV_C = NKV / NC, QC = NH_C * HD, KVC = NKV_C * HD, FF_C = FF / NC, NV_C = NV / NC;
constexpr int KC = 128;      // keys a block of the attention (k_attn_part)

static void ck(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s at %d: %s\n", what, line, cudaGetErrorString(e));
        std::exit(2);
    }
}
#define CK(x) ck((x), #x, __LINE__)

// Q8_0 kept split as in the split engine's Q8S: q = int8 [rows][cols], d = f16 [rows][cols / 32]
struct Mat {
    uint32_t type = 0;
    int rows = 0, cols = 0;
    const void* d = nullptr;
    const int8_t* q = nullptr;
};

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float s = 0;
    if (threadIdx.x < 32) {
        s = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
        s = warp_sum(s);
        if (threadIdx.x == 0) red[0] = s;
    }
    __syncthreads();
    s = red[0];
    __syncthreads();
    return s;
}
__device__ __forceinline__ float block_max(float v, float* red) {
    v = warp_max(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float s = -INFINITY;
    if (threadIdx.x < 32) {
        s = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY;
        s = warp_max(s);
        if (threadIdx.x == 0) red[0] = s;
    }
    __syncthreads();
    s = red[0];
    __syncthreads();
    return s;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

// ------------------------------------------------------------------------------------------------ GEMV (from hy4_kern.cuh)
// y[t][r] = W[r] . x[t]: a block of 8 warps holds GW_R rows a warp, the input goes through shared memory in chunks of 512 columns
constexpr int GW_R = 2, GW_BR = 8 * GW_R;
__global__ void __launch_bounds__(256) k_gemv8(const int8_t* __restrict__ q, const __half* __restrict__ d, int rows, int cols, const float* x,
                                                int ldx, float* y, int ldy, int T) {
    extern __shared__ float xs[];   // [T][16][32]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rb = blockIdx.x * GW_BR, r0 = rb + warp * GW_R;
    float acc[GW_R][MAXT];
#pragma unroll
    for (int a = 0; a < GW_R; ++a)
#pragma unroll
        for (int t = 0; t < MAXT; ++t) acc[a][t] = 0.0f;
    for (int c0 = 0; c0 < cols; c0 += 512) {
        __syncthreads();
        for (int i = threadIdx.x; i < T * 512; i += 256) {
            const int t = i >> 9, c = i & 511;
            xs[t * 512 + (c & 15) * 32 + (c >> 4)] = c0 + c < cols ? x[(size_t) t * ldx + c0 + c] : 0.0f;
        }
        __syncthreads();
        const int c = c0 + lane * 16;
        if (c >= cols) continue;
#pragma unroll
        for (int a = 0; a < GW_R; ++a) {
            const int r = r0 + a;
            if (r >= rows) break;
            const int4 w4 = __ldg((const int4*) (q + (size_t) r * cols + c));
            const float sc = __half2float(d[(size_t) r * (cols / 32) + c / 32]);
            const int8_t* w = (const int8_t*) &w4;
            float wf[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) wf[j] = (float) w[j];
#pragma unroll
            for (int t = 0; t < MAXT; ++t) {
                if (t >= T) break;
                const float* xt = xs + t * 512 + lane;
                float s = 0;
#pragma unroll
                for (int j = 0; j < 16; ++j) s += wf[j] * xt[j * 32];
                acc[a][t] += sc * s;
            }
        }
    }
#pragma unroll
    for (int a = 0; a < GW_R; ++a) {
        const int r = r0 + a;
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= T) break;
            const float s = warp_sum(acc[a][t]);
            if (lane == 0 && r < rows) y[(size_t) t * ldy + r] = s;
        }
    }
}
// F32 rows (router, loop controllers, queries): 4 values a lane a chunk of 128 columns
__global__ void __launch_bounds__(256) k_gemv32(const float* __restrict__ W, int rows, int cols, const float* x, int ldx, float* y, int ldy, int T) {
    extern __shared__ float xs[];   // [T][4][32]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rb = blockIdx.x * GW_BR, r0 = rb + warp * GW_R;
    float acc[GW_R][MAXT];
#pragma unroll
    for (int a = 0; a < GW_R; ++a)
#pragma unroll
        for (int t = 0; t < MAXT; ++t) acc[a][t] = 0.0f;
    for (int c0 = 0; c0 < cols; c0 += 128) {
        __syncthreads();
        for (int i = threadIdx.x; i < T * 128; i += 256) {
            const int t = i >> 7, c = i & 127;
            xs[t * 128 + (c & 3) * 32 + (c >> 2)] = c0 + c < cols ? x[(size_t) t * ldx + c0 + c] : 0.0f;
        }
        __syncthreads();
        const int c = c0 + lane * 4;
        if (c >= cols) continue;
#pragma unroll
        for (int a = 0; a < GW_R; ++a) {
            const int r = r0 + a;
            if (r >= rows) break;
            const float4 w = __ldg((const float4*) (W + (size_t) r * cols + c));
#pragma unroll
            for (int t = 0; t < MAXT; ++t) {
                if (t >= T) break;
                const float* xt = xs + t * 128 + lane;
                acc[a][t] += w.x * xt[0] + w.y * xt[32] + w.z * xt[64] + w.w * xt[96];
            }
        }
    }
#pragma unroll
    for (int a = 0; a < GW_R; ++a) {
        const int r = r0 + a;
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= T) break;
            const float s = warp_sum(acc[a][t]);
            if (lane == 0 && r < rows) y[(size_t) t * ldy + r] = s;
        }
    }
}

// ------------------------------------------------------------------------------------------------ norms, mixing, control
// y[t] = x[t] / rms(x[t]) (* w): one block of 256 per row
__global__ void k_rms(const float* x, int ldx, const float* w, float* y, int ldy, int n, float eps) {
    __shared__ float red[32];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float ss = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    const float k = 1.0f / sqrtf(ss / n + eps);
    float* yr = y + (size_t) blockIdx.x * ldy;
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = w ? xr[i] * k * w[i] : xr[i] * k;
}
// control's gate input: [rms(x); rms(m)] (eps 1e-6, no weight), 2N a row
__global__ void k_ctrl_in(const float* x, const float* m, int ldm, float* xm) {
    __shared__ float red[32];
    const int t = blockIdx.x;
    const float* a = x + (size_t) t * N;
    const float* b = m + (size_t) t * ldm;
    float sa = 0, sb = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) { sa += a[i] * a[i]; sb += b[i] * b[i]; }
    sa = block_sum(sa, red);
    sb = block_sum(sb, red);
    const float ka = 1.0f / sqrtf(sa / N + 1e-6f), kb = 1.0f / sqrtf(sb / N + 1e-6f);
    float* o = xm + (size_t) t * 2 * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) { o[i] = a[i] * ka; o[N + i] = b[i] * kb; }
}
// x' = x - tanh(f) x + tanh(g) m   (llama.cpp: (cur - forget*cur) + gain*message)
__global__ void k_ctrl_out(const float* x, const float* m, int ldm, const float* gp, const float* fp, float* y, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int t = i / N, c = i % N;
    const float xv = x[i], f = tanhf(fp[i]), g = tanhf(gp[i]);
    y[i] = (xv - f * xv) + g * m[(size_t) t * ldm + c];
}
// attention over the depth (llama.cpp mix()): states [blocks..., cur] normalized by rms (eps 1e-6), scores state.query / sqrt(N),
// softmax over the states, output = the weighted sum of the normalized states; nq queries (query j at q + j*N), out [nq][T][N]
struct States {
    const float* p[MAXB + 2];
    int n;
};
__global__ void __launch_bounds__(256) k_mix(States S, const float* q, int nq, float* out) {
    __shared__ float red[32];
    __shared__ float inv[MAXB + 2], w[4][MAXB + 2];   // nq <= 4 (the loop messages)
    const int t = blockIdx.x;
    for (int s = 0; s < S.n; ++s) {
        const float* x = S.p[s] + (size_t) t * N;
        float ss = 0;
        for (int i = threadIdx.x; i < N; i += blockDim.x) ss += x[i] * x[i];
        ss = block_sum(ss, red);
        if (threadIdx.x == 0) inv[s] = 1.0f / sqrtf(ss / N + 1e-6f);
    }
    __syncthreads();
    for (int j = 0; j < nq; ++j)
        for (int s = 0; s < S.n; ++s) {
            const float* x = S.p[s] + (size_t) t * N;
            float d = 0;
            for (int i = threadIdx.x; i < N; i += blockDim.x) d += x[i] * inv[s] * q[(size_t) j * N + i];
            d = block_sum(d, red);
            if (threadIdx.x == 0) w[j][s] = d / sqrtf((float) N);
        }
    __syncthreads();
    if (threadIdx.x < nq) {
        const int j = threadIdx.x;
        float m = -INFINITY;
        for (int s = 0; s < S.n; ++s) m = fmaxf(m, w[j][s]);
        float l = 0;
        for (int s = 0; s < S.n; ++s) { w[j][s] = expf(w[j][s] - m); l += w[j][s]; }
        for (int s = 0; s < S.n; ++s) w[j][s] /= l;
    }
    __syncthreads();
    for (int j = 0; j < nq; ++j) {   // out [nq][T][N]: each query's rows contiguous
        float* o = out + ((size_t) j * gridDim.x + t) * N;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float v = 0;
            for (int s = 0; s < S.n; ++s) v += w[j][s] * (S.p[s][(size_t) t * N + i] * inv[s]);
            o[i] = v;
        }
    }
}

// ------------------------------------------------------------------------------------------------ attention
// causal short convolution with a residual (llama.cpp convolve()): channel c of T rows x[t][c] (ld C) with the slot's state st[3][C]
// (the 3 inputs before, oldest first): out = w[c][0..3] . (x_{t-3}..x_t) [+ dyn[t][h][lag] x_{t-lag}], y = (x + out) * k; the state
// keeps the last 3 inputs. h = c / hs (channels a head, dynamic taps only)
__global__ void k_conv(const float* x, int C, float* st, const float* w, const float* dyn, int hs, float k, float* y, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s0 = st[c], s1 = st[C + c], s2 = st[2 * C + c];
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; ++t) {
        const float xt = x[(size_t) t * C + c];
        float o = 0.0f;
        o += s0 * w0;
        o += s1 * w1;
        o += s2 * w2;
        o += xt * w3;
        if (dyn) {
            const float* dt = dyn + ((size_t) t * (C / hs) + c / hs) * NCONV;
            o = o + dt[0] * xt;
            o = o + dt[1] * s2;
            o = o + dt[2] * s1;
            o = o + dt[3] * s0;
        }
        y[(size_t) t * C + c] = (xt + o) * k;
        s0 = s1;
        s1 = s2;
        s2 = xt;
    }
    st[c] = s0;
    st[C + c] = s1;
    st[2 * C + c] = s2;
}
// clip (mode 0): v *= s[h][.], then the head's L2 norm to at most thr, then * tau[t] when given; rms (mode 1): v / rms(v) (eps).
// One warp per (t, head) of D values (D = 128 or 16); s is [heads_all][D] with this card's first head h0
__global__ void k_headnorm(float* v, int H, int D, const float* s, int h0, float thr, const float* tau, int mode, float eps, int T) {
    const int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if (g >= T * H) return;
    const int t = g / H, h = g % H;
    float* x = v + (size_t) g * D;
    float a[4];
    float ss = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = l + 32 * k;
        a[k] = 0;
        if (i < D) {
            a[k] = x[i];
            if (mode == 0 && s) a[k] *= s[(size_t) (h0 + h) * D + i];
            ss += a[k] * a[k];
        }
    }
    ss = warp_sum(ss);
    float f;
    if (mode == 0) {
        const float nrm = sqrtf(ss) + 1e-6f;
        f = fminf(fmaxf(thr / nrm, 0.0f), 1.0f);
    } else {
        f = 1.0f / sqrtf(ss / D + eps);
    }
    const float tt = (mode == 0 && tau) ? tau[t] : 1.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = l + 32 * k;
        if (i < D) x[i] = (a[k] * f) * tt;
    }
}
// p18: a full slot's cache in int8 (INK2_KV8, on by itself past 256k: the 512k profile). Rows from f0 on: values in groups of
// KQG = 32 (one warp) with an f16 scale absmax / 127, k, v [row - f0][KVC], ks, vs [row - f0][KQN]; rows below f0 stay in the f16
// cache (kc / vc), so a conversation shorter than f0 is computed as in the f16 mode. A null k or v = that part all in f16 (window
// slots: both). rot: the slot's rows (int8 and f16) hold H x (orthonormal Walsh-Hadamard per head): q . k = Hq . Hk, and the
// attention's sum of V rows is turned back with H.
constexpr int KQG = 32, KQN = KVC / KQG;
struct KV8 {
    int8_t *k = nullptr, *v = nullptr;
    __half *ks = nullptr, *vs = nullptr;
    int f0 = 0, rot = 0;
};
__device__ __forceinline__ void kv8_put(int8_t* q, __half* s, size_t row, int c, float y) {   // all 32 lanes of group c / KQG
    const __half hs = __float2half(warp_max(fabsf(y)) * (1.0f / 127.0f));
    const float sc = __half2float(hs);
    q[row * KVC + c] = sc > 0.0f ? (int8_t) __float2int_rn(fminf(fmaxf(y / sc, -127.0f), 127.0f)) : (int8_t) 0;
    if ((c & (KQG - 1)) == 0) s[row * KQN + c / KQG] = hs;
}
__device__ __forceinline__ float kv8_get(const int8_t* q, const __half* s, size_t row, int c) {
    return (float) q[row * KVC + c] * __half2float(s[row * KQN + c / KQG]);
}
// a K (isk) or V value of a row, int8 or f16 by the row (a warp puts 32 values of one row)
__device__ __forceinline__ void kv_put(const KV8& q8, bool isk, __half* fc, size_t row, int c, float y) {
    int8_t* q = isk ? q8.k : q8.v;
    if (q && row >= (size_t) q8.f0) kv8_put(q, isk ? q8.ks : q8.vs, row - q8.f0, c, y);
    else fc[row * KVC + c] = __float2half(y);
}
__device__ __forceinline__ float kv_get(const KV8& q8, bool isk, const __half* fc, size_t row, int c) {
    const int8_t* q = isk ? q8.k : q8.v;
    if (q && row >= (size_t) q8.f0) return kv8_get(q, isk ? q8.ks : q8.vs, row - q8.f0, c);
    return __half2float(fc[row * KVC + c]);
}
// the orthonormal Walsh-Hadamard transform of a head's 128 values, a value a thread (all 128 threads of the block call it)
__device__ __forceinline__ float fwht128(float x, float* sh) {
    const int t = threadIdx.x & 127;
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, x, h);
        x = (t & h) ? o - x : x + o;
    }
#pragma unroll
    for (int h = 32; h < 128; h <<= 1) {
        __syncthreads();
        sh[t] = x;
        __syncthreads();
        const float o = sh[t ^ h];
        x = (t & h) ? o - x : x + o;
    }
    __syncthreads();
    return x * 0.088388347648318447f;   // 1 / sqrt(128)
}
// K and V rows of T tokens into the slot's cache (f16): row = pos (full) or pos % RING (window); p18: int8 when q8.k (a warp is
// 32 values of one row: KVC is a multiple of 32)
__global__ void k_kv_store(const float* k, const float* v, const int* pos, int ring, __half* kc, __half* vc, int T, KV8 q8) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * KVC) return;
    const int t = i / KVC, c = i % KVC;
    const size_t row = ring ? (size_t) (pos[t] % RING) : (size_t) pos[t];
    kv_put(q8, true, kc, row, c, k[i]);
    kv_put(q8, false, vc, row, c, v[i]);
}
// p18: k_kv_store of a full slot with q8.rot: a block a (head, token, K or V), the head turned by H before it is stored
__global__ void __launch_bounds__(128) k_kv_store_rot(const float* k, const float* v, const int* pos, __half* kc, __half* vc, KV8 q8) {
    __shared__ float sh[HD];
    const int h = blockIdx.x, t = blockIdx.y, isv = blockIdx.z, c = h * HD + threadIdx.x;
    const float y = fwht128((isv ? v : k)[(size_t) t * KVC + c], sh);
    kv_put(q8, !isv, isv ? vc : kc, (size_t) pos[t], c, y);
}
// p18: x = H x for every head (blockIdx.x) of row blockIdx.y (rows of ld floats): the reading path's queries into the int8 cache's
// basis and the attention's outputs back
__global__ void __launch_bounds__(128) k_rot(float* x, int ld) {
    __shared__ float sh[HD];
    float* p = x + (size_t) blockIdx.y * ld + blockIdx.x * HD + threadIdx.x;
    *p = fwht128(*p, sh);
}
// one block of 128 threads per (t, head, part of KC keys): scores q.k / sqrt(HD) + R.B[min(d, EXT)] - sink, masked to the
// window or the causal range, their max m, sum l and the unnormalized sum of V rows; parts are joined by k_attn_comb
__global__ void __launch_bounds__(128) k_attn_part(const float* q, const float* R, const float* B, const float* sink, const __half* kc,
                                                   const __half* vc, const int* pos, int window, float* part, int nparts) {
    __shared__ float qs[HD], rs[NREL], sc[KC];
    __shared__ float red[32];
    const int t = blockIdx.z, h = blockIdx.y, p = blockIdx.x, kvh = h / (NH / NKV);
    const int P = pos[t];
    const int lo = window ? max(0, P - (SWA - 1)) : 0, hi = P + 1;
    const int j0 = lo + p * KC, j1 = min(hi, j0 + KC);
    float* o = part + (((size_t) t * NH_C + h) * nparts + p) * (HD + 2);
    if (j0 >= hi) {   // nothing of this part: an empty result
        if (threadIdx.x == 0) { o[0] = -INFINITY; o[1] = 0.0f; }
        o[2 + threadIdx.x] = 0.0f;
        return;
    }
    qs[threadIdx.x] = q[((size_t) t * NH_C + h) * HD + threadIdx.x];
    if (threadIdx.x < NREL) rs[threadIdx.x] = R[((size_t) t * NH_C + h) * NREL + threadIdx.x];
    __syncthreads();
    const float tsink = sink[t * NH_C + h], scale = 1.0f / sqrtf((float) HD);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int j = j0 + warp; j < j1; j += 4) {
        const size_t row = window ? (size_t) (j % RING) : (size_t) j;
        const __half* kr = kc + row * KVC + kvh * HD;
        const __half2 k01 = *(const __half2*) (kr + lane * 4), k23 = *(const __half2*) (kr + lane * 4 + 2);
        float d = __low2float(k01) * qs[lane * 4] + __high2float(k01) * qs[lane * 4 + 1] + __low2float(k23) * qs[lane * 4 + 2] +
                  __high2float(k23) * qs[lane * 4 + 3];
        d = warp_sum(d);
        if (lane == 0) {
            const int dist = P - j;
            float b = 0.0f;
            if (dist < EXT) {
                const float* bb = B + (size_t) dist * NREL;
#pragma unroll
                for (int r = 0; r < NREL; ++r) b += rs[r] * bb[r];
            }
            sc[j - j0] = d * scale + (b - tsink);   // the bias and the sink are not scaled (transformers eager_attention_forward)
        }
    }
    __syncthreads();
    const int n = j1 - j0;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < n; j += 128) m = fmaxf(m, sc[j]);
    m = block_max(m, red);
    float l = 0;
    for (int j = threadIdx.x; j < n; j += 128) {
        const float e = expf(sc[j] - m);
        sc[j] = e;
        l += e;
    }
    l = block_sum(l, red);
    float acc = 0;
    for (int j = 0; j < n; ++j) {
        const int jj = j0 + j;
        const size_t row = window ? (size_t) (jj % RING) : (size_t) jj;
        acc += sc[j] * __half2float(vc[row * KVC + kvh * HD + threadIdx.x]);
    }
    if (threadIdx.x == 0) { o[0] = m; o[1] = l; }
    o[2 + threadIdx.x] = acc;
}
// the parts of (t, head) and the zero sink logit (ggml_soft_max_add_sinks: one more logit 0, not scaled), times sigmoid(gate)
__global__ void __launch_bounds__(128) k_attn_comb(const float* part, int nparts, const float* gate, float* out, int rot) {
    __shared__ float sh[HD];
    const int t = blockIdx.y, h = blockIdx.x;
    const float* p = part + ((size_t) t * NH_C + h) * nparts * (HD + 2);
    float M = 0.0f;   // the sink
    for (int i = 0; i < nparts; ++i) M = fmaxf(M, p[(size_t) i * (HD + 2)]);
    float L = expf(0.0f - M), acc = 0.0f;
    for (int i = 0; i < nparts; ++i) {
        const float* pi = p + (size_t) i * (HD + 2);
        if (pi[1] == 0.0f) continue;
        const float f = expf(pi[0] - M);
        L += pi[1] * f;
        acc += pi[2 + threadIdx.x] * f;
    }
    const size_t o = ((size_t) t * NH_C + h) * HD + threadIdx.x;
    float y = acc / L;
    if (rot) y = fwht128(y, sh);   // p18: the sum of H v rows (a full slot with q8.rot) back with H, before the gate
    out[o] = y * sigm(gate[o]);
}

// p57: k_attn_comb's numbers bit for bit (INK2_COMB2): the max over the parts by the block (exact in any order), each part's weight
// once (the same expf of the same value) in shared memory a chunk of parts at a time, then the same sums in part order
__global__ void __launch_bounds__(128) k_attn_comb2(const float* part, int nparts, const float* gate, float* out, int rot) {
    constexpr int CH = 1024;
    __shared__ float sh[HD];
    __shared__ float fw[CH], lw[CH], red[4];
    const int t = blockIdx.y, h = blockIdx.x, tid = threadIdx.x;
    const float* p = part + ((size_t) t * NH_C + h) * nparts * (HD + 2);
    float m = 0.0f;   // the sink, as k_attn_comb's start
    for (int i = tid; i < nparts; i += 128) m = fmaxf(m, p[(size_t) i * (HD + 2)]);
    m = warp_max(m);
    if ((tid & 31) == 0) red[tid >> 5] = m;
    __syncthreads();
    const float M = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    float L = expf(0.0f - M), acc = 0.0f;
    for (int c0 = 0; c0 < nparts; c0 += CH) {
        const int n = min(CH, nparts - c0);
        __syncthreads();   // the previous chunk's weights are used up
        for (int i = tid; i < n; i += 128) {
            const float* pi = p + (size_t) (c0 + i) * (HD + 2);
            const float l = pi[1];
            lw[i] = l;
            fw[i] = l == 0.0f ? 0.0f : expf(pi[0] - M);
        }
        __syncthreads();
        const float* pd = p + (size_t) c0 * (HD + 2) + 2 + tid;
#pragma unroll 8
        for (int i = 0; i < n; ++i) {
            const float l = lw[i];
            if (l == 0.0f) continue;
            const float f = fw[i];
            L += l * f;
            acc += pd[(size_t) i * (HD + 2)] * f;
        }
    }
    const size_t o = ((size_t) t * NH_C + h) * HD + threadIdx.x;
    float y = acc / L;
    if (rot) y = fwht128(y, sh);
    out[o] = y * sigm(gate[o]);
}

// p59: k_attn_comb2's numbers with S blocks of 128 / S threads a (row, head) (INK2_COMB2=3, T <= 2): block b of head b / S does
// dimensions (b % S) * 128 / S ..., a thread a dimension, each block the same max and weights, the values U parts ahead in
// registers; acc / L to out, then k_attn_comb4b the FWHT (rot) and the gate in place
template <int S, int U>
__global__ void __launch_bounds__(128 / S) k_attn_comb4(const float* part, int nparts, float* out) {
    constexpr int NT = 128 / S, CH = 1024;
    __shared__ float fw[CH], lw[CH], red[NT / 32 > 0 ? NT / 32 : 1];
    const int t = blockIdx.y, h = blockIdx.x / S, d = (blockIdx.x % S) * NT + threadIdx.x, tid = threadIdx.x;
    const float* p = part + ((size_t) t * NH_C + h) * nparts * (HD + 2);
    float m = 0.0f;   // the sink, as k_attn_comb's start
    for (int i = tid; i < nparts; i += NT) m = fmaxf(m, p[(size_t) i * (HD + 2)]);
    m = warp_max(m);
    if (NT > 32) {
        if ((tid & 31) == 0) red[tid >> 5] = m;
        __syncthreads();
        m = red[0];
        for (int w = 1; w < NT / 32; ++w) m = fmaxf(m, red[w]);
    }
    const float M = m;
    float L = expf(0.0f - M), acc = 0.0f;
    for (int c0 = 0; c0 < nparts; c0 += CH) {
        const int n = min(CH, nparts - c0);
        __syncthreads();   // the previous chunk's weights are used up
        for (int i = tid; i < n; i += NT) {
            const float* pi = p + (size_t) (c0 + i) * (HD + 2);
            const float l = pi[1];
            lw[i] = l;
            fw[i] = l == 0.0f ? 0.0f : expf(pi[0] - M);
        }
        __syncthreads();
        const float* pd = p + (size_t) c0 * (HD + 2) + 2 + d;
        int i = 0;
        for (; i + U <= n; i += U) {
            float vv[U];
#pragma unroll
            for (int j = 0; j < U; ++j) vv[j] = __ldg(pd + (size_t) (i + j) * (HD + 2));
#pragma unroll
            for (int j = 0; j < U; ++j) {
                const float l = lw[i + j];
                if (l == 0.0f) continue;
                const float f = fw[i + j];
                L += l * f;
                acc += vv[j] * f;
            }
        }
        for (; i < n; ++i) {
            const float l = lw[i];
            if (l == 0.0f) continue;
            const float f = fw[i];
            L += l * f;
            acc += pd[(size_t) i * (HD + 2)] * f;
        }
    }
    out[((size_t) t * NH_C + h) * HD + d] = acc / L;
}
__global__ void __launch_bounds__(128) k_attn_comb4b(const float* gate, float* out, int rot) {
    __shared__ float sh[HD];
    const size_t o = ((size_t) blockIdx.y * NH_C + blockIdx.x) * HD + threadIdx.x;
    float y = out[o];
    if (rot) y = fwht128(y, sh);
    out[o] = y * sigm(gate[o]);
}
// p59: k_attn_comb2 with the values 16 parts ahead in registers (INK2_COMB2=3, T >= 3), the FWHT and the gate in the same kernel
__global__ void __launch_bounds__(128) k_attn_comb5(const float* part, int nparts, const float* gate, float* out, int rot) {
    constexpr int CH = 1024, U = 16;
    __shared__ float sh[HD];
    __shared__ float fw[CH], lw[CH], red[4];
    const int t = blockIdx.y, h = blockIdx.x, tid = threadIdx.x;
    const float* p = part + ((size_t) t * NH_C + h) * nparts * (HD + 2);
    float m = 0.0f;   // the sink, as k_attn_comb's start
    for (int i = tid; i < nparts; i += 128) m = fmaxf(m, p[(size_t) i * (HD + 2)]);
    m = warp_max(m);
    if ((tid & 31) == 0) red[tid >> 5] = m;
    __syncthreads();
    const float M = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    float L = expf(0.0f - M), acc = 0.0f;
    for (int c0 = 0; c0 < nparts; c0 += CH) {
        const int n = min(CH, nparts - c0);
        __syncthreads();   // the previous chunk's weights are used up
        for (int i = tid; i < n; i += 128) {
            const float* pi = p + (size_t) (c0 + i) * (HD + 2);
            const float l = pi[1];
            lw[i] = l;
            fw[i] = l == 0.0f ? 0.0f : expf(pi[0] - M);
        }
        __syncthreads();
        const float* pd = p + (size_t) c0 * (HD + 2) + 2 + tid;
        int i = 0;
        for (; i + U <= n; i += U) {
            float vv[U];
#pragma unroll
            for (int j = 0; j < U; ++j) vv[j] = __ldg(pd + (size_t) (i + j) * (HD + 2));
#pragma unroll
            for (int j = 0; j < U; ++j) {
                const float l = lw[i + j];
                if (l == 0.0f) continue;
                const float f = fw[i + j];
                L += l * f;
                acc += vv[j] * f;
            }
        }
        for (; i < n; ++i) {
            const float l = lw[i];
            if (l == 0.0f) continue;
            const float f = fw[i];
            L += l * f;
            acc += pd[(size_t) i * (HD + 2)] * f;
        }
    }
    const size_t o = ((size_t) t * NH_C + h) * HD + threadIdx.x;
    float y = acc / L;
    if (rot) y = fwht128(y, sh);
    out[o] = y * sigm(gate[o]);
}

// ------------------------------------------------------------------------------------------------ MoE
// router: p = sigmoid(logit), the top TOPK by p + bias (ties: the lower index), weights p / sum * scale. One block of NE threads a row
__global__ void k_route(const float* logits, const float* bias, float wscale, int* topi, float* topw) {
    __shared__ float sv[NE];
    __shared__ int si[NE];
    __shared__ float pr[NE];
    const int t = blockIdx.x, e = threadIdx.x;
    const float p = sigm(logits[(size_t) t * NE + e]);
    pr[e] = p;
    sv[e] = p + (bias ? bias[e] : 0.0f);
    si[e] = e;
    __syncthreads();
    for (int k = 2; k <= NE; k <<= 1)   // bitonic sort, descending by value, ascending by index on ties
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ix = e ^ j;
            if (ix > e) {
                const bool up = (e & k) == 0;
                const float a = sv[e], b = sv[ix];
                const int ia = si[e], ib = si[ix];
                const bool a_first = a > b || (a == b && ia < ib);
                if (up ? !a_first : a_first) {
                    sv[e] = b; sv[ix] = a; si[e] = ib; si[ix] = ia;
                }
            }
            __syncthreads();
        }
    if (e == 0) {
        float s = 0;
        for (int k = 0; k < TOPK; ++k) s += pr[si[k]];
        s = s + 1e-20f;
        for (int k = 0; k < TOPK; ++k) {
            topi[t * TOPK + k] = si[k];
            topw[t * TOPK + k] = (pr[si[k]] / s) * wscale;
        }
    }
}
// NVFP4 (ggml type 40): a block of 64 values = 4 UE4M3 scales (one per 16, x 0.5 as in ggml) + 32 bytes of E2M1 codes; sub-block s
// of 8 bytes: byte i low nibble = value i, high nibble = value i + 8
__device__ __forceinline__ float ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 7;
    const float raw = e == 0 ? ldexpf((float) m, -9) : ldexpf(1.0f + m / 8.0f, e - 7);
    return raw * 0.5f;
}
__constant__ float c_fp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
// one expert row block per warp: rows [rows] of nsub sub-blocks (NVFP4) or cols int8 + cols/32 scales (Q8S) for the job's expert;
// the job's input row (cols floats) in shared memory; y[job][row] = (W[e][row] . x) * escale[e]
struct EJob { int t, k, e; };
// p11: the order the expert kernels' blocks take a window's jobs: by expert, then by job (a stable rank), so two picks of one expert
// run one after the other and the second reads the weights from L2. One block, nj <= MAXT * TOPK threads
__global__ void k_job_perm(const EJob* jobs, int nj, int* perm) {
    const int j = threadIdx.x;
    if (j >= nj) return;
    const int e = jobs[j].e;
    int r = 0;
    for (int i = 0; i < nj; ++i) {
        const int f = jobs[i].e;
        r += f < e || (f == e && i < j);
    }
    perm[r] = j;
}
template <bool FP4>
__global__ void __launch_bounds__(256) k_exp_rows(const uint8_t* __restrict__ W, size_t estride, const __half* __restrict__ Dq, size_t dstride,
                                                  int rows, int cols, const float* x, int ldx, int xbyjob, const float* escale,
                                                  const EJob* jobs, float* y, int ldy, const int* perm = nullptr,
                                                  const uint8_t* const* ET = nullptr) {   // p30: expert bases (Q8: rows, then scales)
    extern __shared__ float xs[];
    __shared__ float tab[16];   // the E2M1 table in shared memory: lanes reading different entries of __constant__ memory serialize
    const int jy = perm ? perm[blockIdx.y] : blockIdx.y;   // p11: blocks by expert, outputs by job
    const EJob jb = jobs[jy];
    const float* xr = x + (size_t) (xbyjob ? jy : jb.t) * ldx;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = xr[i];
    if (threadIdx.x < 16) tab[threadIdx.x] = c_fp4[threadIdx.x];
    __syncthreads();
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= rows) return;
    float acc = 0;
    if (FP4) {
        const uint8_t* row = (ET ? ET[jb.e] : W + (size_t) jb.e * estride) + (size_t) r * (cols / 64) * 36;
        const int nsub = cols / 16;
        for (int s = lane; s < nsub; s += 32) {
            const uint8_t* blk = row + (size_t) (s >> 2) * 36;
            const float d = ue4m3(blk[s & 3]);
            const uint8_t* qs = blk + 4 + (s & 3) * 8;
            const float* xv = xs + s * 16;
            float a = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint8_t b = qs[i];
                a += tab[b & 15] * xv[i] + tab[b >> 4] * xv[i + 8];
            }
            acc += a * d;
        }
    } else {
        const int8_t* row = (const int8_t*) (ET ? ET[jb.e] : W + (size_t) jb.e * estride) + (size_t) r * cols;
        const __half* dr = (ET ? (const __half*) (ET[jb.e] + estride) : Dq + (size_t) jb.e * dstride) + (size_t) r * (cols / 32);
        for (int c = lane * 16; c < cols; c += 512) {
            const int4 w4 = __ldg((const int4*) (row + c));
            const int8_t* w = (const int8_t*) &w4;
            float a = 0;
#pragma unroll
            for (int j = 0; j < 16; ++j) a += (float) w[j] * xs[c + j];
            acc += a * __half2float(dr[c / 32]);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) jy * ldy + r] = escale ? acc * escale[jb.e] : acc;
}
// the expert jobs of T rows from the router's picks, row-major (job t * TOPK + k)
__global__ void k_jobs(const int* topi, EJob* jobs, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * TOPK) jobs[i] = EJob{i / TOPK, i % TOPK, topi[i]};
}
// SITU on this card's FF part: gu[job] = [gate FF_C | up FF_C], h = 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25)
__global__ void k_situ(const float* gu, float* h, int njobs, float beta, float lb) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= njobs * FF_C) return;
    const int j = i / FF_C, c = i % FF_C;
    const float g = gu[(size_t) j * 2 * FF_C + c], u = gu[(size_t) j * 2 * FF_C + FF_C + c];
    const float act = (beta * tanhf(g / beta)) * sigm(g);
    const float up = lb > 0.0f ? lb * tanhf(u / lb) : u;
    h[i] = act * up;
}
// moe[t] = sum_k w[t][k] * y[t * TOPK + k]
__global__ void k_moe_sum(const float* y, const float* topw, float* moe, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int t = i / N, c = i % N;
    float v = 0;
    for (int k = 0; k < TOPK; ++k) v += y[((size_t) t * TOPK + k) * N + c] * topw[t * TOPK + k];
    moe[i] = v;
}

// ------------------------------------------------------------------------------------------------ small
__global__ void k_add(float* y, const float* a, const float* b, size_t n) {   // y = a + b
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
__global__ void k_softcap(float* x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] / cap) * cap;
}
// p36: the softcap of nr rows of ld logits, then with invT > 0 logit / T + Gumbel noise (row r predicts position pos0 + r, column j
// is vocabulary index col0 + j): the argmax is then a sample of softmax(logits / T)
__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__global__ void k_softcap_g(float* x, int ld, int nr, float cap, float invT, unsigned long long seed, int pos0, int col0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ld * nr) return;
    float v = tanhf(x[i] / cap) * cap;
    if (invT > 0.0f) {
        const int r = i / ld, j = i % ld;
        const unsigned long long h = mix64(seed ^ mix64((unsigned long long) (pos0 + r) * 0x100000000ull + (unsigned long long) (col0 + j)));
        const float u = ((float) (h >> 40) + 0.5f) * (1.0f / 16777216.0f);
        v = v * invT - logf(-logf(u));
    }
    x[i] = v;
}
// p76: each row's largest logit of this card into its half of mx2 [nr][2] (the other half 0, so the cross-card sum gives both)
__global__ void __launch_bounds__(1024) k_rowmax2(const float* x, int ld, int valid, float* mx2, int c) {
    __shared__ float red[32];
    const float* xr = x + (size_t) blockIdx.x * ld;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < valid; i += 1024) m = fmaxf(m, xr[i]);
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; ++w) m = fmaxf(m, red[w]);
        m = fmaxf(m, red[0]);
        mx2[blockIdx.x * 2 + c] = m;
        mx2[blockIdx.x * 2 + (1 - c)] = 0.0f;
    }
}
// p76: min-p and p36's noise: a (softcapped) logit below max - T ln(1/min_p) of its row (the max of both cards' halves) is -inf,
// the others logit / T + Gumbel noise of (seed, position, token) as k_softcap_g does
__global__ void k_gumbel_minp(float* x, int ld, int nr, float invT, unsigned long long seed, int pos0, int col0, const float* mx2,
                              int ncards, float lnminp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ld * nr) return;
    const int r = i / ld, j = i % ld;
    float m = mx2[r * 2];
    if (ncards > 1) m = fmaxf(m, mx2[r * 2 + 1]);
    float v = x[i];
    if (v < m + lnminp / invT) {
        x[i] = -INFINITY;
        return;
    }
    const unsigned long long h = mix64(seed ^ mix64((unsigned long long) (pos0 + r) * 0x100000000ull + (unsigned long long) (col0 + j)));
    const float u = ((float) (h >> 40) + 0.5f) * (1.0f / 16777216.0f);
    x[i] = v * invT - logf(-logf(u));
}
// the cross-card sum (Hy4 v1): page-locked mapped buffers, a flag a card; each card writes its part and adds the other's
__global__ void k_xput(const float* y, int n, float* out, volatile unsigned* flag, unsigned seq) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = y[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *flag = seq;
}
__global__ void k_xget(float* y, int n, const volatile float* in, const volatile unsigned* flag, unsigned seq, int card) {
    if (threadIdx.x == 0)
        while (*flag < seq) {}
    __syncthreads();
    __threadfence_system();
    // both cards add in the same order (card 0's part first), so both hold the same bits
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = card == 0 ? y[i] + in[i] : in[i] + y[i];
}


// ------------------------------------------------------------------------------------------------ p4a: faster writing kernels
// (21. 11., after nvprof of v1: F32 GEMV 9.5 ms a token and card, NVFP4 experts 7.9, Q8 GEMV 6.3, k_mix 5.4). The same math in
// another order of the f32 sums; INK2_STARE=1 runs v1's kernels.

// NVFP4 expert rows by 32-bit words: lane l reads words l, l+32, ... of the row (coalesced); word o of a 9-word block is the 4
// scales (o = 0) or 4 code bytes of sub-block (o-1)/2, half (o-1)&1. y[job][row] = (W[e][row] . x) * escale[e]
__global__ void __launch_bounds__(256) k_fp4_rows2(const uint8_t* __restrict__ W, size_t estride, int rows, int cols, const float* x, int ldx,
                                                   int xbyjob, const float* escale, const EJob* jobs, float* y, int ldy) {
    extern __shared__ float xs[];
    __shared__ float tab[16];
    const EJob jb = jobs[blockIdx.y];
    const float* xr = x + (size_t) (xbyjob ? blockIdx.y : jb.t) * ldx;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = xr[i];
    if (threadIdx.x < 16) tab[threadIdx.x] = c_fp4[threadIdx.x];
    __syncthreads();
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= rows) return;
    const int rb = (cols / 64) * 36, nw = rb / 4;
    const uint8_t* row = W + (size_t) jb.e * estride + (size_t) r * rb;
    const uint32_t* rw = (const uint32_t*) row;
    float acc = 0;
    for (int w = lane; w < nw; w += 32) {
        const int b = w / 9, o = w - 9 * b;
        if (o == 0) continue;
        const uint32_t q = __ldg(rw + w);
        const int sb = (o - 1) >> 1;
        const float d = ue4m3(__ldg(row + b * 36 + sb));
        const float* xv = xs + b * 64 + sb * 16 + ((o - 1) & 1) * 4;
        float a = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t c = (q >> (8 * j)) & 0xFFu;
            a += tab[c & 15] * xv[j] + tab[c >> 4] * xv[j + 8];
        }
        acc += a * d;
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) blockIdx.y * ldy + r] = escale ? acc * escale[jb.e] : acc;
}

// Q8S at T = 1 (from Hy4's k_gemv8_t1): 4 rows a warp, x through L1, columns in S parts (blocks.y) when the rows give too few blocks;
// the last part of a row block to finish adds the parts in a fixed order. Rows of one concatenated matrix go to up to 8 outputs
// (segments: rows [r0, r0 + n) to dst + (r - r0)), so the matrices with the same input run in one launch
constexpr int G1_R = 4, G1_BR = 8 * G1_R, G1_PART = 1 << 17;
__device__ float g_part[G1_PART];
__device__ unsigned g_g1_cnt[4096];
struct Segs {
    int n;
    int r0[8];
    float* dst[8];
};
__device__ __forceinline__ void seg_store(const Segs& S, int r, float v) {
    int i = 0;
    while (i + 1 < S.n && r >= S.r0[i + 1]) ++i;
    S.dst[i][r - S.r0[i]] = v;
}
template <int WPB = 8>   // p49: WPB warps a block (4 rows each); S is the launcher's, from 32-row blocks
__global__ void __launch_bounds__(32 * WPB) k_gemv8_t1(const int8_t* __restrict__ q, const __half* __restrict__ d, int rows, int cols,
                                                   const float* __restrict__ x, Segs segs, int S) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rb = blockIdx.x * (WPB * G1_R), r0 = rb + warp * G1_R;
    const int nch = (cols + 511) / 512, s = blockIdx.y, ch0 = s * nch / S, ch1 = (s + 1) * nch / S;
    float acc[G1_R] = {};
    for (int ch = ch0; ch < ch1; ++ch) {
        const int c = ch * 512 + lane * 16;
        if (c >= cols) continue;
        const float4 x0 = __ldg((const float4*) (x + c)), x1 = __ldg((const float4*) (x + c + 4)), x2 = __ldg((const float4*) (x + c + 8)),
                     x3 = __ldg((const float4*) (x + c + 12));
#pragma unroll
        for (int a = 0; a < G1_R; ++a) {
            const int r = r0 + a;
            if (r >= rows) break;
            const int4 w4 = __ldg((const int4*) (q + (size_t) r * cols + c));
            const float sc = __half2float(d[(size_t) r * (cols / 32) + c / 32]);
            const int8_t* w = (const int8_t*) &w4;
            const float v = w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w +
                            w[8] * x2.x + w[9] * x2.y + w[10] * x2.z + w[11] * x2.w + w[12] * x3.x + w[13] * x3.y + w[14] * x3.z + w[15] * x3.w;
            acc[a] += sc * v;
        }
    }
#pragma unroll
    for (int a = 0; a < G1_R; ++a) {
        const float v = warp_sum(acc[a]);
        const int r = r0 + a;
        if (lane == 0 && r < rows) {
            if (S == 1) seg_store(segs, r, v);
            else g_part[(size_t) s * rows + r] = v;
        }
    }
    if (S == 1) return;
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&g_g1_cnt[blockIdx.x], 1u) == (unsigned) S - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < WPB * G1_R && rb + (int) threadIdx.x < rows) {
        const int r = rb + threadIdx.x;
        float v = 0;
        for (int k = 0; k < S; ++k) v += ((volatile float*) g_part)[(size_t) k * rows + r];
        seg_store(segs, r, v);
    }
    if (threadIdx.x == 0) g_g1_cnt[blockIdx.x] = 0;
}

// p53: k_gemv8_t1's sums with RPW rows a warp and all S column parts of a row in that warp (INK2_G1W): part s's chunks as in
// k_gemv8_t1 (lane l the 16 columns ch * 512 + 16 l, the same expression, acc += sc * v), warp_sum per part, the parts added in
// order from 0 (k_gemv8_t1's last block); S = 1 stores the warp's sum as it is. 8 warps a block
template <int RPW>
__global__ void __launch_bounds__(256) k_gemv8_t1w(const int8_t* __restrict__ q, const __half* __restrict__ d, int rows, int cols,
                                                    const float* __restrict__ x, Segs segs, int S) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r0 = (blockIdx.x * 8 + warp) * RPW;
    if (r0 >= rows) return;
    const int nch = (cols + 511) / 512;
    float tot[RPW];
#pragma unroll
    for (int a = 0; a < RPW; ++a) tot[a] = 0.0f;
    for (int s = 0; s < S; ++s) {
        const int ch0 = s * nch / S, ch1 = (s + 1) * nch / S;
        float acc[RPW] = {};
        for (int ch = ch0; ch < ch1; ++ch) {
            const int c = ch * 512 + lane * 16;
            if (c >= cols) continue;
            const float4 x0 = __ldg((const float4*) (x + c)), x1 = __ldg((const float4*) (x + c + 4)), x2 = __ldg((const float4*) (x + c + 8)),
                         x3 = __ldg((const float4*) (x + c + 12));
#pragma unroll
            for (int a = 0; a < RPW; ++a) {
                const int r = r0 + a;
                if (r >= rows) break;
                const int4 w4 = __ldg((const int4*) (q + (size_t) r * cols + c));
                const float sc = __half2float(d[(size_t) r * (cols / 32) + c / 32]);
                const int8_t* w = (const int8_t*) &w4;
                const float v = w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w +
                                w[8] * x2.x + w[9] * x2.y + w[10] * x2.z + w[11] * x2.w + w[12] * x3.x + w[13] * x3.y + w[14] * x3.z + w[15] * x3.w;
                acc[a] += sc * v;
            }
        }
#pragma unroll
        for (int a = 0; a < RPW; ++a) {
            const float pv = warp_sum(acc[a]);
            if (S == 1) tot[a] = pv;
            else tot[a] += pv;
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int a = 0; a < RPW; ++a)
            if (r0 + a < rows) seg_store(segs, r0 + a, tot[a]);
    }
}
// F32 matrices with few rows and many columns (loop controllers' down 64 x 4608, router 128 x 2304): a block per row
__global__ void __launch_bounds__(256) k_gemv32_rows(const float* __restrict__ W, int cols, const float* x, int ldx, float* y, int ldy, int T) {
    __shared__ float red[32];
    const int r = blockIdx.x;
    const float* w = W + (size_t) r * cols;
    float acc[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) acc[t] = 0;
    for (int c = threadIdx.x * 4; c < cols; c += blockDim.x * 4) {
        const float4 wv = __ldg((const float4*) (w + c));
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= T) break;
            const float4 xv = *(const float4*) (x + (size_t) t * ldx + c);
            acc[t] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
    }
    for (int t = 0; t < T; ++t) {
        const float v = block_sum(acc[t], red);
        if (threadIdx.x == 0) y[(size_t) t * ldy + r] = v;
    }
}
// the loop controller's gain and forget (64 -> N each, F32) and x' = (x - tanh(f) x) + tanh(g) m in one: a warp per output column
__global__ void __launch_bounds__(256) k_ctrl_gf(const float* __restrict__ Wg, const float* __restrict__ Wf, const float* g64, const float* x,
                                                 const float* m, int ldm, float* y, int T) {
    const int c = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (c >= N) return;
    const float2 wg = *(const float2*) (Wg + (size_t) c * RANK + lane * 2), wf = *(const float2*) (Wf + (size_t) c * RANK + lane * 2);
    for (int t = 0; t < T; ++t) {
        const float2 gv = *(const float2*) (g64 + (size_t) t * RANK + lane * 2);
        const float gp = warp_sum(wg.x * gv.x + wg.y * gv.y), fp = warp_sum(wf.x * gv.x + wf.y * gv.y);
        if (lane == 0) {
            const float xv = x[(size_t) t * N + c];
            y[(size_t) t * N + c] = (xv - tanhf(fp) * xv) + tanhf(gp) * m[(size_t) t * ldm + c];
        }
    }
}
// depth mixing in one pass of reductions: a block of 768 threads a row (3 values a thread), the S norms and S x nq dots at once
__global__ void __launch_bounds__(768) k_mix2(States S, const float* q, int nq, float* out) {
    constexpr int NS = MAXB + 2, NR = NS * 5;   // per state: its sum of squares and up to 4 dots
    __shared__ float red[24][NR];
    __shared__ float inv[NS], w[4][NS];
    const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) acc[i] = 0;
    float xv[NS][3];
#pragma unroll
    for (int s = 0; s < NS; ++s)
#pragma unroll
        for (int k = 0; k < 3; ++k) xv[s][k] = 0;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int i = threadIdx.x + 768 * k;
        float qv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) qv[j] = j < nq ? q[(size_t) j * N + i] : 0.0f;
#pragma unroll
        for (int s = 0; s < NS; ++s) {
            if (s < S.n) {
                const float v = S.p[s][(size_t) t * N + i];
                xv[s][k] = v;
                acc[s * 5] += v * v;
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[s * 5 + 1 + j] += v * qv[j];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < NR; ++i) {
        const float v = warp_sum(acc[i]);
        if (lane == 0) red[warp][i] = v;
    }
    __syncthreads();
    if (threadIdx.x < NR) {
        float v = 0;
        for (int k = 0; k < 24; ++k) v += red[k][threadIdx.x];
        red[0][threadIdx.x] = v;
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        const int j = threadIdx.x;
        float mx = -INFINITY;
        for (int s = 0; s < S.n; ++s) {
            const float iv = 1.0f / sqrtf(red[0][s * 5] / N + 1e-6f);
            if (j == 0) inv[s] = iv;
            w[j][s] = red[0][s * 5 + 1 + j] * iv / sqrtf((float) N);
            mx = fmaxf(mx, w[j][s]);
        }
        float l = 0;
        for (int s = 0; s < S.n; ++s) { w[j][s] = expf(w[j][s] - mx); l += w[j][s]; }
        for (int s = 0; s < S.n; ++s) w[j][s] /= l;
    }
    __syncthreads();
    for (int j = 0; j < nq; ++j)
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int i = threadIdx.x + 768 * k;
            float v = 0;
#pragma unroll
            for (int s = 0; s < NS; ++s)
                if (s < S.n) v += w[j][s] * (xv[s][k] * inv[s]);
            out[((size_t) j * gridDim.x + t) * N + i] = v;
        }
}


// p51: k_mix2 in two launches for few rows (INK2_MIX3=1, T <= MAXT); the same sums in the same order (see popravki/p51)
__device__ float g_mix_part[MAXT * 24 * (MAXB + 2) * 5];
__global__ void __launch_bounds__(32) k_mix3a(States S, const float* q, int nq) {   // block (w, t): k_mix2's warp w of row t
    constexpr int NS = MAXB + 2, NR = NS * 5;
    const int t = blockIdx.y, warp = blockIdx.x, lane = threadIdx.x, tid = warp * 32 + lane;
    float acc[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) acc[i] = 0;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int i = tid + 768 * k;
        float qv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) qv[j] = j < nq ? q[(size_t) j * N + i] : 0.0f;
#pragma unroll
        for (int s = 0; s < NS; ++s) {
            if (s < S.n) {
                const float v = S.p[s][(size_t) t * N + i];
                acc[s * 5] += v * v;
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[s * 5 + 1 + j] += v * qv[j];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < NR; ++i) {
        const float v = warp_sum(acc[i]);
        if (lane == 0) g_mix_part[((size_t) t * 24 + warp) * NR + i] = v;
    }
}
__global__ void __launch_bounds__(256) k_mix3b(States S, int nq, float* out, int T) {   // block (b, t): outputs b * 256 .. of row t
    constexpr int NS = MAXB + 2, NR = NS * 5;
    __shared__ float red[NR];
    __shared__ float inv[NS], w[4][NS];
    const int t = blockIdx.y;
    if (threadIdx.x < NR) {
        float v = 0;
        for (int k = 0; k < 24; ++k) v += g_mix_part[((size_t) t * 24 + k) * NR + threadIdx.x];
        red[threadIdx.x] = v;
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        const int j = threadIdx.x;
        float mx = -INFINITY;
        for (int s = 0; s < S.n; ++s) {
            const float iv = 1.0f / sqrtf(red[s * 5] / N + 1e-6f);
            if (j == 0) inv[s] = iv;
            w[j][s] = red[s * 5 + 1 + j] * iv / sqrtf((float) N);
            mx = fmaxf(mx, w[j][s]);
        }
        float l = 0;
        for (int s = 0; s < S.n; ++s) { w[j][s] = expf(w[j][s] - mx); l += w[j][s]; }
        for (int s = 0; s < S.n; ++s) w[j][s] /= l;
    }
    __syncthreads();
    const int i = blockIdx.x * 256 + threadIdx.x;
    float xv[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s) xv[s] = s < S.n ? S.p[s][(size_t) t * N + i] : 0.0f;
    for (int j = 0; j < nq; ++j) {
        float v = 0;
#pragma unroll
        for (int s = 0; s < NS; ++s)
            if (s < S.n) v += w[j][s] * (xv[s] * inv[s]);
        out[((size_t) j * T + t) * N + i] = v;
    }
}

// p4b: NVFP4 expert rows at writing by contiguous tiles: a warp takes R consecutive rows of NBLK blocks (R * NBLK = 288 = 9 * 32
// blocks, one contiguous run of bytes); lane l decodes whole 36-byte blocks l, l + 32, ... (9 independent 32-bit loads each), the
// blocks' sums go to shared memory and each row adds its NBLK sums in order. gate_up: NBLK 36, R 8; down: NBLK 9, R 32
template <int NBLK, int R>
__global__ void __launch_bounds__(128) k_fp4_rows3(const uint8_t* __restrict__ W, size_t estride, int rows, const float* x, int ldx,
                                                   int xbyjob, const float* escale, const EJob* jobs, float* y, int ldy) {
    static_assert(NBLK * R == 288, "a warp's tile is 9 blocks a lane");
    constexpr int COLS = NBLK * 64, WPB = 4;
    __shared__ float xs[NBLK * 65];   // a 64-value block every 65 floats: lanes reading the same offset of different blocks hit different banks
    __shared__ float part[WPB][288];
    __shared__ float tab[16];
    const EJob jb = jobs[blockIdx.y];
    const float* xr = x + (size_t) (xbyjob ? blockIdx.y : jb.t) * ldx;
    for (int i = threadIdx.x; i < COLS; i += blockDim.x) xs[(i >> 6) * 65 + (i & 63)] = xr[i];
    if (threadIdx.x < 16) tab[threadIdx.x] = c_fp4[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * WPB + warp) * R;
    if (row0 >= rows) return;
    const uint8_t* tile = W + (size_t) jb.e * estride + (size_t) row0 * NBLK * 36;
#pragma unroll 3
    for (int k = 0; k < 9; ++k) {
        const int gb = k * 32 + lane, bl = gb % NBLK;
        const uint32_t* bw = (const uint32_t*) (tile + (size_t) gb * 36);
        uint32_t w[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) w[i] = __ldg(bw + i);
        const float* xv = xs + bl * 65;
        float acc = 0;
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            const float d = ue4m3((w[0] >> (8 * sb)) & 0xFFu);
            float a = 0;
#pragma unroll
            for (int hw = 0; hw < 2; ++hw) {
                const uint32_t q = w[1 + sb * 2 + hw];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t c = (q >> (8 * j)) & 0xFFu;
                    const int i = hw * 4 + j;
                    a += tab[c & 15] * xv[sb * 16 + i] + tab[c >> 4] * xv[sb * 16 + i + 8];
                }
            }
            acc += a * d;
        }
        part[warp][gb] = acc;
    }
    __syncwarp();
    if (lane < R) {
        float v = 0;
#pragma unroll
        for (int b = 0; b < NBLK; ++b) v += part[warp][lane * NBLK + b];
        const int r = row0 + lane;
        if (r < rows) y[(size_t) blockIdx.y * ldy + r] = escale ? v * escale[jb.e] : v;
    }
}


// p4c: the writing path as one CUDA graph a token: the exchanges' sequence numbers come from a device counter (k_tick at the start
// of every forward; exchange i of that forward has seq = tok * 128 + i + 1), so a replay changes no kernel argument
__global__ void k_tick(unsigned* tok) { *tok += 1; }
__global__ void k_xput2(const float* y, int n, float* out, volatile unsigned* flag, const unsigned* tok, unsigned i) {
    const unsigned seq = *tok * 128u + i + 1u;
    for (int j = threadIdx.x; j < n; j += blockDim.x) out[j] = y[j];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *flag = seq;
}
__global__ void k_xget2(float* y, int n, const volatile float* in, const volatile unsigned* flag, const unsigned* tok, unsigned i, int card) {
    const unsigned seq = *tok * 128u + i + 1u;
    if (threadIdx.x == 0)
        while (*flag < seq) {}
    __syncthreads();
    __threadfence_system();
    for (int j = threadIdx.x; j < n; j += blockDim.x) y[j] = card == 0 ? y[j] + in[j] : in[j] + y[j];
}


// p32: the writing path's cache of RAM experts (one card), see ink2_v1.cu place_experts
struct ECache {
    int S = 0;                            // slots
    size_t eb = 0, gb = 0;                // an expert's bytes (gate_up, then down) and its gate_up part
    uint8_t* mem = nullptr;               // S x eb
    int* slot_of = nullptr;               // [NE] the expert's slot or -1
    int* exp_of = nullptr;                // [S] the slot's expert or -1
    unsigned* used = nullptr;             // [S] the stamp (forward counter) of the last forward that picked it
    const char* host = nullptr;           // [NE] 1 = the expert lives in RAM
    const uint8_t* const* hb = nullptr;   // [NE] the experts' RAM bases (whole expert)
    int* fill = nullptr;                  // [2 x MAXT * TOPK] (expert, destination) to copy (p34: destination < 0: scratch byte offset -1 - off / 16)
    int* nfill = nullptr;
    uint8_t* scr = nullptr;               // p34: the scratch shared by all layers, scr_bytes
    size_t scr_bytes = 0;
    int* scr_list = nullptr;              // p34: [MAXT * TOPK] this layer's experts the tables point at the scratch for, nscr of them
    int* nscr = nullptr;
    uint8_t* const* sp = nullptr;         // p63: [S] the slots' addresses (the cache's block, then pieces of the reading path's buffers)
    int S0 = 0;                           // p63: slots from S0 on lie in the reading path's buffers (emptied before a prompt is read)
};
__global__ void k_cache_assign(ECache c, const int* topi, int nj, const unsigned* stamp, const uint8_t** gt, const uint8_t** dt) {
    if (threadIdx.x || blockIdx.x) return;
    const unsigned st = *stamp;
    int nf = 0;
    if (c.nscr) {   // p34: the last forward's scratch picks get their RAM bases back
        for (int i = 0; i < *c.nscr; ++i) {
            const int e = c.scr_list[i];
            gt[e] = c.hb[e];
            dt[e] = c.hb[e] + c.gb;
        }
        *c.nscr = 0;
    }
    size_t soff = 0;
    int ns = 0;
    for (int i = 0; i < nj; ++i) {
        const int e = topi[i];
        if (!c.host[e]) continue;
        const int s = c.slot_of[e];
        if (s >= 0) {
            c.used[s] = st;
            continue;
        }
        int v = -1;
        unsigned best = 0xffffffffu;
        for (int k = 0; k < c.S; ++k) {   // an empty slot, else the least recently picked one this forward did not pick
            const unsigned u = c.used[k];
            if (u == st) continue;
            if (c.exp_of[k] < 0) {
                v = k;
                break;
            }
            if (u < best) {
                best = u;
                v = k;
            }
        }
        if (v < 0) {   // p34: no slot: the scratch, once for the whole window
            bool seen = false;
            for (int k = 0; k < ns; ++k) seen |= c.scr_list[k] == e;
            if (seen || !c.scr || soff + c.eb > c.scr_bytes) continue;
            gt[e] = c.scr + soff;
            dt[e] = c.scr + soff + c.gb;
            c.scr_list[ns++] = e;
            c.fill[2 * nf] = e;
            c.fill[2 * nf + 1] = -1 - (int) (soff / 16);
            ++nf;
            soff += c.eb;
            continue;
        }
        const int old = c.exp_of[v];
        if (old >= 0) {
            c.slot_of[old] = -1;
            gt[old] = c.hb[old];
            dt[old] = c.hb[old] + c.gb;
        }
        c.exp_of[v] = e;
        c.slot_of[e] = v;
        c.used[v] = st;
        gt[e] = c.sp[v];   // p63
        dt[e] = c.sp[v] + c.gb;
        c.fill[2 * nf] = e;
        c.fill[2 * nf + 1] = v;
        ++nf;
    }
    *c.nfill = nf;
    if (c.nscr) *c.nscr = ns;
}
__global__ void __launch_bounds__(256) k_cache_fill(ECache c) {   // block (x, j): a share of fill j's bytes, 16 at a time
    const int j = blockIdx.y;
    if (j >= *c.nfill) return;
    const uint4* src = (const uint4*) c.hb[c.fill[2 * j]];
    const int d = c.fill[2 * j + 1];
    uint4* dst = (uint4*) (d >= 0 ? c.sp[d] : c.scr + (size_t) (-1 - d) * 16);   // p63
    const size_t n16 = c.eb / 16;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}
// p63: the slots in the reading path's buffers emptied (their experts' tables back to the RAM bases) before a prompt is read
__global__ void k_cache_drop(ECache c, const uint8_t** gt, const uint8_t** dt) {
    for (int s = c.S0 + (int) threadIdx.x; s < c.S; s += blockDim.x) {
        const int e = c.exp_of[s];
        if (e < 0) continue;
        c.slot_of[e] = -1;
        gt[e] = c.hb[e];
        dt[e] = c.hb[e] + c.gb;
        c.exp_of[s] = -1;
        c.used[s] = 0;
    }
}

// planar NVFP4 rows (p4d): a row of NBLK blocks stored as all 32-byte code runs, then all 4-byte scale words (the same 36 bytes a
// block), so lane l can read block l's codes as two aligned 16-byte loads and its scales as one word, neighbouring lanes neighbours
static void fp4_to_planar(const uint8_t* src, uint8_t* dst, size_t rows, int nblk, int nexp) {   // per expert: codes, then scales
    for (int e = 0; e < nexp; ++e) {
        const uint8_t* se = src + (size_t) e * rows * nblk * 36;
        uint8_t* de = dst + (size_t) e * rows * nblk * 36;
        for (size_t r = 0; r < rows; ++r)
            for (int b = 0; b < nblk; ++b) {
                const size_t i = r * nblk + b;
                std::memcpy(de + i * 32, se + i * 36 + 4, 32);
                std::memcpy(de + rows * nblk * 32 + i * 4, se + i * 36, 4);
            }
    }
}
// k_fp4_rows6: k_fp4_rows3's tiles (R rows a warp, lane l the blocks l, l + 32, ... of the tile) on planar rows
template <int NBLK, int R, int WPB = 4>   // p49: WPB warps a block
__global__ void __launch_bounds__(32 * WPB) k_fp4_rows6(const uint8_t* __restrict__ W, size_t estride, int rows, const float* x, int ldx,
                                                   int xbyjob, const float* escale, const EJob* jobs, float* y, int ldy, const int* perm = nullptr,
                                                   const uint8_t* const* ET = nullptr) {   // p30: expert bases
    static_assert(NBLK * R == 288, "a warp's tile is 9 blocks a lane");
    constexpr int COLS = NBLK * 64;
    __shared__ float xs[NBLK * 65];
    __shared__ float part[WPB][288];
    __shared__ float tab[16], sct[256];
    const int jy = perm ? perm[blockIdx.y] : blockIdx.y;   // p11: blocks by expert, outputs by job
    const EJob jb = jobs[jy];
    const float* xr = x + (size_t) (xbyjob ? jy : jb.t) * ldx;
    for (int i = threadIdx.x; i < COLS; i += blockDim.x) xs[(i >> 6) * 65 + (i & 63)] = xr[i];
    if (threadIdx.x < 16) tab[threadIdx.x] = c_fp4[threadIdx.x];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) sct[i] = ue4m3((uint8_t) i);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * WPB + warp) * R;
    if (row0 >= rows) return;
    const uint8_t* EB = ET ? ET[jb.e] : W + (size_t) jb.e * estride;   // p30
    const uint8_t* codes = EB + (size_t) row0 * NBLK * 32;
    const uint32_t* scales = (const uint32_t*) (EB + (size_t) rows * NBLK * 32) + (size_t) row0 * NBLK;
#pragma unroll 3
    for (int k = 0; k < 9; ++k) {
        const int gb = k * 32 + lane, bl = gb % NBLK;
        const uint4 c0 = __ldg((const uint4*) (codes + (size_t) gb * 32)), c1 = __ldg((const uint4*) (codes + (size_t) gb * 32 + 16));
        const uint32_t sc = __ldg(scales + gb);
        const uint32_t q[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        const float* xv = xs + bl * 65;
        float acc = 0;
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            const float d = sct[(sc >> (8 * sb)) & 0xFFu];
            float a = 0;
#pragma unroll
            for (int hw = 0; hw < 2; ++hw) {
                const uint32_t qq = q[sb * 2 + hw];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t c = (qq >> (8 * j)) & 0xFFu;
                    const int i = hw * 4 + j;
                    a += tab[c & 15] * xv[sb * 16 + i] + tab[c >> 4] * xv[sb * 16 + i + 8];
                }
            }
            acc += a * d;
        }
        part[warp][gb] = acc;
    }
    __syncwarp();
    if (lane < R) {
        float v = 0;
#pragma unroll
        for (int b = 0; b < NBLK; ++b) v += part[warp][lane * NBLK + b];
        const int r = row0 + lane;
        if (r < rows) y[(size_t) jy * ldy + r] = escale ? v * escale[jb.e] : v;
    }
}


// k_fp4_rows7: planar rows, LPR lanes a row (gate_up 4, down 1), each lane 9 blocks of its row, the row's sum by shuffles (no shared
// partial sums: more blocks an SM); TPB threads a block
template <int NBLK, int LPR, int TPB>
__global__ void __launch_bounds__(TPB) k_fp4_rows7(const uint8_t* __restrict__ W, size_t estride, int rows, const float* x, int ldx,
                                                   int xbyjob, const float* escale, const EJob* jobs, float* y, int ldy, const int* perm = nullptr,
                                                   const uint8_t* const* ET = nullptr) {   // p30: expert bases
    constexpr int COLS = NBLK * 64, BPL = NBLK / LPR, RPW = 32 / LPR;
    __shared__ float xs[NBLK * 65];
    __shared__ float tab[16], sct[256];
    const int jy = perm ? perm[blockIdx.y] : blockIdx.y;   // p11: blocks by expert, outputs by job
    const EJob jb = jobs[jy];
    const float* xr = x + (size_t) (xbyjob ? jy : jb.t) * ldx;
    for (int i = threadIdx.x; i < COLS; i += TPB) xs[(i >> 6) * 65 + (i & 63)] = xr[i];
    if (threadIdx.x < 16) tab[threadIdx.x] = c_fp4[threadIdx.x];
    for (int i = threadIdx.x; i < 256; i += TPB) sct[i] = ue4m3((uint8_t) i);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (blockIdx.x * (TPB / 32) + warp) * RPW + lane / LPR;
    if (row >= rows) return;
    const uint8_t* EB = ET ? ET[jb.e] : W + (size_t) jb.e * estride;   // p30
    const uint8_t* codes = EB + (size_t) row * NBLK * 32;
    const uint32_t* scales = (const uint32_t*) (EB + (size_t) rows * NBLK * 32) + (size_t) row * NBLK;
    float acc = 0;
#pragma unroll 3
    for (int k = 0; k < BPL; ++k) {
        const int b = (lane % LPR) + LPR * k;
        const uint4 c0 = __ldg((const uint4*) (codes + b * 32)), c1 = __ldg((const uint4*) (codes + b * 32 + 16));
        const uint32_t sc = __ldg(scales + b);
        const uint32_t q[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        const float* xv = xs + b * 65;
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            const float d = sct[(sc >> (8 * sb)) & 0xFFu];
            float a = 0;
#pragma unroll
            for (int hw = 0; hw < 2; ++hw) {
                const uint32_t qq = q[sb * 2 + hw];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t c = (qq >> (8 * j)) & 0xFFu;
                    const int i = hw * 4 + j;
                    a += tab[c & 15] * xv[sb * 16 + i] + tab[c >> 4] * xv[sb * 16 + i + 8];
                }
            }
            acc += a * d;
        }
    }
#pragma unroll
    for (int o = LPR / 2; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane % LPR == 0) y[(size_t) jy * ldy + row] = escale ? acc * escale[jb.e] : acc;
}

// ------------------------------------------------------------------------------------------------ p4e: fused writing kernels (T = 1)
// The same math as the separate kernels; INK2_FUSE_OFF=1 runs them separately.

// the cross-card sum, the short convolution of its result and the residual in one block: own = y (or, with ey, the routed experts'
// weighted sum sum_k w[k] ey[k]), written to page-locked RAM, the other card's part added (card 0's first on both), conv with the
// slot's state (3 inputs before), then cur += a, or (push) bpush = cur and cur = a
__global__ void __launch_bounds__(1024) k_xchg_conv(const float* y, const float* ey, const float* topw, float* out, volatile unsigned* flag,
                                                    const volatile float* in, const volatile unsigned* oflag, const unsigned* tok, unsigned idx,
                                                    int card, float* st, const float* w, float k, float* cur, float* bpush) {
    const unsigned seq = *tok * 128u + idx + 1u;
    float own[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = threadIdx.x + 1024 * j;
        own[j] = 0.0f;
        if (i < N) {
            float v;
            if (ey) {
                v = 0.0f;
                for (int kk = 0; kk < TOPK; ++kk) v += ey[(size_t) kk * N + i] * topw[kk];
            } else {
                v = y[i];
            }
            own[j] = v;
            if (NC > 1) out[i] = v;
        }
    }
    __shared__ __align__(16) float oth[NC > 1 ? N : 4];
    if (NC > 1) {   // p30: one card (NC = 1) has nothing to exchange
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *flag = seq;
        while (*oflag < seq) {}
    }
    __syncthreads();
    __threadfence_system();
    // p6e: the other card's part through shared memory, read by float4 (576 threads, one load each, all issued at once)
    if (threadIdx.x < N / 4) {
        const float4 v4 = __ldcv((const float4*) (const float*) in + threadIdx.x);   // not cached: the other card wrote it
        *(float4*) (oth + threadIdx.x * 4) = v4;
    }
    __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = threadIdx.x + 1024 * j;
        if (i >= N) continue;
        const float a = NC == 1 ? own[j] : card == 0 ? own[j] + oth[i] : oth[i] + own[j];
        const float s0 = st[i], s1 = st[N + i], s2 = st[2 * N + i];
        float o = 0.0f;
        o += s0 * w[i * 4 + 0];
        o += s1 * w[i * 4 + 1];
        o += s2 * w[i * 4 + 2];
        o += a * w[i * 4 + 3];
        const float r = (a + o) * k;
        st[i] = s1;
        st[N + i] = s2;
        st[2 * N + i] = a;
        if (bpush) {
            bpush[i] = cur[i];
            cur[i] = r;
        } else {
            cur[i] = cur[i] + r;
        }
    }
}

// after the attention's projections (T = 1): q clip (x q_clip, L2 <= thr, x tau), r clip (x rel_clip, L2 <= 8, x tau), K and V convolved
// with the slot's states (+ dynamic taps), K clipped, V rms-normed, both stored in the cache. Blocks: NH_C q heads, NH_C r heads, NKV_C
// k heads, NKV_C v heads, 128 threads (a value each)
__global__ void __launch_bounds__(128) k_post_qkv(float* q, float* r, const float* k, const float* v, const float* dk, const float* dv,
                                                  float* sk, float* sv, const float* kw, const float* vw, const float* qclip, const float* rclip,
                                                  int h0, float qk_thr, float rel_thr, const float* tau, float eps, float kconv, const int* pos,
                                                  int ring, __half* kc, __half* vc, KV8 q8) {
    __shared__ float red[32], rsh[HD];
    const int b = blockIdx.x, i = threadIdx.x;
    const float tt = tau ? tau[0] : 1.0f;
    if (b < NH_C) {   // q head b
        float x = q[b * HD + i] * qclip[(size_t) (h0 + b) * HD + i];
        const float ss = block_sum(x * x, red);
        const float f = fminf(fmaxf(qk_thr / (sqrtf(ss) + 1e-6f), 0.0f), 1.0f);
        float y = (x * f) * tt;
        if (q8.rot) y = fwht128(y, rsh);   // p18: Hq . Hk = q . k
        q[b * HD + i] = y;
        return;
    }
    if (b < 2 * NH_C) {   // r head b - NH_C (16 values)
        const int h = b - NH_C;
        float x = 0.0f;
        if (i < NREL) x = r[h * NREL + i] * rclip[(size_t) (h0 + h) * NREL + i];
        const float ss = block_sum(x * x, red);
        const float f = fminf(fmaxf(rel_thr / (sqrtf(ss) + 1e-6f), 0.0f), 1.0f);
        if (i < NREL) r[h * NREL + i] = (x * f) * tt;
        return;
    }
    const bool isk = b < 2 * NH_C + NKV_C;
    const int h = isk ? b - 2 * NH_C : b - 2 * NH_C - NKV_C, c = h * HD + i;
    const float* x = isk ? k : v;
    const float* dyn = isk ? dk : dv;
    float* st = isk ? sk : sv;
    const float* w = isk ? kw : vw;
    const float xt = x[c], s0 = st[c], s1 = st[KVC + c], s2 = st[2 * KVC + c];
    float o = 0.0f;
    o += s0 * w[c * 4 + 0];
    o += s1 * w[c * 4 + 1];
    o += s2 * w[c * 4 + 2];
    o += xt * w[c * 4 + 3];
    if (dyn) {
        const float* dt = dyn + h * NCONV;
        o = o + dt[0] * xt;
        o = o + dt[1] * s2;
        o = o + dt[2] * s1;
        o = o + dt[3] * s0;
    }
    float y = (xt + o) * kconv;
    st[c] = s1;
    st[KVC + c] = s2;
    st[2 * KVC + c] = xt;
    const float ss = block_sum(y * y, red);
    if (isk) {
        const float f = fminf(fmaxf(qk_thr / (sqrtf(ss) + 1e-6f), 0.0f), 1.0f);
        y = (y * f) * 1.0f;
    } else {
        y = y * (1.0f / sqrtf(ss / HD + eps));
    }
    const size_t row = ring ? (size_t) (pos[0] % RING) : (size_t) pos[0];
    if (q8.rot) y = fwht128(y, rsh);   // p18 (the block is one head; a warp = 32 values of it)
    kv_put(q8, isk, isk ? kc : vc, row, c, y);
}

// the loop controller's down projection with the rms of x and m inside (T = 1): a block per output row, [rms(x); rms(m)] . W[row]
__global__ void __launch_bounds__(256) k_ctrl_down1(const float* __restrict__ W, const float* x, const float* m, float* g64) {
    __shared__ float red[32];
    const int r = blockIdx.x;
    float sx = 0, sm = 0, dx = 0, dm = 0;
    const float* w = W + (size_t) r * 2 * N;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        const float a = x[c], b = m[c];
        sx += a * a;
        sm += b * b;
        dx += w[c] * a;
        dm += w[N + c] * b;
    }
    sx = block_sum(sx, red);
    sm = block_sum(sm, red);
    dx = block_sum(dx, red);
    dm = block_sum(dm, red);
    if (threadIdx.x == 0) g64[r] = dx * (1.0f / sqrtf(sx / N + 1e-6f)) + dm * (1.0f / sqrtf(sm / N + 1e-6f));
}

// the router's top-k with the experts' jobs written too (T rows)
__global__ void k_route_jobs(const float* logits, const float* bias, float wscale, int* topi, float* topw, EJob* jobs) {
    __shared__ float sv[NE];
    __shared__ int si[NE];
    __shared__ float pr[NE];
    const int t = blockIdx.x, e = threadIdx.x;
    const float p = sigm(logits[(size_t) t * NE + e]);
    pr[e] = p;
    sv[e] = p + (bias ? bias[e] : 0.0f);
    si[e] = e;
    __syncthreads();
    for (int kk = 2; kk <= NE; kk <<= 1)
        for (int j = kk >> 1; j > 0; j >>= 1) {
            const int ix = e ^ j;
            if (ix > e) {
                const bool up = (e & kk) == 0;
                const float a = sv[e], b = sv[ix];
                const int ia = si[e], ib = si[ix];
                const bool a_first = a > b || (a == b && ia < ib);
                if (up ? !a_first : a_first) {
                    sv[e] = b; sv[ix] = a; si[e] = ib; si[ix] = ia;
                }
            }
            __syncthreads();
        }
    if (e == 0) {
        float s = 0;
        for (int k = 0; k < TOPK; ++k) s += pr[si[k]];
        s = s + 1e-20f;
        for (int k = 0; k < TOPK; ++k) {
            topi[t * TOPK + k] = si[k];
            topw[t * TOPK + k] = (pr[si[k]] / s) * wscale;
            jobs[t * TOPK + k] = EJob{t, k, si[k]};
        }
    }
}

// p4f: the writing path's attention (T = 1) by KV head: a block = one of the card's KV heads and KC2 keys, for its 3 query heads at once
// (K and V rows read once); 8 lanes a key (16 dims each, two 16-byte loads), 4 keys a warp at a time, the 3 dots reduced over the 8
// lanes; scores q.k / sqrt(HD) + R.B[min(d, EXT)] - sink (one lane a key adds the bias); then each head's max, exp and sum, and P.V
// with a thread a dimension. The parts go to k_attn_comb in its layout ([t = 0][q head][part]).
// p18: KCB keys a block (64; 128 past half of a 512k context, where K.part holds half the parts) and Q8 = the int8 cache (q8)
constexpr int KC2 = 64, GQ = NH / NKV;
template <int KCB, bool Q8, bool SV = false>   // p70: SV = an int8 part's V from shared memory
__global__ void __launch_bounds__(128) k_attn_dec(const float* q, const float* R, const float* B, const float* sink, const __half* kc,
                                                  const __half* vc, const int* pos, int window, float* part, int nparts, KV8 q8, int poff,
                                                  int nt = 1) {   // p38: nt rows of a window side by side in grid x
    __shared__ float qs[GQ][HD];
    __shared__ float rs[GQ][NREL];
    __shared__ float sc[GQ][KCB];
    __shared__ float red[GQ][4];
    constexpr bool STAGE = SV && Q8 && KCB * HD <= 16384;   // p70: an int8 part's V rows staged in shared memory
    constexpr int VU = STAGE ? KCB * HD / 16 / 128 : 1, SPT = STAGE ? KCB * (HD / KQG) / 128 : 1;
    __shared__ __align__(16) int8_t vsh[STAGE ? KCB * HD : 16];
    __shared__ __half vss[STAGE ? KCB * (HD / KQG) : 1];
    const int g = blockIdx.y, p = (int) blockIdx.x / nt + poff, tid = threadIdx.x, t = (int) blockIdx.x % nt + (int) blockIdx.z;   // p38   // p5a: token t of a window (grid z); p18: parts from poff
    const int P = pos[t];
    q += (size_t) t * QC;
    R += (size_t) t * NH_C * NREL;
    sink += t * NH_C;
    part += (size_t) t * NH_C * nparts * (HD + 2);
    const int lo = window ? max(0, P - (SWA - 1)) : 0, hi = P + 1;
    const int j0 = lo + p * KCB, j1 = min(hi, j0 + KCB);
    if (j0 >= hi) {
        for (int hh = 0; hh < GQ; ++hh) {
            float* o = part + ((size_t) (g * GQ + hh) * nparts + p) * (HD + 2);
            if (tid == 0) { o[0] = -INFINITY; o[1] = 0.0f; }
            o[2 + tid] = 0.0f;
        }
        return;
    }
    const int nv = j1 - j0;
    const bool v8 = STAGE && !window && q8.v && (size_t) j0 >= (size_t) q8.f0;   // a part lies on one side of f0
    // p70: the part's V rows (16-byte pieces) and scales issued now, so they arrive while the K part runs
    uint4 vr[VU];
    __half vsr[SPT];
    if (v8) {
#pragma unroll
        for (int k = 0; k < VU; ++k) {
            const int u = tid + 128 * k, jj = u / (HD / 16), c = u % (HD / 16);
            vr[k] = jj < nv ? __ldg((const uint4*) (q8.v + ((size_t) (j0 + jj) - q8.f0) * KVC + g * HD) + c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int s2 = 0; s2 < SPT; ++s2) {
            const int u = tid * SPT + s2, jj = u / (HD / KQG), c = u % (HD / KQG);
            vsr[s2] = jj < nv ? q8.vs[((size_t) (j0 + jj) - q8.f0) * KQN + (g * HD) / KQG + c] : __float2half(0.0f);
        }
    }
    constexpr int KIT = KCB / 16;   // p72: rounds of the K loop (16 keys a round: 4 warps x 4 keys)
    uint4 kpr[STAGE ? KIT : 1];
    __half kps[STAGE ? KIT : 1];
    const bool k8p = v8 && q8.k;   // the part's K pieces issued now, next to p70's V
    if (k8p) {
        const int w_ = tid >> 5, s_ = (tid & 31) >> 3, l_ = tid & 7;
#pragma unroll
        for (int it = 0; it < (STAGE ? KIT : 1); ++it) {
            const int jj = w_ * 4 + s_ + 16 * it;
            kpr[it] = make_uint4(0, 0, 0, 0);
            kps[it] = __float2half(1.0f);
            if (jj < nv) {
                const size_t r8 = (size_t) (j0 + jj) - q8.f0;
                kpr[it] = __ldg((const uint4*) (q8.k + r8 * KVC + g * HD + l_ * 16));
                kps[it] = q8.ks[r8 * KQN + (g * HD + l_ * 16) / KQG];
            }
        }
    }
    for (int hh = 0; hh < GQ; ++hh) qs[hh][tid] = q[(g * GQ + hh) * HD + tid];   // p18 rot: q is H q already (post_qkv)
    if (tid < GQ * NREL) rs[tid / NREL][tid % NREL] = R[(g * GQ + tid / NREL) * NREL + tid % NREL];
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31, sub = lane >> 3, l8 = lane & 7;
    const float scale = 1.0f / sqrtf((float) HD);
    float qv[GQ][16];
#pragma unroll
    for (int hh = 0; hh < GQ; ++hh)
#pragma unroll
        for (int i = 0; i < 16; ++i) qv[hh][i] = qs[hh][l8 * 16 + i];
    const int n = j1 - j0;
    if (k8p) {   // p72: an int8 part, its K pieces issued at the start: the loop unrolled over its rounds
#pragma unroll
    for (int it = 0; it < KIT; ++it) {
        const int base = warp * 4 + 16 * it;
        if (base >= n) break;
        const int jj = base + sub;
        const bool ok = jj < n;
        const int j = j0 + (ok ? jj : 0);
        const size_t row = window ? (size_t) (j % RING) : (size_t) j;
        float kf[16], ksc = 1.0f;
        if (Q8 && q8.k && row >= (size_t) q8.f0) {   // p18: the lane's 16 values (one 16-byte load) lie in one group of 32
            uint4 a = make_uint4(0, 0, 0, 0);
            if (k8p) {   // p72: issued at the start
                a = kpr[STAGE ? it : 0];
                ksc = __half2float(kps[STAGE ? it : 0]);
            } else if (ok) {
                const size_t r8 = row - q8.f0;
                a = __ldg((const uint4*) (q8.k + r8 * KVC + g * HD + l8 * 16));
                ksc = __half2float(q8.ks[r8 * KQN + (g * HD + l8 * 16) / KQG]);
            }
            const int8_t* b8 = (const int8_t*) &a;
#pragma unroll
            for (int i = 0; i < 16; ++i) kf[i] = (float) b8[i];
        } else {
            const uint4* kr = (const uint4*) (kc + row * KVC + g * HD + l8 * 16);
            uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
            if (ok) {
                a = __ldg(kr);
                b = __ldg(kr + 1);
            }
            const __half2* h2a = (const __half2*) &a;
            const __half2* h2b = (const __half2*) &b;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 fa = __half22float2(h2a[i]), fb = __half22float2(h2b[i]);
                kf[2 * i] = fa.x; kf[2 * i + 1] = fa.y; kf[8 + 2 * i] = fb.x; kf[8 + 2 * i + 1] = fb.y;
            }
        }
        float d[GQ];
#pragma unroll
        for (int hh = 0; hh < GQ; ++hh) {
            float s = 0;
#pragma unroll
            for (int i = 0; i < 16; ++i) s += kf[i] * qv[hh][i];
            if (Q8) s *= ksc;
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            s += __shfl_xor_sync(0xffffffffu, s, 4);
            d[hh] = s;
        }
        if (ok && l8 < GQ) {   // lane l8 = hh adds head hh's bias
            const int hh = l8, dist = P - j;
            float bias = 0.0f;
            if (dist < EXT) {
                const float* bb = B + (size_t) dist * NREL;
#pragma unroll
                for (int r = 0; r < NREL; ++r) bias += rs[hh][r] * bb[r];
            }
            const float dd = hh == 0 ? d[0] : (hh == 1 ? d[1] : d[2]);
            sc[hh][jj] = dd * scale + (bias - sink[g * GQ + hh]);
        }
    }
    } else
    for (int base = warp * 4; base < n; base += 16) {   // the warp's 4 keys of this round (all lanes reach the shuffles)
        const int it = 0;
        const int jj = base + sub;
        const bool ok = jj < n;
        const int j = j0 + (ok ? jj : 0);
        const size_t row = window ? (size_t) (j % RING) : (size_t) j;
        float kf[16], ksc = 1.0f;
        if (Q8 && q8.k && row >= (size_t) q8.f0) {   // p18: the lane's 16 values (one 16-byte load) lie in one group of 32
            uint4 a = make_uint4(0, 0, 0, 0);
            if (k8p) {   // p72: issued at the start
                a = kpr[STAGE ? it : 0];
                ksc = __half2float(kps[STAGE ? it : 0]);
            } else if (ok) {
                const size_t r8 = row - q8.f0;
                a = __ldg((const uint4*) (q8.k + r8 * KVC + g * HD + l8 * 16));
                ksc = __half2float(q8.ks[r8 * KQN + (g * HD + l8 * 16) / KQG]);
            }
            const int8_t* b8 = (const int8_t*) &a;
#pragma unroll
            for (int i = 0; i < 16; ++i) kf[i] = (float) b8[i];
        } else {
            const uint4* kr = (const uint4*) (kc + row * KVC + g * HD + l8 * 16);
            uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
            if (ok) {
                a = __ldg(kr);
                b = __ldg(kr + 1);
            }
            const __half2* h2a = (const __half2*) &a;
            const __half2* h2b = (const __half2*) &b;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 fa = __half22float2(h2a[i]), fb = __half22float2(h2b[i]);
                kf[2 * i] = fa.x; kf[2 * i + 1] = fa.y; kf[8 + 2 * i] = fb.x; kf[8 + 2 * i + 1] = fb.y;
            }
        }
        float d[GQ];
#pragma unroll
        for (int hh = 0; hh < GQ; ++hh) {
            float s = 0;
#pragma unroll
            for (int i = 0; i < 16; ++i) s += kf[i] * qv[hh][i];
            if (Q8) s *= ksc;
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            s += __shfl_xor_sync(0xffffffffu, s, 4);
            d[hh] = s;
        }
        if (ok && l8 < GQ) {   // lane l8 = hh adds head hh's bias
            const int hh = l8, dist = P - j;
            float bias = 0.0f;
            if (dist < EXT) {
                const float* bb = B + (size_t) dist * NREL;
#pragma unroll
                for (int r = 0; r < NREL; ++r) bias += rs[hh][r] * bb[r];
            }
            const float dd = hh == 0 ? d[0] : (hh == 1 ? d[1] : d[2]);
            sc[hh][jj] = dd * scale + (bias - sink[g * GQ + hh]);
        }
    }
    if (v8) {   // p70: the V rows into shared memory (read after the barrier below)
#pragma unroll
        for (int k = 0; k < VU; ++k) ((uint4*) vsh)[tid + 128 * k] = vr[k];
#pragma unroll
        for (int s2 = 0; s2 < SPT; ++s2) vss[tid * SPT + s2] = vsr[s2];
    }
    __syncthreads();
    // each head's max and sum over the n scores: warp hh (hh < GQ) does head hh
    if (warp < GQ) {
        const int hh = warp;
        float m = -INFINITY;
        for (int j = lane; j < n; j += 32) m = fmaxf(m, sc[hh][j]);
        m = warp_max(m);
        float l = 0;
        for (int j = lane; j < n; j += 32) {
            const float e = expf(sc[hh][j] - m);
            sc[hh][j] = e;
            l += e;
        }
        l = warp_sum(l);
        if (lane == 0) { red[hh][0] = m; red[hh][1] = l; }
    }
    __syncthreads();
    float acc[GQ] = {};
    if (v8)   // p70: the same products in the same order, V from shared memory
        for (int jj = 0; jj < n; ++jj) {
            const float vv = (float) vsh[jj * HD + tid] * __half2float(vss[jj * (HD / KQG) + tid / KQG]);
#pragma unroll
            for (int hh = 0; hh < GQ; ++hh) acc[hh] += sc[hh][jj] * vv;
        }
    else
    for (int jj = 0; jj < n; ++jj) {
        const int j = j0 + jj;
        const size_t row = window ? (size_t) (j % RING) : (size_t) j;
        const float vv = Q8 ? kv_get(q8, false, vc, row, g * HD + tid) : __half2float(vc[row * KVC + g * HD + tid]);
#pragma unroll
        for (int hh = 0; hh < GQ; ++hh) acc[hh] += sc[hh][jj] * vv;
    }
#pragma unroll
    for (int hh = 0; hh < GQ; ++hh) {
        float* o = part + ((size_t) (g * GQ + hh) * nparts + p) * (HD + 2);
        if (tid == 0) { o[0] = red[hh][0]; o[1] = red[hh][1]; }
        o[2 + tid] = acc[hh];
    }
}


// p61: a full layer's attention for up to TT rows of a draft window in one block (INK2_ATT_W): the K and V rows read once
template <int KCB, bool Q8, int TT, bool SV = false>   // p70: SV as k_attn_dec's
__global__ void __launch_bounds__(128) k_attn_decw(const float* q, const float* R, const float* B, const float* sink, const __half* kc,
                                                   const __half* vc, const int* pos, float* part, int nparts, KV8 q8, int poff, int T) {
    constexpr int QP = 17;   // a lane's 16 query values, padded: 8 lanes read 8 banks apart
    __shared__ float qs[TT][GQ][8 * QP];
    __shared__ float rs[TT][GQ][NREL];
    __shared__ float sc[TT][GQ][KCB];
    __shared__ float red[TT][GQ][2];
    constexpr bool STAGE = SV && Q8 && KCB * HD <= 16384;   // p70: an int8 part's V rows staged in shared memory
    constexpr int VU = STAGE ? KCB * HD / 16 / 128 : 1, SPT = STAGE ? KCB * (HD / KQG) / 128 : 1;
    __shared__ __align__(16) int8_t vsh[STAGE ? KCB * HD : 16];
    __shared__ __half vss[STAGE ? KCB * (HD / KQG) : 1];
    const int g = blockIdx.y, p = (int) blockIdx.x + poff, tid = threadIdx.x, t0 = (int) blockIdx.z * TT;
    const int nt = min(TT, T - t0);
    const int j0 = p * KCB;
    int nr[TT];   // keys of this part a row
    int nmax = 0;
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        nr[t] = t < nt ? max(0, min(pos[t0 + t] + 1, j0 + KCB) - j0) : 0;
        nmax = max(nmax, nr[t]);
    }
    const bool v8 = STAGE && q8.v && (size_t) j0 >= (size_t) q8.f0;   // p70: full layers only, a part on one side of f0
    // p70: the part's V rows (16-byte pieces) and scales issued now, so they arrive while the K part runs
    uint4 vr[VU];
    __half vsr[SPT];
    if (v8) {
#pragma unroll
        for (int k = 0; k < VU; ++k) {
            const int u = tid + 128 * k, jj = u / (HD / 16), c = u % (HD / 16);
            vr[k] = jj < nmax ? __ldg((const uint4*) (q8.v + ((size_t) (j0 + jj) - q8.f0) * KVC + g * HD) + c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int s2 = 0; s2 < SPT; ++s2) {
            const int u = tid * SPT + s2, jj = u / (HD / KQG), c = u % (HD / KQG);
            vsr[s2] = jj < nmax ? q8.vs[((size_t) (j0 + jj) - q8.f0) * KQN + (g * HD) / KQG + c] : __float2half(0.0f);
        }
    }
    constexpr int KIT = KCB / 16;   // p72: rounds of the K loop (16 keys a round: 4 warps x 4 keys)
    uint4 kpr[STAGE ? KIT : 1];
    __half kps[STAGE ? KIT : 1];
    const bool k8p = v8 && q8.k;   // the part's K pieces issued now, next to p70's V
    if (k8p) {
        const int w_ = tid >> 5, s_ = (tid & 31) >> 3, l_ = tid & 7;
#pragma unroll
        for (int it = 0; it < (STAGE ? KIT : 1); ++it) {
            const int jj = w_ * 4 + s_ + 16 * it;
            kpr[it] = make_uint4(0, 0, 0, 0);
            kps[it] = __float2half(1.0f);
            if (jj < nmax) {
                const size_t r8 = (size_t) (j0 + jj) - q8.f0;
                kpr[it] = __ldg((const uint4*) (q8.k + r8 * KVC + g * HD + l_ * 16));
                kps[it] = q8.ks[r8 * KQN + (g * HD + l_ * 16) / KQG];
            }
        }
    }
    for (int t = 0; t < nt; ++t)
        for (int hh = 0; hh < GQ; ++hh) qs[t][hh][(tid >> 4) * QP + (tid & 15)] = q[(size_t) (t0 + t) * QC + (g * GQ + hh) * HD + tid];
    for (int i = tid; i < nt * GQ * NREL; i += 128) {
        const int t = i / (GQ * NREL), r = i % (GQ * NREL);
        rs[t][r / NREL][r % NREL] = R[(size_t) (t0 + t) * NH_C * NREL + (g * GQ + r / NREL) * NREL + r % NREL];
    }
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31, sub = lane >> 3, l8 = lane & 7;
    const float scale = 1.0f / sqrtf((float) HD);
    if (k8p) {   // p72: an int8 part, its K pieces issued at the start: the loop unrolled over its rounds
#pragma unroll
    for (int it = 0; it < KIT; ++it) {
        const int base = warp * 4 + 16 * it;
        if (base >= nmax) break;
        const int jj = base + sub;
        const bool ok = jj < nmax;
        const int j = j0 + (ok ? jj : 0);
        const size_t row = (size_t) j;
        float kf[16], ksc = 1.0f;
        if (Q8 && q8.k && row >= (size_t) q8.f0) {
            uint4 a = make_uint4(0, 0, 0, 0);
            if (k8p) {   // p72: issued at the start
                a = kpr[STAGE ? it : 0];
                ksc = __half2float(kps[STAGE ? it : 0]);
            } else if (ok) {
                const size_t r8 = row - q8.f0;
                a = __ldg((const uint4*) (q8.k + r8 * KVC + g * HD + l8 * 16));
                ksc = __half2float(q8.ks[r8 * KQN + (g * HD + l8 * 16) / KQG]);
            }
            const int8_t* b8 = (const int8_t*) &a;
#pragma unroll
            for (int i = 0; i < 16; ++i) kf[i] = (float) b8[i];
        } else {
            const uint4* kr = (const uint4*) (kc + row * KVC + g * HD + l8 * 16);
            uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
            if (ok) {
                a = __ldg(kr);
                b = __ldg(kr + 1);
            }
            const __half2* h2a = (const __half2*) &a;
            const __half2* h2b = (const __half2*) &b;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 fa = __half22float2(h2a[i]), fb = __half22float2(h2b[i]);
                kf[2 * i] = fa.x; kf[2 * i + 1] = fa.y; kf[8 + 2 * i] = fb.x; kf[8 + 2 * i + 1] = fb.y;
            }
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t >= nt) break;
            float d[GQ];
#pragma unroll
            for (int hh = 0; hh < GQ; ++hh) {
                const float* qv = &qs[t][hh][l8 * QP];
                float s = 0;
#pragma unroll
                for (int i = 0; i < 16; ++i) s += kf[i] * qv[i];
                if (Q8) s *= ksc;
                s += __shfl_xor_sync(0xffffffffu, s, 1);
                s += __shfl_xor_sync(0xffffffffu, s, 2);
                s += __shfl_xor_sync(0xffffffffu, s, 4);
                d[hh] = s;
            }
            if (jj < nr[t] && l8 < GQ) {   // lane l8 = hh adds head hh's bias, as k_attn_dec
                const int hh = l8, dist = pos[t0 + t] - j;
                float bias = 0.0f;
                if (dist < EXT) {
                    const float* bb = B + (size_t) dist * NREL;
#pragma unroll
                    for (int r = 0; r < NREL; ++r) bias += rs[t][hh][r] * bb[r];
                }
                const float dd = hh == 0 ? d[0] : (hh == 1 ? d[1] : d[2]);
                sc[t][hh][jj] = dd * scale + (bias - sink[(t0 + t) * NH_C + g * GQ + hh]);
            }
        }
    }
    } else
    for (int base = warp * 4; base < nmax; base += 16) {
        const int it = 0;
        const int jj = base + sub;
        const bool ok = jj < nmax;
        const int j = j0 + (ok ? jj : 0);
        const size_t row = (size_t) j;
        float kf[16], ksc = 1.0f;
        if (Q8 && q8.k && row >= (size_t) q8.f0) {
            uint4 a = make_uint4(0, 0, 0, 0);
            if (k8p) {   // p72: issued at the start
                a = kpr[STAGE ? it : 0];
                ksc = __half2float(kps[STAGE ? it : 0]);
            } else if (ok) {
                const size_t r8 = row - q8.f0;
                a = __ldg((const uint4*) (q8.k + r8 * KVC + g * HD + l8 * 16));
                ksc = __half2float(q8.ks[r8 * KQN + (g * HD + l8 * 16) / KQG]);
            }
            const int8_t* b8 = (const int8_t*) &a;
#pragma unroll
            for (int i = 0; i < 16; ++i) kf[i] = (float) b8[i];
        } else {
            const uint4* kr = (const uint4*) (kc + row * KVC + g * HD + l8 * 16);
            uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
            if (ok) {
                a = __ldg(kr);
                b = __ldg(kr + 1);
            }
            const __half2* h2a = (const __half2*) &a;
            const __half2* h2b = (const __half2*) &b;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 fa = __half22float2(h2a[i]), fb = __half22float2(h2b[i]);
                kf[2 * i] = fa.x; kf[2 * i + 1] = fa.y; kf[8 + 2 * i] = fb.x; kf[8 + 2 * i + 1] = fb.y;
            }
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t >= nt) break;
            float d[GQ];
#pragma unroll
            for (int hh = 0; hh < GQ; ++hh) {
                const float* qv = &qs[t][hh][l8 * QP];
                float s = 0;
#pragma unroll
                for (int i = 0; i < 16; ++i) s += kf[i] * qv[i];
                if (Q8) s *= ksc;
                s += __shfl_xor_sync(0xffffffffu, s, 1);
                s += __shfl_xor_sync(0xffffffffu, s, 2);
                s += __shfl_xor_sync(0xffffffffu, s, 4);
                d[hh] = s;
            }
            if (jj < nr[t] && l8 < GQ) {   // lane l8 = hh adds head hh's bias, as k_attn_dec
                const int hh = l8, dist = pos[t0 + t] - j;
                float bias = 0.0f;
                if (dist < EXT) {
                    const float* bb = B + (size_t) dist * NREL;
#pragma unroll
                    for (int r = 0; r < NREL; ++r) bias += rs[t][hh][r] * bb[r];
                }
                const float dd = hh == 0 ? d[0] : (hh == 1 ? d[1] : d[2]);
                sc[t][hh][jj] = dd * scale + (bias - sink[(t0 + t) * NH_C + g * GQ + hh]);
            }
        }
    }
    if (v8) {   // p70: the V rows into shared memory (read after the barrier below)
#pragma unroll
        for (int k = 0; k < VU; ++k) ((uint4*) vsh)[tid + 128 * k] = vr[k];
#pragma unroll
        for (int s2 = 0; s2 < SPT; ++s2) vss[tid * SPT + s2] = vsr[s2];
    }
    __syncthreads();
    for (int pp = warp; pp < nt * GQ; pp += 4) {   // each (row, head)'s max and sum by one warp, as k_attn_dec's warp hh
        const int t = pp / GQ, hh = pp % GQ;
        int n = 0;
#pragma unroll
        for (int u = 0; u < TT; ++u) if (u == t) n = nr[u];
        float m = -INFINITY;
        for (int j = lane; j < n; j += 32) m = fmaxf(m, sc[t][hh][j]);
        m = warp_max(m);
        float l = 0;
        for (int j = lane; j < n; j += 32) {
            const float e = expf(sc[t][hh][j] - m);
            sc[t][hh][j] = e;
            l += e;
        }
        l = warp_sum(l);
        if (lane == 0) { red[t][hh][0] = m; red[t][hh][1] = l; }
    }
    __syncthreads();
    float acc[TT][GQ] = {};
    if (v8)   // p70: the same products in the same order, V from shared memory
        for (int jj = 0; jj < nmax; ++jj) {
            const float vv = (float) vsh[jj * HD + tid] * __half2float(vss[jj * (HD / KQG) + tid / KQG]);
#pragma unroll
            for (int t = 0; t < TT; ++t)
                if (jj < nr[t])
#pragma unroll
                    for (int hh = 0; hh < GQ; ++hh) acc[t][hh] += sc[t][hh][jj] * vv;
        }
    else
    for (int jj = 0; jj < nmax; ++jj) {
        const int j = j0 + jj;
        const size_t row = (size_t) j;
        const float vv = Q8 ? kv_get(q8, false, vc, row, g * HD + tid) : __half2float(vc[row * KVC + g * HD + tid]);
#pragma unroll
        for (int t = 0; t < TT; ++t)
            if (jj < nr[t])
#pragma unroll
                for (int hh = 0; hh < GQ; ++hh) acc[t][hh] += sc[t][hh][jj] * vv;
    }
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        if (t >= nt) break;
        float* pt = part + (size_t) (t0 + t) * NH_C * nparts * (HD + 2);
#pragma unroll
        for (int hh = 0; hh < GQ; ++hh) {
            float* o = pt + ((size_t) (g * GQ + hh) * nparts + p) * (HD + 2);
            if (nr[t] == 0) {   // nothing of this part for the row: k_attn_dec's empty result
                if (tid == 0) { o[0] = -INFINITY; o[1] = 0.0f; }
                o[2 + tid] = 0.0f;
            } else {
                if (tid == 0) { o[0] = red[t][hh][0]; o[1] = red[t][hh][1]; }
                o[2 + tid] = acc[t][hh];
            }
        }
    }
}

// p5a: draft windows. The convolutions' states of all slots in one buffer (per slot: K, V 3 x KVC, attention and FFN outputs 3 x N);
// after a window of T tokens of which keep were valid, the state = rows keep .. keep + 2 of [the state before (3 rows), the window's
// inputs (T rows, saved in hist)]
constexpr int SLOTST = 6 * KVC + 6 * N, HISTS = 2 * MAXT * KVC + 2 * MAXT * N;
__global__ void k_conv_roll(float* st, const float* bak, const float* hist, int keep) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) NSLOT * SLOTST) return;
    const int s = (int) (i / SLOTST), o = (int) (i % SLOTST);
    int C, kind, r, c;
    if (o < 6 * KVC) { C = KVC; kind = o / (3 * KVC); r = (o % (3 * KVC)) / KVC; c = o % KVC; }
    else { const int o2 = o - 6 * KVC; C = N; kind = 2 + o2 / (3 * N); r = (o2 % (3 * N)) / N; c = o2 % N; }
    const int e = keep + r;   // index into [state (3), window inputs]
    float v;
    if (e < 3) {
        v = bak[(size_t) s * SLOTST + (o - r * C) + e * C];
    } else {
        const size_t hoff = kind == 0 ? 0 : kind == 1 ? (size_t) MAXT * KVC : kind == 2 ? (size_t) 2 * MAXT * KVC : (size_t) 2 * MAXT * KVC + MAXT * N;
        v = hist[(size_t) s * HISTS + hoff + (size_t) (e - 3) * C + c];
    }
    st[i] = v;
}

// p5b: draft windows read each picked expert once for all its tokens. k_mt_jobs (one block) groups a window's T * TOPK picks by
// expert into jobs of up to MT picks (pairs t * TOPK + k); the planar NVFP4 kernels below take a job, decode every block once and
// multiply it with each pick's input (from L1, float4), writing each pick's output row
constexpr int MT = 4;
struct MJob {
    int e, n;
    int pair[MT];
};
__global__ void __launch_bounds__(128) k_mt_jobs(const int* topi, int npairs, MJob* jobs, int* njobs) {
    __shared__ int cnt[NE], off[NE], cur[NE];
    __shared__ int list[MAXT * TOPK];
    const int tid = threadIdx.x;
    cnt[tid] = 0;
    cur[tid] = 0;
    __syncthreads();
    for (int i = tid; i < npairs; i += 128) atomicAdd(&cnt[topi[i]], 1);
    __syncthreads();
    if (tid == 0) {
        int s = 0;
        for (int e = 0; e < NE; ++e) { off[e] = s; s += cnt[e]; }
    }
    __syncthreads();
    if (tid == 0)   // in pair order within an expert (one thread: deterministic)
        for (int j = 0; j < npairs; ++j) list[off[topi[j]] + cur[topi[j]]++] = j;
    __syncthreads();
    if (tid == 0) {
        int nj = 0;
        for (int e = 0; e < NE; ++e)
            for (int a = 0; a < cnt[e]; a += MT) {
                MJob J;
                J.e = e;
                J.n = min(MT, cnt[e] - a);
                for (int b = 0; b < MT; ++b) J.pair[b] = b < J.n ? list[off[e] + a + b] : 0;
                jobs[nj++] = J;
            }
        *njobs = nj;
    }
}
// gate_up (NBLK 36, LPR 4) or down (NBLK 9, LPR 1) of a job's picks: input row of pick j = x + (xbypair ? pair : pair / TOPK) * ldx
template <int NBLK, int LPR, int TPB>
__global__ void __launch_bounds__(TPB) k_fp4_mt(const uint8_t* __restrict__ W, size_t estride, int rows, const float* __restrict__ x, int ldx,
                                                int xbypair, const float* escale, const MJob* jobs, const int* njobs, float* y, int ldy) {
    constexpr int BPL = NBLK / LPR, RPW = 32 / LPR;
    __shared__ float tab[16], sct[256];
    if (blockIdx.y >= *njobs) return;
    const MJob jb = jobs[blockIdx.y];
    if (threadIdx.x < 16) tab[threadIdx.x] = c_fp4[threadIdx.x];
    for (int i = threadIdx.x; i < 256; i += TPB) sct[i] = ue4m3((uint8_t) i);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (blockIdx.x * (TPB / 32) + warp) * RPW + lane / LPR;
    if (row >= rows) return;
    const float* xr[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) xr[j] = x + (size_t) (xbypair ? jb.pair[j] : jb.pair[j] / TOPK) * ldx;
    const uint8_t* codes = W + (size_t) jb.e * estride + (size_t) row * NBLK * 32;
    const uint32_t* scales = (const uint32_t*) (W + (size_t) jb.e * estride + (size_t) rows * NBLK * 32) + (size_t) row * NBLK;
    float acc[MT] = {};
    for (int k = 0; k < BPL; ++k) {
        const int b = (lane % LPR) + LPR * k;
        const uint4 c0 = __ldg((const uint4*) (codes + b * 32)), c1 = __ldg((const uint4*) (codes + b * 32 + 16));
        const uint32_t sc = __ldg(scales + b);
        const uint32_t q[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        float wv[64];   // the block's 64 code values (without the scales), decoded once
        float dsb[4];
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            dsb[sb] = sct[(sc >> (8 * sb)) & 0xFFu];
#pragma unroll
            for (int hw = 0; hw < 2; ++hw) {
                const uint32_t qq = q[sb * 2 + hw];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t c = (qq >> (8 * j)) & 0xFFu;
                    wv[sb * 16 + hw * 4 + j] = tab[c & 15];
                    wv[sb * 16 + 8 + hw * 4 + j] = tab[c >> 4];
                }
            }
        }
        // each pick's sum in the order of k_fp4_rows7 / k_fp4_rows6 (T = 1): per sub-block, low and high value of each byte, times
        // the scale, so a window gives the same bits as single tokens
#pragma unroll
        for (int j = 0; j < MT; ++j) {
            if (j >= jb.n) break;
            const float* xv = xr[j] + b * 64;
            float bsum = 0;
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const float4 l0 = __ldg((const float4*) (xv + sb * 16)), l1 = __ldg((const float4*) (xv + sb * 16 + 4));
                const float4 h0 = __ldg((const float4*) (xv + sb * 16 + 8)), h1 = __ldg((const float4*) (xv + sb * 16 + 12));
                const float xl[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w}, xh[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
                float a = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) a += wv[sb * 16 + i] * xl[i] + wv[sb * 16 + 8 + i] * xh[i];
                if (LPR == 1) bsum += a * dsb[sb];   // down: k_fp4_rows6 adds whole blocks' sums
                else acc[j] += a * dsb[sb];         // gate_up: k_fp4_rows7 adds each sub-block into the lane's sum
            }
            if (LPR == 1) acc[j] += bsum;
        }
    }
#pragma unroll
    for (int j = 0; j < MT; ++j) {
        float a = acc[j];
#pragma unroll
        for (int o = LPR / 2; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (j < jb.n && lane % LPR == 0) y[(size_t) jb.pair[j] * ldy + row] = escale ? a * escale[jb.e] : a;
    }
}

// ------------------------------------------------------------------------------------------------ p6b: fused kernels for windows
// (T <= MAXT rows: MTP / n-gram verification and short prompts). Row t of each fused kernel does what the T = 1 kernel does.

// Q8S for T rows (k_gemv8's tiles) with the rows of one concatenated matrix going to up to 8 outputs, output s row-major [T][n_s]
struct SegsT {
    int n;
    int r0[8], ld[8];
    float* dst[8];
};
__global__ void __launch_bounds__(256) k_gemv8_seg(const int8_t* __restrict__ q, const __half* __restrict__ d, int rows, int cols, const float* x,
                                                   int ldx, SegsT sg, int T) {
    extern __shared__ float xs[];   // [T][16][32]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rb = blockIdx.x * GW_BR, r0 = rb + warp * GW_R;
    float acc[GW_R][MAXT];
#pragma unroll
    for (int a = 0; a < GW_R; ++a)
#pragma unroll
        for (int t = 0; t < MAXT; ++t) acc[a][t] = 0.0f;
    for (int c0 = 0; c0 < cols; c0 += 512) {
        __syncthreads();
        for (int i = threadIdx.x; i < T * 512; i += 256) {
            const int t = i >> 9, c = i & 511;
            xs[t * 512 + (c & 15) * 32 + (c >> 4)] = c0 + c < cols ? x[(size_t) t * ldx + c0 + c] : 0.0f;
        }
        __syncthreads();
        const int c = c0 + lane * 16;
        if (c >= cols) continue;
#pragma unroll
        for (int a = 0; a < GW_R; ++a) {
            const int r = r0 + a;
            if (r >= rows) break;
            const int4 w4 = __ldg((const int4*) (q + (size_t) r * cols + c));
            const float sc = __half2float(d[(size_t) r * (cols / 32) + c / 32]);
            const int8_t* w = (const int8_t*) &w4;
            float wf[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) wf[j] = (float) w[j];
#pragma unroll
            for (int t = 0; t < MAXT; ++t) {
                if (t >= T) break;
                const float* xt = xs + t * 512 + lane;
                float s = 0;
#pragma unroll
                for (int j = 0; j < 16; ++j) s += wf[j] * xt[j * 32];
                acc[a][t] += sc * s;
            }
        }
    }
#pragma unroll
    for (int a = 0; a < GW_R; ++a) {
        const int r = r0 + a;
        int si = 0;
        while (si + 1 < sg.n && r >= sg.r0[si + 1]) ++si;
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= T) break;
            const float s = warp_sum(acc[a][t]);
            if (lane == 0 && r < rows) sg.dst[si][(size_t) t * sg.ld[si] + (r - sg.r0[si])] = s;
        }
    }
}

// k_post_qkv for T rows (grid y = t): the convolution reads the window's earlier raw rows (k, v [T][KVC]) or the state for the
// rows before the window; the state is advanced afterwards by k_post_state; raw rows go to hist (draft rollback) when given
__global__ void __launch_bounds__(128) k_post_qkvT(float* q, float* r, const float* k, const float* v, const float* dk, const float* dv,
                                                   const float* sk, const float* sv, const float* kw, const float* vw, const float* qclip,
                                                   const float* rclip, int h0, float qk_thr, float rel_thr, const float* tau, float eps,
                                                   float kconv, const int* pos, int ring, __half* kc, __half* vc, float* hk, float* hv,
                                                   KV8 q8) {
    __shared__ float red[32], rsh[HD];
    const int b = blockIdx.x, i = threadIdx.x, t = blockIdx.y;
    const float tt = tau ? tau[t] : 1.0f;
    if (b < NH_C) {
        float* qr = q + (size_t) t * QC;
        float x = qr[b * HD + i] * qclip[(size_t) (h0 + b) * HD + i];
        const float ss = block_sum(x * x, red);
        const float f = fminf(fmaxf(qk_thr / (sqrtf(ss) + 1e-6f), 0.0f), 1.0f);
        float y = (x * f) * tt;
        if (q8.rot) y = fwht128(y, rsh);   // p18
        qr[b * HD + i] = y;
        return;
    }
    if (b < 2 * NH_C) {
        const int h = b - NH_C;
        float* rr = r + (size_t) t * NH_C * NREL;
        float x = 0.0f;
        if (i < NREL) x = rr[h * NREL + i] * rclip[(size_t) (h0 + h) * NREL + i];
        const float ss = block_sum(x * x, red);
        const float f = fminf(fmaxf(rel_thr / (sqrtf(ss) + 1e-6f), 0.0f), 1.0f);
        if (i < NREL) rr[h * NREL + i] = (x * f) * tt;
        return;
    }
    const bool isk = b < 2 * NH_C + NKV_C;
    const int h = isk ? b - 2 * NH_C : b - 2 * NH_C - NKV_C, c = h * HD + i;
    const float* x = isk ? k : v;
    const float* dyn = isk ? dk : dv;
    const float* st = isk ? sk : sv;
    const float* w = isk ? kw : vw;
    auto in = [&](int tt2) { return tt2 >= 0 ? x[(size_t) tt2 * KVC + c] : st[(size_t) (3 + tt2) * KVC + c]; };
    const float xt = x[(size_t) t * KVC + c], s0 = in(t - 3), s1 = in(t - 2), s2 = in(t - 1);
    float* hh = isk ? hk : hv;
    if (hh) hh[(size_t) t * KVC + c] = xt;
    float o = 0.0f;
    o += s0 * w[c * 4 + 0];
    o += s1 * w[c * 4 + 1];
    o += s2 * w[c * 4 + 2];
    o += xt * w[c * 4 + 3];
    if (dyn) {
        const float* dt = dyn + ((size_t) t * NKV_C + h) * NCONV;
        o = o + dt[0] * xt;
        o = o + dt[1] * s2;
        o = o + dt[2] * s1;
        o = o + dt[3] * s0;
    }
    float y = (xt + o) * kconv;
    const float ss = block_sum(y * y, red);
    if (isk) {
        const float f = fminf(fmaxf(qk_thr / (sqrtf(ss) + 1e-6f), 0.0f), 1.0f);
        y = (y * f) * 1.0f;
    } else {
        y = y * (1.0f / sqrtf(ss / HD + eps));
    }
    const size_t row = ring ? (size_t) (pos[t] % RING) : (size_t) pos[t];
    if (q8.rot) y = fwht128(y, rsh);   // p18
    kv_put(q8, isk, isk ? kc : vc, row, c, y);
}
__global__ void k_post_state(const float* k, const float* v, float* sk, float* sv, int T) {   // the K and V states after T rows
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * KVC) return;
    const bool isk = i < KVC;
    const int c = isk ? i : i - KVC;
    const float* x = isk ? k : v;
    float* st = isk ? sk : sv;
    float nv[3];
    for (int r = 0; r < 3; ++r) {
        const int tt = T - 3 + r;
        nv[r] = tt >= 0 ? x[(size_t) tt * KVC + c] : st[(size_t) (3 + tt) * KVC + c];
    }
    for (int r = 0; r < 3; ++r) st[(size_t) r * KVC + c] = nv[r];
}

// p52: k_xchg_conv with the exchange as single 8-byte words (value bits | sequence << 32) in page-locked RAM (INK2_XLL=1): the
// other card's word is polled until its sequence is this exchange's; no fence, no flag. The rest is k_xchg_conv's code
__global__ void __launch_bounds__(1024) k_xchg_conv_ll(const float* y, const float* ey, const float* topw, unsigned long long* out_ll,
                                                       const unsigned long long* in_ll, const unsigned* tok, unsigned idx, int card,
                                                       float* st, const float* w, float k, float* cur, float* bpush) {
    const unsigned seq = *tok * 128u + idx + 1u;
    const unsigned long long tag = (unsigned long long) seq << 32;
    float own[3], oth[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = threadIdx.x + 1024 * j;
        own[j] = 0.0f;
        if (i < N) {
            float v;
            if (ey) {
                v = 0.0f;
                for (int kk = 0; kk < TOPK; ++kk) v += ey[(size_t) kk * N + i] * topw[kk];
            } else {
                v = y[i];
            }
            own[j] = v;
            ((volatile unsigned long long*) out_ll)[i] = tag | __float_as_uint(v);
        }
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = threadIdx.x + 1024 * j;
        oth[j] = 0.0f;
        if (i < N) {
            unsigned long long wv;
            do {
                wv = ((const volatile unsigned long long*) in_ll)[i];
            } while ((unsigned) (wv >> 32) != seq);
            oth[j] = __uint_as_float((unsigned) wv);
        }
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = threadIdx.x + 1024 * j;
        if (i >= N) continue;
        const float a = card == 0 ? own[j] + oth[j] : oth[j] + own[j];
        const float s0 = st[i], s1 = st[N + i], s2 = st[2 * N + i];
        float o = 0.0f;
        o += s0 * w[i * 4 + 0];
        o += s1 * w[i * 4 + 1];
        o += s2 * w[i * 4 + 2];
        o += a * w[i * 4 + 3];
        const float r = (a + o) * k;
        st[i] = s1;
        st[N + i] = s2;
        st[2 * N + i] = a;
        if (bpush) {
            bpush[i] = cur[i];
            cur[i] = r;
        } else {
            cur[i] = cur[i] + r;
        }
    }
}
// k_xchg_conv for T rows: the exchange of T x N, then per channel the convolution over the T rows in order (state in registers), the
// residual (or the push) of each row; the exchanged rows go to hist (draft rollback) when given
__global__ void __launch_bounds__(1024) k_xchg_convT(const float* y, const float* ey, const float* topw, float* out, volatile unsigned* flag,
                                                     const volatile float* in, const volatile unsigned* oflag, const unsigned* tok,
                                                     unsigned idx, int card, float* st, const float* w, float k, float* cur, float* bpush,
                                                     float* hist, int T, float* scratch) {
    const unsigned seq = *tok * 128u + idx + 1u;
    auto own = [&](int e) {   // this card's part (recomputed instead of read back from page-locked RAM)
        if (!ey) return y[e];
        const int t = e / N, i = e % N;
        float v = 0.0f;
        for (int kk = 0; kk < TOPK; ++kk) v += ey[((size_t) t * TOPK + kk) * N + i] * topw[t * TOPK + kk];
        return v;
    };
    if (NC > 1) {   // p30: one card (NC = 1) has nothing to exchange
    for (int e = threadIdx.x; e < T * N; e += 1024) out[e] = own(e);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *flag = seq;
        while (*oflag < seq) {}
    }
    __syncthreads();
    __threadfence_system();
    // p6e: the other card's rows by float4 into scratch (the loads issued together), then the convolution reads them from L2
    for (int e4 = threadIdx.x; e4 < T * N / 4; e4 += 1024) {
        const float4 v4 = __ldcv((const float4*) (const float*) in + e4);
        ((float4*) scratch)[e4] = v4;
    }
    __syncthreads();
    }
    for (int i = threadIdx.x; i < N; i += 1024) {
        float s0 = st[i], s1 = st[N + i], s2 = st[2 * N + i];
        const float w0 = w[i * 4 + 0], w1 = w[i * 4 + 1], w2 = w[i * 4 + 2], w3 = w[i * 4 + 3];
        for (int t = 0; t < T; ++t) {
            const size_t e = (size_t) t * N + i;
            const float mine = own((int) e);
            const float a = NC == 1 ? mine : card == 0 ? mine + scratch[e] : scratch[e] + mine;
            if (hist) hist[e] = a;
            float o = 0.0f;
            o += s0 * w0;
            o += s1 * w1;
            o += s2 * w2;
            o += a * w3;
            const float rr = (a + o) * k;
            if (bpush) {
                bpush[e] = cur[e];
                cur[e] = rr;
            } else {
                cur[e] = cur[e] + rr;
            }
            s0 = s1;
            s1 = s2;
            s2 = a;
        }
        st[i] = s0;
        st[N + i] = s1;
        st[2 * N + i] = s2;
    }
}
// k_ctrl_down1 for T rows (grid y = t)
__global__ void __launch_bounds__(256) k_ctrl_downT(const float* __restrict__ W, const float* x, const float* m, float* g64) {
    __shared__ float red[32];
    const int r = blockIdx.x, t = blockIdx.y;
    x += (size_t) t * N;
    m += (size_t) t * N;
    float sx = 0, sm = 0, dx = 0, dm = 0;
    const float* w = W + (size_t) r * 2 * N;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        const float a = x[c], b = m[c];
        sx += a * a;
        sm += b * b;
        dx += w[c] * a;
        dm += w[N + c] * b;
    }
    sx = block_sum(sx, red);
    sm = block_sum(sm, red);
    dx = block_sum(dx, red);
    dm = block_sum(dm, red);
    if (threadIdx.x == 0) g64[(size_t) t * RANK + r] = dx * (1.0f / sqrtf(sx / N + 1e-6f)) + dm * (1.0f / sqrtf(sm / N + 1e-6f));
}


// p9: k_gemv8_t1T with T known at compile time: acc[G1_R][TT] instead of [MAXT] (p6e's convert-once spilled with 64 accumulators),
// the 16 weights of each row converted once per chunk and used by every row t. Same rows a block, same split S, the same products
// in the same order as k_gemv8_t1: the bits of single tokens
template <int TT>
__global__ void __launch_bounds__(256) k_gemv8_t1TT(const int8_t* __restrict__ q, const __half* __restrict__ d, int rows, int cols,
                                                     const float* __restrict__ x, int ldx, SegsT segs, int S) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rb = blockIdx.x * G1_BR, r0 = rb + warp * G1_R;
    const int nch = (cols + 511) / 512, s = blockIdx.y, ch0 = s * nch / S, ch1 = (s + 1) * nch / S;
    float acc[G1_R][TT];
#pragma unroll
    for (int a = 0; a < G1_R; ++a)
#pragma unroll
        for (int t = 0; t < TT; ++t) acc[a][t] = 0.0f;
    for (int ch = ch0; ch < ch1; ++ch) {
        const int c = ch * 512 + lane * 16;
        if (c >= cols) continue;
        float wf[G1_R][16], sc[G1_R];
#pragma unroll
        for (int a = 0; a < G1_R; ++a) {
            const int r = r0 + a;
            const int4 w4 = r < rows ? __ldg((const int4*) (q + (size_t) r * cols + c)) : make_int4(0, 0, 0, 0);
            sc[a] = r < rows ? __half2float(d[(size_t) r * (cols / 32) + c / 32]) : 0.0f;
            const int8_t* w = (const int8_t*) &w4;
#pragma unroll
            for (int j = 0; j < 16; ++j) wf[a][j] = (float) w[j];
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const float* xt = x + (size_t) t * ldx;
            const float4 x0 = __ldg((const float4*) (xt + c)), x1 = __ldg((const float4*) (xt + c + 4)), x2 = __ldg((const float4*) (xt + c + 8)),
                         x3 = __ldg((const float4*) (xt + c + 12));
#pragma unroll
            for (int a = 0; a < G1_R; ++a) {
                const float* w = wf[a];
                const float v = w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w +
                                w[8] * x2.x + w[9] * x2.y + w[10] * x2.z + w[11] * x2.w + w[12] * x3.x + w[13] * x3.y + w[14] * x3.z + w[15] * x3.w;
                acc[a][t] += sc[a] * v;
            }
        }
    }
    auto store = [&](int r, int t, float v) {
        int i = 0;
        while (i + 1 < segs.n && r >= segs.r0[i + 1]) ++i;
        segs.dst[i][(size_t) t * segs.ld[i] + (r - segs.r0[i])] = v;
    };
#pragma unroll
    for (int a = 0; a < G1_R; ++a) {
        const int r = r0 + a;
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const float v = warp_sum(acc[a][t]);
            if (lane == 0 && r < rows) {
                if (S == 1) store(r, t, v);
                else g_part[((size_t) s * TT + t) * rows + r] = v;
            }
        }
    }
    if (S == 1) return;
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&g_g1_cnt[blockIdx.x], 1u) == (unsigned) S - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int e = threadIdx.x; e < G1_BR * TT; e += blockDim.x) {
        const int r = rb + e % G1_BR, t = e / G1_BR;
        if (r >= rows) continue;
        float v = 0;
        for (int kk = 0; kk < S; ++kk) v += ((volatile float*) g_part)[((size_t) kk * TT + t) * rows + r];
        store(r, t, v);
    }
    if (threadIdx.x == 0) g_g1_cnt[blockIdx.x] = 0;
}

// p6c: k_gemv8_t1 for T rows: every row's sum exactly as k_gemv8_t1 makes it (the same column parts S, the same lanes, the same order),
// so a window gives the bits of single tokens; outputs by segments with a row length each ([T][ld])
__global__ void __launch_bounds__(256) k_gemv8_t1T(const int8_t* __restrict__ q, const __half* __restrict__ d, int rows, int cols,
                                                    const float* __restrict__ x, int ldx, SegsT segs, int S, int T) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rb = blockIdx.x * G1_BR, r0 = rb + warp * G1_R;
    const int nch = (cols + 511) / 512, s = blockIdx.y, ch0 = s * nch / S, ch1 = (s + 1) * nch / S;
    float acc[G1_R][MAXT];
#pragma unroll
    for (int a = 0; a < G1_R; ++a)
#pragma unroll
        for (int t = 0; t < MAXT; ++t) acc[a][t] = 0.0f;
    for (int ch = ch0; ch < ch1; ++ch) {
        const int c = ch * 512 + lane * 16;
        if (c >= cols) continue;
        int4 w4[G1_R];
        float sc[G1_R];
#pragma unroll
        for (int a = 0; a < G1_R; ++a) {
            const int r = r0 + a;
            w4[a] = r < rows ? __ldg((const int4*) (q + (size_t) r * cols + c)) : make_int4(0, 0, 0, 0);
            sc[a] = r < rows ? __half2float(d[(size_t) r * (cols / 32) + c / 32]) : 0.0f;
        }
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {   // unrolled with a break: acc[a][t] stays in registers
            if (t >= T) break;
            const float* xt = x + (size_t) t * ldx;
            const float4 x0 = __ldg((const float4*) (xt + c)), x1 = __ldg((const float4*) (xt + c + 4)), x2 = __ldg((const float4*) (xt + c + 8)),
                         x3 = __ldg((const float4*) (xt + c + 12));
#pragma unroll
            for (int a = 0; a < G1_R; ++a) {
                const int8_t* w = (const int8_t*) &w4[a];
                const float v = w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w +
                                w[8] * x2.x + w[9] * x2.y + w[10] * x2.z + w[11] * x2.w + w[12] * x3.x + w[13] * x3.y + w[14] * x3.z + w[15] * x3.w;
                acc[a][t] += sc[a] * v;
            }
        }
    }
    auto store = [&](int r, int t, float v) {
        int i = 0;
        while (i + 1 < segs.n && r >= segs.r0[i + 1]) ++i;
        segs.dst[i][(size_t) t * segs.ld[i] + (r - segs.r0[i])] = v;
    };
#pragma unroll
    for (int a = 0; a < G1_R; ++a) {
        const int r = r0 + a;
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= T) break;
            const float v = warp_sum(acc[a][t]);
            if (lane == 0 && r < rows) {
                if (S == 1) store(r, t, v);
                else g_part[((size_t) s * T + t) * rows + r] = v;
            }
        }
    }
    if (S == 1) return;
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&g_g1_cnt[blockIdx.x], 1u) == (unsigned) S - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int e = threadIdx.x; e < G1_BR * T; e += blockDim.x) {
        const int r = rb + e % G1_BR, t = e / G1_BR;
        if (r >= rows) continue;
        float v = 0;
        for (int kk = 0; kk < S; ++kk) v += ((volatile float*) g_part)[((size_t) kk * T + t) * rows + r];
        store(r, t, v);
    }
    if (threadIdx.x == 0) g_g1_cnt[blockIdx.x] = 0;
}

}  // namespace ink2
