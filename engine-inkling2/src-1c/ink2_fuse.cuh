// ink2_fuse.cuh (p22, 2026-11-22): fewer kernel launches a token on the writing path, the same bits.
// A token is 48 passes of ~16 (plain) or ~29 (looped) small launches, back to back (the profile of 21. 11.). These kernels are
// several launches of one kernel made one (the grid's y or z says which), with every block doing exactly what it did before:
// the same arithmetic in the same order, so INK2_FUSE2=0 (p21's launches) and any other mask give the same logits bit for bit.
//   bit 0  k_mix2x          a looped layer's two depth mixes (the layer's queries, the loop messages)
//   bit 1  k_ctrl_down1x/Tx the three loop controllers' down projections (q, k, v); k_ctrl_gfx their gains and forgets
//   bit 2  k_rms3           the three inputs' rms norms; k_gemv8_p1/pT their Q8S matrices (each part keeps k_gemv8_t1's split)
//   bit 3  k_route2         the router's rows with the routing by the last block to finish (k_route_jobs's sort)
// p22a (the norms computed inside every block of the matrices' and the router's launch) gave the same bits but was slower:
// single tokens +1.5 %, windows -8 to -13 % (the matrices' blocks are bound by their instructions, not by the launch): not kept.
#pragma once

namespace ink2 {

// ---------------------------------------------------------------------------------------------------------------- depth mixes
__global__ void __launch_bounds__(768) k_mix2x(States S0, const float* q0, int nq0, float* out0, States S1, const float* q1, int nq1, float* out1) {
    const bool second = blockIdx.y != 0;
    const States S = second ? S1 : S0;
    const float* q = second ? q1 : q0;
    const int nq = second ? nq1 : nq0;
    float* out = second ? out1 : out0;
    // k_mix2's body from here on
    constexpr int NS = MAXB + 2, NR = NS * 5;
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

// ------------------------------------------------------------------------------------------------------------ loop controllers
struct Ctl3 {
    const float *W[3], *Wg[3], *Wf[3];   // down [RANK][2N], gain and forget [N][RANK]
    const float *x[3], *m[3];            // [T][N] each
    float *g[3], *y[3];                  // down's output [T][RANK], the controlled x [T][N]
};
// k_ctrl_down1 for the three controllers: grid (RANK, 3)
__global__ void __launch_bounds__(256) k_ctrl_down1x(Ctl3 A) {
    __shared__ float red[32];
    const int r = blockIdx.x, j = blockIdx.y;
    const float *x = A.x[j], *m = A.m[j];
    float sx = 0, sm = 0, dx = 0, dm = 0;
    const float* w = A.W[j] + (size_t) r * 2 * N;
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
    if (threadIdx.x == 0) A.g[j][r] = dx * (1.0f / sqrtf(sx / N + 1e-6f)) + dm * (1.0f / sqrtf(sm / N + 1e-6f));
}
// k_ctrl_downT for the three controllers: grid (RANK, T, 3)
__global__ void __launch_bounds__(256) k_ctrl_downTx(Ctl3 A) {
    __shared__ float red[32];
    const int r = blockIdx.x, t = blockIdx.y, j = blockIdx.z;
    const float *x = A.x[j] + (size_t) t * N, *m = A.m[j] + (size_t) t * N;
    float sx = 0, sm = 0, dx = 0, dm = 0;
    const float* w = A.W[j] + (size_t) r * 2 * N;
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
    if (threadIdx.x == 0) A.g[j][(size_t) t * RANK + r] = dx * (1.0f / sqrtf(sx / N + 1e-6f)) + dm * (1.0f / sqrtf(sm / N + 1e-6f));
}
// k_ctrl_gf for the three controllers: grid (N / 8, 3)
__global__ void __launch_bounds__(256) k_ctrl_gfx(Ctl3 A, int T) {
    const int c = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31, j = blockIdx.y;
    if (c >= N) return;
    const float *Wg = A.Wg[j], *Wf = A.Wf[j], *g64 = A.g[j], *x = A.x[j], *m = A.m[j];
    float* y = A.y[j];
    const float2 wg = *(const float2*) (Wg + (size_t) c * RANK + lane * 2), wf = *(const float2*) (Wf + (size_t) c * RANK + lane * 2);
    for (int t = 0; t < T; ++t) {
        const float2 gv = *(const float2*) (g64 + (size_t) t * RANK + lane * 2);
        const float gp = warp_sum(wg.x * gv.x + wg.y * gv.y), fp = warp_sum(wf.x * gv.x + wf.y * gv.y);
        if (lane == 0) {
            const float xv = x[(size_t) t * N + c];
            y[(size_t) t * N + c] = (xv - tanhf(fp) * xv) + tanhf(gp) * m[(size_t) t * N + c];
        }
    }
}

// ---------------------------------------------------------------------------------------------------- the three rms norms
// k_rms for three inputs: grid (T, 3)
__global__ void k_rms3(const float* x0, const float* x1, const float* x2, const float* w0, const float* w1, const float* w2, float* y0, float* y1,
                       float* y2, float eps) {
    __shared__ float red[32];
    const int j = blockIdx.y;
    const float* x = j == 0 ? x0 : j == 1 ? x1 : x2;
    const float* w = j == 0 ? w0 : j == 1 ? w1 : w2;
    float* y = j == 0 ? y0 : j == 1 ? y1 : y2;
    const float* xr = x + (size_t) blockIdx.x * N;
    float ss = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    const float k = 1.0f / sqrtf(ss / N + eps);
    float* yr = y + (size_t) blockIdx.x * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) yr[i] = w ? xr[i] * k * w[i] : xr[i] * k;
}

// --------------------------------------------------------------------------------------- Q8S matrices of up to three inputs
// A part: a (view of a) matrix with N columns and its input. Blocks b0 .. b0 + blocks * S - 1 of the launch are the part's (row
// block bx = local % blocks, column part s = local / blocks: k_gemv8_t1's grid (blocks, S)), with their own place in the buffers
// of the column parts. Every block runs k_gemv8_t1's (k_gemv8_t1TT's) code on its part.
constexpr int GM_PART = 3 << 17;
__device__ float g_partx[GM_PART];
__device__ unsigned g_cntx[4096];
struct GPart {
    const int8_t* q;
    const __half* d;
    const float* x;
    int rows, blocks, S, b0, p0, c0;   // p0: its place in g_partx, c0: in g_cntx
};
struct GParts1 {
    int n;
    GPart p[3];
    Segs sg[3];
};
struct GPartsT {
    int n;
    GPart p[3];
    SegsT sg[3];
};
__global__ void __launch_bounds__(256) k_gemv8_p1(GParts1 A) {
    __shared__ bool last;
    int pi = 0;
    while (pi + 1 < A.n && (int) blockIdx.x >= A.p[pi + 1].b0) ++pi;
    const GPart P = A.p[pi];
    const int local = blockIdx.x - P.b0, bx = local % P.blocks, s = local / P.blocks, S = P.S, rows = P.rows;
    const int8_t* __restrict__ q = P.q;
    const __half* __restrict__ d = P.d;
    const float* __restrict__ x = P.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rb = bx * G1_BR, r0 = rb + warp * G1_R;
    const int cols = N, nch = (cols + 511) / 512, ch0 = s * nch / S, ch1 = (s + 1) * nch / S;
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
    const Segs& segs = A.sg[pi];
#pragma unroll
    for (int a = 0; a < G1_R; ++a) {
        const float v = warp_sum(acc[a]);
        const int r = r0 + a;
        if (lane == 0 && r < rows) {
            if (S == 1) seg_store(segs, r, v);
            else g_partx[P.p0 + (size_t) s * rows + r] = v;
        }
    }
    if (S == 1) return;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&g_cntx[P.c0 + bx], 1u) == (unsigned) S - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < G1_BR && rb + (int) threadIdx.x < rows) {
        const int r = rb + threadIdx.x;
        float v = 0;
        for (int k = 0; k < S; ++k) v += ((volatile float*) g_partx)[P.p0 + (size_t) k * rows + r];
        seg_store(segs, r, v);
    }
    if (threadIdx.x == 0) g_cntx[P.c0 + bx] = 0;
}
template <int TT>
__global__ void __launch_bounds__(256) k_gemv8_pT(GPartsT A) {
    __shared__ bool last;
    int pi = 0;
    while (pi + 1 < A.n && (int) blockIdx.x >= A.p[pi + 1].b0) ++pi;
    const GPart P = A.p[pi];
    const int local = blockIdx.x - P.b0, bx = local % P.blocks, s = local / P.blocks, S = P.S, rows = P.rows;
    const int8_t* __restrict__ q = P.q;
    const __half* __restrict__ d = P.d;
    const float* __restrict__ x = P.x;
    const int ldx = N;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rb = bx * G1_BR, r0 = rb + warp * G1_R;
    const int cols = N, nch = (cols + 511) / 512, ch0 = s * nch / S, ch1 = (s + 1) * nch / S;
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
    const SegsT& segs = A.sg[pi];
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
                else g_partx[P.p0 + ((size_t) s * TT + t) * rows + r] = v;
            }
        }
    }
    if (S == 1) return;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&g_cntx[P.c0 + bx], 1u) == (unsigned) S - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int e = threadIdx.x; e < G1_BR * TT; e += blockDim.x) {
        const int r = rb + e % G1_BR, t = e / G1_BR;
        if (r >= rows) continue;
        float v = 0;
        for (int k = 0; k < S; ++k) v += ((volatile float*) g_partx)[P.p0 + ((size_t) k * TT + t) * rows + r];
        store(r, t, v);
    }
    if (threadIdx.x == 0) g_cntx[P.c0 + bx] = 0;
}

// ------------------------------------------------------------------------------------------------------ the router's rows
// k_gemv32_rows on the router's matrix (a block per row), then the last block to finish (a counter, as the column parts above)
// sorts every row's 128 probabilities as k_route_jobs does (its threads 0 .. 127, the same compare-exchanges) and writes topi,
// topw and the jobs
__device__ unsigned g_route_cnt;
__global__ void __launch_bounds__(256) k_route2(const float* __restrict__ W, const float* x, float* y, int T, const float* bias, float wscale,
                                                int* topi, float* topw, EJob* jobs) {
    __shared__ float red[32];
    __shared__ float sv[NE], pr[NE];
    __shared__ int si[NE];
    __shared__ bool last;
    const int r = blockIdx.x, cols = N, ldx = N;
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
        if (threadIdx.x == 0) y[(size_t) t * NE + r] = v;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&g_route_cnt, 1u) == (unsigned) gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int e = threadIdx.x;
    for (int t = 0; t < T; ++t) {
        if (e < NE) {
            const float p = sigm(((volatile float*) y)[(size_t) t * NE + e]);
            pr[e] = p;
            sv[e] = p + (bias ? bias[e] : 0.0f);
            si[e] = e;
        }
        __syncthreads();
        for (int k2 = 2; k2 <= NE; k2 <<= 1)
            for (int j = k2 >> 1; j > 0; j >>= 1) {
                const int ix = e ^ j;
                if (e < NE && ix > e) {
                    const bool up = (e & k2) == 0;
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
        __syncthreads();
    }
    if (threadIdx.x == 0) g_route_cnt = 0;
}

}  // namespace ink2
