// ink2_dspark.cuh (2026-11-21, p6): Inkling 2's DSpark drafter ("MTP", thinkingmachines mtp.safetensors / Unsloth
// dspark-Inkling-2-Mini-Q8_0.gguf), after llama.cpp PR #32218 (src/models/dspark.cpp graph_inkling2, common/speculative.cpp).
// The drafter runs whole on both cards (the same inputs give the same bits, no exchanges); only the target's output head is split by
// vocabulary. A wrong draft only costs acceptance: every draft is checked by the target.
//  - context: per committed position p the target's features F[4] (the input of layer visits 5, 23, 41 = after slots 4, 28, 46, and
//    the final normed hidden); drafter layer il keeps K = rope(rms_head(k_ctx . c_il) * k_ctx_norm), V = v_ctx . c_il with
//    c_il = rms(sum_j softmax(fuse[il])_j F_j) * dflare_norm[il]; f_rnn = rms(sum_j softmax(fuse[3])_j F_j) * dspark_norm
//  - block: [rms(tok_embd[anchor]) * tok_norm, mask_embd x 7] at positions P..P+7 through 3 layers (q/k head norms, NeoX RoPE
//    base 1e6, attention over the cache [lo, P+7], no causal mask inside the block, layers 0 and 2 a window of 1024), final rms
//  - Markov head, position i = 1..k: joint = [state | w1[prev] | hid_i | f_rnn(P-1)] -> joint_proj (+b) -> gate, candidate, latent;
//    state = cand + gate (state - cand); y = hid_i + hidden_out . latent (+b); draft_i = argmax(output . y) (softcap is monotonic)
#pragma once
#include "ink2_kern.cuh"

namespace ink2 {

constexpr int DSL = 3, DFF = 9216, DRANK = 256, DBLK = 8, DJ = 2 * DRANK + 2 * N;   // layers, FFN, Markov rank, block, joint width

__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

// c = rms(sum_j softmax(fuse_row)_j F[t][j]) * w: a block per row (F rows of 4 x N)
__global__ void k_ds_mixfeat(const float* F, const float* fuse_row, const float* w, float eps, float* out) {
    __shared__ float red[32];
    __shared__ float sw[4];
    const int t = blockIdx.x;
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int j = 0; j < 4; ++j) m = fmaxf(m, fuse_row[j]);
        float s = 0;
        for (int j = 0; j < 4; ++j) { sw[j] = expf(fuse_row[j] - m); s += sw[j]; }
        for (int j = 0; j < 4; ++j) sw[j] /= s;
    }
    __syncthreads();
    const float* f = F + (size_t) t * 4 * N;
    float v[N / 256], ss = 0;
#pragma unroll
    for (int k = 0; k < N / 256; ++k) {
        const int i = threadIdx.x + 256 * k;
        const float a = sw[0] * f[i] + sw[1] * f[N + i] + sw[2] * f[2 * N + i] + sw[3] * f[3 * N + i];
        v[k] = a;
        ss += a * a;
    }
    ss = block_sum(ss, red);
    const float kk = 1.0f / sqrtf(ss / N + eps);
#pragma unroll
    for (int k = 0; k < N / 256; ++k) {
        const int i = threadIdx.x + 256 * k;
        out[(size_t) t * N + i] = v[k] * kk * w[i];
    }
}
// per head (HD values): optional rms * w, then NeoX RoPE at pos[t] (pairs i, i + 64), stored f32 in place or f16 into a cache row
__global__ void __launch_bounds__(128) k_ds_headrope(float* x, int H, const float* w, float eps, const int* pos, int pos_add, float base,
                                                     __half* cache, int ldc, int ring = 0) {   // p14: ring = cache rows by position % ring
    __shared__ float red[32];
    __shared__ float hv[HD];
    const int t = blockIdx.y, h = blockIdx.x, i = threadIdx.x;
    float* xr = x + ((size_t) t * H + h) * HD;
    float a = xr[i];
    if (w) {
        const float ss = block_sum(a * a, red);
        a = a * (1.0f / sqrtf(ss / HD + eps)) * w[i];
    }
    hv[i] = a;
    __syncthreads();
    const float p = (float) (pos[t] + pos_add);
    const int j = i & 63;
    const float th = p * powf(base, -2.0f * j / HD);
    float c, s;
    sincosf(th, &s, &c);
    const float o = i < 64 ? hv[i] * c - hv[i + 64] * s : hv[i - 64] * s + hv[i] * c;
    if (cache) cache[(size_t) (ring ? (pos[t] + pos_add) % ring : pos[t] + pos_add) * ldc + h * HD + i] = __float2half(o);
    else xr[i] = o;
}
__global__ void k_ds_vstore(const float* v, const int* pos, int pos_add, __half* cache, int T, int ring = 0) {   // V rows (NKVD) into the cache
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * NKVD) return;
    const int t = i / NKVD, c = i % NKVD;
    cache[(size_t) (ring ? (pos[t] + pos_add) % ring : pos[t] + pos_add) * NKVD + c] = __float2half(v[i]);
}
// the block's attention: query t (of DBLK) of head h over cache rows [lo, hi) of KV head h / 3; parts of KC2 keys, then k_ds_comb
__global__ void __launch_bounds__(128) k_ds_attn(const float* q, const __half* kc, const __half* vc, int lo, int hi, float* part, int nparts,
                                                 int ring = 0) {
    __shared__ float qs[HD], sc[KC2];
    __shared__ float red[32];
    const int p = blockIdx.x, h = blockIdx.y, t = blockIdx.z, tid = threadIdx.x, g = h / (NH / NKV);
    const int j0 = lo + p * KC2, j1 = min(hi, j0 + KC2);
    float* o = part + (((size_t) t * NH + h) * nparts + p) * (HD + 2);
    if (j0 >= hi) {
        if (tid == 0) { o[0] = -INFINITY; o[1] = 0.0f; }
        o[2 + tid] = 0.0f;
        return;
    }
    qs[tid] = q[((size_t) t * NH + h) * HD + tid];
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31;
    for (int j = j0 + warp; j < j1; j += 4) {
        const __half* kr = kc + (size_t) (ring ? j % ring : j) * NKVD + g * HD;
        const __half2 k01 = *(const __half2*) (kr + lane * 4), k23 = *(const __half2*) (kr + lane * 4 + 2);
        float d = __low2float(k01) * qs[lane * 4] + __high2float(k01) * qs[lane * 4 + 1] + __low2float(k23) * qs[lane * 4 + 2] +
                  __high2float(k23) * qs[lane * 4 + 3];
        d = warp_sum(d);
        if (lane == 0) sc[j - j0] = d * (1.0f / sqrtf((float) HD));
    }
    __syncthreads();
    const int n = j1 - j0;
    float m = -INFINITY;
    for (int j = tid; j < n; j += 128) m = fmaxf(m, sc[j]);
    m = block_max(m, red);
    float l = 0;
    for (int j = tid; j < n; j += 128) {
        const float e = expf(sc[j] - m);
        sc[j] = e;
        l += e;
    }
    l = block_sum(l, red);
    float acc = 0;
    for (int j = 0; j < n; ++j) acc += sc[j] * __half2float(vc[(size_t) (ring ? (j0 + j) % ring : j0 + j) * NKVD + g * HD + tid]);
    if (tid == 0) { o[0] = m; o[1] = l; }
    o[2 + tid] = acc;
}
__global__ void __launch_bounds__(128) k_ds_comb(const float* part, int nparts, float* out) {   // no sink, no gate
    const int t = blockIdx.y, h = blockIdx.x;
    const float* p = part + ((size_t) t * NH + h) * nparts * (HD + 2);
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
    out[((size_t) t * NH + h) * HD + threadIdx.x] = acc / L;
}
__global__ void k_ds_swiglu(const float* g, const float* u, float* h, int n) {   // h = silu(g) * u
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}
// BF16 rows (the Markov head's matrices): y[r] = W[r] . x (+ b), a warp per row
__global__ void __launch_bounds__(256) k_gemv_bf16(const uint16_t* __restrict__ W, int rows, int cols, const float* x, const float* b, float* y) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= rows) return;
    const uint16_t* w = W + (size_t) r * cols;
    float a = 0;
    for (int c = lane * 8; c < cols; c += 256) {
        const uint4 v = __ldg((const uint4*) (w + c));
        const uint16_t* h = (const uint16_t*) &v;
#pragma unroll
        for (int j = 0; j < 8; ++j) a += bf2f(h[j]) * x[c + j];
    }
    a = warp_sum(a);
    if (lane == 0) y[r] = b ? a + b[r] : a;
}
// the joint input [state | w1[prev] | hid | ctx] (DJ floats)
__global__ void k_ds_joint_in(const float* state, const uint16_t* w1, const int* prev, const float* hid, const float* ctx, float* jin) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DJ) return;
    float v;
    if (i < DRANK) v = state[i];
    else if (i < 2 * DRANK) v = bf2f(w1[(size_t) prev[0] * DRANK + (i - DRANK)]);
    else if (i < 2 * DRANK + N) v = hid[i - 2 * DRANK];
    else v = ctx[i - 2 * DRANK - N];
    jin[i] = v;
}
__global__ void k_ds_gates(const float* j, float* state, float* latent) {   // j = [gate | candidate | latent] pre-activations
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DRANK) return;
    const float g = sigm(j[i]), cand = tanhf(j[DRANK + i]);
    state[i] = cand + g * (state[i] - cand);
    latent[i] = tanhf(j[2 * DRANK + i]);
}
// argmax of n logits (the first maximum), one block: (value, index) to out
// p12: with outs, also sum_i exp(sc(l_i) - sc(max)) (sc = the softcap): the host gets the drafted token's probability from both cards
__global__ void __launch_bounds__(1024) k_argmax_n(const float* l, int n, float* outv, int* outi, float* outs = nullptr, float cap = 30.0f) {
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float ss[32];
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float v = l[i];
        if (v > bv) { bv = v; bi = i; }
    }
    for (int o = 16; o; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, bv, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, bi, o);
        if (v2 > bv || (v2 == bv && i2 < bi)) { bv = v2; bi = i2; }
    }
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) { sv[w] = bv; si[w] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < 32; ++k)
            if (sv[k] > sv[0] || (sv[k] == sv[0] && si[k] < si[0])) { sv[0] = sv[k]; si[0] = si[k]; }
        *outv = sv[0];
        *outi = si[0];
    }
    if (!outs) return;
    __syncthreads();
    const float m = cap * tanhf(sv[0] / cap);
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += 1024) acc += expf(cap * tanhf(l[i] / cap) - m);
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) ss[w] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int k = 0; k < 32; ++k) t += ss[k];
        *outs = t;
    }
}

}  // namespace ink2
