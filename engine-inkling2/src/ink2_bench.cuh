// ink2_bench.cuh (2026-11-21, p4): candidate writing kernels, measured by bench_dec.cu before they go into the engine
#pragma once
#include "ink2_kern.cuh"

#include <algorithm>
#include <cstring>
#include <vector>

namespace ink2 {

// k_fp4_rows4: LPR lanes per row (gate_up 4, down 1), each lane 9 whole 36-byte blocks of its row (lanes of a row on neighbouring
// blocks), the row's sum by shuffles; the input every 65 floats a block (no bank conflicts), the 256 UE4M3 scales from a table
template <int NBLK, int LPR>
__global__ void __launch_bounds__(128) k_fp4_rows4(const uint8_t* __restrict__ W, size_t estride, int rows, const float* x, int ldx,
                                                   int xbyjob, const float* escale, const EJob* jobs, float* y, int ldy) {
    constexpr int COLS = NBLK * 64, BPL = NBLK / LPR, RPW = 32 / LPR;
    static_assert(NBLK % LPR == 0 && 32 % LPR == 0, "lanes per row");
    __shared__ float xs[NBLK * 65];
    __shared__ float tab[16], sct[256];
    const EJob jb = jobs[blockIdx.y];
    const float* xr = x + (size_t) (xbyjob ? blockIdx.y : jb.t) * ldx;
    for (int i = threadIdx.x; i < COLS; i += blockDim.x) xs[(i >> 6) * 65 + (i & 63)] = xr[i];
    if (threadIdx.x < 16) tab[threadIdx.x] = c_fp4[threadIdx.x];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) sct[i] = ue4m3((uint8_t) i);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (blockIdx.x * (blockDim.x >> 5) + warp) * RPW + lane / LPR;
    if (row >= rows) return;
    const uint8_t* rp = W + (size_t) jb.e * estride + (size_t) row * NBLK * 36;
    float acc = 0;
#pragma unroll 3
    for (int k = 0; k < BPL; ++k) {
        const int b = (lane % LPR) + LPR * k;
        const uint32_t* bw = (const uint32_t*) (rp + (size_t) b * 36);
        uint32_t w[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) w[i] = __ldg(bw + i);
        const float* xv = xs + b * 65;
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            const float d = sct[(w[0] >> (8 * sb)) & 0xFFu];
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
    }
#pragma unroll
    for (int o = LPR / 2; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane % LPR == 0) y[(size_t) blockIdx.y * ldy + row] = escale ? acc * escale[jb.e] : acc;
}


// k_fp4_rows5: rows4 with a byte -> (low, high) value pair table (one shared load a code byte) and the input read as float4 (values
// i..i+3 of a sub-block's low and high halves are contiguous); a block every 68 floats keeps the float4 reads conflict-free
template <int NBLK, int LPR>
__global__ void __launch_bounds__(128) k_fp4_rows5(const uint8_t* __restrict__ W, size_t estride, int rows, const float* x, int ldx,
                                                   int xbyjob, const float* escale, const EJob* jobs, float* y, int ldy) {
    constexpr int COLS = NBLK * 64, BPL = NBLK / LPR, RPW = 32 / LPR, XP = 68;
    __shared__ __align__(16) float xs[NBLK * XP];
    __shared__ float2 pair[256];
    __shared__ float sct[256];
    const EJob jb = jobs[blockIdx.y];
    const float* xr = x + (size_t) (xbyjob ? blockIdx.y : jb.t) * ldx;
    for (int i = threadIdx.x; i < COLS; i += blockDim.x) xs[(i >> 6) * XP + (i & 63)] = xr[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        pair[i] = make_float2(c_fp4[i & 15], c_fp4[i >> 4]);
        sct[i] = ue4m3((uint8_t) i);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (blockIdx.x * (blockDim.x >> 5) + warp) * RPW + lane / LPR;
    if (row >= rows) return;
    const uint8_t* rp = W + (size_t) jb.e * estride + (size_t) row * NBLK * 36;
    float acc = 0;
#pragma unroll 3
    for (int k = 0; k < BPL; ++k) {
        const int b = (lane % LPR) + LPR * k;
        const uint32_t* bw = (const uint32_t*) (rp + (size_t) b * 36);
        uint32_t w[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) w[i] = __ldg(bw + i);
        const float* xv = xs + b * XP;
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            const float d = sct[(w[0] >> (8 * sb)) & 0xFFu];
            float a = 0;
#pragma unroll
            for (int hw = 0; hw < 2; ++hw) {
                const uint32_t q = w[1 + sb * 2 + hw];
                const float4 lo = *(const float4*) (xv + sb * 16 + hw * 4), hi = *(const float4*) (xv + sb * 16 + 8 + hw * 4);
                const float2 p0 = pair[q & 0xFFu], p1 = pair[(q >> 8) & 0xFFu], p2 = pair[(q >> 16) & 0xFFu], p3 = pair[q >> 24];
                a += p0.x * lo.x + p0.y * hi.x;
                a += p1.x * lo.y + p1.y * hi.y;
                a += p2.x * lo.z + p2.y * hi.z;
                a += p3.x * lo.w + p3.y * hi.w;
            }
            acc += a * d;
        }
    }
#pragma unroll
    for (int o = LPR / 2; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane % LPR == 0) y[(size_t) blockIdx.y * ldy + row] = escale ? acc * escale[jb.e] : acc;
}


// k_gemv8_t1b: Q8S at T = 1 with R rows a warp and all NCH chunks of 512 columns loaded before the products (more loads in flight)
template <int NCH, int R>
__global__ void __launch_bounds__(256) k_gemv8_t1b(const int8_t* __restrict__ q, const __half* __restrict__ d, int rows, int cols,
                                                    const float* __restrict__ x, Segs segs) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r0 = (blockIdx.x * 8 + warp) * R;
    int4 wv[R][NCH];
    float sc[R][NCH];
#pragma unroll
    for (int a = 0; a < R; ++a)
#pragma unroll
        for (int ch = 0; ch < NCH; ++ch) {
            const int c = ch * 512 + lane * 16, r = r0 + a;
            if (r < rows && c < cols) {
                wv[a][ch] = __ldg((const int4*) (q + (size_t) r * cols + c));
                sc[a][ch] = __half2float(d[(size_t) r * (cols / 32) + c / 32]);
            } else {
                wv[a][ch] = make_int4(0, 0, 0, 0);
                sc[a][ch] = 0.0f;
            }
        }
    float acc[R] = {};
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) {
        const int c = ch * 512 + lane * 16;
        if (c >= cols) continue;
        const float4 x0 = __ldg((const float4*) (x + c)), x1 = __ldg((const float4*) (x + c + 4)), x2 = __ldg((const float4*) (x + c + 8)),
                     x3 = __ldg((const float4*) (x + c + 12));
#pragma unroll
        for (int a = 0; a < R; ++a) {
            const int8_t* w = (const int8_t*) &wv[a][ch];
            const float v = w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w +
                            w[8] * x2.x + w[9] * x2.y + w[10] * x2.z + w[11] * x2.w + w[12] * x3.x + w[13] * x3.y + w[14] * x3.z + w[15] * x3.w;
            acc[a] += sc[a][ch] * v;
        }
    }
#pragma unroll
    for (int a = 0; a < R; ++a) {
        const float v = warp_sum(acc[a]);
        if (lane == 0 && r0 + a < rows) seg_store(segs, r0 + a, v);
    }
}

template <class Report, class Time>
static void bench_more(const uint8_t* gu, const uint8_t* dn, size_t gu_e, size_t dn_e, const float* x, const float* h, float* y, float* y2,
                       const float* esc, const EJob* jobs, int nj, Report report, Time timeit) {
    report("rows4 (LPR 4 / 1)",
           timeit([&] { k_fp4_rows4<36, 4><<<dim3(2 * FF_C / 32, nj), 128>>>(gu, gu_e, 2 * FF_C, x, N, 0, esc, jobs, y2, 2 * FF_C); }),
           timeit([&] { k_fp4_rows4<9, 1><<<dim3(N / 128, nj), 128>>>(dn, dn_e, N, h, FF_C, 1, esc, jobs, y2, N); }));
    report("rows5 (pairs, float4 x)",
           timeit([&] { k_fp4_rows5<36, 4><<<dim3(2 * FF_C / 32, nj), 128>>>(gu, gu_e, 2 * FF_C, x, N, 0, esc, jobs, y2, 2 * FF_C); }),
           timeit([&] { k_fp4_rows5<9, 1><<<dim3(N / 128, nj), 128>>>(dn, dn_e, N, h, FF_C, 1, esc, jobs, y2, N); }));
    report("rows5 LPR 2 / down",
           timeit([&] { k_fp4_rows5<36, 2><<<dim3(2 * FF_C / 64, nj), 128>>>(gu, gu_e, 2 * FF_C, x, N, 0, esc, jobs, y2, 2 * FF_C); }),
           timeit([&] { k_fp4_rows5<9, 1><<<dim3(N / 128, nj), 128>>>(dn, dn_e, N, h, FF_C, 1, esc, jobs, y2, N); }));
    {
        std::vector<float> a((size_t) nj * 2 * FF_C), b((size_t) nj * 2 * FF_C);
        k_fp4_rows3<36, 8><<<dim3(2 * FF_C / 32, nj), 128>>>(gu, gu_e, 2 * FF_C, x, N, 0, esc, jobs, y, 2 * FF_C);
        k_fp4_rows5<36, 4><<<dim3(2 * FF_C / 32, nj), 128>>>(gu, gu_e, 2 * FF_C, x, N, 0, esc, jobs, y2, 2 * FF_C);
        CK(cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * (double) (a[i] - b[i]); den += (double) a[i] * a[i]; }
        std::printf("rows5 proti rows3 (gate_up): relativna razlika %.2e\n", std::sqrt(num / (den + 1e-30)));
    }
    {   // planar copies of the same random weights
        std::vector<uint8_t> hg(gu_e * NE), hd(dn_e * NE), pg(gu_e * NE), pd(dn_e * NE);
        CK(cudaMemcpy(hg.data(), gu, hg.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hd.data(), dn, hd.size(), cudaMemcpyDeviceToHost));
        fp4_to_planar(hg.data(), pg.data(), (size_t) 2 * FF_C, N / 64, NE);
        fp4_to_planar(hd.data(), pd.data(), (size_t) N, FF_C / 64, NE);
        uint8_t *gp, *dp;
        CK(cudaMalloc(&gp, pg.size()));
        CK(cudaMalloc(&dp, pd.size()));
        CK(cudaMemcpy(gp, pg.data(), pg.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dp, pd.data(), pd.size(), cudaMemcpyHostToDevice));
        report("rows6 (planar, rows3 tiles)",
               timeit([&] { k_fp4_rows6<36, 8><<<dim3(2 * FF_C / 32, nj), 128>>>(gp, gu_e, 2 * FF_C, x, N, 0, esc, jobs, y2, 2 * FF_C); }),
               timeit([&] { k_fp4_rows6<9, 32><<<dim3(N / 128, nj), 128>>>(dp, dn_e, N, h, FF_C, 1, esc, jobs, y2, N); }));
        report("rows7 planar LPR 4/1, 128",
               timeit([&] { k_fp4_rows7<36, 4, 128><<<dim3(2 * FF_C / 32, nj), 128>>>(gp, gu_e, 2 * FF_C, x, N, 0, esc, jobs, y2, 2 * FF_C); }),
               timeit([&] { k_fp4_rows7<9, 1, 128><<<dim3(N / 128, nj), 128>>>(dp, dn_e, N, h, FF_C, 1, esc, jobs, y2, N); }));
        report("rows7 planar LPR 4/1, 256",
               timeit([&] { k_fp4_rows7<36, 4, 256><<<dim3(2 * FF_C / 64, nj), 256>>>(gp, gu_e, 2 * FF_C, x, N, 0, esc, jobs, y2, 2 * FF_C); }),
               timeit([&] { k_fp4_rows7<9, 1, 256><<<dim3(N / 256, nj), 256>>>(dp, dn_e, N, h, FF_C, 1, esc, jobs, y2, N); }));
        report("rows7 planar LPR 2/1, 128",
               timeit([&] { k_fp4_rows7<36, 2, 128><<<dim3(2 * FF_C / 64, nj), 128>>>(gp, gu_e, 2 * FF_C, x, N, 0, esc, jobs, y2, 2 * FF_C); }),
               timeit([&] { k_fp4_rows7<9, 1, 64><<<dim3(N / 64, nj), 64>>>(dp, dn_e, N, h, FF_C, 1, esc, jobs, y2, N); }));
        std::vector<float> a((size_t) nj * 2 * FF_C), b((size_t) nj * 2 * FF_C);
        k_fp4_rows3<36, 8><<<dim3(2 * FF_C / 32, nj), 128>>>(gu, gu_e, 2 * FF_C, x, N, 0, esc, jobs, y, 2 * FF_C);
        k_fp4_rows6<36, 8><<<dim3(2 * FF_C / 32, nj), 128>>>(gp, gu_e, 2 * FF_C, x, N, 0, esc, jobs, y2, 2 * FF_C);
        CK(cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * (double) (a[i] - b[i]); den += (double) a[i] * a[i]; }
        std::printf("rows6 proti rows3 (gate_up): relativna razlika %.2e\n", std::sqrt(num / (den + 1e-30)));
    }
    // the same bits as rows3? (both add a row's blocks in another order: compare loosely)
    std::vector<float> a((size_t) nj * N), b((size_t) nj * N);
    k_fp4_rows3<9, 32><<<dim3(N / 128, nj), 128>>>(dn, dn_e, N, h, FF_C, 1, esc, jobs, y, N);
    k_fp4_rows4<9, 1><<<dim3(N / 128, nj), 128>>>(dn, dn_e, N, h, FF_C, 1, esc, jobs, y2, N);
    CK(cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * (double) (a[i] - b[i]); den += (double) a[i] * a[i]; }
    std::printf("rows4 proti rows3 (down): relativna razlika %.2e\n", std::sqrt(num / (den + 1e-30)));
}
template <class Time>
static void bench_gemv(const int8_t* q, const __half* d, int rows, int cols, const float* x, float* y, double gb, Time timeit) {
    Segs sg{};
    sg.n = 1;
    sg.dst[0] = y;
    float us;
    if (cols == N) {
        us = timeit([&] { k_gemv8_t1b<5, 2><<<(rows + 15) / 16, 256>>>(q, d, rows, cols, x, sg); });
        std::printf("  t1b<5,2>                     %7.1f us %6.0f GB/s\n", us, gb * 1e6 / us);
        us = timeit([&] { k_gemv8_t1b<5, 1><<<(rows + 7) / 8, 256>>>(q, d, rows, cols, x, sg); });
        std::printf("  t1b<5,1>                     %7.1f us %6.0f GB/s\n", us, gb * 1e6 / us);
    } else {
        us = timeit([&] { k_gemv8_t1b<3, 2><<<(rows + 15) / 16, 256>>>(q, d, rows, cols, x, sg); });
        std::printf("  t1b<3,2>                     %7.1f us %6.0f GB/s\n", us, gb * 1e6 / us);
        us = timeit([&] { k_gemv8_t1b<3, 1><<<(rows + 7) / 8, 256>>>(q, d, rows, cols, x, sg); });
        std::printf("  t1b<3,1>                     %7.1f us %6.0f GB/s\n", us, gb * 1e6 / us);
    }
}

}  // namespace ink2
