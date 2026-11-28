// bench_dec.cu (2026-11-21, p4): the writing kernels alone on one card, random weights in the engine's layouts (NVFP4 expert halves,
// Q8S matrices), times by CUDA events, 200 runs each: GB/s of the weights read. No model, no second card.
//   nvcc -O2 -std=c++17 -arch=sm_60 -ccbin g++-13 -o bench_dec bench_dec.cu && ./bench_dec
#include "ink2_kern.cuh"
#include "ink2_bench.cuh"

#include <algorithm>
#include <random>
#include <vector>

using namespace ink2;

template <class F> static float timeit(F f, int n = 200) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int i = 0; i < 5; ++i) f();
    CK(cudaEventRecord(a));
    for (int i = 0; i < n; ++i) f();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms * 1000.0f / n;   // us
}
static void* rnd(size_t bytes, unsigned seed) {
    std::vector<uint8_t> h(bytes);
    std::mt19937 g(seed);
    for (auto& x : h) x = (uint8_t) (g() & 0x77);   // small E4M3 scales and codes: no inf
    void* d;
    CK(cudaMalloc(&d, bytes));
    CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
    return d;
}

int main() {
    const size_t gu_e = (size_t) 2 * FF_C * (N / 64) * 36, dn_e = (size_t) N * (FF_C / 64) * 36;
    const uint8_t* gu = (const uint8_t*) rnd(gu_e * NE, 1);
    const uint8_t* dn = (const uint8_t*) rnd(dn_e * NE, 2);
    float *x, *h, *y, *y2, *esc;
    CK(cudaMalloc(&x, N * 4 * MAXT));
    CK(cudaMalloc(&h, (size_t) MAXT * TOPK * FF_C * 4));
    CK(cudaMalloc(&y, (size_t) MAXT * TOPK * N * 4));
    CK(cudaMalloc(&y2, (size_t) MAXT * TOPK * N * 4));
    CK(cudaMalloc(&esc, NE * 4));
    {
        std::vector<float> hx(N * MAXT), hh((size_t) MAXT * TOPK * FF_C), he(NE, 1.0f);
        std::mt19937 g(3);
        std::uniform_real_distribution<float> u(-1, 1);
        for (auto& v : hx) v = u(g);
        for (auto& v : hh) v = u(g);
        CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(h, hh.data(), hh.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(esc, he.data(), NE * 4, cudaMemcpyHostToDevice));
    }
    EJob hj[TOPK];
    for (int k = 0; k < TOPK; ++k) hj[k] = EJob{0, k, 7 + 13 * k};
    EJob* jobs;
    CK(cudaMalloc(&jobs, sizeof hj));
    CK(cudaMemcpy(jobs, hj, sizeof hj, cudaMemcpyHostToDevice));
    const int nj = TOPK;
    const double gb_gu = (double) nj * gu_e / 1e9 * 1e6, gb_dn = (double) nj * dn_e / 1e9 * 1e6;   // GB * 1e6 (us -> GB/s)
    auto report = [&](const char* name, float us_gu, float us_dn) {
        std::printf("%-28s gate_up %7.1f us %6.0f GB/s | down %7.1f us %6.0f GB/s\n", name, us_gu, gb_gu / us_gu, us_dn, gb_dn / us_dn);
    };
    report("rows2 (p4a)",
           timeit([&] { k_fp4_rows2<<<dim3(2 * FF_C / 8, nj), 256, N * 4>>>(gu, gu_e, 2 * FF_C, N, x, N, 0, esc, jobs, y, 2 * FF_C); }),
           timeit([&] { k_fp4_rows2<<<dim3(N / 8, nj), 256, FF_C * 4>>>(dn, dn_e, N, FF_C, h, FF_C, 1, esc, jobs, y, N); }));
    report("rows3 (p4b)",
           timeit([&] { k_fp4_rows3<36, 8><<<dim3(2 * FF_C / 32, nj), 128>>>(gu, gu_e, 2 * FF_C, x, N, 0, esc, jobs, y, 2 * FF_C); }),
           timeit([&] { k_fp4_rows3<9, 32><<<dim3(N / 128, nj), 128>>>(dn, dn_e, N, h, FF_C, 1, esc, jobs, y, N); }));
    bench_more(gu, dn, gu_e, dn_e, x, h, y, y2, esc, jobs, nj, report, [&](auto f) { return timeit(f); });
    // Q8S T = 1 GEMV: a layer's concatenated attention matrix, wo, the output head half
    struct M { const char* name; int rows, cols; };
    for (M m : {M{"cat 4332 x 2304", 4332, N}, M{"wo 2304 x 1536", N, QC}, M{"head 100512 x 2304", NV_C, N}}) {
        const int8_t* q = (const int8_t*) rnd((size_t) m.rows * m.cols, 5);
        const __half* d = (const __half*) rnd((size_t) m.rows * (m.cols / 32) * 2, 6);
        Segs sg{};
        sg.n = 1;
        sg.dst[0] = y;
        const int blocks = (m.rows + G1_BR - 1) / G1_BR, nch = (m.cols + 511) / 512;
        const int S = blocks >= 112 ? 1 : std::min(nch, (112 + blocks - 1) / blocks);
        const float us = timeit([&] { k_gemv8_t1<<<dim3(blocks, S), 256>>>(q, d, m.rows, m.cols, x, sg, S); });
        const double gb = (double) m.rows * m.cols * (1 + 2.0 / 32) / 1e9;
        std::printf("gemv8_t1 %-20s %7.1f us %6.0f GB/s\n", m.name, us, gb * 1e6 / us);
        bench_gemv(q, d, m.rows, m.cols, x, y, gb, [&](auto f) { return timeit(f); });
        CK(cudaFree((void*) q));
        CK(cudaFree((void*) d));
    }
    return 0;
}
