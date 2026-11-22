// cpu_moe_bench.c (2026-11-15 night): how fast can one Xeon Bronze 3106 socket compute Flash-Next Q4 experts
// (gate|up 1280 x 2560 Q4_K, down 2560 x 640 Q5_1, as UD-Q4_K_XL stores most layers) for T rows, with ggml's CPU dots?
// Idea: compute some missing experts on the CPU instead of copying them over PCIe (IDEJE 7, re-checked for Q4).
// usage: cpu_moe_bench E T THREADS MODE [REPS]
//   E experts in the pool (distinct memory, so the reads come from RAM), T rows (tokens) per expert,
//   MODE 0: every expert split over all threads by rows (latency of one expert), 1: threads take whole experts (throughput)
#include "ggml.h"
#include "ggml-cpu.h"
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define H 2560
#define F 640

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s E T THREADS MODE [REPS]\n", argv[0]); return 1; }
    const int E = atoi(argv[1]), T = atoi(argv[2]), NT = atoi(argv[3]), MODE = atoi(argv[4]);
    const int REPS = argc > 5 ? atoi(argv[5]) : 3;
    ggml_cpu_init();
    const struct ggml_type_traits_cpu * tq4 = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    const struct ggml_type_traits_cpu * tq5 = ggml_get_type_traits_cpu(GGML_TYPE_Q5_1);
    const size_t gu_row = ggml_row_size(GGML_TYPE_Q4_K, H), dn_row = ggml_row_size(GGML_TYPE_Q5_1, F);
    const size_t gu_sz = gu_row * 2 * F, dn_sz = dn_row * H, ex_sz = gu_sz + dn_sz;
    const size_t a_row = ggml_row_size(tq4->vec_dot_type, H), h_row = ggml_row_size(tq5->vec_dot_type, F);
    fprintf(stderr, "expert %.3f MiB (gate|up %zu B/row, down %zu B/row), dot types %s / %s, pool %.1f MiB\n",
            ex_sz / 1048576.0, gu_row, dn_row, ggml_type_name(tq4->vec_dot_type), ggml_type_name(tq5->vec_dot_type),
            E * ex_sz / 1048576.0);
    // one expert quantized from random weights, copied to all E (distinct memory, same values)
    float * w = malloc(sizeof(float) * (size_t) H * 2 * F);
    srand(1);
    for (size_t i = 0; i < (size_t) H * 2 * F; i++) w[i] = ((rand() / (float) RAND_MAX) - 0.5f) * 0.05f;
    uint8_t * pool = aligned_alloc(64, ((E * ex_sz) + 63) / 64 * 64);
    ggml_quantize_chunk(GGML_TYPE_Q4_K, w, pool, 0, 2 * F, H, NULL);
    ggml_quantize_chunk(GGML_TYPE_Q5_1, w, pool + gu_sz, 0, H, F, NULL);
    // touch every page from the threads that will read it (first touch under numactl --membind keeps it local anyway)
    #pragma omp parallel for num_threads(NT) schedule(static)
    for (int e = 1; e < E; e++) memcpy(pool + e * ex_sz, pool, ex_sz);
    // activations
    float * x = malloc(sizeof(float) * H * T);
    for (int i = 0; i < H * T; i++) x[i] = ((rand() / (float) RAND_MAX) - 0.5f);
    uint8_t * xa = aligned_alloc(64, (a_row * T + 63) / 64 * 64);
    // the activations by the dot type's own quantizer (a weight type's from_float makes that weight type: the first runs printed nan)
    const struct ggml_type_traits_cpu * tx = ggml_get_type_traits_cpu(tq4->vec_dot_type), * th = ggml_get_type_traits_cpu(tq5->vec_dot_type);
    for (int t = 0; t < T; t++) tx->from_float(x + t * H, xa + t * a_row, H);
    // per thread scratch for MODE 1, shared for MODE 0
    float * g = malloc(sizeof(float) * 2 * F * T * NT);
    float * hbuf = malloc(sizeof(float) * F * T * NT);
    uint8_t * ha = aligned_alloc(64, (h_row * T * NT + 63) / 64 * 64);
    float * y = malloc(sizeof(float) * H * T * NT);
    double best = 1e30;
    for (int rep = 0; rep < REPS; rep++) {
        const double t0 = now();
        if (MODE == 0) {
            #pragma omp parallel num_threads(NT)
            {
                for (int e = 0; e < E; e++) {
                    const uint8_t * W = pool + e * ex_sz;
                    #pragma omp for schedule(static)
                    for (int r = 0; r < 2 * F; r++)
                        for (int t = 0; t < T; t++) tq4->vec_dot(H, g + t * 2 * F + r, 0, W + r * gu_row, 0, xa + t * a_row, 0, 1);
                    #pragma omp for schedule(static)
                    for (int t = 0; t < T; t++) {
                        for (int c = 0; c < F; c++) { const float a = g[t * 2 * F + c]; hbuf[t * F + c] = a / (1.0f + expf(-a)) * g[t * 2 * F + F + c]; }
                        th->from_float(hbuf + t * F, ha + t * h_row, F);
                    }
                    #pragma omp for schedule(static)
                    for (int r = 0; r < H; r++)
                        for (int t = 0; t < T; t++) tq5->vec_dot(F, y + t * H + r, 0, W + gu_sz + r * dn_row, 0, ha + t * h_row, 0, 1);
                }
            }
        } else {
            #pragma omp parallel for num_threads(NT) schedule(dynamic, 1)
            for (int e = 0; e < E; e++) {
                const int me = omp_get_thread_num();
                const uint8_t * W = pool + e * ex_sz;
                float * gg = g + me * 2 * F * T, * hh = hbuf + me * F * T, * yy = y + me * H * T;
                uint8_t * hq = ha + me * h_row * T;
                for (int r = 0; r < 2 * F; r++)
                    for (int t = 0; t < T; t++) tq4->vec_dot(H, gg + t * 2 * F + r, 0, W + r * gu_row, 0, xa + t * a_row, 0, 1);
                for (int t = 0; t < T; t++) {
                    for (int c = 0; c < F; c++) { const float a = gg[t * 2 * F + c]; hh[t * F + c] = a / (1.0f + expf(-a)) * gg[t * 2 * F + F + c]; }
                    th->from_float(hh + t * F, hq + t * h_row, F);
                }
                for (int r = 0; r < H; r++)
                    for (int t = 0; t < T; t++) tq5->vec_dot(F, yy + t * H + r, 0, W + gu_sz + r * dn_row, 0, hq + t * h_row, 0, 1);
            }
        }
        const double dt = now() - t0;
        if (dt < best) best = dt;
    }
    // reference for expert 0, token 0 (MODE 1 thread 0 did expert 0 first only by chance: recompute it here in f32 from dequantized rows)
    {
        const struct ggml_type_traits * d4 = ggml_get_type_traits(GGML_TYPE_Q4_K), * d5 = ggml_get_type_traits(GGML_TYPE_Q5_1);
        float row[H], gref[2 * F], href[F], err = 0, mag = 0;
        for (int r = 0; r < 2 * F; r++) { d4->to_float(pool + r * gu_row, row, H); double s = 0; for (int k = 0; k < H; k++) s += row[k] * x[k]; gref[r] = (float) s; }
        for (int q = 0; q < F; q++) href[q] = gref[q] / (1.0f + expf(-gref[q])) * gref[F + q];
        uint8_t hq0[1024]; float g0[2 * F], h0[F], y0;
        for (int r = 0; r < 2 * F; r++) tq4->vec_dot(H, g0 + r, 0, pool + r * gu_row, 0, xa, 0, 1);
        for (int q = 0; q < F; q++) h0[q] = g0[q] / (1.0f + expf(-g0[q])) * g0[F + q];
        th->from_float(h0, hq0, F);
        for (int r = 0; r < H; r++) {
            float rw[F]; d5->to_float(pool + gu_sz + r * dn_row, rw, F); double s = 0; for (int q = 0; q < F; q++) s += rw[q] * href[q];
            tq5->vec_dot(F, &y0, 0, pool + gu_sz + r * dn_row, 0, hq0, 0, 1);
            if (fabs(y0 - s) > err) err = fabs(y0 - s);
            if (fabs(s) > mag) mag = fabs(s);
        }
        printf("check expert 0 token 0: max |cpu - f32| %.3g of max |f32| %.3g\n", err, mag);
    }
    double chk = 0; for (int i = 0; i < H * T; i++) chk += y[i];
    printf("E %d T %d threads %d mode %d: %.1f us per expert, %.1f GB/s of weights, %.1f GMAC/s (chk %.3g)\n", E, T, NT, MODE,
           best / E * 1e6, E * ex_sz / best / 1e9, (double) E * T * 3 * H * F / best / 1e9, chk);
    return 0;
}
